from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import RMSNorm, DecoderModel, SwiGLU, apply_rotary, rotary_cache


class GQAMultiHeadAttention(nn.Module):
    """
    Grouped-query / multi-query attention variant where K/V heads < Q heads.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        dropout: float,
        rope_base: int,
        use_flash: bool = False,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope_base = rope_base
        self.use_flash = use_flash
        self.register_buffer("cos_cache", None, persistent=False)
        self.register_buffer("sin_cache", None, persistent=False)

    def _maybe_refresh_cache(self, seq_len: int, device, dtype):
        if (
            self.cos_cache is None
            or self.cos_cache.size(0) < seq_len
            or self.cos_cache.device != device
            or self.cos_cache.dtype != dtype
        ):
            cos, sin = rotary_cache(seq_len, self.head_dim, self.rope_base, device, dtype)
            self.cos_cache = cos
            self.sin_cache = sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        self._maybe_refresh_cache(T, x.device, x.dtype)
        q, k = apply_rotary(q, k, self.cos_cache, self.sin_cache)

        repeat_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        sdpa_kwargs = {
            "dropout_p": self.dropout if self.training else 0.0,
            "is_causal": True,
        }
        if self.use_flash and x.is_cuda and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                attn = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(attn)


class TransformerBlockGQA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        rope_base: int,
        resid_scale: float,
        use_flash: bool,
    ):
        super().__init__()
        head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAMultiHeadAttention(
            d_model, n_heads, n_kv_heads, head_dim, attn_dropout, rope_base, use_flash=use_flash
        )
        self.norm2 = RMSNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = SwiGLU(d_model, hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        self.resid_scale = resid_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.norm1(x))
        x = x + self.dropout(y) * self.resid_scale
        y = self.mlp(self.norm2(x))
        x = x + self.dropout(y) * self.resid_scale
        return x


class DecoderModelGQA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        rope_base: int,
        max_seq_len: int,
        resid_scale: float = 1.0,
        use_flash: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockGQA(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    rope_base=rope_base,
                    resid_scale=resid_scale,
                    use_flash=use_flash,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_residual_scaling_model(cfg: dict, use_flash: bool) -> DecoderModel:
    model_cfg = cfg["model"]
    resid = float(model_cfg.get("resid_scale", 1.0))
    learnable = bool(model_cfg.get("learnable_resid_scale", False))
    return DecoderModel(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        attn_dropout=model_cfg.get("attn_dropout", 0.0),
        rope_base=model_cfg.get("rope_base", 10000),
        max_seq_len=model_cfg["seq_len"],
        use_flash=use_flash,
        resid_scale=resid,
        learnable_resid=learnable,
    )


def build_gqa_model(cfg: dict, use_flash: bool) -> DecoderModelGQA:
    model_cfg = cfg["model"]
    n_kv_heads = model_cfg.get("n_kv_heads", 1)
    return DecoderModelGQA(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=n_kv_heads,
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        attn_dropout=model_cfg.get("attn_dropout", 0.0),
        rope_base=model_cfg.get("rope_base", 10000),
        max_seq_len=model_cfg["seq_len"],
        resid_scale=model_cfg.get("resid_scale", 1.0),
        use_flash=use_flash,
    )
