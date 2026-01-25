import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (self.weight * x).type_as(x)


def rotary_cache(max_seq_len: int, head_dim: int, base: int, device, dtype):
    # based on GPT-NeoX style RoPE
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, heads, T, D)
    cos = cos[: q.shape[2]].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = sin[: q.shape[2]].unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        rope_base: int = 10000,
        use_flash: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.use_flash = use_flash
        self.rope_base = rope_base
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
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        self._maybe_refresh_cache(T, x.device, x.dtype)
        q, k = apply_rotary(q, k, self.cos_cache, self.sin_cache)

        sdpa_kwargs = {
            "dropout_p": self.dropout if self.training else 0.0,
            "is_causal": True,
        }
        if self.use_flash and x.is_cuda and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                attn = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w12 = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.out = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.dropout(self.out(F.silu(x1) * x2))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        rope_base: int,
        use_flash: bool,
        resid_scale: float = 1.0,
        learnable_resid: bool = False,
    ):
        super().__init__()
        head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, head_dim, attn_dropout, rope_base, use_flash)
        self.norm2 = RMSNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = SwiGLU(d_model, hidden, dropout)
        self.resid_scale_attn = nn.Parameter(torch.ones(1) * resid_scale) if learnable_resid else resid_scale
        self.resid_scale_mlp = nn.Parameter(torch.ones(1) * resid_scale) if learnable_resid else resid_scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.norm1(x))
        x = x + self.dropout(y) * self.resid_scale_attn
        y = self.mlp(self.norm2(x))
        x = x + self.dropout(y) * self.resid_scale_mlp
        return x


class DecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        rope_base: int,
        max_seq_len: int,
        use_flash: bool = False,
        resid_scale: float = 1.0,
        learnable_resid: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    rope_base=rope_base,
                    use_flash=use_flash,
                    resid_scale=resid_scale,
                    learnable_resid=learnable_resid,
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
        logits = self.lm_head(x)
        return logits
