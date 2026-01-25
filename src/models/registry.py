from typing import Callable, Dict

from .baseline import DecoderModel
from .variants import build_gqa_model, build_residual_scaling_model

MODEL_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    def decorator(fn: Callable):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


@register("baseline")
def build_baseline(cfg: dict, use_flash: bool) -> DecoderModel:
    model_cfg = cfg["model"]
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
    )


@register("variant_residual_scaling")
def build_residual(cfg: dict, use_flash: bool):
    return build_residual_scaling_model(cfg, use_flash)


@register("variant_attention_change")
def build_attention_variant(cfg: dict, use_flash: bool):
    return build_gqa_model(cfg, use_flash)


def build_model(cfg: dict, use_flash: bool = False):
    name = cfg["model"]["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name {name}")
    return MODEL_REGISTRY[name](cfg, use_flash)
