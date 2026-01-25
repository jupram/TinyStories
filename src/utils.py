import argparse
import json
import os
import random
import time
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update mapping."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_override(override: str) -> Tuple[List[str], Any]:
    """
    Parse override strings like 'model.d_model=256' into (['model','d_model'], 256)
    using yaml parsing for the value.
    """
    if "=" not in override:
        raise ValueError(f"Invalid override {override}, expected key=value")
    key, raw_val = override.split("=", 1)
    path = key.split(".")
    value = yaml.safe_load(raw_val)
    return path, value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    for ov in overrides:
        path, value = parse_override(ov)
        cursor = cfg
        for p in path[:-1]:
            if p not in cursor or not isinstance(cursor[p], dict):
                cursor[p] = {}
            cursor = cursor[p]
        cursor[path[-1]] = value
    return cfg


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = apply_overrides(data, overrides)
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def maybe_enable_deterministic(enabled: bool) -> None:
    if not enabled:
        return
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def resolve_precision(precision: str, device: torch.device) -> torch.dtype:
    if precision.lower() == "bf16":
        return torch.bfloat16
    if precision.lower() in {"fp16", "float16", "16"}:
        return torch.float16
    return torch.float32


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def default_run_name(model_name: str) -> str:
    return f"{model_name}-{now_timestamp()}"


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def create_scheduler(optimizer: torch.optim.Optimizer, warmup: int, max_steps: int, min_lr_ratio: float = 0.1):
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, max_steps - warmup))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_torch_compile(model: torch.nn.Module, use_compile: bool) -> torch.nn.Module:
    if use_compile and hasattr(torch, "compile"):
        return torch.compile(model)  # type: ignore[attr-defined]
    return model


def check_flash_attention(enabled: bool) -> bool:
    if not enabled:
        return False
    return torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled()


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyStories Transformer training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values, e.g. training.batch_size=16 model.d_model=512",
    )
    return parser
