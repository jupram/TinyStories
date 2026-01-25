import json
import os
from itertools import cycle
from typing import Dict

import torch
import torch.nn.functional as F
from torch import amp
from torch.optim import AdamW
from tqdm import trange

from . import data
from .metrics import MetricsLogger, grad_norms, plot_run_curves, weight_norms
from .models import build_model
from .utils import (
    apply_overrides,
    check_flash_attention,
    count_parameters,
    create_scheduler,
    default_run_name,
    ensure_dir,
    load_config,
    maybe_enable_deterministic,
    maybe_torch_compile,
    resolve_precision,
    select_device,
    set_seed,
    write_json,
)


def add_train_defaults(cfg: Dict) -> Dict:
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "auto")
    cfg.setdefault("precision", "bf16" if torch.cuda.is_available() else "fp32")
    cfg.setdefault("torch_compile", False)
    cfg.setdefault("use_flash", True)
    cfg.setdefault("deterministic", True)
    cfg.setdefault("output_dir", "runs")
    cfg.setdefault("run_name", None)
    cfg.setdefault("log_every", 50)
    cfg.setdefault("eval_every", 500)
    cfg.setdefault("save_every", None)
    cfg.setdefault("max_steps", 1000)
    cfg.setdefault("grad_accum_steps", 1)
    cfg.setdefault("clip_grad", None)
    cfg.setdefault("batch_size", 32)
    cfg.setdefault("val_batch_size", cfg["batch_size"])
    cfg.setdefault("max_eval_batches", 100)
    cfg.setdefault("max_train_samples", None)
    cfg.setdefault("max_val_samples", None)

    cfg.setdefault(
        "optimizer",
        {
            "lr": 3e-4,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
    )
    cfg.setdefault(
        "scheduler",
        {"warmup_steps": 100, "min_lr_ratio": 0.1},
    )
    cfg.setdefault(
        "data",
        {
            "dataset_name": "roneneldan/TinyStories",
            "tokenizer_name": "gpt2",
            "seq_len": 256,
            "num_workers": 2,
            "val_split": "validation",
            "val_fraction": 0.01,
            "cache_dir": "data/cache",
            "dataset_variant": "full",  # options: full, small
            "small_train_samples": 50_000,
            "small_val_samples": 5_000,
        },
    )
    if "model" not in cfg:
        raise ValueError("config must contain a 'model' section")
    model_cfg = cfg["model"]
    model_cfg.setdefault("name", "baseline")
    model_cfg.setdefault("vocab_size", 50257)
    model_cfg.setdefault("d_model", 512)
    model_cfg.setdefault("n_layers", 8)
    model_cfg.setdefault("n_heads", 8)
    model_cfg.setdefault("mlp_ratio", 4.0)
    model_cfg.setdefault("dropout", 0.0)
    model_cfg.setdefault("attn_dropout", 0.0)
    model_cfg.setdefault("rope_base", 10000)
    model_cfg.setdefault("seq_len", cfg["data"].get("seq_len", 256))
    return cfg


def loss_fn(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # shift for next-token prediction
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def evaluate(model, loader, device, dtype, max_batches: int) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            with amp.autocast(device_type="cuda", enabled=dtype != torch.float32, dtype=dtype):
                logits = model(input_ids)
                loss = loss_fn(logits, input_ids)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)


def save_checkpoint(model, optimizer, scheduler, scaler, step: int, path: str) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
    }
    torch.save(state, path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="Override config values key=val")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    cfg = apply_overrides(raw_cfg, args.override)
    cfg = add_train_defaults(cfg)
    set_seed(cfg["seed"])

    device = select_device(cfg["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; no CUDA device is available.")
    maybe_enable_deterministic(cfg["deterministic"])

    tokenizer = data.load_tokenizer(cfg["data"]["tokenizer_name"])
    cfg["model"]["vocab_size"] = tokenizer.vocab_size
    # keep data seq length aligned with model
    cfg["data"]["seq_len"] = cfg["model"]["seq_len"]

    train_ds, val_ds = data.prepare_datasets(
        dataset_name=cfg["data"]["dataset_name"],
        tokenizer=tokenizer,
        seq_len=cfg["data"]["seq_len"],
        seed=cfg["seed"],
        val_split=cfg["data"].get("val_split", "validation"),
        val_fraction=cfg["data"].get("val_fraction", 0.01),
        max_train_samples=cfg.get("max_train_samples"),
        max_val_samples=cfg.get("max_val_samples"),
        num_proc=cfg["data"].get("num_workers", 1),
        cache_dir=cfg["data"].get("cache_dir", "data/cache"),
        dataset_variant=cfg["data"].get("dataset_variant", "full"),
        small_train_samples=cfg["data"].get("small_train_samples", 50_000),
        small_val_samples=cfg["data"].get("small_val_samples", 5_000),
    )

    train_loader = data.make_dataloader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 1),
        pin_memory=device.type == "cuda",
    )
    val_loader = data.make_dataloader(
        val_ds,
        batch_size=cfg["val_batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 1),
        pin_memory=device.type == "cuda",
    )

    use_flash = check_flash_attention(cfg["use_flash"])
    model = build_model(cfg, use_flash=use_flash).to(device)
    model = maybe_torch_compile(model, cfg["torch_compile"])

    dtype = resolve_precision(cfg["precision"], device)
    scaler = amp.GradScaler(device="cuda", enabled=(dtype == torch.float16))

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),
        eps=cfg["optimizer"]["eps"],
    )
    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup=cfg["scheduler"]["warmup_steps"],
        max_steps=cfg["max_steps"],
        min_lr_ratio=cfg["scheduler"].get("min_lr_ratio", 0.1),
    )

    run_name = cfg["run_name"] or default_run_name(cfg["model"]["name"])
    cfg["run_name"] = run_name
    run_dir = os.path.join(cfg["output_dir"], run_name)
    ensure_dir(run_dir)

    write_json(os.path.join(run_dir, "config.json"), cfg)

    metrics_logger = MetricsLogger(
        csv_path=os.path.join(run_dir, "metrics.csv"),
        tb_dir=os.path.join(run_dir, "tensorboard"),
    )

    best_val = float("inf")
    tokens_seen = 0
    grad_accum = cfg["grad_accum_steps"]
    log_every = cfg["log_every"]
    eval_every = cfg["eval_every"]
    save_every = cfg["save_every"]
    clip_grad = cfg["clip_grad"]

    train_iter = cycle(train_loader)

    for step in trange(1, cfg["max_steps"] + 1, desc="train"):
        optimizer.zero_grad()
        train_loss = 0.0
        for _ in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            with amp.autocast(device_type="cuda", enabled=dtype != torch.float32, dtype=dtype):
                logits = model(input_ids)
                loss = loss_fn(logits, input_ids) / grad_accum
            train_loss += loss.item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            tokens_seen += input_ids.numel()

        if clip_grad:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        metrics = {"train/loss": train_loss}

        if step % log_every == 0 or step == 1:
            metrics.update(grad_norms(model, per_layer=True))
            metrics.update(weight_norms(model, per_layer=True))

        if step % eval_every == 0 or step == 1:
            val_loss = evaluate(model, val_loader, device, dtype, cfg["max_eval_batches"])
            metrics["val/loss"] = val_loss
            best_val = min(best_val, val_loss)

        if step % log_every == 0 or step % eval_every == 0 or step == 1:
            metrics["lr"] = scheduler.get_last_lr()[0]
            metrics["tokens_seen"] = tokens_seen
            metrics_logger.log(step, metrics)

        if save_every and step % save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                path=os.path.join(run_dir, f"checkpoint_{step}.pt"),
            )

    # end training loop
    metrics_logger.close()

    summary = {
        "final_step": cfg["max_steps"],
        "final_train_loss": train_loss,
        "best_val_loss": best_val,
        "tokens_seen": tokens_seen,
        "parameters": count_parameters(model),
        "run_name": run_name,
        "model": cfg["model"]["name"],
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # generate quick plots
    plot_run_curves(os.path.join(run_dir, "metrics.csv"), run_dir)


if __name__ == "__main__":
    main()
