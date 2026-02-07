import csv
import math
import os
from typing import Dict, Iterable, Iterator, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


def _norm_from_params(params: Iterable[torch.nn.Parameter], use_grad: bool = False) -> Dict[str, float]:
    l2_sum = 0.0
    linf = 0.0
    count = 0
    for p in params:
        data = p.grad if use_grad else p.data
        if data is None:
            continue
        data_f = data.float()
        l2_sum += torch.sum(data_f * data_f).item()
        linf = max(linf, torch.max(torch.abs(data_f)).item())
        count += data.numel()
    l2 = math.sqrt(l2_sum) if l2_sum > 0 else 0.0
    rms = math.sqrt(l2_sum / count) if count > 0 else 0.0
    return {"l2": l2, "linf": linf, "rms": rms}


def _iter_params(model: torch.nn.Module, exclude_embeddings: bool = False) -> Iterator[torch.nn.Parameter]:
    """
    Yield parameters, optionally skipping embeddings (including tied lm_head weight).
    """
    skipped: set[int] = set()
    for name, param in model.named_parameters():
        if exclude_embeddings and (
            "token_emb" in name
            or "embedding" in name
            or name.endswith("lm_head.weight")
        ):
            skipped.add(id(param))
            continue
        if exclude_embeddings and id(param) in skipped:
            continue
        yield param


def grad_norms(model: torch.nn.Module, per_layer: bool = True) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    norms = _norm_from_params(_iter_params(model, exclude_embeddings=True), True)
    metrics["grad/global_rms"] = norms["rms"]
    if per_layer and hasattr(model, "blocks"):
        for idx, block in enumerate(getattr(model, "blocks")):
            norms = _norm_from_params(block.parameters(), True)
            metrics[f"grad/layer.{idx}.rms"] = norms["rms"]
    return metrics


def weight_norms(model: torch.nn.Module, per_layer: bool = True) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    global_norms = _norm_from_params(_iter_params(model, exclude_embeddings=True), False)
    metrics["weights/global_rms"] = global_norms["rms"]
    if per_layer and hasattr(model, "blocks"):
        for idx, block in enumerate(getattr(model, "blocks")):
            norms = _norm_from_params(block.parameters(), False)
            metrics[f"weights/layer.{idx}.rms"] = norms["rms"]
    return metrics


class MetricsLogger:
    def __init__(self, csv_path: str, tb_dir: Optional[str] = None):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self.file = open(csv_path, "w", newline="", encoding="utf-8")
        self.writer: Optional[csv.DictWriter] = None
        self.header: List[str] = []
        self.tb = SummaryWriter(tb_dir) if tb_dir else None

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        metrics = {"step": step, **metrics}
        if self.writer is None:
            self.header = list(metrics.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.header)
            self.writer.writeheader()
        # ensure consistent columns
        for key in self.header:
            if key not in metrics:
                metrics[key] = None
        self.writer.writerow(metrics)
        self.file.flush()

        if self.tb:
            for k, v in metrics.items():
                if k == "step" or v is None:
                    continue
                self.tb.add_scalar(k, v, global_step=step)

    def close(self) -> None:
        if self.tb:
            self.tb.flush()
            self.tb.close()
        self.file.close()


def plot_run_curves(csv_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    x = df["step"]
    plots = {
        "loss_train.png": [("train/loss", "Train Loss")],
        "loss_val.png": [("val/loss", "Val Loss")],
        "grad_global_rms.png": [("grad/global_rms", "Grad RMS")],
        "weights_global_rms.png": [("weights/global_rms", "Weights RMS")],
    }
    for filename, items in plots.items():
        plt.figure()
        plotted_cols: List[str] = []
        plotted_values: List[float] = []
        for col, label in items:
            if col not in df:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            plt.plot(df.loc[series.index, "step"], series, label=label)
            plotted_cols.append(col)
            plotted_values.extend(float(v) for v in series.tolist())
        if plotted_cols and any("loss" in col for col in plotted_cols):
            positive = [v for v in plotted_values if v > 0]
            if positive and (max(positive) / max(min(positive), 1e-12)) >= 50.0:
                plt.yscale("log")
        plt.xlabel("Step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()
