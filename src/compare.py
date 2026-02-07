import argparse
import json
import os
from glob import glob
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd


def load_run(run_dir: str) -> Dict:
    metrics_path = os.path.join(run_dir, "metrics.csv")
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"No metrics.csv in {run_dir}")
    df = pd.read_csv(metrics_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {"name": os.path.basename(run_dir), "df": df, "config": cfg}


def plot_overlay(runs: List[Dict], column: str, ylabel: str, out_path: str) -> None:
    fig, ax = plt.subplots()
    plotted_values: List[float] = []
    for run in runs:
        df = run["df"]
        if column not in df:
            continue
        series = df[["step", column]].dropna()
        if series.empty:
            continue
        ax.plot(series["step"], series[column], label=run["name"], marker="o", markersize=3, linewidth=1.5)
        plotted_values.extend(float(v) for v in series[column].tolist())

    if "loss" in column:
        positive = [v for v in plotted_values if v > 0]
        if positive and (max(positive) / max(min(positive), 1e-12)) >= 50.0:
            ax.set_yscale("log")

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    # Avoid confusing offset notation (e.g., "1e-7 + 1" that makes small deltas look negative)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summarize_run(run: Dict, last_n: int) -> Dict[str, float]:
    df = run["df"]
    metrics = {
        "run_name": run["name"],
        "model": run["config"]["model"]["name"],
        "steps": int(df["step"].max()),
    }
    if "train/loss" in df:
        metrics["final_train_loss"] = float(df["train/loss"].iloc[-1])
    if "val/loss" in df:
        metrics["best_val_loss"] = float(df["val/loss"].min())
    if "tokens_seen" in df:
        metrics["tokens_seen"] = float(df["tokens_seen"].max())
    tail = df.tail(last_n)
    for col, key in [
        ("grad/global_rms", "mean_grad_rms"),
        ("weights/global_rms", "mean_weight_rms"),
    ]:
        if col in tail:
            metrics[key] = float(tail[col].mean())
    return metrics


def _format_summary_table(df: pd.DataFrame) -> str:
    """Return a nicely formatted summary table (Markdown when available)."""
    column_order = [
        "run_name",
        "model",
        "steps",
        "final_train_loss",
        "best_val_loss",
        "tokens_seen",
        "mean_grad_rms",
        "mean_weight_rms",
    ]
    cols = [c for c in column_order if c in df.columns]
    fmt = df[cols].copy()

    for col in ["final_train_loss", "best_val_loss", "mean_grad_rms", "mean_weight_rms"]:
        if col in fmt:
            fmt[col] = fmt[col].map(lambda x: f"{x:.4f}")
    if "tokens_seen" in fmt:
        fmt["tokens_seen"] = fmt["tokens_seen"].map(lambda x: f"{int(x):,}")
    if "steps" in fmt:
        fmt["steps"] = fmt["steps"].astype(int)

    try:
        return fmt.to_markdown(index=False)  # pandas uses tabulate if installed
    except Exception:
        return fmt.to_string(index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing run folders")
    parser.add_argument("--out_dir", type=str, default="comparisons", help="Where to save comparison outputs")
    parser.add_argument("--last_n", type=int, default=100, help="Number of tail steps for norm averages")
    args = parser.parse_args()

    run_dirs = [p for p in glob(os.path.join(args.runs_dir, "*")) if os.path.isdir(p)]
    runs = [load_run(rd) for rd in run_dirs]
    os.makedirs(args.out_dir, exist_ok=True)

    plot_overlay(runs, "train/loss", "Train Loss", os.path.join(args.out_dir, "loss_train.png"))
    plot_overlay(runs, "val/loss", "Val Loss", os.path.join(args.out_dir, "loss_val.png"))
    plot_overlay(runs, "grad/global_rms", "Grad RMS", os.path.join(args.out_dir, "grad_global_rms.png"))
    plot_overlay(runs, "weights/global_rms", "Weights RMS", os.path.join(args.out_dir, "weights_global_rms.png"))

    summary_rows = [summarize_run(run, args.last_n) for run in runs]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.out_dir, "summary_table.csv"), index=False)
    table_str = _format_summary_table(summary_df)
    with open(os.path.join(args.out_dir, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(table_str)


if __name__ == "__main__":
    main()
