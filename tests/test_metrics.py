import os

import torch

from src.metrics import MetricsLogger, grad_norms, weight_norms
from src.models.baseline import DecoderModel


def small_model():
    return DecoderModel(
        vocab_size=100,
        d_model=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        rope_base=10000,
        max_seq_len=16,
        use_flash=False,
    )


def test_metric_columns(tmp_path):
    model = small_model()
    # run a tiny forward/backward to create grads
    x = torch.randint(0, 100, (2, 8))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, 100), x[:, 1:].reshape(-1))
    loss.backward()

    metrics = {}
    metrics.update(grad_norms(model))
    metrics.update(weight_norms(model))
    metrics["train/loss"] = loss.item()

    csv_path = os.path.join(tmp_path, "metrics.csv")
    logger = MetricsLogger(csv_path)
    logger.log(1, metrics)
    logger.close()

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    assert "train/loss" in header
    assert any(col.startswith("grad/layer.0") for col in header)
    assert "weights/global_rms" in header
