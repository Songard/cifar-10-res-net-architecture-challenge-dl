"""Read-only helpers for the results notebook and report figures: logs, checkpoints, held-out predictions."""

import json
from pathlib import Path

import numpy as np
import torch

from train import Data, make_model, predict_logits

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_metrics(run):
    return [json.loads(line) for line in (Path("runs") / run / "metrics.jsonl").read_text().splitlines() if line.strip()]


def load_config(run):
    return json.loads((Path("runs") / run / "config.json").read_text())


def load_submissions(path="artifacts/kaggle_submissions.jsonl"):
    """Kaggle receipts recorded after each upload (public/private scores come from Kaggle, never computed here)."""
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_checkpoint_model(path, device, weights=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = weights or ("ema" if "ema" in checkpoint else "model")
    model = make_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint[weights])
    return model.eval(), checkpoint


@torch.no_grad()
def validation_predictions(model, device, tta="none", batch_size=500):
    """Predicted and true labels on the fixed 5,000-image validation split, preprocessed exactly as in training."""
    data = Data(Path("data"), device, "batched", False)
    predicted, labels = [], []
    for x, y in data.batches(data.valid, batch_size):
        predicted.append(predict_logits(model, x, tta).argmax(1).cpu())
        labels.append(y.cpu())
    return torch.cat(predicted).numpy(), torch.cat(labels).numpy()


def confusion(predicted, labels):
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (labels, predicted), 1)
    return matrix
