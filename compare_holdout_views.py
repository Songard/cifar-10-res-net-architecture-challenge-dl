"""Compare predeclared translation TTA and candidate agreement on the fixed holdout only."""
import argparse
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train import Data, make_model


def offsets(radius):
    return [(0, 0), (-radius, 0), (radius, 0), (0, -radius), (0, radius)]


def translated(x, dy, dx):
    """Positive offsets select a crop below/right of the reflect-padded center."""
    pad = max(abs(dy), abs(dx))
    if not pad:
        return x
    height, width = x.shape[-2:]
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return padded[:, :, pad + dy:pad + dy + height, pad + dx:pad + dx + width]


@torch.no_grad()
def batch_predictions(model, x, flip_only=False):
    base = (model(x) + model(x.flip(-1))) / 2
    result = {"flip": base.argmax(1)}
    if flip_only:
        return result
    for radius in (1, 2):
        total = base * 2
        for dy, dx in offsets(radius)[1:]:
            view = translated(x, dy, dx)
            total = total + model(view) + model(view.flip(-1))
        result[f"translate{radius}_flip"] = (total / 10).argmax(1)
    return result


def pair_statistics(a, b, labels):
    ca, cb = a == labels, b == labels
    ea, eb = ~ca, ~cb
    return {
        "count": len(labels), "prediction_agreement": float(np.mean(a == b)),
        "prediction_disagreements": int(np.sum(a != b)),
        "both_wrong": int(np.sum(ea & eb)), "only_a_correct": int(np.sum(ca & eb)),
        "only_b_correct": int(np.sum(ea & cb)),
        "error_jaccard": float(np.sum(ea & eb) / np.sum(ea | eb)) if np.any(ea | eb) else 1.0,
        "error_correlation": float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--flip-only", action="store_true", help="Audit existing flip predictions without exploring additional TTA")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    root = Path(__file__).resolve().parent
    data = Data(root / "data", device, "batch", True, workers=0)
    assert len(data.valid) == 5000 and len(data.train) == 45000
    assert not set(data.train.tolist()).intersection(data.valid.tolist())
    labels = data.labels[data.valid].numpy()
    protocols = {"flip": [(0, 0)], "translate1_flip": offsets(1), "translate2_flip": offsets(2)}
    if args.flip_only:
        protocols = {"flip": [(0, 0)]}
    report = {
        "device": str(device), "batch_size": args.batch_size, "weights": "ema",
        "hidden_test_read": False, "validation_held_out": True, "validation_count": 5000,
        "padding": "reflect", "aggregation": "equal mean of logits, crop then horizontal flip",
        "offsets_dy_dx": protocols, "views": {k: 2 * len(v) for k, v in protocols.items()},
        "data_audit": data.audit, "models": [], "pairs": [],
    }
    # The comparison set and the two translation protocols are fixed before any evaluation.
    (args.output / "protocol.json").write_text(json.dumps({**report, "checkpoints": [str(p) for p in args.checkpoints]}, indent=2) + "\n")
    baseline = {}
    for path in args.checkpoints:
        name = path.parent.name
        assert name not in baseline
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        assert config["full_train"] is False and checkpoint["data_audit"] == data.audit
        assert checkpoint["epoch"] == config["epochs"]
        model = make_model(config).to(device).to(memory_format=torch.channels_last).eval()
        model.load_state_dict(checkpoint["ema"], strict=True)
        count = sum(p.numel() for p in model.parameters())
        assert count <= 5000000
        predictions = {key: [] for key in protocols}
        started = time.perf_counter()
        for x, _ in data.batches(data.valid, args.batch_size):
            for key, value in batch_predictions(model, x, args.flip_only).items():
                predictions[key].append(value.cpu().numpy())
        predictions = {key: np.concatenate(value) for key, value in predictions.items()}
        baseline[name] = predictions["flip"]
        summary = {
            "run": name, "checkpoint": str(path), "epoch": checkpoint["epoch"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "parameter_count": count,
            "evaluation_seconds": time.perf_counter() - started, "protocols": {},
        }
        for key, pred in predictions.items():
            correct = int(np.sum(pred == labels))
            summary["protocols"][key] = {
                "correct": correct, "accuracy": correct / len(labels),
                "delta_correct_vs_flip": correct - int(np.sum(predictions["flip"] == labels)),
                **pair_statistics(predictions["flip"], pred, labels),
            }
        with (args.output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["training_dataset_index", "true_label", *protocols])
            writer.writerows(zip(data.valid.tolist(), labels.tolist(), *(predictions[k].tolist() for k in protocols)))
        report["models"].append(summary)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        del model, checkpoint
    for a, b in itertools.combinations(baseline, 2):
        report["pairs"].append({"a": a, "b": b, **pair_statistics(baseline[a], baseline[b], labels)})
    with (args.output / "agreement.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", *baseline])
        for a in baseline:
            writer.writerow([a, *[float(np.mean(baseline[a] == baseline[b])) for b in baseline]])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
        for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "SHA256SUMS"))


if __name__ == "__main__":
    main()
