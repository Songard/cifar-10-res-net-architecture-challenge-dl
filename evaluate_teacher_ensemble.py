"""Evaluate fixed-split teacher ensembles without loading hidden-test data."""

import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from train import Data, make_model, predict_logits


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(logits, labels, names, temperature=1.0):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels)
    if logits.ndim != 3 or logits.shape[0] != len(names) or logits.shape[1] != len(labels):
        raise ValueError("Expected member x sample x class logits and aligned labels")
    if temperature <= 0 or not torch.isfinite(logits).all():
        raise ValueError("Temperature must be positive and logits finite")
    probabilities = (logits / temperature).softmax(-1)
    predictions = logits.argmax(-1)
    subsets = []
    for count in range(1, len(names) + 1):
        for indices in itertools.combinations(range(len(names)), count):
            predicted = probabilities[list(indices)].mean(0).argmax(-1)
            correct = int((predicted == labels).sum())
            subsets.append({"members": [names[i] for i in indices], "correct": correct,
                            "accuracy": correct / len(labels)})
    subsets.sort(key=lambda item: (-item["correct"], len(item["members"]), item["members"]))
    pairs = []
    for i, j in itertools.combinations(range(len(names)), 2):
        a, b = predictions[i] == labels, predictions[j] == labels
        pairs.append({"members": [names[i], names[j]],
                      "disagreements": int((predictions[i] != predictions[j]).sum()),
                      "a_only_correct": int((a & ~b).sum()), "b_only_correct": int((~a & b).sum()),
                      "both_wrong": int((~a & ~b).sum())})
    return {"temperature": temperature, "subsets": subsets, "pairs": pairs}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    data = Data(Path(args.data), device, "batched", False)
    names = [str(Path(p).parent) for p in args.checkpoints]
    if len(set(names)) != len(names):
        raise ValueError("Use distinct run directories")
    records, logits = [], []
    for name, path in zip(names, args.checkpoints):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["data_audit"] != data.audit or checkpoint["config"].get("validation_held_out") is not True:
            raise ValueError(f"Teacher split or held-out provenance mismatch: {path}")
        if "ema" not in checkpoint:
            raise ValueError(f"EMA weights required: {path}")
        model = make_model(checkpoint["config"]).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["ema"])
        begin = time.perf_counter()
        values = [predict_logits(model, x, "flip").float().cpu()
                  for x, _ in data.batches(data.valid, args.batch_size)]
        values = torch.cat(values)
        logits.append(values)
        record = {"name": name, "checkpoint": str(path), "checkpoint_sha256": sha256(path),
                  "epoch": checkpoint["epoch"], "config": checkpoint["config"],
                  "evaluation_seconds": time.perf_counter() - begin,
                  "accuracy": float((values.argmax(1) == data.labels[data.valid]).float().mean())}
        records.append(record)
        print(json.dumps({k: record[k] for k in ("name", "epoch", "evaluation_seconds", "accuracy")}), flush=True)
        del model, checkpoint
    logits = torch.stack(logits)
    labels = data.labels[data.valid]
    np.savez_compressed(output / "validation_logits.npz", logits=logits.numpy(), labels=labels.numpy(),
                        indices=data.valid.numpy(), names=np.array(names))
    result = {"device": args.device, "torch_version": str(torch.__version__), "platform": platform.platform(),
              "validation_count": len(labels), "weights": "ema", "tta": "flip",
              "data_audit": data.audit, "models": records,
              "selection_warning": "Exploratory selection on a repeatedly used validation split; not an independent confirmation.",
              "probability_rule": "softmax(mean(original and flip logits) / temperature), then average members",
              "comparisons": [compare(logits, labels, names, t) for t in (1.0, 4.0)]}
    result["files_sha256"] = {"validation_logits.npz": sha256(output / "validation_logits.npz"),
                              "evaluate_teacher_ensemble.py": sha256(__file__)}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"best_t1": result["comparisons"][0]["subsets"][0],
                      "best_t4": result["comparisons"][1]["subsets"][0]}), flush=True)


if __name__ == "__main__":
    main()
