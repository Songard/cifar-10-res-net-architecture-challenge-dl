"""Validate official files, cache decoded pixels, and freeze a held-out split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def stratified_split(labels, hashes, seed=2026, per_class=500):
    rng = np.random.default_rng(seed)
    train, valid = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        valid.extend(indices[:per_class])
        train.extend(indices[per_class:])
    valid_hashes = {hashes[i] for i in valid}
    excluded = [i for i in train if hashes[i] in valid_hashes]
    train = [i for i in train if hashes[i] not in valid_hashes]
    return np.array(sorted(train)), np.array(sorted(valid)), excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.root
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    training, testing, sample = [rows(root / name) for name in ("train_labels.csv", "test.csv", "sample_submission.csv")]
    assert len(training) == 50000 and len(testing) == 2000
    assert [r["id"] for r in testing] == [r["id"] for r in sample]
    all_hashes = {}
    file_digest = hashlib.sha256()
    for name, records in (("train", training), ("test", testing)):
        assert len({r["id"] for r in records}) == len(records)
        assert len({r["filename"] for r in records}) == len(records)
        images = np.empty((len(records), 32, 32, 3), dtype=np.uint8)
        hashes = []
        for i, record in enumerate(records):
            path = root / f"{name}_images" / record["filename"]
            raw = path.read_bytes()
            file_digest.update(path.name.encode() + hashlib.sha256(raw).digest())
            with Image.open(path) as image:
                if image.size != (32, 32) or image.mode != "RGB":
                    raise ValueError(f"Unexpected image shape/mode: {path}")
                images[i] = np.asarray(image)
            hashes.append(hashlib.sha256(images[i].tobytes()).hexdigest())
        np.save(cache / f"{name}_images.npy", images)
        all_hashes[name] = hashes
    labels = np.array([int(row["label"]) for row in training], dtype=np.int64)
    assert set(labels) == set(range(10))
    train, valid, excluded = stratified_split(labels, all_hashes["train"])
    np.savez(cache / "split.npz", train=train, valid=valid, labels=labels)
    images = np.load(cache / "train_images.npy", mmap_mode="r")
    pixels = images[train].astype(np.float32) / 255
    mean = pixels.mean(axis=(0, 1, 2), dtype=np.float64)
    std = pixels.std(axis=(0, 1, 2), dtype=np.float64)
    audit = {
        "source": "https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/data",
        "train_images": len(training), "test_images": len(testing),
        "train_class_counts": np.bincount(labels).tolist(),
        "training_split": len(train), "validation_split": len(valid), "split_seed": 2026,
        "validation_class_counts": np.bincount(labels[valid]).tolist(),
        "train_duplicates_excluded_to_avoid_validation_leakage": excluded,
        "normalization_training_only_mean": mean.tolist(),
        "normalization_training_only_std": std.tolist(),
        "ordered_image_file_digest_sha256": file_digest.hexdigest(),
        "csv_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in ("train_labels.csv", "test.csv", "sample_submission.csv")},
        "test_labels_accessed": False,
    }
    (cache / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
