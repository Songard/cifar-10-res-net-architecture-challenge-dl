"""Prepare training-only fold caches for label review, without changing the main split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stratified_folds(labels, folds, seed):
    if folds < 2:
        raise ValueError("At least two folds are required")
    rng = np.random.default_rng(seed)
    pieces = [[] for _ in range(folds)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < folds:
            raise ValueError("Each class must contain at least one sample per fold")
        for fold, part in enumerate(np.array_split(rng.permutation(indices), folds)):
            pieces[fold].append(part)
    return [np.sort(np.concatenate(parts)) for parts in pieces]


def normalization(images, indices):
    total = np.zeros(images.shape[-1], dtype=np.float64)
    squares = total.copy()
    count = 0
    for offset in range(0, len(indices), 256):
        pixels = images[indices[offset:offset + 256]].astype(np.float64) / 255.0
        pixels = pixels.reshape(-1, images.shape[-1])
        total += pixels.sum(axis=0)
        squares += np.square(pixels).sum(axis=0)
        count += len(pixels)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0))
    if np.any(std == 0):
        raise ValueError("Training channels must have nonzero variance")
    return mean.tolist(), std.tolist()


def prepare(source, output, folds=5, seed=2030):
    source = Path(source).resolve()
    output = Path(output)
    image_path = source / "cache/train_images.npy"
    split_path = source / "cache/split.npz"
    images = np.load(image_path, mmap_mode="r")
    with np.load(split_path) as split:
        labels = split["labels"].copy()
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) != len(labels):
        raise ValueError("Expected matching NHWC RGB training images and labels")
    validation_folds = stratified_folds(labels, folds, seed)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "purpose": "out_of_fold_training_label_review_only",
        "source_train_images_sha256": sha256(image_path),
        "source_split_sha256": sha256(split_path),
        "folds": folds,
        "split_seed": seed,
        "train_images": len(labels),
        "hidden_test_accessed": False,
    }
    all_indices = np.arange(len(labels))
    for fold, valid in enumerate(validation_folds):
        train = np.setdiff1d(all_indices, valid)
        cache = output / f"fold-{fold}" / "cache"
        cache.mkdir(parents=True)
        (cache / "train_images.npy").symlink_to(image_path)
        np.savez(cache / "split.npz", labels=labels, train=train, valid=valid)
        mean, std = normalization(images, train)
        audit = dict(provenance, fold=fold, training_split=len(train),
                     validation_split=len(valid),
                     fold_split_sha256=sha256(cache / "split.npz"),
                     normalization_training_only_mean=mean,
                     normalization_training_only_std=std)
        (cache / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.folds, args.seed), indent=2))
