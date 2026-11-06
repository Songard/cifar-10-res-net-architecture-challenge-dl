import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_oof import prepare, stratified_folds


class OutOfFoldTests(unittest.TestCase):
    def test_stratified_partition(self):
        labels = np.repeat(np.arange(3), 11)
        folds = stratified_folds(labels, 5, 2030)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(33))
        for label in range(3):
            counts = [np.sum(labels[fold] == label) for fold in folds]
            self.assertLessEqual(max(counts) - min(counts), 1)
        for actual, repeated in zip(folds, stratified_folds(labels, 5, 2030)):
            np.testing.assert_array_equal(actual, repeated)

    def test_cache_without_hidden_test_and_training_only_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "source/cache"
            cache.mkdir(parents=True)
            images = np.random.default_rng(7).integers(0, 256, (20, 4, 4, 3), dtype=np.uint8)
            labels = np.repeat(np.arange(2), 10)
            np.save(cache / "train_images.npy", images)
            np.savez(cache / "split.npz", labels=labels, train=np.arange(16), valid=np.arange(16, 20))
            original = (cache / "split.npz").read_bytes()
            output = root / "oof"
            prepare(root / "source", output)
            seen = []
            for fold in range(5):
                fold_cache = output / f"fold-{fold}/cache"
                with np.load(fold_cache / "split.npz") as split:
                    train, valid = split["train"], split["valid"]
                    self.assertEqual(len(np.intersect1d(train, valid)), 0)
                    self.assertEqual(len(train), 16)
                    seen.extend(valid)
                    pixels = images[train].astype(np.float64).reshape(-1, 3) / 255.0
                audit = json.loads((fold_cache / "audit.json").read_text())
                np.testing.assert_allclose(audit["normalization_training_only_mean"], pixels.mean(0))
                np.testing.assert_allclose(audit["normalization_training_only_std"], pixels.std(0))
                self.assertFalse((fold_cache / "test_images.npy").exists())
                self.assertEqual((fold_cache / "train_images.npy").resolve(), (cache / "train_images.npy").resolve())
            np.testing.assert_array_equal(np.sort(seen), np.arange(20))
            self.assertEqual(original, (cache / "split.npz").read_bytes())
            with self.assertRaises(FileExistsError):
                prepare(root / "source", output)

    def test_reject_too_few_samples(self):
        with self.assertRaises(ValueError):
            stratified_folds(np.array([0, 0, 1]), 2, 2030)
