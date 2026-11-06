import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from model import PreActBlock, build_model, parameter_count
from train import Data, PolicyImages, fold_batchnorm, main, seed_all


def synthetic_cache(root):
    cache = root / "cache"
    cache.mkdir()
    rng = np.random.default_rng(41)
    np.save(cache / "train_images.npy", rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8))
    np.savez(cache / "split.npz", train=np.arange(12), valid=np.arange(12, 16),
             labels=np.arange(16, dtype=np.int64) % 10)
    (cache / "audit.json").write_text(json.dumps({
        "test_fixture": True, "normalization_training_only_mean": [.5] * 3,
        "normalization_training_only_std": [.25] * 3,
    }))


class Round2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_new_models_budget_forward_backward_and_reload(self):
        for arch, count in (("resnet-highres", 3527082), ("preact-highres", 3526314)):
            with self.subTest(arch=arch):
                model = build_model(arch)
                self.assertEqual(parameter_count(model), count)
                self.assertLess(count, 5_000_000)
                x = torch.randn(2, 3, 32, 32)
                loss = F.cross_entropy(model(x), torch.tensor([1, 2]))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                restored = build_model(arch)
                restored.load_state_dict(model.state_dict())
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_preact_identity_preserves_negative_values_and_rejects_old_fold(self):
        block = PreActBlock(3, 3).eval()
        with torch.no_grad():
            block.conv2.weight.zero_()
            x = torch.randn(2, 3, 8, 8)
            torch.testing.assert_close(block(x), x, rtol=0, atol=0)
        self.assertEqual(PreActBlock(3, 5, stride=2)(x).shape, (2, 5, 4, 4))
        with self.assertRaises(ValueError):
            fold_batchnorm(build_model("preact-highres").eval())

    def test_policy_shapes_labels_and_unaugmented_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            synthetic_cache(root)
            baseline = Data(root, torch.device("cpu"), "batched", False)
            expected = next(baseline.batches(baseline.valid, 4))
            for policy in ("autoaugment", "randaugment"):
                data = Data(root, torch.device("cpu"), policy, False, workers=0)
                dataset = PolicyImages(root / "cache/train_images.npy", np.array([7, 2]),
                                       np.arange(16) % 10, policy)
                image, label = dataset[0]
                self.assertEqual((image.shape, image.dtype, label), ((3, 32, 32), torch.uint8, 7))
                actual = next(data.batches(data.valid, 4))
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_worker_epoch_restart_matches_fresh_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            synthetic_cache(root)
            first = Data(root, torch.device("cpu"), "randaugment", False, workers=1)
            first.policy_seed = 2026
            list(first.batches(first.train, 4, training=True))
            first.policy_epoch = 1
            seed_all(77)
            continued = list(first.batches(first.train, 4, training=True))
            second = Data(root, torch.device("cpu"), "randaugment", False, workers=1)
            second.policy_seed, second.policy_epoch = 2026, 1
            seed_all(77)
            resumed = list(second.batches(second.train, 4, training=True))
            for (x, y), (xx, yy) in zip(continued, resumed):
                torch.testing.assert_close(x, xx, rtol=0, atol=0)
                torch.testing.assert_close(y, yy, rtol=0, atol=0)

    def test_tiny_training_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            synthetic_cache(data)
            for policy in ("autoaugment", "randaugment"):
                with self.subTest(policy=policy):
                    complete, split = root / f"{policy}-complete", root / f"{policy}-split"
                    common = ["train.py", "train", "--data", str(data), "--device", "cpu",
                              "--epochs", "3", "--batch-size", "4", "--workers", "0",
                              "--pipeline", policy, "--epoch-seeded-policy", "--ema-decay", ".9",
                              "--lr-per-step", "--warmup-epochs", "1", "--no-eval-raw"]
                    def fit(run, extra=()):
                        with patch("sys.argv", common + ["--run", str(run)] + list(extra)):
                            with contextlib.redirect_stdout(io.StringIO()):
                                main()
                    fit(complete)
                    fit(split, ("--stop-after", "1"))
                    fit(split, ("--resume", str(split / "last.pt")))
                    a = torch.load(complete / "last.pt", weights_only=False)
                    b = torch.load(split / "last.pt", weights_only=False)
                    self.assertEqual((a["epoch"], b["epoch"], a["ema_updates"], b["ema_updates"]),
                                     (3, 3, 9, 9))
                    for key in ("model", "ema"):
                        for name in a[key]:
                            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
                    for index, state in a["optimizer"]["state"].items():
                        torch.testing.assert_close(state["momentum_buffer"],
                                                   b["optimizer"]["state"][index]["momentum_buffer"],
                                                   rtol=0, atol=0)
                    rows = [json.loads(line) for line in (split / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([r["epoch"] for r in rows], [1, 2, 3])
                    self.assertTrue(all(np.isfinite(r["train_loss"]) for r in rows))
                    with self.assertRaisesRegex(ValueError, "ra_magnitude"):
                        fit(split, ("--resume", str(split / "last.pt"), "--ra-magnitude", "10"))


if __name__ == "__main__":
    unittest.main()
