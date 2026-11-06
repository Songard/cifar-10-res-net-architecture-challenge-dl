import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests.test_round2 import synthetic_cache
from train import main, sam_perturbed


class SAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_quadratic_gradient_and_exact_restoration(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[3., 4.]]))
        original = model.weight.detach().clone()
        (.5 * model.weight.square().sum()).backward()
        with sam_perturbed(model, .5):
            torch.testing.assert_close(model.weight, original * 1.1)
            model.zero_grad(set_to_none=True)
            (.5 * model.weight.square().sum()).backward()
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
        torch.testing.assert_close(model.weight.grad, original * 1.1)
        torch.optim.SGD(model.parameters(), lr=.1).step()
        torch.testing.assert_close(model.weight, original * .89)

    def test_bn_updates_once_and_exception_restores(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Flatten(), torch.nn.Linear(48, 2)).train()
        x = torch.randn(4, 3, 4, 4)
        model(x).square().mean().backward()
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with sam_perturbed(model, .05):
                model.zero_grad(set_to_none=True)
                model(x).square().mean().backward()
                raise RuntimeError("test failure")
        self.assertTrue(model[0].track_running_stats)
        self.assertEqual(model[0].num_batches_tracked.item(), 1)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, saved[k], rtol=0, atol=0)

    def test_synthetic_training_resume_and_bf16(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            synthetic_cache(data)
            for precision in ("none", "bf16"):
                whole, split = root / f"whole-{precision}", root / f"split-{precision}"
                common = ["train.py", "train", "--data", str(data), "--device", "cpu",
                          "--epochs", "2", "--batch-size", "4", "--workers", "0",
                          "--pipeline", "trivialaugment", "--epoch-seeded-policy", "--ema-decay", ".9",
                          "--lr-per-step", "--warmup-epochs", "1", "--no-eval-raw", "--sam-rho", ".05",
                          "--amp", precision]
                def fit(run, extra=()):
                    with patch("sys.argv", common + ["--run", str(run)] + list(extra)):
                        with contextlib.redirect_stdout(io.StringIO()):
                            main()
                fit(whole)
                fit(split, ("--stop-after", "1"))
                fit(split, ("--resume", str(split / "last.pt")))
                a = torch.load(whole / "last.pt", weights_only=False)
                b = torch.load(split / "last.pt", weights_only=False)
                self.assertEqual(a["ema_updates"], 6)
                for key in ("model", "ema"):
                    for name in a[key]:
                        torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
                        if name.endswith("num_batches_tracked") and key == "model":
                            self.assertEqual(a[key][name].item(), 6)
                for index, state in a["optimizer"]["state"].items():
                    torch.testing.assert_close(state["momentum_buffer"], b["optimizer"]["state"][index]["momentum_buffer"], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "sam_rho"):
                    fit(split, ("--resume", str(split / "last.pt"), "--sam-rho", ".1"))


if __name__ == "__main__":
    unittest.main()
