import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from model import ARCHITECTURES, CifarResNet, build_model, parameter_count
from prepare_data import stratified_split
from train import (EMA, PARAMETER_LIMIT, Data, PolicyImages, augment_batch, capture_rng, cutout_batch, distillation_loss,
                   load_teachers, lr_at_epoch, make_model, predict_logits, restore_rng, save_checkpoint)


class PipelineTests(unittest.TestCase):
    def test_parameter_budget_and_forward_backward(self):
        torch.set_num_threads(2)
        model = CifarResNet()
        self.assertEqual(parameter_count(model), 2797610)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.shape, (2, 10))
        F.cross_entropy(output, torch.tensor([0, 1])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_layout_preserves_predictions(self):
        model = CifarResNet().eval()
        x = torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            original = model(x)
            model.to(memory_format=torch.channels_last)
            changed = model(x.to(memory_format=torch.channels_last))
        torch.testing.assert_close(original, changed, rtol=1e-4, atol=1e-5)

    def test_split_has_no_id_or_pixel_leakage(self):
        labels = np.repeat(np.arange(10), 20)
        hashes = [str(i // 2) for i in range(200)]
        train, valid, _ = stratified_split(labels, hashes, per_class=5)
        self.assertEqual(len(valid), 50)
        self.assertFalse(set(train) & set(valid))
        self.assertFalse({hashes[i] for i in train} & {hashes[i] for i in valid})
        again = stratified_split(labels, hashes, per_class=5)
        np.testing.assert_array_equal(train, again[0])

    def test_augmentation_preserves_image_identity_and_bounds(self):
        x = torch.stack([torch.full((3, 32, 32), i / 4) for i in range(4)])
        torch.manual_seed(17)
        out = augment_batch(x)
        self.assertEqual(out.shape, x.shape)
        for i in range(4):
            self.assertTrue(((out[i] == 0) | (out[i] == i / 4)).all())
        self.assertTrue((out[1:].sum(dim=(1, 2, 3)) > 0).all())

    def test_lr_schedule(self):
        self.assertAlmostEqual(lr_at_epoch(0, 100, .1), .02)
        self.assertAlmostEqual(lr_at_epoch(4, 100, .1), .1)
        self.assertLess(lr_at_epoch(99, 100, .1), .0001)
        per_step = [lr_at_epoch(i / 176, 200, .2) for i in range(200 * 176)]
        self.assertAlmostEqual(max(per_step), .2)
        self.assertTrue(all(a >= b for a, b in zip(per_step[4 * 176:], per_step[4 * 176 + 1:])))

    def test_checkpoint_roundtrip_and_rng(self):
        device = torch.device("cpu")
        model = CifarResNet().eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        data = SimpleNamespace(device=device, audit={"split_seed": 2026})
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            save_checkpoint(path, model, optimizer, {"width": 32}, 3, .5, data)
            checkpoint = torch.load(path, weights_only=False)
            restored = CifarResNet().eval()
            restored.load_state_dict(checkpoint["model"])
            with torch.no_grad():
                torch.testing.assert_close(restored(x), expected)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["data_audit"], data.audit)
        state = capture_rng(device)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        restore_rng(state, device)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(2), expected[2])

    @unittest.skipUnless(Path("data/cache/audit.json").exists(), "Needs the verified official dataset")
    def test_real_validation_is_deterministic_and_split_is_complete(self):
        data = Data(Path("data"), torch.device("cpu"), "batched", False)
        self.assertEqual(len(data.train), 45000)
        self.assertEqual(len(data.valid), 5000)
        self.assertFalse(set(data.train.tolist()) & set(data.valid.tolist()))
        x1, y1 = next(data.batches(data.valid, 16, training=False))
        x2, y2 = next(data.batches(data.valid, 16, training=False))
        torch.testing.assert_close(x1, x2, rtol=0, atol=0)
        torch.testing.assert_close(y1, y2, rtol=0, atol=0)
        torch.testing.assert_close(y1, data.labels[data.valid[:16]])

    def test_registered_architectures_fit_budget_and_rebuild(self):
        expected = {"resnet18-w32": 2797610, "resnet18-w42": 4815940, "resnet18-64-352": 4797002, "resnet20-x4": 4327754}
        for arch, count in expected.items():
            model = build_model(arch)
            self.assertEqual(parameter_count(model), count)
            self.assertLessEqual(parameter_count(model), PARAMETER_LIMIT)
        self.assertLessEqual(set(expected), set(ARCHITECTURES))
        old = make_model({"width": 32})
        self.assertEqual(old.state_dict().keys(), build_model("resnet18-w32").state_dict().keys())

    def test_reflect_crop_and_cutout(self):
        torch.manual_seed(3)
        x = torch.stack([torch.full((3, 32, 32), i / 4) for i in range(1, 5)])
        out = augment_batch(x, padding_mode="reflect")
        torch.testing.assert_close(out, x)  # reflection of a constant image never introduces padding zeros
        erased = cutout_batch(torch.ones(64, 3, 32, 32), 16)
        zeros = (erased == 0).all(dim=1).sum(dim=(1, 2))
        self.assertTrue(((zeros >= 64) & (zeros <= 256)).all())  # clipped squares keep at least 8 x 8 pixels
        self.assertTrue(((erased == 0) | (erased == 1)).all())
        self.assertGreater(len(set(zeros.tolist())), 1)

    def test_ema_tracks_weights_and_buffers(self):
        model = CifarResNet()
        ema = EMA(model, 0.5)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1)
        model.train()
        model(torch.randn(4, 3, 32, 32))  # changes BN running statistics
        before = ema.model.head.bias.clone()
        ema.update()
        decay = min(0.5, 2 / 11)
        torch.testing.assert_close(ema.model.head.bias, before + (1 - decay))
        torch.testing.assert_close(ema.model.stem[1].num_batches_tracked, model.stem[1].num_batches_tracked)
        self.assertFalse(any(p.requires_grad for p in ema.model.parameters()))

    def test_flip_tta_is_mirror_invariant(self):
        model = CifarResNet().eval()
        x = torch.randn(3, 3, 32, 32)
        torch.testing.assert_close(predict_logits(model, x, "flip"), predict_logits(model, x.flip(-1), "flip"), rtol=1e-4, atol=1e-5)

    def test_distillation_targets_and_leakage_guard(self):
        model = CifarResNet().eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            self.assertLess(abs(distillation_loss(model(x), [model], x, 2.0).item()), 1e-5)
            self.assertGreater(distillation_loss(torch.zeros(4, 10), [model], x, 2.0).item(), 0)
        device = torch.device("cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with tempfile.TemporaryDirectory() as folder:
            for held_out in (True, False):
                data = SimpleNamespace(device=device, audit={"split_seed": 2026}, held_out=True)
                path = Path(folder) / f"teacher-{held_out}.pt"
                save_checkpoint(path, model, optimizer, {"arch": "resnet18-w32", "validation_held_out": held_out}, 1, .5, data)
                if held_out:
                    teacher, = load_teachers([path], device, data)
                    torch.testing.assert_close(teacher(x), model(x))
                else:
                    with self.assertRaises(ValueError):
                        load_teachers([path], device, data)

    @unittest.skipUnless(Path("data/cache/audit.json").exists(), "Needs the verified official dataset")
    def test_predict_writes_csv_and_audit_for_registry_checkpoints(self):
        import json
        from train import predict
        audit = json.loads(Path("data/cache/audit.json").read_text())
        model = build_model("resnet18-w32")
        data = SimpleNamespace(device=torch.device("cpu"), audit=audit, held_out=True)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "last.pt"
            save_checkpoint(checkpoint, model, torch.optim.SGD(model.parameters(), lr=.1),
                            {"arch": "resnet18-w32", "validation_held_out": True}, 1, .1, data)
            output = Path(folder) / "submission.csv"
            predict(SimpleNamespace(checkpoint=str(checkpoint), output=str(output), weights=None, tta="flip",
                                    data="data", batch_size=500, device="cpu"))
            receipt = json.loads(output.with_suffix(".audit.json").read_text())
            self.assertEqual((receipt["rows"], receipt["arch"], receipt["tta"]), (2000, "resnet18-w32", "flip"))

    @unittest.skipUnless(Path("data/cache/audit.json").exists(), "Needs the verified official dataset")
    def test_full_train_split_and_policy_images(self):
        data = Data(Path("data"), torch.device("cpu"), "batched", False, full_train=True)
        self.assertEqual(len(data.train), 50000)
        self.assertFalse(data.held_out)
        self.assertTrue(set(data.valid.tolist()) <= set(data.train.tolist()))
        dataset = PolicyImages(Path("data/cache/train_images.npy"), data.train.numpy(), data.labels.numpy())
        image, label = dataset[7]
        self.assertEqual((image.shape, image.dtype), ((3, 32, 32), torch.uint8))
        self.assertEqual(label, int(data.labels[data.train[7]]))


if __name__ == "__main__":
    unittest.main()
