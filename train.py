"""Reproducible local training, validation, resume, and CSV generation.

Defaults reproduce the original baseline recipe. The improved recipe is opt-in through flags or a
JSON file passed with --config (see configs/); every resolved setting is written to the run's config.json.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision import transforms

from model import CifarResNet, build_model, parameter_count
from prepare_data import rows

PARAMETER_LIMIT = 5_000_000
# Settings that define an experiment; resuming with any of them changed would mix two experiments in one log.
EXPERIMENT_KEYS = ("arch", "epochs", "batch_size", "pipeline", "channels_last", "lr", "seed", "weight_decay",
                   "bn_weight_decay", "warmup_epochs", "lr_per_step", "padding_mode", "cutout",
                   "label_smoothing", "ema_decay", "amp", "full_train", "teachers", "kd_alpha", "kd_temperature",
                   "randconv", "mix", "mix_alpha", "mix_prob", "ra_ops", "ra_magnitude", "epoch_seeded_policy", "sam_rho")
# Keys added after earlier runs started; a checkpoint without them trained with the feature off.
LATER_DEFAULTS = {"randconv": 0.0, "mix": "none", "mix_alpha": 1.0, "mix_prob": 0.5,
                  "ra_ops": 2, "ra_magnitude": 9, "epoch_seeded_policy": False, "sam_rho": 0.0}
AMP_DTYPES = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}
POLICY_PIPELINES = ("trivialaugment", "autoaugment", "randaugment")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_batch(x, padding=4, padding_mode="constant"):
    """Independent integer crop and flip per image, using the same policy as the baseline."""
    n = len(x)
    padded = F.pad(x, (padding,) * 4, mode=padding_mode)
    top = torch.randint(2 * padding + 1, (n, 1, 1), device=x.device)
    left = torch.randint(2 * padding + 1, (n, 1, 1), device=x.device)
    y = top + torch.arange(32, device=x.device).reshape(1, 32, 1)
    z = left + torch.arange(32, device=x.device).reshape(1, 1, 32)
    cropped = padded.permute(0, 2, 3, 1)[torch.arange(n, device=x.device)[:, None, None], y, z]
    cropped = cropped.permute(0, 3, 1, 2)
    flip = torch.rand(n, 1, 1, 1, device=x.device) < 0.5
    return torch.where(flip, cropped.flip(-1), cropped)


def cutout_batch(x, size):
    """Zero one size x size square per image, centre uniform over the image (DeVries & Taylor, 2017).

    Applied after normalization, so the erased pixels equal the training-set mean colour.
    """
    n, _, height, width = x.shape
    top = torch.randint(height, (n, 1, 1), device=x.device) - size // 2
    left = torch.randint(width, (n, 1, 1), device=x.device) - size // 2
    y = torch.arange(height, device=x.device).reshape(1, height, 1)
    z = torch.arange(width, device=x.device).reshape(1, 1, width)
    inside = (y >= top) & (y < top + size) & (z >= left) & (z < left + size)
    return x * (~inside).unsqueeze(1)


def randconv_batch(x, probability, sizes=(1, 3, 5)):
    """Random convolution, RC-mix variant (Xu et al., 2021): selected images are blended with a random filter's response.

    Each image gets its own 3->3 channel kernel with N(0, 1/(3k^2)) weights; the size k is drawn once per batch.
    k=1 remixes colours and larger k perturbs local texture, while object shape survives. Applied after normalization.
    """
    n, c, height, width = x.shape
    chosen = torch.rand(n, 1, 1, 1, device=x.device) < probability
    k = sizes[int(torch.randint(len(sizes), ()))]
    weight = torch.randn(n * c, c, k, k, device=x.device) / math.sqrt(c * k * k)
    response = F.conv2d(x.reshape(1, n * c, height, width), weight, padding=k // 2, groups=n).reshape(n, c, height, width)
    alpha = torch.rand(n, 1, 1, 1, device=x.device)
    return torch.where(chosen, alpha * x + (1 - alpha) * response, x)


def mix_batch(x, y, mode, alpha):
    """Mixup (Zhang et al., 2018) or CutMix (Yun et al., 2019) against a shuffled copy of the batch.

    Returns the mixed images, the partner labels, and the weight that the original labels keep.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(len(x), device=x.device)
    if mode == "mixup":
        return lam * x + (1 - lam) * x[index], y[index], lam
    height, width = x.shape[2:]
    cut_h, cut_w = int(height * math.sqrt(1 - lam)), int(width * math.sqrt(1 - lam))
    cy, cx = int(torch.randint(height, ())), int(torch.randint(width, ()))
    top, bottom = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    left, right = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    x = x.clone()
    x[:, :, top:bottom, left:right] = x[index, :, top:bottom, left:right]
    return x, y[index], 1 - (bottom - top) * (right - left) / (height * width)


class PolicyImages(torch.utils.data.Dataset):
    """Torchvision image policies on CPU workers; crop, flip and Cutout stay on the device."""

    def __init__(self, path, indices, labels, pipeline="trivialaugment", ra_ops=2, ra_magnitude=9):
        self.path, self.indices, self.labels = path, indices, labels
        self.images = None
        if pipeline == "trivialaugment":
            self.policy = transforms.TrivialAugmentWide()
        elif pipeline == "autoaugment":
            self.policy = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        elif pipeline == "randaugment":
            self.policy = transforms.RandAugment(num_ops=ra_ops, magnitude=ra_magnitude)
        else:
            raise ValueError(f"Unknown image policy: {pipeline}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode="r")
        index = int(self.indices[position])
        image = self.policy(Image.fromarray(np.asarray(self.images[index])))
        return torch.from_numpy(np.array(image)).permute(2, 0, 1), int(self.labels[index])


class Data:
    def __init__(self, root, device, pipeline, channels_last, full_train=False, padding_mode="constant", cutout=0, workers=6,
                 randconv=0.0, ra_ops=2, ra_magnitude=9):
        self.root = root
        self.device = device
        self.pipeline = pipeline
        self.channels_last = channels_last
        self.padding_mode = padding_mode
        self.cutout = cutout
        self.workers = workers
        self.randconv = randconv
        self.ra_ops, self.ra_magnitude = ra_ops, ra_magnitude
        self.audit = json.loads((root / "cache/audit.json").read_text())
        self.images = torch.from_numpy(np.load(root / "cache/train_images.npy")).permute(0, 3, 1, 2)
        split = np.load(root / "cache/split.npz")
        self.labels = torch.from_numpy(split["labels"])
        self.valid = torch.from_numpy(split["valid"])
        # A full-train run also fits the 5,000 validation images, so its validation numbers are not held out.
        self.train = torch.from_numpy(np.sort(np.concatenate([split["train"], split["valid"]])) if full_train else split["train"])
        self.held_out = not full_train
        self.mean = torch.tensor(self.audit["normalization_training_only_mean"], device=device).float()[None, :, None, None]
        self.std = torch.tensor(self.audit["normalization_training_only_std"], device=device).float()[None, :, None, None]
        self.single_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        self.loader = None
        self.policy_seed = None
        self.policy_epoch = 0

    def finish(self, x, training):
        """Device-side part of every pipeline: scale, crop/flip, normalize, random convolution, Cutout, memory layout."""
        x = x.to(self.device, dtype=torch.float32) / 255
        if training and self.pipeline != "per-image":
            x = augment_batch(x, padding_mode=self.padding_mode)
        x = (x - self.mean) / self.std
        if training and self.randconv:
            x = randconv_batch(x, self.randconv)
        if training and self.cutout:
            x = cutout_batch(x, self.cutout)
        memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
        return x.contiguous(memory_format=memory_format)

    def batches(self, indices, batch_size, training=False):
        if training and self.pipeline in POLICY_PIPELINES:
            yield from self.policy_batches(batch_size)
            return
        if training:
            indices = indices[torch.randperm(len(indices))]
        for start in range(0, len(indices), batch_size):
            selection = indices[start:start + batch_size]
            x = self.images[selection]
            if training and self.pipeline == "per-image":
                x = torch.stack([self.single_transform(image) for image in x])
            yield self.finish(x, training), self.labels[selection].to(self.device)

    def policy_batches(self, batch_size):
        if self.loader is None:
            # The shuffling generator is seeded from the run's torch RNG, so a fresh run is repeatable.
            generator = torch.Generator().manual_seed(
                self.policy_seed if self.policy_seed is not None else int(torch.randint(2**31 - 1, ())))
            dataset = PolicyImages(self.root / "cache/train_images.npy", self.train.numpy(), self.labels.numpy(),
                                   self.pipeline, self.ra_ops, self.ra_magnitude)
            parallel = dict(num_workers=self.workers, persistent_workers=self.policy_seed is None, prefetch_factor=4) if self.workers else {}
            self.loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, generator=generator, **parallel)
        if self.policy_seed is not None:
            # Restart worker RNG streams at epoch boundaries so segmented and uninterrupted runs agree.
            self.loader.generator.manual_seed(self.policy_seed + self.policy_epoch)
        for x, y in self.loader:
            yield self.finish(x, True), y.to(self.device)


class EMA:
    """Exponential moving average of all floating-point weights and BN statistics (Polyak averaging)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.pairs = [(e, m) for e, m in zip(self.model.state_dict().values(), model.state_dict().values())]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        floats = [(e, m) for e, m in self.pairs if e.dtype.is_floating_point]
        torch._foreach_lerp_([e for e, _ in floats], [m for _, m in floats], 1 - decay)
        for e, m in self.pairs:
            if not e.dtype.is_floating_point:
                e.copy_(m)


@torch.no_grad()
def predict_logits(model, x, tta):
    output = model(x)
    if tta == "flip":
        output = (output + model(x.flip(-1))) / 2
    return output


@torch.no_grad()
def evaluate(model, data, batch_size, tta="none"):
    model.eval()
    sums = torch.zeros(2, device=data.device)
    for x, y in data.batches(data.valid, batch_size):
        prediction = predict_logits(model, x, tta)
        sums[0] += F.cross_entropy(prediction, y, reduction="sum")
        sums[1] += (prediction.argmax(1) == y).sum()
    values = sums.cpu().tolist()
    return values[0] / len(data.valid), values[1] / len(data.valid)


@torch.no_grad()
def evaluate_views(model, data, batch_size, flip):
    """(loss, accuracy) of the plain logits and, when `flip`, of the flip-TTA logits, sharing the un-flipped pass.

    Same arithmetic as evaluate(model, ...) and evaluate(model, ..., "flip") in one sweep over the validation set.
    """
    model.eval()
    sums = torch.zeros(4, device=data.device)
    for x, y in data.batches(data.valid, batch_size):
        plain = model(x)
        sums[0] += F.cross_entropy(plain, y, reduction="sum")
        sums[1] += (plain.argmax(1) == y).sum()
        if flip:
            both = (plain + model(x.flip(-1))) / 2
            sums[2] += F.cross_entropy(both, y, reduction="sum")
            sums[3] += (both.argmax(1) == y).sum()
    values = [value / len(data.valid) for value in sums.cpu().tolist()]
    return (values[0], values[1]), ((values[2], values[3]) if flip else None)


def capture_rng(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, config, epoch, best, data, ema=None, scaler=None):
    snapshot = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": config, "epoch": epoch, "best_valid_accuracy": best,
        "rng": capture_rng(data.device), "data_audit": data.audit,
    }
    if ema is not None:
        snapshot.update(ema=ema.model.state_dict(), ema_updates=ema.updates)
    if scaler is not None:
        snapshot["scaler"] = scaler.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(snapshot, temporary)
    os.replace(temporary, path)


def lr_at_epoch(epoch, total, base, warmup=5):
    """Linear warmup then cosine decay; `epoch` may be fractional when the rate is updated every step."""
    warmup = min(warmup, total)
    if epoch < warmup:
        # Equals the baseline's (epoch + 1) / warmup ramp at whole epochs and never exceeds `base` in between.
        return base * min(1.0, (epoch + 1) / warmup)
    return base * 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, total - warmup)))


def memory_snapshot(device):
    result = {"rss_gb": psutil.Process().memory_info().rss / 1e9}
    if device.type == "mps":
        result.update(mps_allocated_gb=torch.mps.current_allocated_memory() / 1e9,
                      mps_driver_gb=torch.mps.driver_allocated_memory() / 1e9)
    if device.type == "cuda":
        result.update(cuda_allocated_gb=torch.cuda.memory_allocated() / 1e9,
                      cuda_peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
    return result


def make_model(config):
    """Rebuild the network a checkpoint was trained with (old baseline checkpoints only record `width`)."""
    return build_model(config["arch"]) if "arch" in config else CifarResNet(config["width"])


def fold_batchnorm(model):
    """Fold each eval-mode BatchNorm of a CifarResNet into the convolution before it (same function, fewer passes).

    Only for frozen networks (teachers): the folded convolutions carry the BN shift as a bias.
    """
    if getattr(model, "preactivation", False):
        raise ValueError("The optional BN-folding optimization does not support pre-activation networks")
    model.stem[0], model.stem[1] = fuse_conv_bn_eval(model.stem[0], model.stem[1]), nn.Identity()
    for block in model.blocks:
        block.conv1, block.bn1 = fuse_conv_bn_eval(block.conv1, block.bn1), nn.Identity()
        block.conv2, block.bn2 = fuse_conv_bn_eval(block.conv2, block.bn2), nn.Identity()
        if isinstance(block.shortcut, nn.Sequential):
            block.shortcut = nn.Sequential(fuse_conv_bn_eval(block.shortcut[0], block.shortcut[1]))
    return model


def load_teachers(paths, device, data, fold_bn=False):
    """Frozen teacher networks for distillation (Hinton et al., 2015), each trained here from scratch on the same data.

    A student that reports held-out validation accuracy may only learn from teachers that never fit those images.
    With `fold_bn`, CifarResNet teachers have BatchNorm folded into their convolutions, checked against the unfolded net.
    """
    teachers = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["data_audit"] != data.audit:
            raise ValueError(f"Teacher {path} was trained on a different dataset/split")
        if data.held_out and not checkpoint["config"].get("validation_held_out", True):
            raise ValueError(f"Teacher {path} fit the validation images; it would leak them into this held-out run")
        teacher = make_model(checkpoint["config"]).to(device)
        teacher.load_state_dict(checkpoint["ema" if "ema" in checkpoint else "model"])
        teacher.eval()
        if fold_bn and isinstance(teacher, CifarResNet):
            reference = copy.deepcopy(teacher)
            fold_batchnorm(teacher)
            with torch.no_grad():
                probe = torch.randn(64, 3, 32, 32, device=device)
                torch.testing.assert_close(teacher(probe), reference(probe), rtol=1e-3, atol=1e-3)
        teachers.append(teacher.requires_grad_(False))
    return teachers


def distillation_loss(output, teachers, x, temperature):
    """KL divergence to the teachers' averaged softened distribution, scaled by T^2 to keep gradient size."""
    with torch.no_grad():
        target = torch.stack([F.softmax(teacher(x).float() / temperature, 1) for teacher in teachers]).mean(0)
    log_student = F.log_softmax(output.float() / temperature, 1)
    return F.kl_div(log_student, target, reduction="batchmean") * temperature ** 2


def make_optimizer(model, args):
    decayed = [p for p in model.parameters() if args.bn_weight_decay or p.ndim > 1]
    groups = [{"params": decayed, "weight_decay": args.weight_decay}]
    if len(decayed) < len(list(model.parameters())):
        # BN scales/shifts and biases are excluded from weight decay in the improved recipe.
        groups.append({"params": [p for p in model.parameters() if p.ndim <= 1], "weight_decay": 0.0})
    return torch.optim.SGD(groups, lr=args.lr, momentum=0.9, nesterov=True)


@contextmanager
def sam_perturbed(model, rho):
    """SAM L2 perturbation from first-pass gradients; restore weights even on failure.

    The second forward uses batch statistics but does not update BN running statistics.
    Weight decay remains in the base SGD update, outside the perturbation objective.
    """
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        raise ValueError("SAM requires first-pass gradients")
    norm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in params]))
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite SAM gradient norm")
    originals = [p.detach().clone() for p in params]
    batchnorm = [(m, m.track_running_stats) for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        with torch.no_grad():
            scale = rho / (norm + 1e-12)
            for p in params:
                p.add_(p.grad * scale)
        for m, _ in batchnorm:
            m.track_running_stats = False
        yield
    finally:
        with torch.no_grad():
            for p, original in zip(params, originals):
                p.copy_(original)
        for m, tracking in batchnorm:
            m.track_running_stats = tracking


def code_hashes():
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ("model.py", "train.py", "prepare_data.py")}


def train(args):
    device = torch.device(args.device)
    torch.set_num_threads(6)
    seed_all(args.seed)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "metrics.jsonl").exists() and not args.resume:
        raise ValueError("Run already exists; choose another name or pass --resume.")
    data = Data(Path(args.data), device, args.pipeline, args.channels_last, args.full_train, args.padding_mode, args.cutout, args.workers,
                args.randconv, args.ra_ops, args.ra_magnitude)
    data.policy_seed = args.seed if args.epoch_seeded_policy else None
    model = build_model(args.arch).to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    count = parameter_count(model)
    assert count <= PARAMETER_LIMIT
    optimizer = make_optimizer(model, args)
    ema = EMA(model, args.ema_decay) if args.ema_decay else None
    teachers = load_teachers(args.teachers, device, data, args.fold_teacher_bn) if args.kd_alpha else []
    amp_dtype = AMP_DTYPES[args.amp]
    scaler = torch.amp.GradScaler(device.type) if args.amp == "fp16" else None
    config = vars(args).copy()
    config.update(parameter_count=count, torch_version=str(torch.__version__), platform=platform.platform(),
                  train_images=len(data.train), validation_held_out=data.held_out, code_sha256=code_hashes())
    start, best = 0, -1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["data_audit"] != data.audit:
            raise ValueError("Resume dataset/split mismatch")
        for key in EXPERIMENT_KEYS:
            if checkpoint["config"].get(key, LATER_DEFAULTS.get(key)) != config[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        if checkpoint["config"]["code_sha256"] != config["code_sha256"]:
            print("warning: source changed since this run started; both hashes are kept in config.json", flush=True)
        config["code_sha256_at_start"] = checkpoint["config"].get("code_sha256_at_start", checkpoint["config"]["code_sha256"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None:
            ema.model.load_state_dict(checkpoint["ema"])
            ema.updates = checkpoint["ema_updates"]
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], device)
        start, best = checkpoint["epoch"], checkpoint["best_valid_accuracy"]
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"device": str(device), "arch": args.arch, "parameters": count, "train": len(data.train),
                      "valid": len(data.valid), "validation_held_out": data.held_out, "start_epoch": start}), flush=True)
    steps_per_epoch = math.ceil(len(data.train) / args.batch_size)
    end = min(args.epochs, args.stop_after if args.stop_after else args.epochs)
    for epoch in range(start, end):
        data.policy_epoch = epoch
        lr = lr_at_epoch(epoch, args.epochs, args.lr, args.warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = torch.zeros(2, device=device)
        maximum = memory_snapshot(device)
        synchronize(device)
        begin = time.perf_counter()
        for batch_index, (x, y) in enumerate(data.batches(data.train, args.batch_size, training=True)):
            if args.lr_per_step:
                lr = lr_at_epoch(epoch + batch_index / steps_per_epoch, args.epochs, args.lr, args.warmup_epochs)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            partner = None
            if args.mix != "none" and float(torch.rand(())) < args.mix_prob:
                mode = args.mix if args.mix != "both" else ("mixup" if float(torch.rand(())) < 0.5 else "cutmix")
                x, partner, lam = mix_batch(x, y, mode, args.mix_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = model(x)
                loss = F.cross_entropy(output, y, label_smoothing=args.label_smoothing)
                if partner is not None:
                    loss = lam * loss + (1 - lam) * F.cross_entropy(output, partner, label_smoothing=args.label_smoothing)
                if teachers:
                    loss = (1 - args.kd_alpha) * loss + args.kd_alpha * distillation_loss(output, teachers, x, args.kd_temperature)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.sam_rho:
                    with sam_perturbed(model, args.sam_rho):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                            perturbed_output = model(x)
                            perturbed_loss = F.cross_entropy(perturbed_output, y, label_smoothing=args.label_smoothing)
                            if partner is not None:
                                perturbed_loss = lam * perturbed_loss + (1 - lam) * F.cross_entropy(
                                    perturbed_output, partner, label_smoothing=args.label_smoothing)
                        perturbed_loss.backward()
                optimizer.step()
            if ema is not None:
                ema.update()
            total[0] += loss.detach().float() * len(y)
            total[1] += (output.detach().argmax(1) == y).sum()
            if batch_index % 10 == 0:
                for key, value in memory_snapshot(device).items():
                    maximum[key] = max(maximum.get(key, 0), value)
            if batch_index % 100 == 0:
                print(f"epoch={epoch+1} batch={batch_index}", flush=True)
        synchronize(device)
        train_seconds = time.perf_counter() - begin
        train_loss, train_correct = total.cpu().tolist()
        if not math.isfinite(train_loss):
            raise FloatingPointError("Non-finite training loss")
        validation_start = time.perf_counter()
        row = {"epoch": epoch+1, "lr": lr, "train_loss": train_loss / len(data.train),
               "train_accuracy": train_correct / len(data.train), "validation_held_out": data.held_out}
        # Validation runs every --eval-every epochs, flip TTA every --tta-every, and both always at the last epoch.
        # With --no-eval-raw and EMA on, only the EMA weights (what predict uses) are validated.
        selected = None
        flip = epoch + 1 == end or (epoch + 1) % args.tta_every == 0
        if flip or (epoch + 1) % args.eval_every == 0:
            if ema is None or args.eval_raw:
                row["valid_loss"], row["valid_accuracy"] = evaluate(model, data, args.batch_size)
                selected = row["valid_accuracy"]
            if ema is not None:
                (row["ema_valid_loss"], row["ema_valid_accuracy"]), tta = evaluate_views(ema.model, data, args.batch_size, flip)
                selected = row["ema_valid_accuracy"]
            elif flip:
                tta = evaluate(model, data, args.batch_size, "flip")
            if flip:
                row["tta_valid_loss"], row["tta_valid_accuracy"] = tta
        synchronize(device)
        valid_seconds = time.perf_counter() - validation_start
        row.update(train_seconds=train_seconds, valid_seconds=valid_seconds,
                   train_images_per_second=len(data.train) / train_seconds, sampled_memory_max=maximum,
                   process_lifetime_peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if platform.system() == "Darwin" else 1e6))
        improved = data.held_out and selected is not None and selected > best
        best = max(best, selected) if data.held_out and selected is not None else best
        checkpoint_start = time.perf_counter()
        save_checkpoint(run / "last.pt", model, optimizer, config, epoch + 1, best, data, ema, scaler)
        if improved:
            shutil.copy2(run / "last.pt", run / "best.pt")
        row["checkpoint_seconds"] = time.perf_counter() - checkpoint_start
        with (run / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    print(json.dumps({"finished_epoch": end, "best_valid_accuracy": best, "run": str(run)}), flush=True)


@torch.no_grad()
def predict(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    root = Path(args.data)
    audit = json.loads((root / "cache/audit.json").read_text())
    if checkpoint["data_audit"] != audit:
        raise ValueError("Prediction dataset does not match the checkpoint")
    config = checkpoint["config"]
    weights = args.weights or ("ema" if "ema" in checkpoint else "model")
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint[weights])
    model.eval()
    count = parameter_count(model)
    assert count <= PARAMETER_LIMIT
    sample = rows(root / "sample_submission.csv")
    test = rows(root / "test.csv")
    assert [r["id"] for r in sample] == [r["id"] for r in test]
    images = np.load(root / "cache/test_images.npy")
    mean = torch.tensor(audit["normalization_training_only_mean"], device=device).float()[None, :, None, None]
    std = torch.tensor(audit["normalization_training_only_std"], device=device).float()[None, :, None, None]
    labels = []
    for start in range(0, len(images), args.batch_size):
        x = torch.from_numpy(images[start:start + args.batch_size]).permute(0, 3, 1, 2).to(device).float() / 255
        labels.extend(predict_logits(model, (x - mean) / std, args.tta).argmax(1).cpu().tolist())
    assert len(labels) == len(sample) and all(0 <= x <= 9 for x in labels)
    held_out = config.get("validation_held_out", True)
    valid = Data(root, device, "batched", False)
    valid_loss, valid_accuracy = evaluate(model, valid, args.batch_size, args.tta)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows((row["id"], label) for row, label in zip(sample, labels))
    receipt = {"file": str(destination), "rows": len(labels), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
               "arch": config["arch"] if "arch" in config else f"resnet18-w{config['width']}", "parameter_count": count,
               "weights": weights, "tta": args.tta, "epoch": checkpoint["epoch"],
               "valid_accuracy_of_these_weights": valid_accuracy, "validation_held_out": held_out,
               "predicted_class_counts": np.bincount(labels, minlength=10).tolist(), "submitted_to_kaggle": False}
    destination.with_suffix(".audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fitting = subparsers.add_parser("train")
    fitting.add_argument("--config", help="JSON file of training settings; explicit flags override it")
    fitting.add_argument("--run", required=True)
    fitting.add_argument("--arch", default="resnet18-w32")
    fitting.add_argument("--epochs", type=int, default=100)
    fitting.add_argument("--stop-after", type=int)
    fitting.add_argument("--seed", type=int, default=2026)
    fitting.add_argument("--lr", type=float, default=0.1)
    fitting.add_argument("--weight-decay", type=float, default=5e-4)
    fitting.add_argument("--sam-rho", type=float, default=0.0,
                         help="SAM L2 radius; 0 preserves SGD (CE only, none/bf16 precision)")
    fitting.add_argument("--no-bn-weight-decay", dest="bn_weight_decay", action="store_false")
    fitting.add_argument("--warmup-epochs", type=float, default=5)
    fitting.add_argument("--lr-per-step", action="store_true")
    fitting.add_argument("--pipeline", choices=("per-image", "batched", *POLICY_PIPELINES), default="batched")
    fitting.add_argument("--ra-ops", type=int, default=2)
    fitting.add_argument("--ra-magnitude", type=int, default=9)
    fitting.add_argument("--epoch-seeded-policy", action="store_true",
                         help="restart policy-worker RNG at each epoch for reproducible segmented training")
    fitting.add_argument("--padding-mode", choices=("constant", "reflect"), default="constant")
    fitting.add_argument("--cutout", type=int, default=0)
    fitting.add_argument("--label-smoothing", type=float, default=0.0)
    fitting.add_argument("--ema-decay", type=float, default=0.0)
    fitting.add_argument("--amp", choices=tuple(AMP_DTYPES), default="none")
    fitting.add_argument("--full-train", action="store_true", help="also fit the 5,000 validation images (final run only)")
    fitting.add_argument("--teachers", nargs="*", default=[], help="checkpoints whose averaged predictions are distilled")
    fitting.add_argument("--kd-alpha", type=float, default=0.0, help="weight of the distillation term (0 disables it)")
    fitting.add_argument("--kd-temperature", type=float, default=1.0)
    fitting.add_argument("--randconv", type=float, default=0.0, help="per-image probability of a random-convolution blend")
    fitting.add_argument("--mix", choices=("none", "mixup", "cutmix", "both"), default="none")
    fitting.add_argument("--mix-alpha", type=float, default=1.0, help="Beta(alpha, alpha) for the mixing weight")
    fitting.add_argument("--mix-prob", type=float, default=0.5, help="fraction of batches that are mixed")
    fitting.add_argument("--channels-last", action="store_true")
    fitting.add_argument("--workers", type=int, default=6)
    fitting.add_argument("--resume")
    fitting.add_argument("--eval-every", type=int, default=1, help="validate every N epochs (always at the last epoch)")
    fitting.add_argument("--tta-every", type=int, default=10, help="flip-TTA validation every N epochs (always at the last epoch)")
    fitting.add_argument("--no-eval-raw", dest="eval_raw", action="store_false",
                         help="with EMA on, skip validating the raw weights (predict uses the EMA weights)")
    fitting.add_argument("--fold-teacher-bn", action="store_true", help="fold teacher BatchNorm into convolutions (checked at load)")
    inference = subparsers.add_parser("predict")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--weights", choices=("model", "ema"), help="default: EMA weights when the checkpoint has them")
    inference.add_argument("--tta", choices=("none", "flip"), default="none")
    for command in (fitting, inference):
        command.add_argument("--data", default="data")
        command.add_argument("--batch-size", type=int, default=128)
        command.add_argument("--device", default="mps")
    args, _ = parser.parse_known_args()
    if args.command == "train" and args.config:
        settings = json.loads(Path(args.config).read_text())
        unknown = set(settings) - set(vars(args))
        if unknown:
            parser.error(f"unknown keys in {args.config}: {sorted(unknown)}")
        fitting.set_defaults(**settings)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.command == "train" and bool(args.kd_alpha) != bool(args.teachers):
        parser.error("distillation needs both --teachers and a positive --kd-alpha")
    if args.command == "train" and (not math.isfinite(args.sam_rho) or args.sam_rho < 0):
        parser.error("--sam-rho must be finite and nonnegative")
    if args.command == "train" and args.sam_rho and (args.amp == "fp16" or args.kd_alpha):
        parser.error("SAM currently supports CE training with --amp none or bf16 only")
    if args.command == "train" and min(args.eval_every, args.tta_every) < 1:
        parser.error("--eval-every and --tta-every must be positive")
    if args.command == "train" and (args.ra_ops < 1 or not 0 <= args.ra_magnitude <= 30):
        parser.error("--ra-ops must be positive and --ra-magnitude must be in [0, 30]")
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
