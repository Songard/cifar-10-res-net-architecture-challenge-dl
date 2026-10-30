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
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from model import CifarResNet, build_model, parameter_count
from prepare_data import rows

PARAMETER_LIMIT = 5_000_000
# Settings that define an experiment; resuming with any of them changed would mix two experiments in one log.
EXPERIMENT_KEYS = ("arch", "epochs", "batch_size", "pipeline", "channels_last", "lr", "seed", "weight_decay",
                   "bn_weight_decay", "warmup_epochs", "lr_per_step", "padding_mode", "cutout",
                   "label_smoothing", "ema_decay", "amp", "full_train", "teachers", "kd_alpha", "kd_temperature")
AMP_DTYPES = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}


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


class PolicyImages(torch.utils.data.Dataset):
    """TrivialAugmentWide (Mueller & Hutter, 2021) per image on CPU workers; crop, flip and Cutout stay batched on the device."""

    def __init__(self, path, indices, labels):
        self.path, self.indices, self.labels = path, indices, labels
        self.images = None
        self.policy = transforms.TrivialAugmentWide()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode="r")
        index = int(self.indices[position])
        image = self.policy(Image.fromarray(np.asarray(self.images[index])))
        return torch.from_numpy(np.array(image)).permute(2, 0, 1), int(self.labels[index])


class Data:
    def __init__(self, root, device, pipeline, channels_last, full_train=False, padding_mode="constant", cutout=0, workers=6):
        self.root = root
        self.device = device
        self.pipeline = pipeline
        self.channels_last = channels_last
        self.padding_mode = padding_mode
        self.cutout = cutout
        self.workers = workers
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

    def finish(self, x, training):
        """Device-side part of every pipeline: scale, crop/flip, normalize, Cutout, memory layout."""
        x = x.to(self.device, dtype=torch.float32) / 255
        if training and self.pipeline != "per-image":
            x = augment_batch(x, padding_mode=self.padding_mode)
        x = (x - self.mean) / self.std
        if training and self.cutout:
            x = cutout_batch(x, self.cutout)
        memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
        return x.contiguous(memory_format=memory_format)

    def batches(self, indices, batch_size, training=False):
        if training and self.pipeline == "trivialaugment":
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
            generator = torch.Generator().manual_seed(int(torch.randint(2**31 - 1, ())))
            dataset = PolicyImages(self.root / "cache/train_images.npy", self.train.numpy(), self.labels.numpy())
            parallel = dict(num_workers=self.workers, persistent_workers=True, prefetch_factor=4) if self.workers else {}
            self.loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, generator=generator, **parallel)
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


def load_teachers(paths, device, data):
    """Frozen teacher networks for distillation (Hinton et al., 2015), each trained here from scratch on the same data.

    A student that reports held-out validation accuracy may only learn from teachers that never fit those images.
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
        teachers.append(teacher.eval().requires_grad_(False))
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
    data = Data(Path(args.data), device, args.pipeline, args.channels_last, args.full_train, args.padding_mode, args.cutout, args.workers)
    model = build_model(args.arch).to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    count = parameter_count(model)
    assert count <= PARAMETER_LIMIT
    optimizer = make_optimizer(model, args)
    ema = EMA(model, args.ema_decay) if args.ema_decay else None
    teachers = load_teachers(args.teachers, device, data) if args.kd_alpha else []
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
            if checkpoint["config"].get(key) != config[key]:
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
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = model(x)
                loss = F.cross_entropy(output, y, label_smoothing=args.label_smoothing)
                if teachers:
                    loss = (1 - args.kd_alpha) * loss + args.kd_alpha * distillation_loss(output, teachers, x, args.kd_temperature)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
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
        valid_loss, valid_accuracy = evaluate(model, data, args.batch_size)
        row = {"epoch": epoch+1, "lr": lr, "train_loss": train_loss / len(data.train),
               "train_accuracy": train_correct / len(data.train), "valid_loss": valid_loss,
               "valid_accuracy": valid_accuracy, "validation_held_out": data.held_out}
        selected = valid_accuracy
        if ema is not None:
            row["ema_valid_loss"], row["ema_valid_accuracy"] = evaluate(ema.model, data, args.batch_size)
            selected = row["ema_valid_accuracy"]
        if epoch + 1 == end or (epoch + 1) % 10 == 0:
            final = ema.model if ema is not None else model
            row["tta_valid_loss"], row["tta_valid_accuracy"] = evaluate(final, data, args.batch_size, "flip")
        synchronize(device)
        valid_seconds = time.perf_counter() - validation_start
        row.update(train_seconds=train_seconds, valid_seconds=valid_seconds,
                   train_images_per_second=len(data.train) / train_seconds, sampled_memory_max=maximum,
                   process_lifetime_peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if platform.system() == "Darwin" else 1e6))
        improved = data.held_out and selected > best
        best = max(best, selected) if data.held_out else best
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
    fitting.add_argument("--no-bn-weight-decay", dest="bn_weight_decay", action="store_false")
    fitting.add_argument("--warmup-epochs", type=float, default=5)
    fitting.add_argument("--lr-per-step", action="store_true")
    fitting.add_argument("--pipeline", choices=("per-image", "batched", "trivialaugment"), default="batched")
    fitting.add_argument("--padding-mode", choices=("constant", "reflect"), default="constant")
    fitting.add_argument("--cutout", type=int, default=0)
    fitting.add_argument("--label-smoothing", type=float, default=0.0)
    fitting.add_argument("--ema-decay", type=float, default=0.0)
    fitting.add_argument("--amp", choices=tuple(AMP_DTYPES), default="none")
    fitting.add_argument("--full-train", action="store_true", help="also fit the 5,000 validation images (final run only)")
    fitting.add_argument("--teachers", nargs="*", default=[], help="checkpoints whose averaged predictions are distilled")
    fitting.add_argument("--kd-alpha", type=float, default=0.0, help="weight of the distillation term (0 disables it)")
    fitting.add_argument("--kd-temperature", type=float, default=1.0)
    fitting.add_argument("--channels-last", action="store_true")
    fitting.add_argument("--workers", type=int, default=6)
    fitting.add_argument("--resume")
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
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
