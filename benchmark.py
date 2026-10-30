"""Short throwaway speed comparison, not an accuracy experiment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from model import CifarResNet, parameter_count
from train import augment_batch, memory_snapshot, seed_all, synchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", default="artifacts/speed_benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(6)
    device = torch.device("mps")
    paths = sorted(Path("data/train_images").glob("*.png"))[:512]
    if len(paths) < 128:
        raise ValueError("Need at least 128 official training images for the speed test")
    pixels = torch.from_numpy(np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths])).permute(0, 3, 1, 2)
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    results = []
    for pipeline, channels_last, size in (("per-image", False, 128), ("batched", False, 128), ("batched", False, 256), ("batched", False, 512), ("batched", True, 128)):
        seed_all(2026)
        model = CifarResNet().to(device)
        layout = torch.channels_last if channels_last else torch.contiguous_format
        model.to(memory_format=layout)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
        maximum = memory_snapshot(device)
        elapsed = []
        for step in range(args.steps + 5):
            synchronize(device)
            begin = time.perf_counter()
            x = pixels[torch.arange(size) % len(pixels)].clone()
            if pipeline == "per-image":
                x = torch.stack([transform(image) for image in x])
            x = x.to(device).float() / 255
            if pipeline == "batched":
                x = augment_batch(x)
            x = ((x - .5) / .25).contiguous(memory_format=layout)
            # Arbitrary labels only drive a backwards pass. No quality claim is made.
            labels = torch.arange(size, device=device) % 10
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), labels)
            loss.backward()
            optimizer.step()
            synchronize(device)
            seconds = time.perf_counter() - begin
            if step >= 5:
                elapsed.append(seconds)
            for key, value in memory_snapshot(device).items():
                maximum[key] = max(maximum.get(key, 0), value)
        result = {"pipeline": pipeline, "channels_last": channels_last, "batch_size": size,
                  "mean_step_seconds": float(np.mean(elapsed)), "median_step_seconds": float(np.median(elapsed)),
                  "images_per_second": size / float(np.mean(elapsed)), "sampled_memory_max": maximum,
                  "parameters": parameter_count(model), "warmup_steps": 5, "measured_steps": len(elapsed)}
        results.append(result)
        print(json.dumps(result), flush=True)
        Path(args.output).write_text(json.dumps({"purpose": "speed only; repeated real training images and arbitrary labels; no retained model", "torch": str(torch.__version__), "results": results}, indent=2) + "\n")
        del model, optimizer, x, labels, loss
        torch.mps.empty_cache()
    Path(args.output).write_text(json.dumps({"purpose": "speed only; repeated real training images and arbitrary labels; no retained model", "torch": str(torch.__version__), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
