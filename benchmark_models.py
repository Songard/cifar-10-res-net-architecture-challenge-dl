"""Short throwaway speed comparison of candidate architectures and precisions, not an accuracy experiment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import ARCHITECTURES, build_model, parameter_count
from train import AMP_DTYPES, augment_batch, memory_snapshot, seed_all, synchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=sorted(ARCHITECTURES))
    parser.add_argument("--amps", nargs="+", choices=tuple(AMP_DTYPES), default=["none", "bf16", "fp16"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[256])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="artifacts/model_speed_benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(6)
    device = torch.device("mps")
    pixels = torch.from_numpy(np.load("data/cache/train_images.npy", mmap_mode="r")[:2048].copy()).permute(0, 3, 1, 2)
    results = []
    for arch in args.archs:
        for amp in args.amps:
            for size in args.batch_sizes:
                seed_all(2026)
                model = build_model(arch).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
                scaler = torch.amp.GradScaler(device.type) if amp == "fp16" else None
                dtype = AMP_DTYPES[amp]
                maximum = memory_snapshot(device)
                elapsed = []
                for step in range(args.steps + 5):
                    synchronize(device)
                    begin = time.perf_counter()
                    x = pixels[torch.arange(step * size, (step + 1) * size) % len(pixels)].to(device).float() / 255
                    x = (augment_batch(x) - .5) / .25
                    # Arbitrary labels only drive a backwards pass. No quality claim is made.
                    labels = torch.arange(size, device=device) % 10
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                        loss = F.cross_entropy(model(x), labels)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    synchronize(device)
                    if step >= 5:
                        elapsed.append(time.perf_counter() - begin)
                    for key, value in memory_snapshot(device).items():
                        maximum[key] = max(maximum.get(key, 0), value)
                result = {"arch": arch, "amp": amp, "batch_size": size, "parameters": parameter_count(model),
                          "median_step_seconds": float(np.median(elapsed)), "images_per_second": size / float(np.median(elapsed)),
                          "finite_loss": bool(torch.isfinite(loss).item()), "sampled_memory_max": maximum,
                          "warmup_steps": 5, "measured_steps": len(elapsed)}
                results.append(result)
                print(json.dumps(result), flush=True)
                del model, optimizer, x, labels, loss
                torch.mps.empty_cache()
    Path(args.output).write_text(json.dumps({"purpose": "speed only; repeated real training images and arbitrary labels; no retained model",
                                             "torch": str(torch.__version__), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
