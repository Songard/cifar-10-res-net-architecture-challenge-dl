# Deep Learning Project 1: a residual CNN under 5M parameters

From-scratch CIFAR-10-style classification for the NYU Fall 2026 Kaggle competition
(`cifar-10-res-net-architecture-challenge-dl`): a residual network with at most 5,000,000 trainable
parameters, trained only on the 50,000 official training images. No pretrained weights or external data are used.
Kaggle team: `jiaqi li nyu`. The task follows the course's Project 1 handout (course material, not redistributed here).

**Final model:** `resnet18-64-352`, **4,797,002 parameters**, EMA weights with horizontal-flip test-time augmentation.
**Kaggle public test accuracy: 86.30%** (submission 57784806, 2026-10-29; the private score is released after the
competition closes). Held-out validation accuracy of the same checkpoint: 96.42%.

The executed notebook [`project1_results.ipynb`](project1_results.ipynb) shows every result below and prints the
final test accuracy and parameter count; the report source is in [`report/`](report/).

## Results on the fixed held-out split

All numbers are accuracy on the fixed seed-2026 split of 5,000 validation images (500 per class) that the
model never trained on. They are local validation results, not Kaggle test scores.

| Run | Architecture | Parameters | Recipe | Epochs | Validation (last epoch) | EMA + flip TTA |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `baseline-100ep` | `resnet18-w32` | 2,797,610 | baseline | 100 | 94.48% | - |
| `improved100-resnet18-64-352` | `resnet18-64-352` | 4,797,002 | improved | 100 | 95.90% (EMA 96.02%) | **96.42%** |
| `improved100-resnet20-x4` | `resnet20-x4` | 4,327,754 | improved | 100 | paused after epoch 35 (unfinished) | |

The architecture and the recipe both changed between the two finished runs, so their gain is not attributed
to either one; a recipe-only ablation (`resnet18-w32` with the improved recipe) was planned but not run.

**Kaggle.** `runs/improved100-resnet18-64-352/submission.csv` was submitted on 2026-10-29 (ref 57784806) and scored
86.30% public test accuracy, versus 96.42% held-out validation. The gap suggests the hidden test images are
distribution-shifted relative to the training images; without test labels that is a hypothesis. Upload receipts
are in `artifacts/kaggle_submissions.jsonl`.

**Unfinished.** Training was paused on 2026-10-29 because the laptop was low on power. The planned final run
`final-resnet18-64-352-50k` (the final recipe for 200 epochs on all 50,000 images) is paused after epoch 41 of
200. Its held-out twin `final-resnet18-64-352-45k` has not started; until it exists, the nearest held-out evidence
for this recipe is the 100-epoch run above. A full-train run also fits the 5,000 validation images, so its
"validation" numbers are flagged `validation_held_out: false` and are not generalization estimates.

`review_runs.py` regenerates `artifacts/experiments_summary.json` and `artifacts/experiment_curves.png` from the logs.

## Architectures (`model.py`)

Every network uses the course's residual block `ReLU(S(x) + F(x))`, `F = conv-BN-ReLU-conv-BN`, with a 1x1
conv + BN shortcut when the shape changes, a 3x3 stem (no ImageNet 7x7 stem or max-pool), global average
pooling and one linear layer. Names are stored in checkpoints so prediction rebuilds the same network.

| Name | Stage widths | Blocks per stage | Parameters | Multiply-accumulates / image |
| --- | --- | --- | ---: | ---: |
| `resnet18-w32` (baseline) | 32, 64, 128, 256 | 2, 2, 2, 2 | 2,797,610 | 139M |
| `resnet18-w42` | 42, 84, 168, 336 | 2, 2, 2, 2 | 4,815,940 | 240M |
| `resnet18-64-352` (final) | 64, 128, 256, 352 | 2, 2, 2, 1 | 4,797,002 | 453M |
| `resnet20-x4` | 64, 128, 256 | 3, 3, 3 | 4,327,754 | 648M |

`resnet18-64-352` keeps ResNet-18's full-width first three stages and trims the last stage, which holds
most of ResNet-18's 11M parameters but little of its compute, to one 352-channel block. On this Apple GPU the
saved benchmark (`artifacts/model_speed_benchmark.json`, bf16, batch 256) runs `resnet18-64-352` at 0.72x the
throughput of `resnet18-w42` (about 1.4x the time per image) despite 1.9x its compute, so the full-width early
stages cost much less time than their compute suggests.

## Training recipes (`train.py`, `configs/`)

`configs/baseline.json` holds the original pilot recipe: batch 128, SGD with Nesterov momentum 0.9,
weight decay 5e-4 on all parameters, learning rate 0.1 with 5 warmup epochs then cosine decay (updated per
epoch), zero-padded random crop and horizontal flip, fp32.

`configs/improved.json` (and the identical `configs/final.json`, which the 200-epoch runs use) is the improved recipe:

- batch 256, peak learning rate 0.2, linear warmup for 5 epochs then cosine decay, updated every step;
- weight decay 5e-4 on convolution and linear weights only (none on BN parameters or biases);
- TrivialAugmentWide (Mueller & Hutter, 2021) per image on CPU workers, then reflect-padded random 32x32 crop,
  horizontal flip and 16x16 Cutout (DeVries & Taylor, 2017) batched on the GPU;
- label smoothing 0.1 (Szegedy et al., 2016);
- an exponential moving average of the weights and BN statistics (decay 0.999 per step), used for evaluation
  and prediction;
- horizontal-flip test-time augmentation (average of the logits of the image and its mirror);
- bf16 autocast on the MPS device (weights and optimizer state stay fp32), 1.6-2.4x faster than fp32 in a short
  benchmark (`artifacts/model_speed_benchmark.json`).

Optional knowledge distillation (`--teachers ... --kd-alpha ... --kd-temperature ...`; Hinton et al., 2015)
averages the softened predictions of teacher checkpoints trained here from scratch on the same data. A run that
reports held-out accuracy refuses teachers that trained on the validation images. It is not used by any result above.

**Provenance.** The reported runs were trained with earlier revisions of `train.py`/`model.py`; the SHA-256 of
each source file is recorded in every run's `config.json` (`code_sha256`). The current code loads their
checkpoints (old baseline checkpoints record only `width`) and the notebook evaluates them with it. Re-running the
commands below repeats the same recipes, but results need not match the reported runs bit for bit.
`runs/baseline-100ep` is the 5-epoch pilot `runs/batched-5ep` (started with the 100-epoch schedule) resumed from its
epoch-5 checkpoint with the same seed.

## Environment

Apple M5 Pro (20-core GPU, 48 GB unified memory), macOS 26.5.1, Python 3.12.14, PyTorch 2.15.0,
torchvision 0.29.2, MPS device. With the improved recipe, `resnet18-64-352` trains at about 2,575 images/s
(about 17.4 s per 45,000-image epoch) when it has the GPU to itself; process RSS stays below 0.8 GB and sampled
MPS driver allocation peaks at about 3.4 GB.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # pins recorded on macOS arm64 (MPS)
```

On Linux/CUDA, install the torch/torchvision build that matches your CUDA version first, then the remaining pins,
and pass `--device cuda` (or `--device cpu`) to `train.py train` and `train.py predict`, which default to `mps`.
`benchmark.py` and `benchmark_models.py` are MPS-only speed tests.

## Data preparation

The competition images and CSVs are not redistributed here (`data/` keeps only `cache/audit.json`, the split and
normalization record, and `official_import_receipt.json`).

1. Download `train_labels.csv`, `test.csv` and `sample_submission.csv` from the competition Data page into `data/`.
2. In Kaggle's Data Explorer, select the `train_images` and `test_images` folders and click each folder's
   **Download** button (on 2026-10-29 the all-files ZIP and the bulk API failed; per-folder download worked).
3. Verify and import the images, then cache decoded pixels and freeze the seed-2026 45,000/5,000 split:

```sh
.venv/bin/python import_official.py --downloads /path/to/folder/with/train_images/and/test_images
.venv/bin/python prepare_data.py
```

`import_official.py` checks all 50,000 training and 2,000 test images against the CSVs (names, counts, 32x32 RGB)
and writes `data/official_import_receipt.json`. `prepare_data.py` writes `data/cache/` and `audit.json`, checks
that no identical-pixel image crosses the split, and computes normalization statistics on the 45,000 training
images only (full-train runs reuse them). Test labels are never accessed. `download_data.py` is an earlier,
abandoned per-file Kaggle API download route (rate-limited); it is kept only as history.

## Commands

Run from this directory.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python train.py train --config configs/baseline.json --run runs/baseline-new
.venv/bin/python train.py train --config configs/improved.json --epochs 100 --arch resnet18-64-352 --run runs/improved100-resnet18-64-352
```

Reproduce the Kaggle-scored file (EMA weights, flip TTA) and check it independently. `predict` writes `id,label`
rows in `sample_submission.csv` order plus a `.audit.json` sidecar (CSV and checkpoint hashes, architecture,
parameter count, weights, TTA, predicted class counts, and the validation accuracy of exactly those weights);
`check_submission.py` re-reads both and lists any Kaggle receipt recorded for that exact file.

```sh
.venv/bin/python train.py predict --checkpoint runs/improved100-resnet18-64-352/last.pt --weights ema --tta flip --output runs/improved100-resnet18-64-352/submission.csv
.venv/bin/python check_submission.py runs/improved100-resnet18-64-352/submission.csv
```

Planned, unfinished: the final recipe for 200 epochs (`logs/final-resnet18-64-352.sh`). The 50k run is paused after
epoch 41; resume it with the same settings plus `--resume runs/final-resnet18-64-352-50k/last.pt`. The 45k twin has
not started, and the 50k predictions need the finished checkpoint.

```sh
.venv/bin/python train.py train --config configs/final.json --full-train --run runs/final-resnet18-64-352-50k
.venv/bin/python train.py train --config configs/final.json --run runs/final-resnet18-64-352-45k
```

`--stop-after N` stops after total epoch N. Re-execute the results notebook (needs the prepared data and the
`runs/` checkpoints, which are not in the repository; it uses MPS when available, otherwise CPU):

```sh
.venv/bin/python -m ipykernel install --user --name project1 --display-name "project1 (.venv)"
.venv/bin/jupytext --to ipynb --set-kernel project1 project1_results.py
.venv/bin/python -c "import nbformat, nbclient; nb = nbformat.read('project1_results.ipynb', 4); nbclient.NotebookClient(nb, timeout=900).execute(); nbformat.write(nb, 'project1_results.ipynb')"
```

## Files

- `model.py`: residual networks, the named architecture registry, parameter and multiply-accumulate counting.
- `train.py`: data pipelines, training, validation, EMA, distillation, checkpoint resume and prediction.
- `configs/*.json`: recipe settings; every resolved setting is also written to each run's `config.json`.
- `import_official.py`, `prepare_data.py`: data verification, caching and the frozen split (see above).
- `project1_results.ipynb` (source `project1_results.py`, jupytext): executed results notebook; it regenerates
  `report/figures/` and prints the final test accuracy and parameter count. `analysis.py`: its read-only helpers.
- `report/`: AAAI-format LaTeX report (`main.tex`, `references.bib`, `figures/`). The included AAAI 2015 style
  and bibliography files are documented in `report/STYLE_SOURCE.md`. Compile with `tectonic main.tex`, or
  use pdfLaTeX and BibTeX. The first-page repository placeholder must be replaced before submission.
- `review_runs.py`: run summary table and curves. `check_submission.py`: independent CSV/audit check.
- `benchmark_models.py`: architecture/precision speed test behind `artifacts/model_speed_benchmark.json`.
- `runs/*/metrics.jsonl`, `config.json`, `submission.audit.json`: per-epoch logs, resolved settings and the
  submission audit (published). Checkpoints (`last.pt`, `best.pt`, about 21-40 MB each), the data cache, the
  official images and all prediction CSVs are kept out of the repository (competition rules and size).
  `logs/`: console output and the queue scripts.
- `tests/test_pipeline.py`: parameter budgets, gradients, layout, augmentation, Cutout, schedule, EMA, TTA,
  distillation leakage guard, split isolation, full-train split, prediction writer, checkpoint/RNG roundtrip.

History (superseded, not results): the 3- and 5-epoch pilots of `resnet18-w32` (`runs/baseline-3ep`,
`runs/batched-5ep`, `artifacts/pilot_summary.json`, `artifacts/pilot_curves.png`, whose 100-epoch times are
extrapolations), and `benchmark.py`, a pilot-only comparison of per-image vs batched augmentation
(`artifacts/speed_benchmark.json`, `artifacts/speed_benchmark_full_download.json`).

## References

- He et al., Deep Residual Learning for Image Recognition, 2016. https://arxiv.org/abs/1512.03385
- Instructor-recommended CIFAR reference code: https://github.com/kuangliu/pytorch-cifar
- DeVries & Taylor, Improved Regularization of CNNs with Cutout, 2017. https://arxiv.org/abs/1708.04552
- Mueller & Hutter, TrivialAugment, 2021. https://arxiv.org/abs/2103.10158
- Szegedy et al., Rethinking the Inception Architecture (label smoothing), 2016. https://arxiv.org/abs/1512.00567
- Polyak & Juditsky, Acceleration of Stochastic Approximation by Averaging, 1992 (iterate averaging).
- Hinton, Vinyals & Dean, Distilling the Knowledge in a Neural Network, 2015. https://arxiv.org/abs/1503.02531
- Loshchilov & Hutter, SGDR (cosine schedule), 2017. https://arxiv.org/abs/1608.03983
- Recht et al., Do ImageNet Classifiers Generalize to ImageNet?, 2019. https://arxiv.org/abs/1902.10811
- PyTorch MPS documentation: https://docs.pytorch.org/docs/stable/notes/mps.html
