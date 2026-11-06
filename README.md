# Deep Learning Project 1: a residual CNN under 5M parameters

From-scratch CIFAR-10 classification for the NYU Fall 2026 Deep Learning Kaggle competition
(`cifar-10-res-net-architecture-challenge-dl`): a residual network with at most 5,000,000 trainable parameters, trained
only on the 50,000 official training images. No pretrained weights and no external data are used. Kaggle team:
`jiaqi li nyu` (Jiaqi Li, NetID jl18215). The report is in [`report/main.tex`](report/main.tex); the executed results
notebook is [`project1_results.ipynb`](project1_results.ipynb).

## Final results

Both final submissions are the same network, `resnet20-x4`: the three-stage CIFAR ResNet-20 layout of He et al. at four
times its width (64, 128, 256 channels), **4,327,754 trainable parameters**, 648M multiply-accumulates per image. Both were
trained from scratch on a single A100 for one 1,800-epoch cosine cycle with the recipe in
[`configs/final-1800.json`](configs/final-1800.json) and are evaluated with their EMA weights and horizontal-flip test-time
augmentation.

| Submission | Training objective | Params | Train images | Held-out (5,000) | Kaggle public (1,000) | Kaggle private | Kaggle ref |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `r2-kd2-ta1800-s2026` (KD2) | cross-entropy + distillation from two of our own networks | 4,327,754 | 45,000 | **97.62%** | **90.1%** | released after the competition closes | 57959285 |
| `sam-x4-1800-s2026` (SAM) | cross-entropy with sharpness-aware minimization, rho = 0.05 | 4,327,754 | 45,000 | **97.58%** | **90.0%** | released after the competition closes | 58002253 |

"Held-out" is the fixed seed-2026 split of 5,000 training images (500 per class) that no submitted model trained on.
"Kaggle public" is Kaggle's score on the public half of the hidden test set. On the public leaderboard the best score ranked
21st of 64 teams on 2026-11-06 at 16:33 UTC. The private-leaderboard scores are not available at the time of writing.

Across more than forty completed runs, schedule length was the largest lever (96.7% held-out at 100 epochs, 97.3% at 400,
97.6% at 1,800); beyond that, distillation, SAM, other augmentation policies and three other residual layouts moved the
held-out accuracy by less than one standard error (0.22 points). The report has the details; every number in it is traceable
to a file in this repository (see [Where the report's numbers come from](#where-the-reports-numbers-come-from)).

## Repository map

| Path | Contents |
| --- | --- |
| `model.py` | Residual networks and the named architecture registry (`ARCHITECTURES`), parameter and multiply-accumulate counting. |
| `train.py` | Data pipelines, training (plain, distillation, SAM), validation, EMA, exact resume, `predict`. One script for every run. |
| `prepare_data.py`, `import_official.py`, `download_data.py` | Data verification, pixel cache and the frozen 45,000/5,000 split (see [Data](#data)). |
| `configs/*.json` | Training recipes. `final-1800.json` is the final recipe; `baseline.json`, `improved.json`, `final.json` are the laptop-phase recipes; `round2-*.json`, `sam1000.json` are the files the cluster jobs were launched with. |
| `check_submission.py` | Independent check of a prediction CSV against its audit sidecar, the sample submission and the Kaggle receipts. |
| `evaluate_teacher_ensemble.py`, `compare_holdout_views.py` | Held-out analyses used in the report: teacher-ensemble accuracy, pairwise agreement of candidates. |
| `prepare_oof.py` | Fold caches for a training-label review (exploratory; not used for any result in the report). |
| `hpc_segment.sbatch`, `hpc_train.sbatch`, `hpc_python.sh` | The Slurm wrappers that ran the cluster jobs in two-hour segments. |
| `runs/<name>/` | Laptop runs: resolved settings (`config.json`), per-epoch log (`metrics.jsonl`), submission audit. Checkpoints and CSVs are not published. |
| `runs/hpc/<name>/` | The same three files for every A100 run cited in the report, indexed in [`runs/hpc/README.md`](runs/hpc/README.md). |
| `artifacts/kaggle_submissions.jsonl` | One receipt per Kaggle upload (21): file hash, checkpoint, held-out accuracy, public score. `kaggle_upload.py` wrote them. |
| `report/` | AAAI-format report source, figure scripts (`figures/make_*.py`) and the figure data (`figures/data/`). |
| `project1_results.ipynb` (`project1_results.py`) | Executed results notebook and its jupytext source. `analysis.py` holds its read-only helpers. |
| `review_runs.py` | Summary table and curves of every run in `runs/` (writes `artifacts/experiments_summary.json`, `artifacts/experiment_curves.png`). |
| `tests/` | Unit tests (`python -m unittest discover -s tests -v`). |

History from the laptop phase that no result in the report depends on: `benchmark.py`, `benchmark_models.py`,
`artifacts/model_speed_benchmark.json`, `artifacts/speed_benchmark*.json`, `artifacts/pilot_*`, and the 3- and 5-epoch pilot
runs. `artifacts/preregistration-2026-10-30-lab.*` holds the public-score predictions that were written down before the
laptop-phase submissions were uploaded.

## Setup

Laptop environment: Apple M5 Pro (20-core GPU, 48 GB), macOS 26.5.1, Python 3.12.14, PyTorch 2.15.0, torchvision 0.29.2, MPS
backend. Cluster environment: NYU HPC cloud-bursting nodes, one A100 per job, PyTorch 2.11.0+cu128 in a Python virtual
environment inside the CUDA 12.9 Singularity image (`hpc_python.sh`).

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # pins recorded on macOS arm64
```

On Linux/CUDA install the `torch`/`torchvision` build for your CUDA version first, then the remaining pins. `train.py train`
and `train.py predict` default to `--device mps`; pass `--device cuda` or `--device cpu`. Everything below is run from the
repository root with the virtual environment's interpreter.

## Data

The competition images and CSVs are not redistributed. `data/` keeps only `cache/audit.json` (the split and normalization
record) and `official_import_receipt.json` (what was downloaded and verified).

1. Download `train_labels.csv`, `test.csv` and `sample_submission.csv` from the competition's Data page into `data/`.
2. Download the `train_images` and `test_images` folders (on 2026-10-29 Kaggle's all-files ZIP and the bulk API failed and
   the per-folder download in the Data Explorer worked; `download_data.py` is the abandoned per-file API route).
3. Verify and import the images, then cache the decoded pixels and freeze the split:

```sh
.venv/bin/python import_official.py --downloads /path/to/folder/with/train_images/and/test_images
.venv/bin/python prepare_data.py
```

`import_official.py` checks all 50,000 training and 2,000 test images against the CSVs and writes
`data/official_import_receipt.json`. `prepare_data.py` writes `data/cache/` (uint8 arrays, `split.npz`, `audit.json`), splits
the training images once, stratified by class with seed 2026, into 45,000 training and 5,000 validation images, checks that no
pixel-identical image crosses the split, and computes per-channel normalization statistics on the 45,000 training images
only. Every checkpoint stores this audit, and `train.py` refuses to resume, distill from or predict with data whose audit
differs. The test images are read by `train.py predict` only, to write the submission file; nothing is adapted to them.

## Reproducing the final models

The final recipe (`configs/final-1800.json`): SGD with Nesterov momentum 0.9, batch 256, peak learning rate 0.2 after a
5-epoch linear warm-up, one cosine decay to zero over 1,800 epochs updated every step, weight decay 5e-4 on convolution and
linear weights only, TrivialAugmentWide per image followed by a reflect-padded random crop, horizontal flip and 16x16 Cutout,
label smoothing 0.1, an EMA of the weights and BN statistics (decay 0.999 per step), bf16 autocast, channels-last memory
layout, validation of the EMA weights every 10 epochs (and always at the last epoch). The augmentation policy's random
stream is re-seeded at every epoch boundary (`epoch_seeded_policy`) so that a run split into segments equals an uninterrupted
one.

### Teachers (two plain-recipe networks, needed by the KD2 model)

```sh
.venv/bin/python train.py train --config configs/improved.json --arch resnet20-x4 --epochs 1000 --seed 2026 --channels-last --eval-every 10 --no-eval-raw --run runs/cloud-x4-1000-s2026 --device cuda
.venv/bin/python train.py train --config configs/improved.json --arch resnet20-x4 --epochs 800 --seed 2027 --channels-last --eval-every 10 --no-eval-raw --run runs/cloud-x4-800-s2027 --device cuda
```

They reach 97.54% and 97.50% held-out; their averaged prediction scores 97.84% (`evaluate_teacher_ensemble.py`). Their
training time (1.03 + 0.84 A100-hours) is the "+1.87" in the report's cost table.

### KD2 (final submission, Kaggle 90.1%)

```sh
.venv/bin/python train.py train --config configs/final-1800.json --seed 2026 --teachers runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --kd-alpha 0.9 --kd-temperature 4 --run runs/r2-kd2-ta1800-s2026 --device cuda
```

Loss: 0.1 cross-entropy (label smoothing 0.1) + 0.9 T^2 KL to the teachers' averaged softened prediction, T = 4; teachers see
the same augmented view as the student. `train.py` refuses a teacher whose checkpoint says it trained on the validation
images, so the student's held-out score stays a held-out score.

### SAM (final submission, Kaggle 90.0%)

```sh
.venv/bin/python train.py train --config configs/final-1800.json --seed 2026 --sam-rho 0.05 --run runs/sam-x4-1800-s2026 --device cuda
```

Each step computes the gradient, moves the weights by rho g/|g| (rho = 0.05), computes the gradient there and applies it at the
original weights; BN statistics are updated in the first pass only and the EMA tracks the base weights.

### Plain recipe and the other runs in the report

```sh
.venv/bin/python train.py train --config configs/final-1800.json --seed 2026 --run runs/r2-ta1800-s2026 --device cuda                                      # CE 1800: 97.66%
.venv/bin/python train.py train --config configs/final-1800.json --seed 2027 --full-train --teachers runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --kd-alpha 0.9 --kd-temperature 4 --run runs/r2-kd2-ta1800-full-s2027 --device cuda   # KD2 on all 50,000 images
.venv/bin/python train.py train --config configs/final-1800.json --arch resnet20-w68 --seed 2026 --full-train --run runs/w68-full-s2026 --device cuda    # r20-w68 on all 50,000 images
```

Every other run is defined by its `runs/hpc/<name>/config.json` or `runs/<name>/config.json`: the `config` entry names the
JSON file it was launched with and every other entry is the resolved value of the corresponding `train.py` flag
(`--epochs`, `--seed`, `--pipeline`, `--sam-rho`, `--teachers`, `--kd-alpha`, `--full-train`, ...). The cluster jobs were
launched with `configs/round2-autoaugment.json --pipeline trivialaugment` or `configs/sam1000.json --epochs 1800`; both
resolve to exactly `configs/final-1800.json` (the test `test_published_final_runs_match_registry_and_consolidated_config`
checks this against the published `config.json` files). The laptop-phase recipes are `configs/baseline.json` (the course
baseline: batch 128, learning rate 0.1, fp32, zero padding, no policy augmentation, Cutout, label smoothing or EMA),
`configs/improved.json` (the final recipe at 200 epochs) and `configs/final.json` (identical to `improved.json`), for example

```sh
.venv/bin/python train.py train --config configs/baseline.json --run runs/baseline-100ep
.venv/bin/python train.py train --config configs/improved.json --epochs 100 --arch resnet20-x4 --run runs/improved100-resnet20-x4
.venv/bin/python train.py train --config configs/improved.json --epochs 400 --arch resnet20-x4 --run runs/x4-400ep-45k
```

### Running in two-hour segments on the cluster

Jobs on the course's cloud-bursting partition are limited to two hours, so every long run was a chain of Slurm jobs,
each stopping at an absolute epoch and the next resuming from `last.pt` (model, EMA, optimizer and RNG states), chained with
`--dependency=afterok`. `hpc_segment.sbatch RUN STOP_EPOCH [train.py options]` does this: it adds `--stop-after STOP_EPOCH`,
`--resume RUN/last.pt` when a checkpoint exists, and `--device cuda`; `--epochs` stays the full schedule length.

```sh
sbatch --job-name=kd2-e600  hpc_segment.sbatch runs/r2-kd2-ta1800-s2026 600  --config configs/final-1800.json --seed 2026 --teachers runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --kd-alpha 0.9 --kd-temperature 4
sbatch --job-name=kd2-e1200 --dependency=afterok:<id of e600>  hpc_segment.sbatch runs/r2-kd2-ta1800-s2026 1200 <same options>
sbatch --job-name=kd2-e1800 --dependency=afterok:<id of e1200> hpc_segment.sbatch runs/r2-kd2-ta1800-s2026 1800 <same options>
```

The script is published as it ran: its account and partition names are the course's, `hpc_python.sh` points at the
cluster's CUDA Singularity image and a `venv/` next to the code, and the third `[[ ... ]]` line is a competition-deadline
guard that refuses to start jobs after 2026-11-05 12:00 New York time; delete that line to reuse the script. Segmented and
uninterrupted training produce identical checkpoints (`tests/test_round2.py`, `tests/test_sam.py`).

### Cost

Training-step time on one A100 at batch 256 (45,000 images): plain recipe 3.9-4.0 s per epoch (about 2.0 A100-hours for 1,800
epochs), KD2 5.0 s (2.53 h) plus 1.87 h for the two teachers, KD5 6.8 s (3.4 h), SAM 6.4 s (3.2 h). On the laptop GPU the
same network trains at about 24 s per epoch with the same recipe (`runs/x4-400ep-45k`). Results need not match the
published runs bit for bit on other hardware; the published per-epoch logs show the expected curves.

## Predicting and checking a submission

`train.py predict` loads a checkpoint, rebuilds the network named in it, predicts the 2,000 test images (EMA weights, flip
TTA), writes `id,label` rows in `sample_submission.csv` order plus a `.audit.json` sidecar (CSV and checkpoint SHA-256,
architecture, parameter count, epoch, predicted class counts, and the held-out accuracy of exactly those weights), and
refuses to run if the parameter count exceeds 5,000,000. `check_submission.py` re-reads the CSV and its audit independently
and lists the Kaggle receipts recorded for that exact file.

```sh
.venv/bin/python train.py predict --checkpoint checkpoints/r2-kd2-ta1800-s2026/last.pt --output predictions/kd2/submission.csv --weights ema --tta flip --device mps --batch-size 256
.venv/bin/python check_submission.py predictions/kd2/submission.csv
```

Expected output for the two final checkpoints (verified on 2026-11-06 with this code on the laptop, on both the MPS and the
CPU device):

| Checkpoint | Checkpoint SHA-256 | Held-out accuracy of these weights | `submission.csv` SHA-256 (= Kaggle receipt) |
| --- | --- | ---: | --- |
| `r2-kd2-ta1800-s2026/last.pt` | `79c348d8011b13954f9a6798ea10a73ae62502a25ea5606803875234b71ce5d4` | 0.9762 | `e50ca98dd471606492eafec432da509c5d5967105c880a0ce98eb162fc7b39a1` |
| `sam-x4-1800-s2026/last.pt` | `4269fb82eca14b0ca9b8df4fb254810a3c89ac0d50c8479e022105ca0172ce7c` | 0.9758 | `7aec582b09645ecdd1c983491c350bb87e08fc6be5408f7ecca8ea6cbf62f2ee` |

**Checkpoints.** The checkpoints (52 MB each) are not in the git history. They are distributed separately (the repository's
Releases page, once published, or from the author). Put them under `checkpoints/<run>/last.pt` (the notebook also looks in
`../project1-hpc/artifacts/candidates/<run>/last.pt`, the working copy of the cluster deliveries, and in the directories
listed in the environment variable `PROJECT1_CHECKPOINTS`, colon-separated). The older laptop checkpoints under `runs/*/`
load with the current code as well (checkpoints of the original baseline record only `width`; `make_model` handles both).

## Parameter count

The count is printed at the start of every training run, stored in every `config.json` and audit sidecar, checked against
the 5,000,000 limit by `train.py` and the tests, and printed by the notebook. To print it directly:

```sh
.venv/bin/python -c "from model import build_model, parameter_count, multiply_accumulates; m = build_model('resnet20-x4'); print(parameter_count(m), multiply_accumulates(m))"
```

prints `4327754 647694848`. All trainable parameters are counted, including batch-normalization scales and shifts and the
classifier. The registry (`model.py`):

| Name | Stage widths | Blocks per stage | Parameters | MACs / image |
| --- | --- | --- | ---: | ---: |
| `resnet18-w32` (course baseline) | 32, 64, 128, 256 | 2, 2, 2, 2 | 2,797,610 | 139M |
| `resnet18-w42` | 42, 84, 168, 336 | 2, 2, 2, 2 | 4,815,940 | 240M |
| `resnet18-64-352` | 64, 128, 256, 352 | 2, 2, 2, 1 | 4,797,002 | 453M |
| `resnet20-x4` (final) | 64, 128, 256 | 3, 3, 3 | 4,327,754 | 648M |
| `resnet20-w68` | 68, 136, 272 | 3, 3, 3 | 4,884,926 | 731M |
| `resnet-highres` | 96, 160, 224 | 3, 3, 2 | 3,527,082 | 956M |
| `preact-highres` (pre-activation blocks) | 96, 160, 224 | 3, 3, 2 | 3,526,314 | 956M |

Every network uses the residual block of the assignment, `ReLU(S(x) + F(x))` with `F = conv-BN-ReLU-conv-BN` and a 1x1
convolution + BN shortcut where the shape changes (`preact-highres` uses full pre-activation blocks instead), a 3x3 stem
without max-pooling as in the CIFAR adaptation of the instructor-recommended reference code
([kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)), global average pooling and one linear layer.

## Where the report's numbers come from

| Report | Source in this repository |
| --- | --- |
| Table 1 (layouts, parameters, MACs, s/epoch) | `model.py` (`parameter_count`, `multiply_accumulates`; the notebook prints the table); `train_seconds` in `runs/hpc/*/metrics.jsonl` |
| Table 3 (progression), Table 4 (long runs), Table 5 (screening), Figure 2 | `report/figures/data/experiments.csv`; its `source` column names the run directory of every number. `project1-hpc/artifacts/candidates/<name>` and `project1-hpc/runs/<name>` are published here as `runs/hpc/<name>`; `runs/<name>` are the laptop runs; the "round-2 screening" rows are `runs/hpc/r2-{ta,aa,ra,highres,preact}400-s{2026,2027}`, `r2-ta1800-s2027` and `r2-aa1800-s2026` |
| Public scores, Table 6, Figure 4 | `artifacts/kaggle_submissions.jsonl` (`report/figures/data/submissions.csv` is the same information in table form) |
| Figure 3 (1,800-epoch curves) | `report/figures/data/curves_1800.json`, extracted from `runs/hpc/{r2-ta1800-s2026,r2-kd2-ta1800-s2026,sam-x4-1800-s2026,r2-kd2-ta1800-s2027}/metrics.jsonl` |
| Figure 5 (confusion matrix), per-class accuracies, pairwise disagreement (65 images, 98.2% either correct) | `report/figures/data/heldout_eval.json`, produced by re-evaluating the final checkpoints on the fixed split (the notebook recomputes and compares) |
| Teacher-ensemble accuracies 97.84% (KD2 teachers) and 97.96% (KD5 teachers) | `evaluate_teacher_ensemble.py runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --output <dir> --device mps` (`compare` averages the members' softened probabilities; both numbers are the T = 1 subset accuracy) |
| Section 5.5 fit over the 18 submissions with a held-out score (public = 1.73 x held-out - 80.1, r = 0.93) | computed by `report/figures/make_results_figures.py` from `submissions.csv`; the notebook prints it |

`report/figures/make_results_figures.py` draws Figures 2-5 from the data files and `make_pipeline.py` draws Figure 1 from
`model.py` and the configs; `report/STYLE_SOURCE.md` records how the PDF is built.

Held-out numbers in the report and in `experiments.csv` are the training log's final-epoch evaluation of the EMA weights
with flip TTA (`tta_valid_accuracy` of the last row of `metrics.jsonl`). Of the 22 held-out cloud runs that were also
evaluated again at prediction time (`submission.audit.json`, and the Kaggle receipt where uploaded), 20 give the same
number both times and two differ by one held-out image: `r2-ta2300-s2027` (log 97.52%, prediction-time 97.54%) and
`ce-x4-1000-epochseed-s2026` (log 97.64%, prediction-time 97.62%); the report uses the log value for both, as its
Limitations paragraph states. The two final checkpoints agree exactly across the cloud log, MPS and CPU; the notebook
prints the full comparison of `experiments.csv` with the published logs. The KD-400 cloud run
(`runs/hpc/cloud-kd-x4-400-s2026`) holds epochs 273-400 only because it resumed the laptop run `runs/kd-x4-400ep-45k`
(epochs 1-272).

## Notebook

`project1_results.ipynb` is executed and committed with its outputs. It rebuilds the final network and prints its parameter
count and layout, loads the two final checkpoints and recomputes their held-out accuracy, regenerates both submission files
with `train.py predict` and compares their SHA-256 with the Kaggle receipts, cross-checks `experiments.csv` against the
published per-epoch logs, draws the report's figures with the report's own code, and prints the final Kaggle test accuracy
and parameter count. It needs the prepared data cache (`data/cache/`) and, for the checkpoint sections, the two final
checkpoints (see above); without them those sections print what is missing and the rest still runs. To re-execute it from
its jupytext source:

```sh
.venv/bin/python -m jupytext --to ipynb --set-kernel python3 --output project1_results.ipynb project1_results.py
.venv/bin/python -c "import nbformat, nbclient; nb = nbformat.read('project1_results.ipynb', 4); nbclient.NotebookClient(nb, timeout=1800, kernel_name='python3').execute(); nbformat.write(nb, 'project1_results.ipynb')"
```

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

32 tests (about one minute on the laptop CPU): parameter budgets of every registered layout, forward/backward, channels-last
layout, split isolation and pixel-duplicate guard, augmentation and Cutout bounds, the learning-rate schedule, EMA, flip
TTA, distillation targets and the teacher leakage guard, checkpoint/RNG round trips, SAM's perturbation and exact weight
restoration, per-epoch seeded policies, and bit-exact resume of segmented runs for every pipeline (including SAM in bf16).
Tests marked "needs the verified official dataset" or "needs the published cluster run records" are skipped when those
files are absent.

## Code provenance

Every `config.json` records `code_sha256`, the SHA-256 of `model.py`, `train.py` and `prepare_data.py` when the run
started (and `code_sha256_at_start` for resumed segments). The published sources are the last revision; the earlier ones
differ only by features added later, with the defaults that reproduce the earlier behaviour (`LATER_DEFAULTS` in
`train.py`), and all published checkpoints load with the current code.

| `train.py` SHA-256 (prefix) | `model.py` | Runs |
| --- | --- | --- |
| `a0591802` (published) | `b7c1165e` | `sam-x4-1000-s2026`, `sam-x4-1800-s2026`, `ce-x4-1000-epochseed-s2026` |
| `bf50b737` (= published minus SAM) | `b7c1165e` (`b093303d` with the `resnet20-w68` entry) | all `r2-*` runs including the KD2 and KD5 models, `w68-full-s2026` |
| `7313ea2a` | `beeef94f` (no `resnet-highres`/`preact-highres`) | `cloud-*` runs including the two KD2 teachers |
| `d0599bf4`, `64480d8b` | `beeef94f` | laptop runs of 2026-10-30 (`runs/x4-400ep-45k`, `runs/final-resnet18-64-352-45k`, `runs/improved100-resnet18-w32`, `runs/kd-x4-400ep-45k`, `runs/lab-*`) |
| `88bb18a1`, `0f35931e`, `8c40bf05` | `6623f36c`, `07ea9703` | laptop runs of 2026-10-29 (`runs/improved100-*`, `runs/final-resnet18-64-352-50k`, the baseline and pilot runs; the oldest checkpoints record only `width`) |

`kaggle_upload.py` is the script that uploaded candidate CSVs through the Kaggle CLI and appended the receipts in
`artifacts/kaggle_submissions.jsonl`; it holds no credentials (the CLI's own token store is used) and is kept as the record
of how the receipts were produced.

## Rules and scope

- The networks are trained from scratch on the competition's training images only; no pretrained weights, no external data.
- Model selection used the fixed held-out split. The leaderboard was consulted only to choose between finalists the
  held-out split could not separate (Kaggle's default selection of the two highest public scores was kept).
- The hidden test images are read once, by `train.py predict`, to write the submission file: no batch-norm adaptation,
  pseudo-labelling or inspection.
- Not in the repository: competition data and caches, checkpoints, prediction CSVs, the course handout, cluster stdout and
  the local console logs (`.gitignore`).

## References

- He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385
- He, Zhang, Ren, Sun. Identity Mappings in Deep Residual Networks. ECCV 2016. https://arxiv.org/abs/1603.05027
- kuangliu/pytorch-cifar (instructor-recommended CIFAR reference code; the 3x3 stem adaptation follows it). https://github.com/kuangliu/pytorch-cifar
- Zagoruyko, Komodakis. Wide Residual Networks. BMVC 2016. https://arxiv.org/abs/1605.07146
- Mueller, Hutter. TrivialAugment. ICCV 2021. https://arxiv.org/abs/2103.10158
- Cubuk et al. AutoAugment (CVPR 2019) and RandAugment (NeurIPS 2020).
- DeVries, Taylor. Improved Regularization of CNNs with Cutout. 2017. https://arxiv.org/abs/1708.04552
- Szegedy et al. Rethinking the Inception Architecture (label smoothing). CVPR 2016. https://arxiv.org/abs/1512.00567
- Loshchilov, Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts (cosine schedule). ICLR 2017. https://arxiv.org/abs/1608.03983
- He et al. Bag of Tricks for Image Classification with CNNs (no bias decay, warm-up). CVPR 2019. https://arxiv.org/abs/1812.01187
- Polyak, Juditsky. Acceleration of Stochastic Approximation by Averaging. 1992 (weight averaging).
- Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015. https://arxiv.org/abs/1503.02531
- Beyer et al. Knowledge Distillation: A Good Teacher Is Patient and Consistent. CVPR 2022. https://arxiv.org/abs/2106.05237
- Foret, Kleiner, Mobahi, Neyshabur. Sharpness-Aware Minimization. ICLR 2021. https://arxiv.org/abs/2010.01412
- Wightman, Touvron, Jegou. ResNet strikes back. 2021. https://arxiv.org/abs/2110.00476
- Recht, Roelofs, Schmidt, Shankar. Do ImageNet Classifiers Generalize to ImageNet? ICML 2019. https://arxiv.org/abs/1902.10811
- Zhang et al. mixup (ICLR 2018); Yun et al. CutMix (ICCV 2019); Xu et al. Robust and Generalizable Visual Representation Learning via Random Convolutions (ICLR 2021). Tried and dropped.
- Micikevicius et al. Mixed Precision Training. ICLR 2018. https://arxiv.org/abs/1710.03740
- PyTorch MPS backend notes: https://docs.pytorch.org/docs/stable/notes/mps.html
