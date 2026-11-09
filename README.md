# Deep Learning Project 1: a residual CNN under 5M parameters

From-scratch CIFAR-10 classification for the NYU Fall 2026 Deep Learning Kaggle competition
(`cifar-10-res-net-architecture-challenge-dl`): a residual network with at most 5,000,000 trainable parameters, trained
only on the 50,000 official training images. No pretrained weights and no external data are used. Kaggle team:
`jiaqi li nyu` (Jiaqi Li, NetID jl18215). The report is in [`report/main.tex`](report/main.tex); the executed results
notebook is [`project1_results.ipynb`](project1_results.ipynb).

## Final results

**Final test accuracy: 87.4%** on the private leaderboard, rank 27 of 81 teams, with **4,327,754 trainable parameters**.

Both final submissions are the same network, `resnet20-x4` (`r20-x4` in the report): the three-stage CIFAR ResNet-20 of
He et al. with four times the original number of channels (64, 128 and 256), 4,327,754 trainable parameters and 648M
multiply-accumulate operations per image. Both were trained from scratch on a single A100 GPU for 1,800 epochs with one
cosine learning-rate schedule and the training procedure of [`configs/final-1800.json`](configs/final-1800.json), one with
sharpness-aware minimization (SAM) and one with knowledge distillation. Predictions use an exponential moving average (EMA)
of the weights and average the logits of each image and its horizontal mirror image (flip test-time augmentation).

| Submission | Training method | Parameters | Training images | Validation (5,000 images) | Public test (1,000) | Private test (1,000) | All test images (2,000) | Kaggle ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sam-x4-1800-s2026` (SAM) | cross-entropy with sharpness-aware minimization, rho = 0.05 | 4,327,754 | 45,000 | 97.58% | 90.0% | **87.4%** | 88.7% | 58002253 |
| `r2-kd2-ta1800-s2026` (KD2) | cross-entropy and distillation from two of our own networks | 4,327,754 | 45,000 | 97.62% | 90.1% | **87.1%** | 88.6% | 57959285 |

The hidden test set has 2,000 images. Kaggle shows the accuracy on a public half during the competition and ranks the teams
on the private half after it closes, by the better private score of the two submissions a team selected. We kept Kaggle's
default selection, our two highest public scores, so the final test accuracy is the 87.4% of the SAM model. It places the
team 27th of 81 on the private leaderboard, where the median is 86.9% and the 16th place, the top 20%, scored 87.9%; on the
final public leaderboard the team is 30th of 81 with 90.1%. Validation accuracy is measured on a fixed set of 5,000 training
images (500 per class) that no submitted model was trained on, and "all test images" is the mean of the two halves.

The main finding of the report is that, within the parameter limit, the number of training epochs matters most. With the
architecture and the training procedure fixed, validation accuracy rises from 96.7% after 100 epochs to 97.3% after 400 and
97.6% after 1,800, whereas other architectures, other augmentation policies, distillation and SAM do not improve it by more
than one standard error of the evaluation (0.22 points). The remaining errors are concentrated in the animal classes, above
all cat and dog, and about half of them are shared by all models trained for 1,800 epochs. On the private half of the test
set the ten models with the highest validation accuracy score between 86.2% and 87.4% and cannot be separated. Every number
in the report is traceable to a file in this repository (see
[Where the report's numbers come from](#where-the-reports-numbers-come-from)).

## Repository map

| Path | Contents |
| --- | --- |
| `model.py` | Residual networks and the registry of named architectures (`ARCHITECTURES`), counting of parameters and multiply-accumulate operations. |
| `train.py` | Data pipelines, training (standard, distillation, SAM), validation, EMA, exact resume, `predict`. One script for every run. |
| `prepare_data.py`, `import_official.py`, `download_data.py` | Data verification, pixel cache and the fixed split into 45,000 training and 5,000 validation images (see [Data](#data)). |
| `configs/*.json` | Training procedures. `final-1800.json` is the final one; `baseline.json`, `improved.json` and `final.json` were used on the laptop; `round2-*.json` and `sam1000.json` are the files the cluster jobs were launched with. |
| `check_submission.py` | Independent check of a prediction CSV against its audit file, the sample submission and the Kaggle receipts. |
| `evaluate_teacher_ensemble.py`, `compare_holdout_views.py` | Analyses on the validation set: accuracy of the averaged predictions of several networks, agreement between models. |
| `prepare_oof.py` | Fold caches for a review of the training labels (exploratory; not used for any result in the report). |
| `hpc_segment.sbatch`, `hpc_train.sbatch`, `hpc_python.sh` | The Slurm scripts that ran each long run as consecutive two-hour jobs on the cluster. |
| `runs/<name>/` | Laptop runs: resolved settings (`config.json`), per-epoch log (`metrics.jsonl`), submission audit. Checkpoints and CSVs are not published. |
| `runs/hpc/<name>/` | The same three files for every A100 run cited in the report, indexed in [`runs/hpc/README.md`](runs/hpc/README.md). |
| `artifacts/kaggle_submissions.jsonl` | One receipt per Kaggle upload (21): file hash, checkpoint, validation accuracy, public and private score. `kaggle_upload.py` wrote them. |
| `artifacts/kaggle_leaderboard.json` | The statistics of the final leaderboard quoted in the report (number of teams, rank, median), read from Kaggle on 2026-11-09. |
| `artifacts/final-checkpoints.sha256` | SHA-256 of the two final checkpoints (the release assets), for `shasum -a 256 -c`. |
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
backend. Cluster environment: NYU's high-performance computing cluster, one A100 GPU per job, PyTorch 2.11.0+cu128 in a
Python virtual environment inside the CUDA 12.9 Singularity image (`hpc_python.sh`).

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # pins recorded on macOS arm64
```

On Linux/CUDA install the `torch`/`torchvision` build for your CUDA version first, then the remaining pins. `train.py train`
and `train.py predict` default to `--device mps`; pass `--device cuda` or `--device cpu`. Everything below is run from the
repository root with the virtual environment's interpreter.

## Data

The competition images and CSVs are not redistributed. `data/` keeps only `cache/audit.json` (the record of the split and
the normalization) and `official_import_receipt.json` (what was downloaded and verified).

1. Download `train_labels.csv`, `test.csv` and `sample_submission.csv` from the competition's Data page into `data/`.
2. Download the `train_images` and `test_images` folders (on 2026-10-29 Kaggle's all-files ZIP and the bulk API failed and
   the per-folder download in the Data Explorer worked; `download_data.py` is the abandoned per-file API route).
3. Verify and import the images, then cache the decoded pixels and fix the split:

```sh
.venv/bin/python import_official.py --downloads /path/to/folder/with/train_images/and/test_images
.venv/bin/python prepare_data.py
```

`import_official.py` checks all 50,000 training and 2,000 test images against the CSVs and writes
`data/official_import_receipt.json`. `prepare_data.py` writes `data/cache/` (uint8 arrays, `split.npz`, `audit.json`), splits
the training images once, stratified by class with random seed 2026, into 45,000 training and 5,000 validation images,
checks that no pixel-identical image appears on both sides of the split, and computes the per-channel normalization
statistics on the 45,000 training images only. Every checkpoint stores this record, and `train.py` refuses to resume,
distill from or predict with data whose record differs. Training and validation never use the test images, and the final
pipeline reads them only through `train.py predict`, which writes the submission files; [Rules and scope](#rules-and-scope)
lists every other use of the test images during the project.

## Reproducing the final models

The final training procedure (`configs/final-1800.json`): SGD with Nesterov momentum 0.9, batch size 256, a peak learning
rate of 0.2 after a five-epoch linear warm-up, one cosine decay to zero over 1,800 epochs updated at every step, weight
decay 5e-4 on the weights of convolutional and linear layers only, TrivialAugmentWide per image followed by a random crop
after reflection padding, a random horizontal flip and a 16x16 Cutout square, label smoothing 0.1, an EMA of the weights and
batch-normalization statistics (decay 0.999 per step), mixed precision with bfloat16, channels-last memory layout, and
evaluation of the EMA weights on the validation set every 10 epochs and at the last epoch. The report calls this procedure,
without distillation or SAM, *standard training*. The random stream of the augmentation policy is re-seeded at every epoch
(`epoch_seeded_policy`), so a run split into consecutive jobs equals an uninterrupted one.

### Teachers (two networks with standard training, needed by the KD2 model)

```sh
.venv/bin/python train.py train --config configs/improved.json --arch resnet20-x4 --epochs 1000 --seed 2026 --channels-last --eval-every 10 --no-eval-raw --run runs/cloud-x4-1000-s2026 --device cuda
.venv/bin/python train.py train --config configs/improved.json --arch resnet20-x4 --epochs 800 --seed 2027 --channels-last --eval-every 10 --no-eval-raw --run runs/cloud-x4-800-s2027 --device cuda
```

They reach 97.54% and 97.50% validation accuracy, and their averaged prediction reaches 97.84%
(`evaluate_teacher_ensemble.py`). Their training time (1.03 + 0.84 hours on an A100) is the "+1.87" of Table 4.

### KD2 (final submission: public 90.1%, private 87.1%)

```sh
.venv/bin/python train.py train --config configs/final-1800.json --seed 2026 --teachers runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --kd-alpha 0.9 --kd-temperature 4 --run runs/r2-kd2-ta1800-s2026 --device cuda
```

Loss: 0.1 times the cross-entropy (label smoothing 0.1) plus 0.9 times T^2 times the KL divergence to the averaged softened
prediction of the teachers, with temperature T = 4; the teachers see the same augmented image as the student. `train.py`
refuses a teacher whose checkpoint says it was trained on the validation images, so the student can still be evaluated on
them.

### SAM (final submission: public 90.0%, private 87.4%, the final test accuracy)

```sh
.venv/bin/python train.py train --config configs/final-1800.json --seed 2026 --sam-rho 0.05 --run runs/sam-x4-1800-s2026 --device cuda
```

Each step computes the gradient, moves the weights by rho g/|g| (rho = 0.05), computes the gradient there and applies it to
the original weights; batch-normalization statistics are updated in the first pass only and the EMA follows the original
weights.

### Standard training and the other runs of the report

```sh
.venv/bin/python train.py train --config configs/final-1800.json --seed 2026 --run runs/r2-ta1800-s2026 --device cuda                                      # standard training, 1,800 epochs: validation 97.66%
.venv/bin/python train.py train --config configs/final-1800.json --seed 2027 --full-train --teachers runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --kd-alpha 0.9 --kd-temperature 4 --run runs/r2-kd2-ta1800-full-s2027 --device cuda   # KD2 on all 50,000 images
.venv/bin/python train.py train --config configs/final-1800.json --arch resnet20-w68 --seed 2026 --full-train --run runs/w68-full-s2026 --device cuda    # r20-w68 on all 50,000 images
```

Every other run is defined by its `runs/hpc/<name>/config.json` or `runs/<name>/config.json`: the `config` entry names the
JSON file it was launched with and every other entry is the resolved value of the corresponding `train.py` flag
(`--epochs`, `--seed`, `--pipeline`, `--sam-rho`, `--teachers`, `--kd-alpha`, `--full-train`, ...). The cluster jobs were
launched with `configs/round2-autoaugment.json --pipeline trivialaugment` or `configs/sam1000.json --epochs 1800`; both
resolve to exactly `configs/final-1800.json` (the test `test_published_final_runs_match_registry_and_consolidated_config`
checks this against the published `config.json` files). The laptop runs used `configs/baseline.json` (the baseline of the
course: batch size 128, learning rate 0.1, 32-bit floating point, zero padding, and no augmentation policy, Cutout, label
smoothing or EMA), `configs/improved.json` (the final training procedure at 200 epochs) and `configs/final.json` (identical
to `improved.json`), for example

```sh
.venv/bin/python train.py train --config configs/baseline.json --run runs/baseline-100ep
.venv/bin/python train.py train --config configs/improved.json --epochs 100 --arch resnet20-x4 --run runs/improved100-resnet20-x4
.venv/bin/python train.py train --config configs/improved.json --epochs 400 --arch resnet20-x4 --run runs/x4-400ep-45k
```

### Running a long run as consecutive two-hour jobs on the cluster

Cluster jobs are limited to two hours, so each long run was split into consecutive Slurm jobs. Each job stops at an absolute
epoch and the next resumes from `last.pt` (model, EMA, optimizer and random-number-generator states), chained with
`--dependency=afterok`. `hpc_segment.sbatch RUN STOP_EPOCH [train.py options]` does this: it adds `--stop-after STOP_EPOCH`,
`--resume RUN/last.pt` when a checkpoint exists, and `--device cuda`; `--epochs` stays the total number of epochs.

```sh
sbatch --job-name=kd2-e600  hpc_segment.sbatch runs/r2-kd2-ta1800-s2026 600  --config configs/final-1800.json --seed 2026 --teachers runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --kd-alpha 0.9 --kd-temperature 4
sbatch --job-name=kd2-e1200 --dependency=afterok:<id of e600>  hpc_segment.sbatch runs/r2-kd2-ta1800-s2026 1200 <same options>
sbatch --job-name=kd2-e1800 --dependency=afterok:<id of e1200> hpc_segment.sbatch runs/r2-kd2-ta1800-s2026 1800 <same options>
```

The script is published as it ran: its account and partition names are the course's, `hpc_python.sh` points at the
cluster's CUDA Singularity image and a `venv/` next to the code, and the third `[[ ... ]]` line refuses to start jobs after
2026-11-05 12:00 New York time; delete that line to reuse the script. Training in consecutive jobs and uninterrupted training produce identical checkpoints (`tests/test_round2.py`,
`tests/test_sam.py`).

### Training time

Training-step time on one A100 with batch size 256 and 45,000 images: standard training 3.9 to 4.0 s per epoch (about 2.0
hours for 1,800 epochs), KD2 5.0 s (2.5 hours) plus 1.87 hours for the two teachers, KD5 6.8 s (3.4 hours), SAM 6.4 s (3.2
hours). On the laptop GPU the same network trains at about 24 s per epoch with the same procedure (`runs/x4-400ep-45k`).
Results on other hardware need not match the published runs bit for bit; the published per-epoch logs show the expected
curves.

## Final checkpoints and the submission files

### Getting the checkpoints

The two final checkpoints are the `last.pt` files of the two runs: model and EMA weights, optimizer and random-number-generator
states, the data record and the resolved settings, 52 MB each. They are not in the git history; they are published as assets
of a GitHub release of this repository (https://github.com/Songard/cifar-10-res-net-architecture-challenge-dl/releases, once
published):

| Release asset | Run | SHA-256 | Size |
| --- | --- | --- | ---: |
| `sam-x4-1800-s2026-last.pt` | `sam-x4-1800-s2026` (SAM, final; public 90.0%, private 87.4%) | `4269fb82eca14b0ca9b8df4fb254810a3c89ac0d50c8479e022105ca0172ce7c` | 52,085,662 bytes |
| `r2-kd2-ta1800-s2026-last.pt` | `r2-kd2-ta1800-s2026` (KD2, final; public 90.1%, private 87.1%) | `79c348d8011b13954f9a6798ea10a73ae62502a25ea5606803875234b71ce5d4` | 52,085,726 bytes |

Download both, place them as `checkpoints/<run>/last.pt` (the folder is ignored by git) and verify them against
[`artifacts/final-checkpoints.sha256`](artifacts/final-checkpoints.sha256):

```sh
mkdir -p checkpoints/r2-kd2-ta1800-s2026 checkpoints/sam-x4-1800-s2026
mv ~/Downloads/r2-kd2-ta1800-s2026-last.pt checkpoints/r2-kd2-ta1800-s2026/last.pt
mv ~/Downloads/sam-x4-1800-s2026-last.pt checkpoints/sam-x4-1800-s2026/last.pt
shasum -a 256 -c artifacts/final-checkpoints.sha256      # sha256sum -c on Linux
```

Any directory organized the same way (`<run>/last.pt`) works instead: set `PROJECT1_CHECKPOINTS=/path/to/dir` (a
colon-separated list is allowed) before running the notebook. By default the notebook looks in `checkpoints/`, then in
`../project1-hpc/artifacts/candidates/` and `../project1-hpc/runs/`, the author's copies of the cluster runs, and in `runs/`.
The older laptop checkpoints under `runs/*/` load with the current code as well (checkpoints of the original baseline record
only `width`; `make_model` handles both).

### Reproducing and checking the submission files

`train.py predict` loads a checkpoint, rebuilds the network named in it, predicts the 2,000 test images (EMA weights, flip
test-time augmentation), writes `id,label` rows in the order of `sample_submission.csv` plus an `.audit.json` file (SHA-256 of
the CSV and the checkpoint, architecture, number of parameters, epoch, predicted class counts, and the validation accuracy of
exactly those weights), and refuses to run if the network has more than 5,000,000 parameters. `check_submission.py` reads the
CSV and its audit file again independently (header, 2,000 rows in the order of `sample_submission.csv`, labels 0 to 9, hashes,
parameter limit) and lists the Kaggle receipts recorded for that exact file. With the checkpoints in place and the data
prepared:

```sh
.venv/bin/python train.py predict --checkpoint checkpoints/r2-kd2-ta1800-s2026/last.pt --output predictions/r2-kd2-ta1800-s2026/submission.csv --weights ema --tta flip --device mps --batch-size 256
.venv/bin/python train.py predict --checkpoint checkpoints/sam-x4-1800-s2026/last.pt --output predictions/sam-x4-1800-s2026/submission.csv --weights ema --tta flip --device mps --batch-size 256
.venv/bin/python check_submission.py predictions/r2-kd2-ta1800-s2026/submission.csv
.venv/bin/python check_submission.py predictions/sam-x4-1800-s2026/submission.csv
shasum -a 256 predictions/*/submission.csv
```

Pass `--device cuda` or `--device cpu` on other machines; the files are identical on MPS and CPU. Each `predict` prints its
audit, including `valid_accuracy_of_these_weights` (the validation accuracy) and the `sha256` of the CSV, which must equal the
hash in the Kaggle receipt of the upload (`artifacts/kaggle_submissions.jsonl`, listed by `check_submission.py` under
`kaggle_receipts_for_this_file`). The receipt holds the public and private scores Kaggle returned for that file. Expected
values, verified with this code on the laptop (MPS and CPU):

| Checkpoint | Validation accuracy of these weights | SHA-256 of `submission.csv` (= Kaggle receipt) | Kaggle ref | Public | Private |
| --- | ---: | --- | --- | ---: | ---: |
| `checkpoints/r2-kd2-ta1800-s2026/last.pt` | 0.9762 | `e50ca98dd471606492eafec432da509c5d5967105c880a0ce98eb162fc7b39a1` | 57959285 | 90.1% | 87.1% |
| `checkpoints/sam-x4-1800-s2026/last.pt` | 0.9758 | `7aec582b09645ecdd1c983491c350bb87e08fc6be5408f7ecca8ea6cbf62f2ee` | 58002253 | 90.0% | 87.4% |

## Parameter count

The number of parameters is printed at the start of every training run, stored in every `config.json` and audit file, checked
against the limit of 5,000,000 by `train.py` and the tests, and printed by the notebook. To print it directly:

```sh
.venv/bin/python -c "from model import build_model, parameter_count, multiply_accumulates; m = build_model('resnet20-x4'); print(parameter_count(m), multiply_accumulates(m))"
```

prints `4327754 647694848`. All trainable parameters are counted, including the scales and shifts of batch normalization and
the classifier. The registry (`model.py`), with the names the report uses:

| Name in `model.py` | In the report | Channels per stage | Blocks per stage | Parameters | Multiply-accumulate operations per image |
| --- | --- | --- | --- | ---: | ---: |
| `resnet18-w32` (baseline of the course) | `r18-w32` | 32, 64, 128, 256 | 2, 2, 2, 2 | 2,797,610 | 139M |
| `resnet18-w42` (laptop speed test only) | | 42, 84, 168, 336 | 2, 2, 2, 2 | 4,815,940 | 240M |
| `resnet18-64-352` | `r18-64-352` | 64, 128, 256, 352 | 2, 2, 2, 1 | 4,797,002 | 453M |
| `resnet20-x4` (final) | `r20-x4` | 64, 128, 256 | 3, 3, 3 | 4,327,754 | 648M |
| `resnet20-w68` | `r20-w68` | 68, 136, 272 | 3, 3, 3 | 4,884,926 | 731M |
| `resnet-highres` | `highres` | 96, 160, 224 | 3, 3, 2 | 3,527,082 | 956M |
| `preact-highres` (pre-activation blocks) | `preact` | 96, 160, 224 | 3, 3, 2 | 3,526,314 | 956M |

Every network uses the residual block of the assignment, `ReLU(S(x) + F(x))` with `F = conv-BN-ReLU-conv-BN`. `S` is the
identity, or a 1x1 convolution with batch normalization where the shape changes (`preact-highres` uses full pre-activation
blocks instead). A 3x3 convolution without max-pooling starts the network, as in the CIFAR adaptation of the reference code
recommended by the instructor ([kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)), and global average
pooling and one linear layer end it.

## Where the report's numbers come from

The notebook recomputes the numbers below and compares each with the value quoted in the report; its last cell prints how
many were compared and whether any differs.

| Report | Source in this repository |
| --- | --- |
| Table 1 and Section 2.2 (architectures: parameters, multiply-accumulate operations, time per epoch) | `model.py` (`parameter_count`, `multiply_accumulates`) and `train_seconds` in `runs/hpc/*/metrics.jsonl`; notebook, "Network architecture" |
| Table 2 and Section 2.3 (training procedures) | `configs/baseline.json` and `configs/final-1800.json`; notebook, "Training procedure" |
| Sections 2.4 to 2.6 (the teachers and their averaged prediction, 97.84% and 97.96%; the extra time per step of distillation and SAM; GPU-hours) | the logs in `runs/hpc/` and `runs/`; `evaluate_teacher_ensemble.py runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --output <dir> --device mps` computes the averaged prediction (subset accuracy at temperature 1); notebook, "Distillation teachers, training time and computing resources" |
| Tables 3, 4 and 5, Sections 3.1 to 3.3, Figure 2 | `report/figures/data/experiments.csv`; its `source` column names the run behind every number. `project1-hpc/artifacts/candidates/<name>` and `project1-hpc/runs/<name>` are published here as `runs/hpc/<name>`; `runs/<name>` are the laptop runs; the "round-2 screening" rows are `runs/hpc/r2-{ta,aa,ra,highres,preact}400-s{2026,2027}`, `r2-ta1800-s2027` and `r2-aa1800-s2026`. Notebook, "Validation results", compares every row with the logs and rebuilds the three tables |
| Figure 3 and Section 3.4 (training curves) | `report/figures/data/curves_1800.json`, extracted from `runs/hpc/{r2-ta1800-s2026,r2-kd2-ta1800-s2026,sam-x4-1800-s2026,r2-kd2-ta1800-s2027}/metrics.jsonl` |
| Table 6 and Section 3.5 (public and private scores, the final test accuracy, the three conclusions drawn from the private scores) | `report/figures/data/submissions.csv` (validation, public and private accuracy of the 21 submissions), the same scores as in `artifacts/kaggle_submissions.jsonl`; the statistics of the leaderboard (81 teams, rank 27, median 86.9%) are in `artifacts/kaggle_leaderboard.json`. Notebook, "Hidden test set", recomputes every number of the section |
| Section 3.6 (error analysis), Figure 4 (confusion matrix), Figure 5 (errors per class, shared errors) | `report/figures/data/error_analysis.json`, written by `report/figures/make_error_analysis.py` from six checkpoints. Figure 4 is drawn by `confusion_figure` from `heldout_eval.json` (the same confusion matrix) and Figure 5 by `errors_figure`, both in `make_results_figures.py`. Notebook, "Error analysis", repeats the computation from the checkpoints |
| Figure 6 (the twelve misclassified validation images) | drawn by `make_error_analysis.py`, which also lists the images in `error_analysis.json` (`hard_examples`) |
| Figure 1 (the final pipeline) | `report/figures/make_pipeline.py`, from `model.py`, the configurations and one training image |

`report/figures/make_results_figures.py` draws Figures 2 to 5 (`schedule_figure`, `curves_figure`, `confusion_figure`,
`errors_figure`); its `scatter_figure`, public against validation accuracy, is no longer used by the report.
`report/STYLE_SOURCE.md` records how the PDF is built and how the report was revised. Of the checkpoints the analyses need,
only those of the two final models are published; the notebook skips the steps whose checkpoints it does not find and reads
those numbers from the data files.

Validation accuracies in the report and in `experiments.csv` are the training log's evaluation at the last epoch, with the
EMA weights and flip test-time augmentation (`tta_valid_accuracy` in the last row of `metrics.jsonl`). Of the 22 cluster runs
with a validation set that were also evaluated at prediction time (`submission.audit.json`, and the Kaggle receipt where
uploaded), 20 give the same number both times and two differ by one validation image: `r2-ta2300-s2027` (log 97.52%,
prediction time 97.54%) and `ce-x4-1000-epochseed-s2026` (log 97.64%, prediction time 97.62%). The report uses the log value
for both. The two final checkpoints give the same accuracy in the cluster log, on MPS and on CPU. The KD run of 400 epochs
on the cluster (`runs/hpc/cloud-kd-x4-400-s2026`) holds epochs 273 to 400 only, because it resumed the laptop run
`runs/kd-x4-400ep-45k` (epochs 1 to 272).

## Notebook

`project1_results.ipynb` is executed and committed with its outputs. It rebuilds the final network and prints its number of
parameters and its architecture; loads the two final checkpoints, recomputes their validation accuracy, regenerates both
submission files with `train.py predict` and compares their SHA-256 with the Kaggle receipts; rebuilds Tables 1 and 3 to 6
from the logs and the data files; recomputes every number of Section 3.5 (hidden test set) and of Section 3.6 (error
analysis); shows the six figures of the report; and ends with the final test accuracy and the number of parameters. It needs
the prepared data cache (`data/cache/`) and, for the steps that load models, the checkpoints (see
[Getting the checkpoints](#getting-the-checkpoints)); a step whose checkpoints are missing prints that it was skipped, and the
rest still runs. To re-execute it from its jupytext source:

```sh
.venv/bin/python -m jupytext --to ipynb --set-kernel python3 --output project1_results.ipynb project1_results.py
.venv/bin/python -c "import nbformat, nbclient; nb = nbformat.read('project1_results.ipynb', 4); nbclient.NotebookClient(nb, timeout=1800, kernel_name='python3').execute(); nbformat.write(nb, 'project1_results.ipynb')"
```

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

32 tests (about one minute on the laptop CPU): the number of parameters of every registered architecture, forward and
backward passes, channels-last memory layout, the separation of training and validation images and the pixel-duplicate
check, the bounds of the augmentation and of Cutout, the learning-rate schedule, the EMA, flip test-time augmentation, the
distillation targets and the refusal of a teacher trained on the validation images, checkpoint and random-number-generator
round trips, SAM's perturbation and exact restoration of the weights, the per-epoch seeding of the augmentation policies, and
bit-exact resume of runs split into consecutive jobs for every pipeline (including SAM with bfloat16). Tests marked "needs
the verified official dataset" or "needs the published cluster run records" are skipped when those files are absent.

## Code provenance

Every `config.json` records `code_sha256`, the SHA-256 of `model.py`, `train.py` and `prepare_data.py` when the run
started (and `code_sha256_at_start` for a run resumed in a later job). The published sources are the last revision; the
earlier ones differ only by features added later, with the defaults that reproduce the earlier behaviour (`LATER_DEFAULTS` in
`train.py`), and all published checkpoints load with the current code.

| `train.py` SHA-256 (prefix) | `model.py` | Runs |
| --- | --- | --- |
| `a0591802` (published) | `b7c1165e` | `sam-x4-1000-s2026`, `sam-x4-1800-s2026`, `ce-x4-1000-epochseed-s2026` |
| `bf50b737` (= published minus SAM) | `b7c1165e` (`b093303d` with the `resnet20-w68` entry) | all `r2-*` runs including the KD2 and KD5 models, `w68-full-s2026` |
| `7313ea2a` | `beeef94f` (no `resnet-highres`/`preact-highres`) | `cloud-*` runs including the two KD2 teachers |
| `d0599bf4`, `64480d8b` | `beeef94f` | laptop runs of 2026-10-30 (`runs/x4-400ep-45k`, `runs/final-resnet18-64-352-45k`, `runs/improved100-resnet18-w32`, `runs/kd-x4-400ep-45k`, `runs/lab-*`) |
| `88bb18a1`, `0f35931e`, `8c40bf05` | `6623f36c`, `07ea9703` | laptop runs of 2026-10-29 (`runs/improved100-*`, `runs/final-resnet18-64-352-50k`, the baseline and pilot runs; the oldest checkpoints record only `width`) |

`kaggle_upload.py` is the script that uploaded the CSVs through the Kaggle CLI and appended the receipts in
`artifacts/kaggle_submissions.jsonl`; it holds no credentials (the CLI's own token store is used) and is kept as the record
of how the receipts were produced.

## Rules and scope

- The networks are trained from scratch on the competition's training images only; no pretrained weights, no external data.
- Every model-selection decision during development used the fixed validation set. The public leaderboard was used only for
  the final choice among submissions that the validation set does not separate: Kaggle's default selection of the two
  highest public scores was kept.
- The final pipeline reads the hidden test images only through `train.py predict`, to write and reproduce the submission
  files, with no batch-norm adaptation or pseudo-labelling.
- On 2026-10-30 we examined the test images without labels to find out why public accuracy was so far below validation
  accuracy. These are all the uses outside `train.py predict`:
  - image statistics compared with the validation set (grayscale, clipping, borders, spectrum, noise, compression), including
    per-channel affine and histogram maps fitted between the two sets;
  - a nearest-neighbour search against the 45,000 training images for near-duplicates, which kept distances only and not
    which training image matched;
  - contact sheets, and a blind rating of a sample of test and validation images for typicality and image type, with no class
    labels;
  - a classifier separating test from validation images on image statistics and on network features, used to reweight
    validation errors;
  - agreement between the predictions of submitted models.

  The findings influenced which experiments we tried next, for example random convolutions, which lowered validation
  accuracy and were dropped. Nothing from this examination enters training or prediction, and its outputs and images are
  not published.
- Not in the repository: competition data and caches, checkpoints, prediction CSVs, the course handout, the cluster's console
  output and the local console logs (`.gitignore`).

## References

- He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385
- He, Zhang, Ren, Sun. Identity Mappings in Deep Residual Networks. ECCV 2016. https://arxiv.org/abs/1603.05027
- kuangliu/pytorch-cifar, the CIFAR reference code recommended by the instructor; the 3x3 first convolution without max-pooling follows it. https://github.com/kuangliu/pytorch-cifar
- Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009.
- Ioffe, Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015. https://arxiv.org/abs/1502.03167
- Zagoruyko, Komodakis. Wide Residual Networks. BMVC 2016. https://arxiv.org/abs/1605.07146
- Sutskever, Martens, Dahl, Hinton. On the Importance of Initialization and Momentum in Deep Learning. ICML 2013.
- Loshchilov, Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts (cosine schedule). ICLR 2017. https://arxiv.org/abs/1608.03983
- He et al. Bag of Tricks for Image Classification with Convolutional Neural Networks (no weight decay on biases, warm-up). CVPR 2019. https://arxiv.org/abs/1812.01187
- Micikevicius et al. Mixed Precision Training. ICLR 2018. https://arxiv.org/abs/1710.03740
- Mueller, Hutter. TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. ICCV 2021. https://arxiv.org/abs/2103.10158
- Cubuk et al. AutoAugment (CVPR 2019, https://arxiv.org/abs/1805.09501) and RandAugment (NeurIPS 2020, https://arxiv.org/abs/1909.13719).
- DeVries, Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017. https://arxiv.org/abs/1708.04552
- Szegedy et al. Rethinking the Inception Architecture for Computer Vision (label smoothing). CVPR 2016. https://arxiv.org/abs/1512.00567
- Polyak, Juditsky. Acceleration of Stochastic Approximation by Averaging. 1992 (averaging of the weights).
- Zhang et al. mixup (ICLR 2018); Yun et al. CutMix (ICCV 2019); Xu et al. Robust and Generalizable Visual Representation Learning via Random Convolutions (ICLR 2021). Tried and not used.
- Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015. https://arxiv.org/abs/1503.02531
- Beyer et al. Knowledge Distillation: A Good Teacher Is Patient and Consistent. CVPR 2022. https://arxiv.org/abs/2106.05237
- Foret, Kleiner, Mobahi, Neyshabur. Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR 2021. https://arxiv.org/abs/2010.01412
- Wightman, Touvron, Jegou. ResNet Strikes Back: An Improved Training Procedure in timm. 2021. https://arxiv.org/abs/2110.00476
- Jordan. 94% on CIFAR-10 in 3.29 Seconds on a Single GPU. 2024. https://arxiv.org/abs/2404.00498
