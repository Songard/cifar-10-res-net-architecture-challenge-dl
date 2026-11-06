# %% [markdown]
# # Deep Learning Project 1: a residual CNN under 5M parameters
#
# Team **jiaqi li nyu** (Jiaqi Li, NetID jl18215), NYU CS-GY 6953 / ECE-GY 7123 Deep Learning, Fall 2026.
# Kaggle competition `cifar-10-res-net-architecture-challenge-dl`.
#
# This notebook is the executed companion of the report (`report/main.tex`). It does not train anything. It
#
# 1. rebuilds the final network from `model.py` and prints its parameter count and layout;
# 2. loads the two final checkpoints, recomputes their held-out accuracy, regenerates both Kaggle submission files with
#    `train.py predict` and checks the files' SHA-256 against the recorded Kaggle receipts;
# 3. cross-checks the report's data tables against the published per-epoch training logs;
# 4. draws the report's result figures from `report/figures/data/` with the report's own plotting code;
# 5. prints the final Kaggle test accuracy and the number of parameters.
#
# Two kinds of accuracy appear below and are kept apart:
#
# - **Held-out accuracy**: the fixed seed-2026 split of 5,000 official training images (500 per class) that no submitted
#   model trained on. It is computed here from the checkpoints and read from the training logs.
# - **Kaggle test accuracy**: the hidden 2,000-image test set, scored by Kaggle (the public half of 1,000 images during the
#   competition, the private half after it closes). We never see its labels; these numbers are copied from the Kaggle receipts.

# %%
import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from IPython.display import Image, display

from analysis import CLASSES, confusion, load_checkpoint_model, load_submissions, validation_predictions
from model import ARCHITECTURES, build_model, multiply_accumulates, parameter_count

ROOT = Path.cwd()
assert (ROOT / "train.py").exists() and (ROOT / "data/cache/audit.json").exists(), "run from the repository root with the prepared data"
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# The checkpoints are not in the repository. They are looked up as <dir>/<run>/last.pt in the colon-separated directories of
# PROJECT1_CHECKPOINTS, by default ./checkpoints and the working copy of the cluster deliveries next to this repository.
CHECKPOINT_DIRS = [Path(p) for p in os.environ.get("PROJECT1_CHECKPOINTS", "checkpoints:../project1-hpc/artifacts/candidates").split(":")]
FINAL = {"KD2 1800": dict(run="r2-kd2-ta1800-s2026", receipt=57959285, eval_key="kd2-s2026",
                          objective="cross-entropy + distillation from two of our own networks (alpha 0.9, T 4)"),
         "SAM 1800": dict(run="sam-x4-1800-s2026", receipt=58002253, eval_key="sam1800-s2026",
                          objective="cross-entropy with sharpness-aware minimization (rho 0.05)")}
for item in FINAL.values():
    item["checkpoint"] = next((d / item["run"] / "last.pt" for d in CHECKPOINT_DIRS if (d / item["run"] / "last.pt").exists()), None)
print("torch", torch.__version__, "| device", device)
for key, item in FINAL.items():
    print(f"{key}: checkpoint {item['checkpoint'] if item['checkpoint'] else 'NOT FOUND (the checkpoint sections below are skipped)'}")

# %% [markdown]
# ## Data and the held-out split
#
# The 50,000 official training images (5,000 per class) were split once, stratified by class with seed 2026, into 45,000
# training and 5,000 validation images. Normalization statistics use the 45,000 training images only, no pixel-identical
# image crosses the split, and the 2,000 hidden test images are only ever read by `train.py predict` to write a submission
# file. Every checkpoint stores this audit and the code refuses data whose audit differs.

# %%
audit = json.loads((ROOT / "data/cache/audit.json").read_text())
print("training split:", audit["training_split"], "| validation split:", audit["validation_split"], "| split seed:", audit["split_seed"])
print("validation images per class:", audit["validation_class_counts"])
print("hidden test images:", audit["test_images"], "| test labels accessed:", audit["test_labels_accessed"])
print("normalization mean:", np.round(audit["normalization_training_only_mean"], 4), "std:", np.round(audit["normalization_training_only_std"], 4))
print("pixel-identical training images excluded to avoid leakage:", len(audit["train_duplicates_excluded_to_avoid_validation_leakage"]))

# %% [markdown]
# ## Architectures and the final network
#
# Every network is built from the assignment's residual block `ReLU(S(x) + F(x))`, `F = conv-BN-ReLU-conv-BN`, with a 1x1
# convolution + BN shortcut `S` where the resolution halves (`preact-highres` uses full pre-activation blocks), a 3x3 stem
# without max-pooling, global average pooling and one linear layer. The final network, `resnet20-x4`, is the three-stage
# CIFAR ResNet-20 layout of He et al. (3 blocks per stage) at four times its width. The table reproduces Table 1 of the report
# (parameters include batch-normalization scales and shifts and the classifier).

# %%
print(f"{'architecture':18s} {'stage widths':22s} {'blocks':14s} {'parameters':>11s} {'MACs/image':>11s}")
for name, spec in ARCHITECTURES.items():
    net = build_model(name)
    print(f"{name:18s} {str(spec['widths']):22s} {str(spec['depths']):14s} {parameter_count(net):>11,} {multiply_accumulates(net) / 1e6:>10.0f}M")

final_model = build_model("resnet20-x4")
count = parameter_count(final_model)
print(f"\nFinal network resnet20-x4: {count:,} trainable parameters (limit 5,000,000; {100 * count / 5e6:.1f}% of the budget), "
      f"{multiply_accumulates(final_model) / 1e6:.0f}M multiply-accumulates per 32x32 image")
assert count <= 5_000_000

# %% [markdown]
# Where the parameters sit: the stem, three stages of three residual blocks (the first block of stages 2 and 3 halves the
# resolution with a stride-2 convolution and a 1x1 projection shortcut), and the classifier. One block of each kind is printed.

# %%
def block_parameters(blocks):
    return sum(p.numel() for block in blocks for p in block.parameters() if p.requires_grad)

stages = [(1, 0, 3, 64, "32x32"), (2, 3, 6, 128, "16x16"), (3, 6, 9, 256, "8x8")]
print(f"{'part':34s} {'parameters':>11s}")
print(f"{'stem (3x3 conv, BN, ReLU)':34s} {block_parameters([final_model.stem]):>11,}")
for stage, start, end, width, resolution in stages:
    print(f"{f'stage {stage}: {end - start} blocks, {width} channels, {resolution}':34s} {block_parameters(final_model.blocks[start:end]):>11,}")
print(f"{'classifier (global pool, linear)':34s} {block_parameters([final_model.head]):>11,}")
print(f"{'total':34s} {count:>11,}\n")
print("first block of stage 1 (identity shortcut):\n", final_model.blocks[0])
print("\nfirst block of stage 2 (stride 2, projection shortcut):\n", final_model.blocks[3])

# %% [markdown]
# ## Training recipe
#
# Both final models share `configs/final-1800.json`: SGD with Nesterov momentum, batch 256, peak learning rate 0.2 after a
# 5-epoch warm-up and one cosine decay to zero over 1,800 epochs (updated every step), weight decay 5e-4 on convolution and
# linear weights only, TrivialAugmentWide, reflect-padded random crop, horizontal flip, 16x16 Cutout, label smoothing 0.1,
# an EMA of the weights (decay 0.999 per step) that is used for evaluation and prediction, bf16 autocast. The KD2 model adds
# the distillation term, the SAM model the sharpness-aware update. The course baseline is listed for comparison.

# %%
baseline = json.loads((ROOT / "configs/baseline.json").read_text())
final_recipe = json.loads((ROOT / "configs/final-1800.json").read_text())
print(f"{'setting':20s} {'baseline':18s} {'final':18s}")
for key in sorted(set(baseline) | set(final_recipe)):
    print(f"{key:20s} {str(baseline.get(key, '-')):18s} {str(final_recipe.get(key, '-')):18s}")
print("\nTraining commands of the two final models (single A100, run as three two-hour segments each, see README):")
print("  teachers: train.py train --config configs/improved.json --arch resnet20-x4 --epochs 1000 --seed 2026 --channels-last --eval-every 10 --no-eval-raw --run runs/cloud-x4-1000-s2026 --device cuda")
print("            train.py train --config configs/improved.json --arch resnet20-x4 --epochs 800 --seed 2027 --channels-last --eval-every 10 --no-eval-raw --run runs/cloud-x4-800-s2027 --device cuda")
print("  KD2 1800: train.py train --config configs/final-1800.json --seed 2026 --teachers runs/cloud-x4-1000-s2026/last.pt runs/cloud-x4-800-s2027/last.pt --kd-alpha 0.9 --kd-temperature 4 --run runs/r2-kd2-ta1800-s2026 --device cuda")
print("  SAM 1800: train.py train --config configs/final-1800.json --seed 2026 --sam-rho 0.05 --run runs/sam-x4-1800-s2026 --device cuda")

# %% [markdown]
# ## The final checkpoints: held-out accuracy recomputed
#
# Each checkpoint holds the model, its EMA weights, the optimizer and RNG states, the data audit and the resolved
# configuration. We rebuild the network from the architecture name stored in the checkpoint, load the EMA weights and
# recompute the accuracy on the 5,000 held-out images with and without horizontal-flip test-time augmentation. The report
# quotes the flip-TTA values: 97.62% (KD2, 4,881 correct) and 97.58% (SAM, 4,879 correct).

# %%
results = {}
for key, item in FINAL.items():
    if item["checkpoint"] is None:
        print(f"{key}: checkpoint not available, skipped")
        continue
    model, checkpoint = load_checkpoint_model(str(item["checkpoint"]), device)  # EMA weights by default
    config = checkpoint["config"]
    assert config["validation_held_out"] and config["train_images"] == 45000 and checkpoint["epoch"] == config["epochs"]
    plain, labels = validation_predictions(model, device, "none")
    flip, _ = validation_predictions(model, device, "flip")
    results[key] = dict(flip=flip, labels=labels, config=config)
    print(f"{key} ({item['run']}): {config['arch']}, {parameter_count(model):,} parameters, epoch {checkpoint['epoch']} of {config['epochs']}, "
          f"trained on {config['train_images']:,} images, validation held out: {config['validation_held_out']}")
    print(f"    objective: {item['objective']}")
    print(f"    held-out accuracy, EMA weights: {100 * (plain == labels).mean():.2f}% plain, "
          f"{100 * (flip == labels).mean():.2f}% with flip TTA ({(flip == labels).sum()} / {len(labels)} correct)")
    print(f"    source revision when trained: train.py {config['code_sha256']['train.py'][:12]}, model.py {config['code_sha256']['model.py'][:12]}")

# %% [markdown]
# ## The final submission files, regenerated and matched to the Kaggle receipts
#
# `train.py predict` is the only code path that reads the hidden test images. It is run here exactly as it was run for the
# uploads (EMA weights, flip TTA, batch 256) and writes the CSV into a temporary directory. The SHA-256 of the regenerated
# file is compared with the hash recorded in the Kaggle receipt of the upload, and `check_submission.py` re-reads the file
# independently (header, 2,000 rows in `sample_submission.csv` order, labels 0-9, parameter limit) and lists the receipts
# that match it.

# %%
receipts = {r["submission_ref"]: r for r in load_submissions()}
for key, item in FINAL.items():
    if item["checkpoint"] is None:
        print(f"{key}: checkpoint not available, skipped")
        continue
    receipt = receipts[item["receipt"]]
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / item["run"] / "submission.csv"
        command = [sys.executable, "train.py", "predict", "--checkpoint", str(item["checkpoint"]), "--output", str(output),
                   "--weights", "ema", "--tta", "flip", "--device", device.type, "--batch-size", "256"]
        print(f"{key}: $ python train.py predict --checkpoint {item['checkpoint']} --output <tmp>/submission.csv --weights ema --tta flip --device {device.type} --batch-size 256")
        prediction = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        check = json.loads(subprocess.run([sys.executable, "check_submission.py", str(output)], check=True, capture_output=True, text=True).stdout)
    identical = prediction["sha256"] == receipt["csv_sha256"]
    print(f"    regenerated: {prediction['rows']} rows, held-out accuracy of these weights {100 * prediction['valid_accuracy_of_these_weights']:.2f}%, "
          f"predicted class counts {prediction['predicted_class_counts']}")
    print(f"    sha256 regenerated {prediction['sha256']}")
    print(f"    sha256 Kaggle upload {receipt['csv_sha256']} (submission {receipt['submission_ref']}, {receipt['submitted_utc'][:10]}, public score {100 * receipt['public_score']:.1f}%)")
    print(f"    -> {'IDENTICAL to the uploaded file' if identical else 'DIFFERENT from the uploaded file'}; independent check: "
          f"ids match {check['ids_match_sample_submission']}, labels valid {check['labels_valid']}, receipts for this file {[r['submission_ref'] for r in check['kaggle_receipts_for_this_file']]}")
    assert identical

# %% [markdown]
# ## Per-class accuracy and agreement of the two final models
#
# The per-class accuracies and confusion matrices computed from the checkpoints above are compared with
# `report/figures/data/heldout_eval.json`, the file the report's error analysis and Figure 5 were drawn from. Cat is the
# hardest class for both models and cat-dog confusions are the largest error group. The two models disagree on 65 of the
# 5,000 held-out images, and at least one of them is right on 98.2% of them; they come from different training families,
# which is one reason both were kept as the final two submissions.

# %%
heldout_eval = json.loads((ROOT / "report/figures/data/heldout_eval.json").read_text())
per_class = {}
for key, item in FINAL.items():
    if key not in results:
        continue
    predicted, labels = results[key]["flip"], results[key]["labels"]
    matrix = confusion(predicted, labels)
    per_class[key] = np.array([(predicted[labels == c] == c).mean() for c in range(10)])
    reference = heldout_eval[item["eval_key"]]
    print(f"{key}: confusion matrix equal to heldout_eval.json: {bool((matrix == np.array(reference['confusion'])).all())}; "
          f"accuracy {100 * matrix.trace() / matrix.sum():.2f}% (file: {100 * reference['accuracy']:.2f}%)")
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    largest = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:3], off.shape))[0]
    print("    hardest classes:", ", ".join(f"{CLASSES[c]} {100 * per_class[key][c]:.1f}%" for c in np.argsort(per_class[key])[:3]),
          "| largest confusions (true -> predicted):", [(CLASSES[i], CLASSES[j], int(off[i, j])) for i, j in largest])
if len(results) == 2:
    kd2, sam = results["KD2 1800"], results["SAM 1800"]
    labels = kd2["labels"]
    disagree = int((kd2["flip"] != sam["flip"]).sum())
    either = int(((kd2["flip"] == labels) | (sam["flip"] == labels)).sum())
    print(f"\nKD2 and SAM disagree on {disagree} held-out images; at least one is correct on {either} ({100 * either / len(labels):.2f}%); "
          f"both wrong on {int(((kd2['flip'] != labels) & (sam['flip'] != labels)).sum())}  "
          f"(heldout_eval.json: {heldout_eval['pairs']['kd2-s2026|sam1800-s2026']})")

# %%
import matplotlib.pyplot as plt  # noqa: E402

if per_class:
    fig, ax = plt.subplots(figsize=(7, 2.6), layout="constrained")
    x = np.arange(10)
    width = 0.8 / len(per_class)
    for i, (key, values) in enumerate(per_class.items()):
        ax.bar(x + (i - (len(per_class) - 1) / 2) * width, 100 * values, width, label=key, color=("#eb6834", "#1baf7a")[i])
    ax.set_xticks(x, CLASSES, rotation=30, ha="right")
    ax.set_ylim(90, 100)
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("Per-class held-out accuracy of the two final models (EMA weights, flip TTA)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    display(Image(data=buffer.getvalue()))

# %% [markdown]
# ## Every number in the report's tables, checked against the published training logs
#
# `report/figures/data/experiments.csv` holds the held-out accuracy and the A100 training-step hours of every completed
# run; its `source` column names the run each number comes from. Those runs are published as `runs/hpc/<name>/` (cluster)
# and `runs/<name>/` (laptop) with their per-epoch `metrics.jsonl`. Below, each row is recomputed from the log: the
# flip-TTA accuracy of the EMA weights at the last epoch and the sum of the logged training-step seconds. Every held-out
# value in the file equals the log; the hours were recorded with two or three decimals and the cell lists any that differ
# from the recomputed sum by more than that rounding. (For two runs, CE 2300 seed 2027 and the CE 1000 control, the
# evaluation of the same weights at prediction time differed from the training log by one held-out image; the report and
# the file use the log value, see the README.)

# %%
SCREENING = {("TA 400", "2026"): "r2-ta400-s2026", ("TA 400", "2027"): "r2-ta400-s2027", ("AA 400", "2026"): "r2-aa400-s2026",
             ("AA 400", "2027"): "r2-aa400-s2027", ("RA 400", "2026"): "r2-ra400-s2026", ("RA 400", "2027"): "r2-ra400-s2027",
             ("HighRes 400", "2026"): "r2-highres400-s2026", ("HighRes 400", "2027"): "r2-highres400-s2027",
             ("PreAct 400", "2026"): "r2-preact400-s2026", ("PreAct 400", "2027"): "r2-preact400-s2027",
             ("CE 1800", "2027"): "r2-ta1800-s2027", ("AA 1800", "2026"): "r2-aa1800-s2026"}


def run_directory(row):
    source = row["source"]
    if "cloud-evidence" in source or "screening" in source:
        return ROOT / "runs/hpc" / SCREENING[(row["label"], row["seed"])]
    name = source.split(" ")[0].rstrip("/").split("/")[-1]
    return ROOT / ("runs/hpc" if source.startswith("project1-hpc") else "runs") / name


def log_summary(directory):
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    last = metrics[-1]
    return 100 * last["tta_valid_accuracy"], math.fsum(r["train_seconds"] for r in metrics) / 3600, last["epoch"]


rows = list(csv.DictReader((ROOT / "report/figures/data/experiments.csv").open()))
print(f"{'family':6s} {'label':12s} {'seed':>4s} {'dev':5s} {'run':30s} {'heldout csv':>11s} {'from log':>8s} {'diff':>6s} {'A100h csv':>9s} {'from log':>8s}")
accuracy_mismatches, hour_differences = [], []
for row in rows:
    directory = run_directory(row)
    accuracy, hours, last_epoch = log_summary(directory)
    diff = accuracy - float(row["heldout"])
    hours_text = f"{hours:8.3f}" if row["device"] == "a100" else "   (mps)"
    print(f"{row['family']:6s} {row['label']:12s} {row['seed']:>4s} {row['device']:5s} {directory.relative_to(ROOT).as_posix():30s} "
          f"{float(row['heldout']):11.2f} {accuracy:8.2f} {diff:+6.2f} {row['a100_hours'] or '':>9s} {hours_text}")
    if abs(diff) > 1e-9:
        accuracy_mismatches.append((row["label"], row["seed"], row["heldout"], round(accuracy, 2)))
    if row["a100_hours"]:
        decimals = len(row["a100_hours"].split(".")[1]) if "." in row["a100_hours"] else 0
        if abs(hours - float(row["a100_hours"])) > 0.5 * 10 ** -decimals + 1e-9:  # the log does not round to the file value
            hour_differences.append((row["label"], row["seed"], row["a100_hours"], round(hours, 3), f"{3600 * (hours - float(row['a100_hours'])):+.0f} s"))
print(f"\n{len(rows)} rows checked. Held-out values that differ from the log: {accuracy_mismatches or 'none'}.")
print(f"A100-hour cells that differ from the recomputed sum by more than their rounding: {hour_differences or 'none'}.")

# %% [markdown]
# ## The report's figures, drawn from the data files with the report's own code
#
# `report/figures/make_results_figures.py` produces Figures 2-5 of the report from `report/figures/data/`. The same functions
# are called here (into a temporary folder, so the committed figure files are left untouched) and the results are shown
# inline.
#
# **Schedule length and cost.** (a) Held-out accuracy of every completed run against the length of its cosine schedule: the
# plain recipe gains about 0.9 points from 100 to 1,800 epochs and then saturates (2,300 and 3,000 epochs do not help);
# distillation, SAM, the other augmentation policies and the other layouts all land within about one standard error (0.22
# points, the bar) of the plain recipe at equal length. (b) The same accuracies against measured A100 training-step hours;
# the hollow markers add the time spent training the distillation teachers, which removes the small advantage of KD.

# %%
import importlib  # noqa: E402

sys.path.insert(0, str(ROOT / "report/figures"))
figures = importlib.import_module("make_results_figures")
figures.HERE = Path(tempfile.mkdtemp())  # output folder of the figure functions; their data folder stays report/figures/data
figures.schedule_figure()
display(Image(filename=figures.HERE / "schedule.png"))

# %% [markdown]
# **Training curves of the three 1,800-epoch seed-2026 runs** (plain, KD2, SAM). (a) Held-out accuracy of the EMA weights
# with flip TTA, evaluated every ten epochs; (b) the last 800 epochs; (c) accuracy on the augmented training batches. The
# three curves are indistinguishable for the first 1,500 epochs and all gain about half a point in the last 300 as the
# learning rate decays to zero. Training-batch accuracy ends at 93-94%, four points below the held-out accuracy: the strong
# augmentation remains the dominant regularizer to the end.

# %%
figures.curves_figure()
display(Image(filename=figures.HERE / "curves.png"))

# %% [markdown]
# **Held-out accuracy predicts the public leaderboard.** For the 18 submissions with a held-out score, the public score
# (1,000 hidden test images) is plotted against the held-out accuracy (5,000 images); error bars are one binomial standard
# error on each axis. The two agree up to sampling noise (slope about 1.7, r = 0.93); the hidden test images are about 8
# points harder than the held-out split. The two outlined points are the final submissions.

# %%
slope, intercept, correlation = figures.scatter_figure()
display(Image(filename=figures.HERE / "scatter.png"))
scored = sum(1 for r in csv.DictReader((ROOT / "report/figures/data/submissions.csv").open()) if r["heldout"])
print(f"least-squares fit over the {scored} submissions with a held-out score: public = {slope:.2f} x held-out {intercept:+.1f}, r = {correlation:.2f}")

# %% [markdown]
# **Confusion matrix of the KD2 model on the held-out split** (EMA weights, flip TTA). Diagonal cells show the per-class
# accuracy in percent; off-diagonal cells show error counts out of 500 images per class. Cat (93.6%) is the hardest class and
# cat-dog confusions make up 33 of the 119 errors; frog and ship are nearly solved.

# %%
figures.confusion_figure()
display(Image(filename=figures.HERE / "confusion.png"))

# %% [markdown]
# ## Kaggle submissions and the final test accuracy
#
# One receipt was recorded per upload (`artifacts/kaggle_submissions.jsonl`, 21 uploads): the CSV hash, the checkpoint, the
# held-out accuracy of the uploaded weights and the public score that Kaggle returned. The public score is Kaggle's accuracy
# on the public half (1,000 images) of the hidden test set; the private half decides the final ranking and is released only
# after the competition closes. Kaggle scores a team on its two selected submissions; ours are the two highest public
# scores, the KD2 and SAM models above.

# %%
submissions = load_submissions()
print(f"{'ref':>9s} {'uploaded':10s} {'model':16s} {'params':>9s} {'train':>6s} {'held-out':>8s} {'public':>7s} {'private':>8s}  file")
for r in submissions:
    held_out = f"{100 * r['local_heldout_valid_accuracy']:.2f}" if r["local_heldout_valid_accuracy"] is not None else "(50k)"
    private = f"{100 * r['private_score']:.1f}" if r["private_score"] is not None else "pending"
    marker = " <- final" if r["submission_ref"] in {item["receipt"] for item in FINAL.values()} else ""
    print(f"{r['submission_ref']:>9d} {r['submitted_utc'][:10]:10s} {r['arch']:16s} {r['parameter_count']:>9,} {r.get('train_images') or 45000:>6,} "
          f"{held_out:>8s} {100 * r['public_score']:>7.1f} {private:>8s}  {Path(r['file']).parent.name}{marker}")

# %%
final_parameters = parameter_count(build_model("resnet20-x4"))
print("FINAL TEST ACCURACY (Kaggle public leaderboard, 1,000 hidden test images):")
for key, item in FINAL.items():
    r = receipts[item["receipt"]]
    assert r["parameter_count"] == final_parameters and r["arch"] == "resnet20-x4"
    private = "not yet released" if r["private_score"] is None else f"{100 * r['private_score']:.1f}%"
    print(f"    {key} ({item['run']}, submission {r['submission_ref']}): {100 * r['public_score']:.1f}%   "
          f"held-out {100 * r['local_heldout_valid_accuracy']:.2f}%   private score: {private}")
print(f"NUMBER OF TRAINABLE PARAMETERS: {final_parameters:,} (resnet20-x4; limit 5,000,000)")
print("Public leaderboard position of the best score (90.1%) on 2026-11-06 at 16:33 UTC: 21st of 64 teams "
      "(median team 89.4%, top-20% cut 90.8%). The private-leaderboard scores are released after the competition closes.")
