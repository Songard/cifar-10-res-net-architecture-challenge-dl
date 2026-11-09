"""Per-image error analysis on the fixed 5,000-image validation set (Section 3.6 of the report).

Reads only the validation split of the labeled training data; it never touches the hidden test images. Every model is
evaluated like the submissions: averaged (EMA) weights, logits averaged over each image and its mirror image.

Inputs: the run checkpoints (not in the repository, see README) and data/cache from prepare_data.py.
Outputs: data/error_analysis.json with every number quoted in Section 3.6, and hard_examples.pdf/.png in this directory.
"""
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CANDIDATES = PROJECT.parent / "project1-hpc" / "artifacts" / "candidates"
sys.path.insert(0, str(PROJECT))
from train import Data, make_model, predict_logits  # noqa: E402

CHECKPOINTS = {
    "r20-x4 standard 100": PROJECT / "runs/improved100-resnet20-x4/last.pt",
    "r18-64-352 standard 100": PROJECT / "runs/improved100-resnet18-64-352/last.pt",
    "r20-x4 standard 1800": CANDIDATES / "r2-ta1800-s2026/last.pt",
    "r20-x4 KD2 1800 s2026": CANDIDATES / "r2-kd2-ta1800-s2026/last.pt",
    "r20-x4 KD2 1800 s2027": CANDIDATES / "r2-kd2-ta1800-s2027/last.pt",
    "r20-x4 SAM 1800": CANDIDATES / "sam-x4-1800-s2026/last.pt",
}
LONG = ["r20-x4 standard 1800", "r20-x4 KD2 1800 s2026", "r20-x4 KD2 1800 s2027", "r20-x4 SAM 1800"]
SHORT = "r20-x4 standard 100"
FINAL = "r20-x4 KD2 1800 s2026"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ANIMALS = {2, 3, 4, 5, 6, 7}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data = Data(PROJECT / "data", device, "batched", False)
valid = data.valid.numpy()
labels = data.labels[data.valid].numpy()

probs = {}
for name, path in CHECKPOINTS.items():
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in data.batches(data.valid, 500):
            out.append(torch.softmax(predict_logits(model, x, "flip"), 1).cpu())
    probs[name] = torch.cat(out).numpy()

preds = {name: p.argmax(1) for name, p in probs.items()}
wrong = {name: pred != labels for name, pred in preds.items()}

result = {"classes": CLASSES, "models": {}}
for name, p in probs.items():
    confusion = np.zeros((10, 10), dtype=int)
    for true, pred in zip(labels, preds[name]):
        confusion[true, pred] += 1
    top = p.max(1)
    second = np.argsort(p, 1)[:, -2]
    result["models"][name] = {
        "accuracy": float((~wrong[name]).mean()),
        "errors": int(wrong[name].sum()),
        "per_class": [float(confusion[i, i] / confusion[i].sum()) for i in range(10)],
        "errors_per_class": [int(confusion[i].sum() - confusion[i, i]) for i in range(10)],
        "confusion": confusion.tolist(),
        "cat_dog_confusions": int(confusion[3, 5] + confusion[5, 3]),
        "confidence_correct_mean": float(top[~wrong[name]].mean()),
        "confidence_wrong_mean": float(top[wrong[name]].mean()),
        "confidence_correct_median": float(np.median(top[~wrong[name]])),
        "errors_above_correct_median": int((top[wrong[name]] >= np.median(top[~wrong[name]])).sum()),
        "errors_with_confidence_at_least_0.5": int((top[wrong[name]] >= 0.5).sum()),
        "errors_where_true_class_is_second_choice": int((second[wrong[name]] == labels[wrong[name]]).sum()),
    }

# Where the errors of the final model fall.
confusion = np.zeros((10, 10), dtype=int)
for true, pred in zip(labels, preds[FINAL]):
    confusion[true, pred] += 1
within_animals = sum(confusion[i, j] for i in ANIMALS for j in ANIMALS if i != j)
vehicles = set(range(10)) - ANIMALS
within_vehicles = sum(confusion[i, j] for i in vehicles for j in vehicles if i != j)
across = sum(confusion[i, j] for i in range(10) for j in range(10) if i != j and ((i in ANIMALS) != (j in ANIMALS)))
pairs = sorted(((int(confusion[i, j] + confusion[j, i]), CLASSES[i], CLASSES[j]) for i, j in combinations(range(10), 2)), reverse=True)
result["final_model_groups"] = {"within_animals": int(within_animals), "within_vehicles": int(within_vehicles), "across": int(across),
                                "largest_unordered_pairs": pairs[:6]}

# How far the errors of different models overlap.
result["pairs"] = {}
for a, b in combinations(LONG, 2):
    both = wrong[a] & wrong[b]
    result["pairs"][f"{a} | {b}"] = {"errors_a": int(wrong[a].sum()), "errors_b": int(wrong[b].sum()), "both_wrong": int(both.sum())}
all_long = np.logical_and.reduce([wrong[m] for m in LONG])
any_long = np.logical_or.reduce([wrong[m] for m in LONG])
counts = np.sum([wrong[m] for m in LONG], axis=0)
same_answer = all_long & np.all([preds[m] == preds[LONG[0]] for m in LONG], axis=0)
result["long_models"] = {
    "wrong_in_all_four": int(all_long.sum()),
    "wrong_in_at_least_one": int(any_long.sum()),
    "wrong_in_exactly": {str(k): int((counts == k).sum()) for k in range(1, 5)},
    "all_four_same_wrong_answer": int(same_answer.sum()),
    "wrong_in_all_four_per_class": {CLASSES[c]: int(n) for c, n in Counter(labels[all_long]).most_common()},
    "wrong_in_all_four_also_wrong_at_100_epochs": int((all_long & wrong[SHORT]).sum()),
}

# What 1,700 more epochs fix, with the same architecture and training procedure.
short, long_ = wrong[SHORT], wrong["r20-x4 standard 1800"]
result["training_length"] = {
    "errors_100": int(short.sum()), "errors_1800": int(long_.sum()), "still_wrong": int((short & long_).sum()),
    "fixed": int((short & ~long_).sum()), "new": int((~short & long_).sum()),
    "errors_100_per_class": [int((short & (labels == c)).sum()) for c in range(10)],
    "errors_1800_per_class": [int((long_ & (labels == c)).sum()) for c in range(10)],
}

# Hard examples: wrong in all four long models with the same answer, most confident first.
mean_probs = np.mean([probs[m] for m in LONG], axis=0)
idx = np.where(same_answer)[0]
idx = idx[np.argsort(-mean_probs[idx].max(1))]
show = idx[:12]
result["hard_examples"] = [{"validation_position": int(i), "train_index": int(valid[i]), "true": CLASSES[labels[i]],
                            "predicted": CLASSES[preds[LONG[0]][i]], "mean_confidence": float(mean_probs[i].max())} for i in show]
(HERE / "data/error_analysis.json").write_text(json.dumps(result, indent=1))

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
fig, axes = plt.subplots(2, 6, figsize=(3.3, 1.45), gridspec_kw={"wspace": 0.08, "hspace": 0.55})
for ax, i in zip(axes.flat, show):
    ax.imshow(data.images[valid[i]].permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.set_title(f"{CLASSES[labels[i]]}\nas {CLASSES[preds[LONG[0]][i]]}", fontsize=4.8, linespacing=1.15, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
fig.savefig(HERE / "hard_examples.pdf")
fig.savefig(HERE / "hard_examples.png")
print("wrote", HERE / "data/error_analysis.json", "and hard_examples.pdf")
