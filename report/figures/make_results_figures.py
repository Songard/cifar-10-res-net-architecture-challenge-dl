"""Result figures for the Project 1 report.

Inputs (all in ./data, every number traceable to a run directory or a Kaggle receipt):
  experiments.csv   validation accuracy (EMA weights + flip TTA, fixed 5,000-image split) and A100 training-step
                    hours of every completed experiment; `source` names the run or evidence directory.
  submissions.csv   the 21 Kaggle uploads with their validation accuracy and public and private leaderboard scores.
  curves_1800.json  per-epoch metrics of the three 1,800-epoch seed-2026 runs (standard, KD2, SAM).
  heldout_eval.json per-class accuracy, confusion matrices and pairwise agreement of the final candidates.
  error_analysis.json  per-image error statistics written by make_error_analysis.py.

Outputs: schedule.pdf, curves.pdf, confusion.pdf, errors.pdf (+ .png previews) in this directory, all used by the report,
and scatter.pdf, public against validation accuracy, which the report no longer uses.
"""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Colors: a validated categorical palette (slots 1-3 pass all-pairs CVD checks), ink and chrome tokens.
C = {"plain": "#2a78d6", "kd": "#eb6834", "sam": "#1baf7a", "other": "#898781",
     "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
FAMILY = {"plain": ("standard training", "o", C["plain"]), "kd": ("+ knowledge distillation", "^", C["kd"]),
          "sam": ("+ SAM", "s", C["sam"]), "aug": ("other augmentation", "D", C["other"]),
          "arch": ("other architecture", "P", C["other"]), "full": ("all 50,000 images (no validation)", "x", C["other"])}
SE5K = 100 * np.sqrt(0.976 * 0.024 / 5000)   # one standard error of a 97.6% accuracy on 5,000 images (0.22 pp)
SE1K = 100 * np.sqrt(0.89 * 0.11 / 1000)     # one standard error of an 89% accuracy on 1,000 images (0.99 pp)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.3, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.edgecolor": C["axis"], "axes.linewidth": 0.6, "xtick.color": C["muted"],
    "ytick.color": C["muted"], "axes.labelcolor": C["ink"], "text.color": C["ink"], "grid.color": C["grid"],
    "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def tidy(ax, y=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y" if y else "both")
    ax.tick_params(length=2.5, width=0.5)


def read_csv(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def fnum(s):
    return float(s) if s not in ("", None) else None


# ----------------------------------------------------------------------------- Figure: schedule length and cost
def schedule_figure():
    rows = read_csv("experiments.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 1.95), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.22})
    rng = np.random.default_rng(0)

    # (a) accuracy against schedule length, every completed resnet20-x4 run plus the other architectures
    for r in rows:
        fam = r["family"]
        name, marker, color = FAMILY[fam]
        x = int(r["epochs"]) * (1 + 0.04 * (1 if r["seed"] == "2027" else -1) * (0.5 + 0.5 * rng.random()))
        hollow = r["device"] == "mps"
        ax1.plot(x, float(r["heldout"]), marker=marker, ms=5.2 if marker != "P" else 6, mfc="white" if hollow else color,
                 mec=color, mew=0.9, ls="none", zorder=4 if fam in ("plain", "kd", "sam") else 3)
    # mean of the plain recipe per schedule length (A100 runs with two seeds where available)
    plain = {}
    for r in rows:
        if r["family"] == "plain" and r["device"] == "a100" and "control" not in r["label"]:
            plain.setdefault(int(r["epochs"]), []).append(float(r["heldout"]))
    xs = sorted(plain)
    ax1.plot(xs, [np.mean(plain[x]) for x in xs], color=C["plain"], lw=1.2, alpha=0.55, zorder=2)
    ax1.set_xscale("log")
    ax1.set_xticks([100, 200, 400, 800, 1600, 3000])
    ax1.set_xticklabels(["100", "200", "400", "800", "1600", "3000"])
    ax1.minorticks_off()
    ax1.set_xlim(85, 3600)
    ax1.set_ylim(96.3, 97.9)
    ax1.set_xlabel("Training epochs")
    ax1.set_ylabel("Validation accuracy (%)")
    ax1.set_title("(a) Accuracy rises with training length, then levels off", loc="left")
    # error-bar key
    ax1.errorbar(110, 97.65, yerr=SE5K, color=C["secondary"], capsize=2, lw=0.8, zorder=5)
    ax1.text(125, 97.65, "±1 standard error\n(5,000 images)", fontsize=5.6, color=C["secondary"], va="center")
    # direct labels on the points the text discusses
    ax1.annotate("SAM, 1,000 epochs", (1000, 97.72), xytext=(440, 97.80), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax1.annotate("KD2, KD5", (1850, 97.66), xytext=(2300, 97.78), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax1.annotate("RandAugment, 400 epochs", (390, 97.00), xytext=(120, 97.08), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax1.annotate("ResNet-18 variant", (104, 96.42), xytext=(118, 96.56), fontsize=6, color=C["secondary"], va="center",
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    handles = [Line2D([], [], marker=m, color=c, ls="none", ms=5, label=n) for n, m, c in FAMILY.values() if n != FAMILY["full"][0]]
    handles.append(Line2D([], [], marker="o", mfc="white", mec=C["ink"], ls="none", ms=5, label="hollow: trained on the laptop"))
    ax1.legend(handles=handles, loc="lower right", ncol=1, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    tidy(ax1)

    # (b) accuracy against measured A100 training-step hours
    for r in rows:
        h = fnum(r["a100_hours"])
        if h is None or r["family"] in ("aug",):
            continue
        name, marker, color = FAMILY[r["family"]]
        y = float(r["heldout"])
        ax2.plot(h, y, marker=marker, ms=5.2 if marker != "P" else 6, color=color, ls="none", zorder=4)
        t = fnum(r["teacher_a100_hours"])
        if t:
            ax2.plot([h, h + t], [y, y], color=color, lw=0.7, alpha=0.6, zorder=2)
            ax2.plot(h + t, y, marker=marker, ms=5.2, mfc="white", mec=color, mew=0.9, ls="none", zorder=3)
    ax2.set_xlim(0, 8.2)
    ax2.set_ylim(97.1, 97.85)
    ax2.set_xlabel("Training time on an A100, hours (hollow: including the teachers)")
    ax2.set_title("(b) Accuracy against training time", loc="left")
    ax2.annotate("standard, 1,800", (1.97, 97.665), xytext=(2.3, 97.80), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax2.annotate("standard, 3,000", (3.31, 97.48), xytext=(3.75, 97.30), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax2.annotate("SAM, 1,800", (3.2, 97.58), xytext=(3.55, 97.47), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax2.annotate("SAM, 1,000", (1.77, 97.72), xytext=(0.5, 97.78), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax2.annotate("KD5 with teachers", (7.68, 97.66), xytext=(5.3, 97.80), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax2.annotate("KD2", (2.5, 97.63), xytext=(2.1, 97.26), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax2.annotate("KD2 with teachers", (4.4, 97.64), xytext=(4.5, 97.52), fontsize=6, color=C["secondary"],
                 arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    tidy(ax2)
    fig.savefig(HERE / "schedule.pdf")
    fig.savefig(HERE / "schedule.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- Figure: 1,800-epoch curves
def curves_figure():
    curves = json.load(open(DATA / "curves_1800.json"))
    runs = [("r2-ta1800-s2026", "standard training", C["plain"]), ("r2-kd2-ta1800-s2026", "KD2 (submitted)", C["kd"]),
            ("sam-x4-1800-s2026", "SAM (submitted)", C["sam"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.42})
    ax_a, ax_b, ax_c = axes
    for run, label, color in runs:
        d = curves[run]
        ep = np.array(d["epoch"])
        tta = np.array([np.nan if v is None else 100 * v for v in d["tta_valid_accuracy"]])
        tr = 100 * np.array(d["train_accuracy"], dtype=float)
        m = ~np.isnan(tta)
        ax_a.plot(ep[m], tta[m], color=color, lw=1.3, label=label)
        z = m & (ep >= 1000)
        ax_b.plot(ep[z], tta[z], color=color, lw=1.3, marker="o", ms=2.2, label=label)
        ax_c.plot(ep, tr, color=color, lw=0.9, alpha=0.9, label=label)
    ax_a.set_xlim(0, 1800)
    ax_a.set_ylim(92, 98.2)
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Validation accuracy (%)")
    ax_a.set_title("(a) All 1,800 epochs", loc="left")
    ax_a.legend(loc="lower right", handlelength=1.4)
    ax_b.set_xlim(1000, 1800)
    ax_b.set_ylim(96.6, 97.9)
    ax_b.set_xlabel("Epoch")
    ax_b.set_title("(b) Last 800 epochs", loc="left")
    ax_b.axhline(97.62, color=C["axis"], lw=0.5, zorder=1)
    ax_b.text(1010, 97.64, "97.6", fontsize=5.8, color=C["muted"], va="bottom")
    ax_c.set_xlim(0, 1800)
    ax_c.set_ylim(80, 100)
    ax_c.set_xlabel("Epoch")
    ax_c.set_ylabel("Training accuracy (%)")
    ax_c.set_title("(c) Accuracy on augmented batches", loc="left")
    for ax in axes:
        ax.set_xticks([0, 600, 1200, 1800] if ax is not ax_b else [1000, 1200, 1400, 1600, 1800])
        tidy(ax)
    fig.savefig(HERE / "curves.pdf")
    fig.savefig(HERE / "curves.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- Figure: held-out vs public
def scatter_figure():
    rows = [r for r in read_csv("submissions.csv") if r["heldout"]]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    xs, ys = [], []
    for r in rows:
        x, y = float(r["heldout"]), float(r["public"])
        xs.append(x)
        ys.append(y)
        name, marker, color = FAMILY[r["family"]]
        sel = r["note"] == "selected"
        ax.errorbar(x, y, xerr=SE5K, yerr=SE1K, fmt="none", ecolor=C["grid"], elinewidth=0.6, capsize=0, zorder=1)
        ax.plot(x, y, marker=marker, ms=6.5 if sel else 4.8, color=color, ls="none", zorder=4,
                mec=C["ink"] if sel else color, mew=0.9 if sel else 0.6)
    xs, ys = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    corr = np.corrcoef(xs, ys)[0, 1]
    grid = np.linspace(94.4, 98.0, 10)
    ax.plot(grid, slope * grid + intercept, color=C["secondary"], lw=0.8, ls="-", alpha=0.7, zorder=2)
    ax.annotate("KD2, 1,800 epochs: 90.1 (submitted)", (97.62, 90.1), xytext=(94.55, 90.55), fontsize=6, color=C["secondary"],
                arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax.annotate("SAM, 1,800 epochs: 90.0 (submitted)", (97.58, 90.0), xytext=(94.55, 89.75), fontsize=6, color=C["secondary"],
                arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax.annotate("SAM, 1,000", (97.72, 89.3), xytext=(96.85, 89.05), fontsize=6, color=C["secondary"],
                arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax.annotate("standard, 1,000", (97.54, 89.5), xytext=(95.75, 89.25), fontsize=6, color=C["secondary"],
                arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.5))
    ax.text(95.85, 83.35, f"least-squares fit: slope {slope:.1f}, r = {corr:.2f}\n({len(rows)} submissions with a validation score)", fontsize=6, color=C["secondary"])
    ax.set_xlim(94.4, 98.0)
    ax.set_ylim(83, 91)
    ax.set_xlabel("Validation accuracy (5,000 images, %)")
    ax.set_ylabel("Public accuracy (1,000 images, %)")
    handles = [Line2D([], [], marker=m, color=c, ls="none", ms=4.5, label=n) for k, (n, m, c) in FAMILY.items() if k in ("plain", "kd", "sam", "arch")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.01, 0.5), handletextpad=0.3, borderaxespad=0.1, labelspacing=0.25)
    tidy(ax, y=False)
    fig.savefig(HERE / "scatter.pdf")
    fig.savefig(HERE / "scatter.png")
    plt.close(fig)
    return slope, intercept, corr


# ----------------------------------------------------------------------------- Figure: confusion matrix
def confusion_figure():
    ev = json.load(open(DATA / "heldout_eval.json"))
    cm = np.array(ev["kd2-s2026"]["confusion"])
    classes = ev["classes"]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    off = cm.copy().astype(float)
    np.fill_diagonal(off, np.nan)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#5598e7", "#1c5cab", "#0d366b"])
    ax.imshow(np.nan_to_num(off), cmap=ramp, vmin=0, vmax=max(1, np.nanmax(off)))
    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j, i, f"{cm[i, j] / 5:.1f}", ha="center", va="center", fontsize=5.6, color=C["ink"])
            elif cm[i, j] > 0:
                dark = off[i, j] >= 0.55 * np.nanmax(off)
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5.6, color="white" if dark else C["ink"])
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(HERE / "confusion.pdf")
    fig.savefig(HERE / "confusion.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- Figure: where the errors are
def errors_figure():
    ev = json.load(open(DATA / "error_analysis.json"))
    classes = ev["classes"]
    length = ev["training_length"]
    short, long_ = np.array(length["errors_100_per_class"]), np.array(length["errors_1800_per_class"])
    order = np.argsort(-short, kind="stable")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.3, 1.5), gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.28})

    # (a) errors per class with the same network and procedure, 100 against 1,800 epochs
    x = np.arange(10)
    ax1.bar(x - 0.19, short[order], width=0.36, color=C["other"], label="100 epochs")
    ax1.bar(x + 0.19, long_[order], width=0.36, color=C["plain"], label="1,800 epochs")
    ax1.set_xticks(x)
    ax1.set_xticklabels([classes[i] for i in order], rotation=45, ha="right", rotation_mode="anchor", fontsize=5.8)
    ax1.set_ylabel("Errors (of 500 images)")
    ax1.set_ylim(0, 37)
    ax1.set_title("(a) Errors per class", loc="left")
    ax1.legend(loc="upper right", handlelength=1.0, labelspacing=0.25, borderaxespad=0.1)
    tidy(ax1)
    ax1.tick_params(axis="x", length=0, colors=C["ink"])

    # (b) how many of the four 1,800-epoch models miss an image
    wrong = ev["long_models"]["wrong_in_exactly"]
    counts = [wrong[str(k)] for k in range(1, 5)]
    ax2.bar(range(1, 5), counts, width=0.62, color=C["plain"])
    for k, n in zip(range(1, 5), counts):
        ax2.text(k, n + 1.5, str(n), ha="center", va="bottom", fontsize=6, color=C["ink"])
    ax2.set_xticks(range(1, 5))
    ax2.set_ylim(0, 80)
    ax2.set_xlabel("Models wrong")
    ax2.set_ylabel("Images")
    ax2.set_title("(b) Shared errors", loc="left")
    tidy(ax2)
    ax2.tick_params(axis="x", length=0)
    fig.savefig(HERE / "errors.pdf")
    fig.savefig(HERE / "errors.png")
    plt.close(fig)


if __name__ == "__main__":
    schedule_figure()
    curves_figure()
    print("scatter fit (slope, intercept, r):", scatter_figure())
    confusion_figure()
    errors_figure()
    print("wrote schedule/curves/scatter/confusion/errors .pdf and .png in", HERE)
