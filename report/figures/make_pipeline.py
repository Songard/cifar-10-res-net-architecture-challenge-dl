"""Draw the end-to-end pipeline figure (data, network, training objectives, inference) as a vector PDF.

Everything shown is read from the project: the augmentation thumbnails are produced by the same torchvision
policies the training code uses, applied to one training image; the network layout, parameter counts and
multiply-accumulates come from model.py; recipe values are the ones in the delivered candidates' config.json.
Run from the project root:  .venv/bin/python report/figures/make_pipeline.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from model import build_model, multiply_accumulates, parameter_count  # noqa: E402

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42})

# Palette: one muted hue per role, a darker edge of the same hue; text stays near-black.
C = {"panel": "#f7f7f9", "panel_ec": "#d9d9df", "ink": "#1f1f24", "mute": "#5a5a64", "arrow": "#3a3a44",
     "data": ("#dde9f6", "#3f6e9e"), "stem": ("#ebebef", "#6a6a74"),
     "s1": ("#e6f1df", "#4f8a3b"), "s2": ("#d3e7c6", "#4f8a3b"), "s3": ("#bddcab", "#4f8a3b"),
     "ce": ("#fde7d2", "#c4661a"), "kd": ("#e9e1f4", "#6f4fa3"), "sam": ("#d6ecea", "#237f76"),
     "opt": ("#ebebef", "#6a6a74"), "ema": ("#fff1c9", "#a7791a")}

FIG_W, FIG_H = 7.0, 3.8                       # inches; include at \textwidth in a two-column layout
XMAX, YMAX = 140.0, 76.0                      # data units with the same aspect ratio as the figure

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, text="", key="stem", fs=4.4, lw=0.6, ls="-", r=1.0, bold=False, color=None, zorder=3,
        fc=None, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc or C[key][0],
                                ec=ec or C[key][1], lw=lw, ls=ls, zorder=zorder))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or C["ink"],
                fontweight="bold" if bold else "normal", zorder=zorder + 1, linespacing=1.2)


def arrow(x1, y1, x2, y2, ls="-", color=None, lw=0.6, ms=4.5, zorder=4, rad=0.0, head=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>" if head else "-", mutation_scale=ms,
                                 lw=lw, ls=ls, color=color or C["arrow"], zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def route(points, ls="-", color=None, lw=0.6, ms=4.5, zorder=4):
    """Orthogonal connector through the given points, arrowhead on the last segment."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], ls=ls, lw=lw, color=color or C["arrow"], zorder=zorder, solid_capstyle="round")
    arrow(xs[-2], ys[-2], xs[-1], ys[-1], ls=ls, color=color, lw=lw, ms=ms, zorder=zorder)


def label(x, y, text, fs=4.0, color=None, ha="center", va="center", bold=False, zorder=5, style="normal",
          rotation=0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color or C["mute"], zorder=zorder, linespacing=1.2,
            fontweight="bold" if bold else "normal", fontstyle=style, rotation=rotation)


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=C["panel"],
                                ec=C["panel_ec"], lw=0.6, zorder=1))
    ax.text(x + 1.6, y + h - 1.6, title, ha="left", va="top", fontsize=5.9, fontweight="bold", color=C["ink"],
            zorder=2)


PY, PH = 1.0, YMAX - 2.0

# ------------------------------------------------------------------------------------------------ (a) data
panel(0.8, PY, 28.6, PH, "(a) Data and augmentation")

rows = list(csv.DictReader((ROOT / "data" / "train_labels.csv").open()))
pick = [r for r in rows if r["label"] == "7"][2]                     # a horse: high contrast at 32×32
image = Image.open(ROOT / "data" / "train_images" / pick["filename"]).convert("RGB")
torch.manual_seed(11)
augmented = transforms.TrivialAugmentWide()(image)                   # CPU-side policy, as in training
crop_flip = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                                transforms.RandomHorizontalFlip(p=1.0)])(augmented)
cut = np.array(crop_flip).copy()
cut[6:22, 13:29] = 0                                                 # one 16×16 Cutout square
thumbs = [np.array(image), np.array(augmented), np.array(crop_flip), cut]

box(2.4, 56.0, 25.4, 12.0, "", key="data", r=1.2)
ax.text(15.1, 65.4, "CIFAR-10 training set", ha="center", va="center", fontsize=5.0, fontweight="bold",
        color=C["ink"], zorder=4)
ax.text(15.1, 60.2, "50,000 images, 32×32, 10 classes\nselection: 45,000 train / 5,000 held-out\n"
        "final models: the same 45,000", ha="center", va="center", fontsize=4.1, color=C["ink"], zorder=4,
        linespacing=1.25)

label(15.1, 52.6, "per-step augmentation (training only)", fs=4.3, bold=True, color=C["ink"])
T, gap, tx0, ty = 4.8, 1.9, 3.4, 44.0
for i, arr in enumerate(thumbs):
    x = tx0 + i * (T + gap)
    ax.imshow(arr, extent=(x, x + T, ty, ty + T), zorder=3, interpolation="nearest")
    ax.add_patch(FancyBboxPatch((x, ty), T, T, boxstyle="round,pad=0,rounding_size=0.3", fc="none",
                                ec=C["data"][1], lw=0.5, zorder=4))
    label(x + T / 2, ty + T + 1.3, "①②③④"[i], fs=4.6, color=C["ink"])
    if i < 3:
        arrow(x + T + 0.25, ty + T / 2, x + T + gap - 0.25, ty + T / 2, ms=3.5)
for j, line in enumerate(["① original image", "② TrivialAugment: one random op, random strength",
                          "③ reflect-pad 4, random 32×32 crop, horizontal flip", "④ Cutout: one 16×16 square zeroed"]):
    label(3.4, 41.6 - 2.5 * j, line, fs=3.6, ha="left", color=C["ink"])
label(15.1, 31.0, "every epoch sees a new random view of each image", fs=3.6, style="italic")

box(2.4, 13.0, 25.4, 14.6, "", key="data", r=1.2)
ax.text(15.1, 24.8, "Held-out 5,000", ha="center", va="center", fontsize=4.6, fontweight="bold", color=C["ink"], zorder=4)
ax.text(15.1, 21.6, "only to compare recipes and seeds;\nnever trained on", ha="center", va="center", fontsize=3.8,
        color=C["ink"], zorder=4, linespacing=1.2)
ax.text(15.1, 17.9, "Hidden test 2,000", ha="center", va="center", fontsize=4.6, fontweight="bold", color=C["ink"], zorder=4)
ax.text(15.1, 15.0, "read once by predict: no adaptation,\nno manual labels, no selection on it",
        ha="center", va="center", fontsize=3.8, color=C["ink"], zorder=4, linespacing=1.2)
label(15.1, 8.4, "one image = 0.02 pp held-out, 0.1 pp public", fs=3.6, style="italic")

# --------------------------------------------------------------------------------------------- (b) network
panel(30.4, PY, 54.6, PH, "(b) Network: ResNet-20 layout at 4× width (4.33M parameters)")
model = build_model("resnet20-x4")
assert parameter_count(model) == 4_327_754 and multiply_accumulates(model) == 647_694_848
stage_params = [sum(p.numel() for b in model.blocks[i * 3:(i + 1) * 3] for p in b.parameters()) for i in range(3)]

ny, bh = 46.4, 13.0
arrow(28.6, ny, 33.2, ny, ms=4.5)                                   # augmented batch into the network
label(30.7, ny - 2.6, "3×32×32", fs=3.6)
box(33.4, ny - bh / 2, 6.4, bh, "3×3 conv\n64, BN\nReLU", key="stem", fs=4.0, r=0.9)
label(36.6, ny - bh / 2 - 1.7, "stem", fs=3.8, style="italic")
arrow(39.8, ny, 41.6, ny, ms=4)

stages = [("Stage 1", "32×32", "64 ch", "s1", 41.8), ("Stage 2", "16×16", "128 ch", "s2", 54.2),
          ("Stage 3", "8×8", "256 ch", "s3", 66.6)]
sw = 10.6
for i, (name, size, ch, key, x) in enumerate(stages):
    box(x, ny - bh / 2, sw, bh, "", key=key, r=0.9)
    ax.text(x + sw / 2, ny + bh / 2 - 1.3, f"{name}  ({size})", ha="center", va="top", fontsize=4.1,
            color=C["ink"], fontweight="bold", zorder=5)
    for j in range(3):
        bx = x + 1.25 + j * 2.85
        box(bx, ny - 3.2, 2.4, 5.6, "", fc="white", ec=C[key][1], lw=0.5, r=0.4, zorder=4)
        ax.text(bx + 1.2, ny - 0.4, "RB", ha="center", va="center", fontsize=3.5, color=C[key][1], zorder=5)
    ax.text(x + sw / 2, ny - bh / 2 + 1.2, f"3 blocks × {ch}", ha="center", va="bottom", fontsize=3.8,
            color=C["ink"], zorder=5)
    label(x + sw / 2, ny - bh / 2 - 1.7, f"{stage_params[i] / 1e6:.2f}M params", fs=3.8, style="italic")
    if i < 2:
        arrow(x + sw + 0.2, ny, x + sw + 1.6, ny, ms=4)
        label(x + sw + 0.9, ny + bh / 2 + 1.5, "stride 2", fs=3.4)
arrow(77.4, ny, 78.6, ny, ms=4)
box(78.8, ny - bh / 2, 4.8, bh, "GAP\n\nFC\n256→10", key="stem", fs=3.8, r=0.9)
label(81.2, ny - bh / 2 - 1.7, "2.6k", fs=3.8, style="italic")

dy = 20.0                                                           # residual block detail
label(35.0, 32.6, "Residual block, used 9 times:  out = ReLU(F(x) + S(x))", fs=4.3, bold=True, color=C["ink"],
      ha="left")
label(52.4, 29.4, "S(x): identity, or 1×1 conv (stride 2) + BN in the first block of stages 2–3", fs=3.5)
box(35.0, dy - 2.6, 3.2, 5.2, "x", key="data", fs=5.0, r=0.6)
arrow(38.2, dy, 41.4, dy, ms=4)
box(41.6, dy - 2.9, 10.4, 5.8, "3×3 conv, BN\nReLU", key="s2", fs=4.0, r=0.7)
arrow(52.0, dy, 54.2, dy, ms=4)
box(54.4, dy - 2.9, 10.4, 5.8, "3×3 conv, BN", key="s2", fs=4.0, r=0.7)
arrow(64.8, dy, 67.0, dy, ms=4)
ax.add_patch(Circle((68.2, dy), 1.2, fc="white", ec=C["ink"], lw=0.6, zorder=4))
ax.text(68.2, dy, "+", ha="center", va="center", fontsize=6.5, color=C["ink"], zorder=5)
arrow(69.4, dy, 71.4, dy, ms=4)
box(71.6, dy - 2.6, 5.2, 5.2, "ReLU", key="stem", fs=4.2, r=0.6)
arrow(76.8, dy, 79.6, dy, ms=4)
label(81.4, dy, "out", fs=4.2, color=C["ink"])
arrow(36.6, dy + 2.6, 68.2, dy + 1.2, rad=-0.33, ms=4)
label(53.0, dy - 4.9, "F(x): two 3×3 convolutions, no bias (BN follows), Kaiming init", fs=3.5)
label(57.7, 9.0, "4,327,754 parameters, 648M MACs per image. Three stages at 32, 16 and 8 px instead of\n"
      "ResNet-18's four: no 4×4 stage, the budget goes to width (3.3M parameters sit in the 8×8 stage)", fs=3.6, style="italic")

# -------------------------------------------------------------------------------------------- (c) training
panel(86.0, PY, 37.4, PH, "(c) Training objectives, one step")
# student logits leave the network and enter the loss row
route([(83.6, ny), (86.9, ny), (86.9, 64.0), (89.6, 64.0)], ms=4.5)
label(87.7, 58.2, "logits z", fs=3.8, ha="left")

ry = 64.0                                                           # loss row
box(89.6, ry - 3.6, 14.4, 7.2, "cross-entropy\nlabel smoothing 0.1", key="ce", fs=4.2, r=0.9)
arrow(104.0, ry, 105.6, ry, ms=4)
box(105.8, ry - 3.6, 15.8, 7.2, "L = 0.1·CE + 0.9·KD\n(CE alone without KD)", key="ce", fs=4.1, r=0.9)

ky = 48.0                                                           # knowledge distillation group
box(88.0, 40.0, 32.6, 14.8, "", fc="none", ec=C["kd"][1], ls=(0, (2.2, 1.6)), lw=0.6, r=1.1, zorder=2)
label(88.9, 53.4, "knowledge distillation (KD models)", fs=4.2, bold=True, color=C["kd"][1], ha="left")
box(88.8, ky - 3.6, 11.6, 7.6, "frozen teachers\n(same augmented x)\nCE1000 s26, CE800 s27", key="kd", fs=3.5, r=0.8)
arrow(100.4, ky + 0.2, 101.8, ky + 0.2, ms=4)
box(102.0, ky - 3.6, 8.6, 7.6, "mean of\nsoftmax(z$_t$ / T)\nT = 4", key="kd", fs=3.8, r=0.8)
arrow(110.6, ky + 0.2, 112.0, ky + 0.2, ms=4)
box(112.2, ky - 3.6, 8.0, 7.6, "KL · T²\nteacher ‖\nstudent", key="kd", fs=3.8, r=0.8)
label(104.3, 42.6, "teachers trained from scratch on the same 45k, never on the held-out 5k", fs=3.2)
arrow(116.2, ky + 4.0, 116.2, ry - 3.8, ms=4)                       # KD term joins the loss

sy = 30.0                                                           # SAM group
box(88.0, 22.4, 32.6, 14.0, "", fc="none", ec=C["sam"][1], ls=(0, (2.2, 1.6)), lw=0.6, r=1.1, zorder=2)
label(88.9, 35.0, "sharpness-aware minimization (SAM models)", fs=4.2, bold=True, color=C["sam"][1], ha="left")
box(88.8, sy - 3.4, 10.0, 6.8, "pass 1: ∇L(w)\nBN stats updated", key="sam", fs=3.6, r=0.8)
arrow(98.8, sy, 100.2, sy, ms=4)
box(100.4, sy - 3.4, 9.4, 6.8, "ε = ρ · g / ‖g‖\nρ = 0.05", key="sam", fs=3.8, r=0.8)
arrow(109.8, sy, 111.2, sy, ms=4)
box(111.4, sy - 3.4, 8.8, 6.8, "pass 2: ∇L(w+ε)\nused for the step", key="sam", fs=3.4, r=0.8)
label(104.3, 24.4, "weights restored after pass 2; weight decay stays in the SGD update", fs=3.2)

oy = 12.0                                                           # optimizer row
box(88.4, oy - 3.8, 21.4, 7.6, "SGD, Nesterov momentum 0.9\nlr 0.2, cosine to 0, 5-epoch warm-up\n"
    "wd 5e-4 (not on BN), batch 256, bf16\n1000–1800 epochs", key="opt", fs=3.5, r=0.8)
arrow(109.8, oy, 111.2, oy, ms=4)
box(111.4, oy - 3.4, 9.2, 6.8, "EMA of weights\ndecay 0.999", key="ema", fs=3.9, r=0.8)
route([(121.6, ry - 2.0), (122.5, ry - 2.0), (122.5, 19.0), (99.1, 19.0), (99.1, oy + 4.0)], ls=(0, (1.6, 1.3)), ms=4)
label(110.0, 20.2, "gradient (two passes with SAM)", fs=3.3, style="italic")
label(104.7, 4.6, "dashed groups: optional, used only by the KD and SAM models", fs=3.3, style="italic")

# ------------------------------------------------------------------------------------------- (d) inference
panel(124.4, PY, 14.8, PH, "(d) Inference")
ix = 126.0
route([(120.6, oy), (123.9, oy), (123.9, 65.8), (ix, 65.8)], ms=4.5)
box(ix, 62.0, 11.6, 7.6, "EMA weights\nno test-time\ntuning", key="ema", fs=3.9, r=0.9)
arrow(ix + 5.8, 62.0, ix + 5.8, 56.6, ms=4)
t2 = 4.4
ax.imshow(np.array(image), extent=(ix + 0.4, ix + 0.4 + t2, 51.0, 51.0 + t2), zorder=3, interpolation="nearest")
ax.imshow(np.array(image)[:, ::-1], extent=(ix + 6.8, ix + 6.8 + t2, 51.0, 51.0 + t2), zorder=3, interpolation="nearest")
for xx in (ix + 0.4, ix + 6.8):
    ax.add_patch(FancyBboxPatch((xx, 51.0), t2, t2, boxstyle="round,pad=0,rounding_size=0.3", fc="none",
                                ec=C["data"][1], lw=0.5, zorder=4))
label(ix + 2.6, 49.6, "x", fs=4.0, va="top", color=C["ink"])
label(ix + 9.0, 49.6, "flip(x)", fs=4.0, va="top", color=C["ink"])
arrow(ix + 2.6, 47.2, ix + 4.8, 43.4, ms=4)
arrow(ix + 9.0, 47.2, ix + 6.8, 43.4, ms=4)
box(ix, 36.0, 11.6, 7.2, "average the\ntwo logit vectors\nargmax → label", key="ema", fs=3.8, r=0.9)
arrow(ix + 5.8, 36.0, ix + 5.8, 31.6, ms=4)
box(ix, 25.0, 11.6, 6.4, "submission.csv\n2,000 ids", key="ema", fs=4.0, r=0.9)
label(ix + 5.8, 20.6, "flip TTA: +0.0 to +0.4 pp\nheld-out; translation\nTTA: no gain, unused", fs=3.4, style="italic", va="top")

out = HERE / "pipeline"
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=400)
print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
