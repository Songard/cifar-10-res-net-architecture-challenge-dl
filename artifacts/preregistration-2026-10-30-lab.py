"""Pre-registered public-score predictions for the transfer-lab arms and x4-400 (written before any of them is uploaded).

Line: probit(public) = a + b * probit(val).  Own fit uses the 5 held-out submitted models (+ final50k at val-eq 97.04),
weighted by binomial public noise; Baek et al. (2022) CIFAR-10.2 line: b = 0.768, a = -0.287.
Paired prediction for an arm = its reference's observed public score moved along the line by the val difference.
Noise band: paired SD = sqrt(k * d_val / 1000) with test disagreement assumed k = 2 .. 3.5 times val disagreement."""
import json, math, sys
from statistics import NormalDist
import numpy as np
N = NormalDist(); z = N.inv_cdf; P = N.cdf
preds = np.load(sys.argv[1]); labels = preds["labels"]
val = {k: float((preds[k] == labels).mean()) for k in preds.files if k != "labels"}
public = {"baseline-100ep": .838, "improved100-resnet18-w32": .850, "improved100-resnet18-64-352": .863,
          "improved100-resnet20-x4": .866, "final-resnet18-64-352-45k": .882}
points = [(val[k], p, k == "baseline-100ep") for k, p in public.items()] + [(0.9704, .878, False)]  # final50k at val-eq
def fit(points, offset):
    X = np.array([[1, z(v)] + ([float(b)] if offset else []) for v, p, b in points])
    y = np.array([z(p) for v, p, b in points])
    w = np.array([N.pdf(z(p)) ** 2 / (p * (1 - p) / 1000) for v, p, b in points])  # 1 / var of probit(public)
    W = np.diag(w); cov = np.linalg.inv(X.T @ W @ X); beta = cov @ X.T @ W @ y
    resid = y - X @ beta; dof = len(y) - X.shape[1]
    scale = max(1.0, float(resid @ W @ resid) / dof)  # inflate if residuals exceed binomial noise
    return beta, cov * scale
fits = {"own_6pt": fit(points, False), "own_6pt_baseline_offset": fit(points, True)}
for name, (beta, cov) in fits.items():
    print(name, "a=%.3f b=%.3f (se %.3f)" % (beta[0], beta[1], math.sqrt(cov[1, 1])), ("offset=%.3f" % beta[2]) if len(beta) > 2 else "")
slopes = {"own": fits["own_6pt_baseline_offset"][0][1], "baek": 0.768}
arms = [("x4-400ep-45k", "improved100-resnet20-x4", "schedule 100->400 epochs (slope anchor at the highest val)"),
        ("lab-r18-200ep-mix", "final-resnet18-64-352-45k", "Mixup/CutMix added (r18, 200 ep)"),
        ("lab-r18-200ep-randconv", "final-resnet18-64-352-45k", "RandConv added (r18, 200 ep)"),
        ("lab-w32-mix", "improved100-resnet18-w32", "Mixup/CutMix added (w32, 100 ep)"),
        ("lab-w32-randconv", "improved100-resnet18-w32", "RandConv added (w32, 100 ep)")]
rows = []
for arm, ref, what in arms:
    d = float((preds[arm] != preds[ref]).mean())
    paired = {k: 100 * P(z(public[ref]) + b * (z(val[arm]) - z(val[ref]))) for k, b in slopes.items()}
    absolute = {"baek": 100 * P(0.768 * z(val[arm]) - 0.287),
                "own": 100 * P(fits["own_6pt_baseline_offset"][0][0] + slopes["own"] * z(val[arm]))}
    sd = [100 * math.sqrt(k * d / 1000) for k in (2, 3.5)]
    point = float(np.mean(list(paired.values())))
    rows.append(dict(arm=arm, reference=ref, change=what, val=val[arm], ref_val=val[ref], ref_public=public[ref],
                     val_disagreement=d, paired_pred=paired, abs_pred=absolute, prediction_pct=round(point, 1),
                     paired_sd_pp=[round(s, 2) for s in sd], band_pp=round(2 * float(np.mean(sd)), 1)))
    print(f"{arm:24s} val {100*val[arm]:.2f} (ref {100*val[ref]:.2f}, pub {public[ref]:.3f}) d={d:.4f} "
          f"paired own {paired['own']:.2f} baek {paired['baek']:.2f} | abs own {absolute['own']:.2f} baek {absolute['baek']:.2f} "
          f"-> {point:.1f} +/- {2*np.mean(sd):.1f}")
# Pooled per-augmentation residual: mean of the two arms' (observed - paired prediction); SD of that mean.
pooled = {}
for aug in ("mix", "randconv"):
    pair = [r for r in rows if r["arm"].endswith(aug)]
    sd = [0.5 * math.sqrt(sum(r["paired_sd_pp"][i] ** 2 for r in pair)) for i in (0, 1)]
    pooled[aug] = {"arms": [r["arm"] for r in pair], "pooled_residual_sd_pp": [round(s, 2) for s in sd]}
    print(aug, "pooled residual SD", [round(s, 2) for s in sd])
json.dump({"written_local": sys.argv[2], "val": val, "fits": {k: {"beta": v[0].tolist(), "cov": v[1].tolist()} for k, v in fits.items()},
           "arms": rows, "pooled": pooled}, open(sys.argv[3], "w"), indent=1)
