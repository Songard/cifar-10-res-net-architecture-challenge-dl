"""Summarize local training runs and plot their curves, without querying the leaderboard.

Every directory under runs/ (laptop runs) and runs/hpc/ (cluster runs, listed as hpc/<name>) with a metrics.jsonl
is summarized. Accuracy on the 5,000 validation images is held out only for runs whose config says so; full-train
runs fit those images and are flagged accordingly. The default plot shows the laptop runs; pass --plot to choose.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RECIPE_KEYS = ("arch", "epochs", "batch_size", "lr", "pipeline", "padding_mode", "cutout", "label_smoothing", "ema_decay", "amp", "full_train")


def load_metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run):
    metrics = load_metrics(run / "metrics.jsonl")
    config = json.loads((run / "config.json").read_text())
    last = metrics[-1]
    # Runs trained with --eval-every N or --no-eval-raw carry validation keys only on the epochs they evaluated.
    evaluated = [r for r in metrics if "valid_accuracy" in r or "ema_valid_accuracy" in r] or metrics
    accuracy = lambda r: r.get("valid_accuracy", r.get("ema_valid_accuracy"))
    timed = metrics[1:] or metrics
    summary = {"run": run.name, **{key: config.get(key) for key in RECIPE_KEYS},
               "arch": config.get("arch", f"resnet18-w{config.get('width')}"),
               "parameter_count": config["parameter_count"], "train_images": config.get("train_images", 45000),
               "validation_held_out": config.get("validation_held_out", True),
               "epochs_completed": last["epoch"], "last_valid_accuracy": accuracy(evaluated[-1]),
               # Picked on the validation set itself, so optimistic; headline numbers are last-epoch values.
               "max_over_epochs_valid_accuracy": max(accuracy(r) for r in evaluated),
               "mean_train_seconds_excluding_first_epoch": statistics.mean(r["train_seconds"] for r in timed),
               "total_hours": sum(r["train_seconds"] + r["valid_seconds"] + r["checkpoint_seconds"] for r in metrics) / 3600,
               "max_process_peak_rss_gb": max(r["process_lifetime_peak_rss_gb"] for r in metrics),
               "max_sampled_mps_driver_gb": max(r["sampled_memory_max"].get("mps_driver_gb", 0) for r in metrics)}
    for key in ("ema_valid_accuracy", "tta_valid_accuracy"):
        if key in evaluated[-1]:
            summary[f"last_{key}"] = evaluated[-1][key]
    return summary, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", nargs="*", default=None, help="run names to plot (default: held-out runs with at least 50 epochs)")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    summaries, curves, local = [], {}, []
    cluster = sorted(Path("runs/hpc").iterdir()) if Path("runs/hpc").is_dir() else []
    for run in sorted(Path("runs").iterdir()) + cluster:
        if (run / "metrics.jsonl").exists() and (run / "config.json").exists():
            summary, metrics = summarize(run)
            summary["run"] = run.name if run.parent == Path("runs") else f"hpc/{run.name}"
            summaries.append(summary)
            curves[summary["run"]] = metrics
            if run.parent == Path("runs"):
                local.append(summary)
    output = Path(args.output)
    output.mkdir(exist_ok=True)
    (output / "experiments_summary.json").write_text(json.dumps({
        "runs": summaries,
        "metric_note": "valid_* are on the fixed seed-2026 5,000-image split; they are held-out estimates only when validation_held_out is true. Headline numbers are last-epoch values; max_over_epochs_valid_accuracy is selected on the same images and therefore optimistic. ema/tta columns use the EMA weights with horizontal-flip test-time augmentation.",
        "memory_note": "Process RSS and MPS driver accounting are not additive on unified memory; sampled GPU accounting is not an exact activation peak.",
        "kaggle_submissions": "see Kaggle submission records; none are inferred here"}, indent=2) + "\n")
    held_out = {s["run"]: s["validation_held_out"] for s in summaries}
    names = args.plot if args.plot is not None else [s["run"] for s in local if s["epochs_completed"] >= 50 and s["validation_held_out"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for name in names:
        metrics = curves[name]
        evaluated = [r for r in metrics if "valid_accuracy" in r or "ema_valid_accuracy" in r]
        accuracy = [r.get("ema_valid_accuracy", r.get("valid_accuracy")) for r in evaluated]
        # Runs that also fit the validation images are drawn dashed and labelled; their curve is not a generalization estimate.
        style, label = ("-", name) if held_out[name] else ("--", f"{name} (fit, not held out)")
        line, = axes[0].plot([r["epoch"] for r in metrics], [r["train_loss"] for r in metrics], style, label=label)
        axes[1].plot([r["epoch"] for r in evaluated], accuracy, style, color=line.get_color(), label=label)
    axes[0].set_ylabel("Training loss (augmented batches)")
    axes[1].set_ylabel("Validation accuracy (EMA weights when used)")
    axes[1].set_ylim(0.85, 1.0)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    title = "Fixed split: 45,000 training / 5,000 held-out validation images"
    if not all(held_out[name] for name in names):
        title += " (dashed runs also trained on them)"
    fig.suptitle(title, fontsize=11)
    fig.savefig(output / "experiment_curves.png", dpi=180)
    plt.close(fig)
    for s in summaries:
        extra = "".join(f" {k[5:]}={s[k]:.4f}" for k in ("last_ema_valid_accuracy", "last_tta_valid_accuracy") if k in s)
        print(f"{s['run']:34s} {s['arch']:16s} {s['parameter_count']:>9,} ep={s['epochs_completed']:>3} "
              f"held_out={s['validation_held_out']!s:5} last={s['last_valid_accuracy']:.4f}{extra} hours={s['total_hours']:.2f}")


if __name__ == "__main__":
    main()
