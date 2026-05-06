"""Render final comparison plots for the Analytic Bench."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from gwbenchmarks.plot_settings import set_nature_style

    set_nature_style()
except Exception:
    pass

CMP = HERE / "comparison"


def _load(name: str):
    with open(CMP / name) as f:
        return json.load(f)


summary = _load("summary_table.json")
error_data = _load("error_data.json")
best = _load("best_model.json") if (CMP / "best_model.json").exists() else sorted(summary, key=lambda r: r["loss"])[0]

cats = sorted({r["category"] for r in summary})
palette = {
    "physics": "#1f77b4",
    "matched_composite": "#2ca02c",
    "functional_optimization": "#ff7f0e",
    "symbolic": "#d62728",
}

names = [r["approach"] for r in summary]
losses = np.array([r["loss"] for r in summary], dtype=float)
runtimes = np.array([max(float(r["runtime_ms"]), 1e-4) for r in summary], dtype=float)
cats_arr = [r["category"] for r in summary]
nums = np.array([r["approach_number"] for r in summary], dtype=int)


def _save(fig, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(CMP / f"{stem}.{ext}", dpi=160)
    plt.close(fig)


def loss_plot():
    order = np.argsort(losses)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(summary))
    ax.barh(y, losses[order], color=[palette.get(cats_arr[i], "gray") for i in order], edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_xlabel("Validation mismatch")
    ax.set_xscale("log")
    for cat, color in palette.items():
        ax.bar([], [], color=color, label=cat)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "loss_only_comparison")


def pareto_plot():
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for i, (r, l, c) in enumerate(zip(runtimes, losses, cats_arr)):
        ax.scatter(r, l, s=65, color=palette.get(c, "gray"), edgecolor="black", linewidth=0.5, alpha=0.9)
        ax.annotate(names[i], (r, l), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Eval time (ms / sample)")
    ax.set_ylabel("Validation mismatch")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for cat, color in palette.items():
        ax.scatter([], [], color=color, label=cat, edgecolor="black", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "pareto_accuracy_speed")


def progress_plot():
    order = np.argsort(nums)
    fig, ax = plt.subplots(figsize=(10, 5))
    running = np.minimum.accumulate(losses[order])
    ax.plot(nums[order], losses[order], "o-", color="C0", label="per approach")
    ax.plot(nums[order], running, "s--", color="C3", label="running best")
    for idx in order:
        ax.annotate(names[idx], (nums[idx], losses[idx]), fontsize=5, rotation=45, xytext=(0, 4), textcoords="offset points")
    ax.set_xlabel("Approach number")
    ax.set_ylabel("Validation mismatch")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "progress")


def histogram_plot():
    n = len(summary)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    axes = np.atleast_1d(axes)
    for i, row in enumerate(summary):
        ax = axes.flat[i]
        ed = error_data.get(row["approach"])
        if ed is None:
            ax.set_visible(False)
            continue
        tr = np.asarray(ed["train_proxy"], dtype=float)
        va = np.asarray(ed["val_proxy"], dtype=float)
        tr = np.clip(tr, 1e-12, None)
        va = np.clip(va, 1e-12, None)
        bins = np.geomspace(min(tr.min(), va.min()), max(tr.max(), va.max(), 1.0), 25)
        ax.hist(tr, bins=bins, alpha=0.55, color="C0", hatch="//", label="train", edgecolor="black", linewidth=0.3)
        ax.hist(va, bins=bins, alpha=0.55, color="C3", label="val", edgecolor="black", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_title(row["approach"], fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(n, axes.size):
        axes.flat[j].set_visible(False)
    fig.tight_layout()
    _save(fig, "error_histograms")


def changelog():
    lines = ["# Analytic Benchmark - CHANGELOG", ""]
    for row in sorted(summary, key=lambda r: r["approach_number"]):
        lines.extend(
            [
                f"## {row['approach_number']:02d} - {row['approach']}",
                f"- Category: {row['category']}",
                f"- Parameterization: {row['parameterization']}",
                f"- Loss: {row['loss']:.6e}",
                f"- Notes: {row['notes']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            f"- Best: {best['approach']} ({best['loss']:.6e})",
            f"- Categories: {sorted(cats)}",
            f"- Approaches: {len(summary)}",
        ]
    )
    (HERE / "CHANGELOG.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote CHANGELOG and plots. Best: {best['approach']} ({best['loss']:.6e})")


if __name__ == "__main__":
    loss_plot()
    pareto_plot()
    progress_plot()
    histogram_plot()
    changelog()
