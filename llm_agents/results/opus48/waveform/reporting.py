"""Comparison artifacts for the Waveform Bench (opus48, original work).

Generates, into comparison/:
  progress.{png,pdf}              - running best loss vs approach index
  error_histograms.{png,pdf}      - per-sample error hist, train vs val, NR floor
  pareto_accuracy_speed.{png,pdf} - loss vs eval time, labelled by short name
  loss_only_comparison.{png,pdf}  - ranked raw-loss bar chart
  summary_table.json              - ranked approaches
  best_model.json                 - the winner
  error_data.json                 - raw per-sample arrays for every approach
All plots: Nature style, PNG+PDF, no titles, descriptive labels.
"""
import json
import numpy as np
from pathlib import Path

import gwbenchmarks.plot_settings as ps

HERE = Path(__file__).resolve().parent
COMP = HERE / "comparison"
COMP.mkdir(exist_ok=True)
NR_FLOOR = 1.4e-3

CAT_COLOR = {
    "decomposition": ps.COLORS["blue"],
    "symbolic": ps.COLORS["red"],
    "interp_kernel": ps.COLORS["green"],
    "ml": ps.COLORS["orange"],
}


def _short(name):
    return name.replace("svd_", "").replace("_", " ")


def update_progress(results):
    ps.apply()
    nums = [r["scorecard"]["approach_number"] for r in results]
    losses = [r["scorecard"]["loss"] for r in results]
    order = np.argsort(nums)
    nums = np.array(nums)[order]
    losses = np.array(losses)[order]
    running_best = np.minimum.accumulate(losses)
    fig, ax = ps.plt.subplots(figsize=(ps.SINGLE_COL, 2.6))
    ax.plot(nums, losses, "o", color=ps.COLORS["gray"], ms=4, label="per approach")
    ax.plot(nums, running_best, "-", color=ps.COLORS["blue"], lw=1.3, label="running best")
    ax.axhline(NR_FLOOR, ls="--", color=ps.COLORS["black"], lw=0.8, label="NR floor")
    ax.set_yscale("log")
    ax.set_xlabel("approach number")
    ax.set_ylabel("mean FD mismatch")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(COMP / f"progress.{ext}", dpi=200)
    ps.plt.close(fig)


def final_plots(results):
    ps.apply()
    results = sorted(results, key=lambda r: r["scorecard"]["loss"])
    names = [r["scorecard"]["approach"] for r in results]
    shorts = [_short(n) for n in names]
    cats = [r["scorecard"]["category"] for r in results]
    losses = np.array([r["scorecard"]["loss"] for r in results])
    rts = np.array([r["scorecard"]["runtime_ms"] for r in results])
    colors = [CAT_COLOR.get(c, ps.COLORS["gray"]) for c in cats]

    # ---- loss_only_comparison ----
    fig, ax = ps.plt.subplots(figsize=(ps.DOUBLE_COL, 3.4))
    y = np.arange(len(names))
    ax.barh(y, losses, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(shorts, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(NR_FLOOR, ls="--", color=ps.COLORS["black"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("mean FD mismatch (raw loss)")
    _cat_legend(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(COMP / f"loss_only_comparison.{ext}", dpi=200)
    ps.plt.close(fig)

    # ---- pareto_accuracy_speed ----
    fig, ax = ps.plt.subplots(figsize=(ps.DOUBLE_COL, 4.2))
    for i in range(len(names)):
        ax.scatter(rts[i], losses[i], color=colors[i], s=28, zorder=3)
        ax.annotate(shorts[i], (rts[i], losses[i]), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(NR_FLOOR, ls="--", color=ps.COLORS["black"], lw=0.8, label="NR floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("evaluation time per waveform (ms)")
    ax.set_ylabel("mean FD mismatch")
    _cat_legend(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(COMP / f"pareto_accuracy_speed.{ext}", dpi=200)
    ps.plt.close(fig)

    # ---- error_histograms ----
    _error_histograms(results)

    # ---- json summaries ----
    summary = [{
        "rank": i + 1,
        "approach": r["scorecard"]["approach"],
        "category": r["scorecard"]["category"],
        "parameterization": r["scorecard"]["parameterization"],
        "loss": r["scorecard"]["loss"],
        "val_median": r["scorecard"]["val_median"],
        "train_loss": r["scorecard"]["train_loss"],
        "runtime_ms": r["scorecard"]["runtime_ms"],
    } for i, r in enumerate(results)]
    with open(COMP / "summary_table.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(COMP / "best_model.json", "w") as f:
        json.dump(results[0]["scorecard"], f, indent=2)

    err_data = {r["scorecard"]["approach"]: {
        "train": list(map(float, r["err_tr"])),
        "val": list(map(float, r["err_va"])),
    } for r in results}
    with open(COMP / "error_data.json", "w") as f:
        json.dump(err_data, f, indent=2)


def _error_histograms(results):
    top = results[:min(6, len(results))]
    n = len(top)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = ps.plt.subplots(nrow, ncol, figsize=(ps.DOUBLE_COL, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    bins = np.logspace(-3.2, 0, 28)
    for k, r in enumerate(top):
        ax = axes[k]
        tr = np.clip(np.asarray(r["err_tr"]), 1e-4, None)
        va = np.clip(np.asarray(r["err_va"]), 1e-4, None)
        ax.hist(tr, bins=bins, alpha=0.55, color=ps.COLORS["blue"],
                label="train", density=True)
        ax.hist(va, bins=bins, alpha=0.5, color=ps.COLORS["red"],
                hatch="///", label="val", density=True)
        ax.axvline(NR_FLOOR, ls="--", color=ps.COLORS["black"], lw=0.8)
        ax.set_xscale("log")
        ax.set_title(_short(r["scorecard"]["approach"]), fontsize=6)
        ax.set_xlabel("FD mismatch", fontsize=7)
        if k % ncol == 0:
            ax.set_ylabel("density", fontsize=7)
        if k == 0:
            ax.legend(frameon=False, fontsize=6)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(COMP / f"error_histograms.{ext}", dpi=200)
    ps.plt.close(fig)


def _cat_legend(ax):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", ls="", color=c, label=cat)
               for cat, c in CAT_COLOR.items()]
    handles.append(Line2D([0], [0], ls="--", color=ps.COLORS["black"], label="NR floor"))
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="best")
