"""Aggregate model scorecards and produce comparison plots and JSON outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import RESULTS_DIR

from gwbenchmarks import plot_settings
plot_settings.apply()
import matplotlib.pyplot as plt
mpl = plot_settings.mpl
COLORS = plot_settings.COLORS
COLOR_CYCLE = plot_settings.COLOR_CYCLE
SINGLE_COL = plot_settings.SINGLE_COL
DOUBLE_COL = plot_settings.DOUBLE_COL


MODELS_DIR = RESULTS_DIR / "models"
COMP_DIR = RESULTS_DIR / "comparison"
COMP_DIR.mkdir(exist_ok=True, parents=True)


def load_all_scorecards():
    rows = []
    for d in sorted(MODELS_DIR.iterdir()):
        sc_file = d / "scorecard.json"
        if not sc_file.exists():
            continue
        sc = json.load(open(sc_file))
        sc["_dir"] = str(d)
        rows.append(sc)
    return rows


def short_label(sc):
    return f"{sc['approach_number']:02d} {sc['approach']}"


def main():
    rows = load_all_scorecards()
    print(f"loaded {len(rows)} scorecards")
    if not rows:
        print("No models — aborting plotting.")
        return

    # Filter only valid closed-form models
    valid_rows = [r for r in rows if r.get("is_closed_form", False)]
    valid_rows = sorted(valid_rows, key=lambda r: r["loss"])
    print(f"valid closed-form: {len(valid_rows)}")

    # Best model
    best = valid_rows[0]
    with open(COMP_DIR / "best_model.json", "w") as f:
        json.dump({k: v for k, v in best.items() if k != "_dir"}, f, indent=2, default=str)

    # Summary table (all models, ranked)
    summary = []
    for r in valid_rows:
        summary.append({
            "approach_number": r["approach_number"],
            "approach": r["approach"],
            "category": r["category"],
            "parameterization": r["parameterization"],
            "loss": r["loss"],
            "loss_components": r["loss_components"],
            "n_params": r["n_params"],
            "runtime_ms": r["runtime_ms"],
            "train_time_s": r["train_time_s"],
            "notes": r.get("notes", ""),
        })
    with open(COMP_DIR / "summary_table.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # all_expressions.json
    expressions = {}
    for r in valid_rows:
        expr_path = Path(r["_dir"]) / "expression.txt"
        if expr_path.exists():
            expressions[f"{r['approach_number']:02d}_{r['approach']}"] = expr_path.read_text()
    with open(COMP_DIR / "all_expressions.json", "w") as f:
        json.dump(expressions, f, indent=2)

    # error_data.json
    error_data = {}
    for r in valid_rows:
        error_data[f"{r['approach_number']:02d}_{r['approach']}"] = {
            "loss": r["loss"],
            "loss_components": r["loss_components"],
            "per_sample_loss": r.get("per_sample_loss", []),
        }
    with open(COMP_DIR / "error_data.json", "w") as f:
        json.dump(error_data, f, indent=2)

    cat_palette = {
        "physics_imr": COLOR_CYCLE[0],
        "composite": COLOR_CYCLE[1],
        "functional_form": COLOR_CYCLE[2],
        "symbolic_regression": COLOR_CYCLE[3],
    }
    fallback_color = COLOR_CYCLE[4]

    # progress.png/pdf — bars by approach (sorted by loss)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.6 * DOUBLE_COL))
    losses = [r["loss"] for r in valid_rows]
    labels = [short_label(r) for r in valid_rows]
    cats = [r["category"] for r in valid_rows]
    cs = [cat_palette.get(c, fallback_color) for c in cats]
    x_pos = np.arange(len(valid_rows))
    ax.bar(x_pos, losses, color=cs, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
    ax.set_ylabel(r"FD mismatch (mean over $M_\mathrm{tot}$)")
    ax.set_xlabel("approach (sorted by loss)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_palette[k]) for k in cat_palette]
    ax.legend(handles, list(cat_palette.keys()), loc="upper left", fontsize=6)
    plt.tight_layout()
    plt.savefig(COMP_DIR / "progress.png", dpi=200, bbox_inches="tight")
    plt.savefig(COMP_DIR / "progress.pdf", bbox_inches="tight")
    plt.close(fig)

    # loss_only_comparison.png/pdf — scatter
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.55 * DOUBLE_COL))
    for k, color in cat_palette.items():
        xs = [i for i, r in enumerate(valid_rows) if r["category"] == k]
        ys = [valid_rows[i]["loss"] for i in xs]
        ax.scatter(xs, ys, c=color, label=k, s=28, edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("rank by loss")
    ax.set_ylabel(r"FD mismatch (mean over $M_\mathrm{tot}$)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(COMP_DIR / "loss_only_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig(COMP_DIR / "loss_only_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # pareto_accuracy_speed.png/pdf — runtime vs loss
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    for k, color in cat_palette.items():
        xs = [r["runtime_ms"] for r in valid_rows if r["category"] == k]
        ys = [r["loss"] for r in valid_rows if r["category"] == k]
        ax.scatter(xs, ys, c=color, label=k, s=30, edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("evaluation time per waveform (ms)")
    ax.set_ylabel(r"FD mismatch")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(COMP_DIR / "pareto_accuracy_speed.png", dpi=200, bbox_inches="tight")
    plt.savefig(COMP_DIR / "pareto_accuracy_speed.pdf", bbox_inches="tight")
    plt.close(fig)

    # error_histograms.png/pdf — per-sample error distributions
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.5 * DOUBLE_COL))
    bins = np.logspace(-4, 0.5, 30)
    n_top = min(8, len(valid_rows))
    for i, r in enumerate(valid_rows[:n_top]):
        per = np.asarray(r.get("per_sample_loss", []))
        if per.size == 0:
            continue
        per = np.clip(per, 1e-6, 1)
        ax.hist(per, bins=bins, alpha=0.5,
                label=f"{r['approach_number']:02d} {r['approach']}",
                histtype="step", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"per-sample FD mismatch (mean over $M_\mathrm{tot}$)")
    ax.set_ylabel("count")
    ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(COMP_DIR / "error_histograms.png", dpi=200, bbox_inches="tight")
    plt.savefig(COMP_DIR / "error_histograms.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote outputs to {COMP_DIR}")
    print(f"Best closed-form model: {best['approach_number']:02d} {best['approach']} loss={best['loss']:.4e}")


if __name__ == "__main__":
    main()
