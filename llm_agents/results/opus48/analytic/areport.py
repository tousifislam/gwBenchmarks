"""Comparison artifacts for the Analytic Bench (opus48, original work)."""
import json
import numpy as np
from pathlib import Path
import gwbenchmarks.plot_settings as ps

HERE = Path(__file__).resolve().parent
COMP = HERE / "comparison"; COMP.mkdir(exist_ok=True)
NR_FLOOR = 1.4e-3
CAT = {"physics": ps.COLORS["blue"], "symbolic": ps.COLORS["red"],
       "functional": ps.COLORS["green"], "matched_asymptotic": ps.COLORS["orange"]}


def _short(n):
    return n.replace("_", " ")


def update_progress(results):
    ps.apply()
    r = sorted(results, key=lambda x: x["scorecard"]["approach_number"])
    nums = np.array([x["scorecard"]["approach_number"] for x in r])
    loss = np.array([x["scorecard"]["loss"] for x in r])
    fig, ax = ps.plt.subplots(figsize=(ps.SINGLE_COL, 2.6))
    ax.plot(nums, loss, "o", color=ps.COLORS["gray"], ms=4, label="per approach")
    ax.plot(nums, np.minimum.accumulate(loss), "-", color=ps.COLORS["blue"], lw=1.3, label="running best")
    ax.axhline(NR_FLOOR, ls="--", color=ps.COLORS["black"], lw=0.8, label="NR floor")
    ax.set_yscale("log"); ax.set_xlabel("approach number"); ax.set_ylabel("mean FD mismatch")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(COMP / f"progress.{e}", dpi=200)
    ps.plt.close(fig)


def final_plots(results):
    ps.apply()
    r = sorted(results, key=lambda x: x["scorecard"]["loss"])
    names = [_short(x["scorecard"]["approach"]) for x in r]
    loss = np.array([x["scorecard"]["loss"] for x in r])
    rts = np.array([max(x["scorecard"]["runtime_ms"], 1e-6) for x in r])
    col = [CAT.get(x["scorecard"]["category"], ps.COLORS["gray"]) for x in r]

    fig, ax = ps.plt.subplots(figsize=(ps.DOUBLE_COL, 3.6))
    y = np.arange(len(names)); ax.barh(y, loss, color=col)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6); ax.invert_yaxis()
    ax.axvline(NR_FLOOR, ls="--", color=ps.COLORS["black"], lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("mean FD mismatch (raw loss)")
    _leg(ax); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(COMP / f"loss_only_comparison.{e}", dpi=200)
    ps.plt.close(fig)

    fig, ax = ps.plt.subplots(figsize=(ps.DOUBLE_COL, 4.2))
    for i in range(len(names)):
        ax.scatter(rts[i], loss[i], color=col[i], s=28, zorder=3)
        ax.annotate(names[i], (rts[i], loss[i]), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax.axhline(NR_FLOOR, ls="--", color=ps.COLORS["black"], lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("evaluation time per waveform (ms)"); ax.set_ylabel("mean FD mismatch")
    _leg(ax); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(COMP / f"pareto_accuracy_speed.{e}", dpi=200)
    ps.plt.close(fig)

    _hist(r)
    summary = [{"rank": i + 1, "approach": x["scorecard"]["approach"],
                "category": x["scorecard"]["category"],
                "parameterization": x["scorecard"]["parameterization"],
                "loss": x["scorecard"]["loss"], "train_loss": x["scorecard"]["train_loss"],
                "runtime_ms": x["scorecard"]["runtime_ms"]} for i, x in enumerate(r)]
    (COMP / "summary_table.json").write_text(json.dumps(summary, indent=2))
    (COMP / "best_model.json").write_text(json.dumps(r[0]["scorecard"], indent=2))
    err = {x["scorecard"]["approach"]: {"train": list(map(float, x["err_tr"])),
                                        "val": list(map(float, x["err_va"]))} for x in results}
    (COMP / "error_data.json").write_text(json.dumps(err, indent=2))
    # collected closed-form expressions
    allexpr = [{"approach": x["scorecard"]["approach"], "category": x["scorecard"]["category"],
                "loss": x["scorecard"]["loss"], "expression": x.get("expr", "")} for x in r]
    (COMP / "all_expressions.json").write_text(json.dumps(allexpr, indent=2))
    (COMP / "best_expression.txt").write_text(
        f"BEST closed-form model: {r[0]['scorecard']['approach']} "
        f"(mean FD mismatch {r[0]['scorecard']['loss']:.4f})\n\n" + r[0].get("expr", ""))


def _hist(r):
    top = r[:min(6, len(r))]; nrow = int(np.ceil(len(top) / 3))
    fig, axes = ps.plt.subplots(nrow, 3, figsize=(ps.DOUBLE_COL, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel(); bins = np.logspace(-3, 0, 26)
    for k, x in enumerate(top):
        ax = axes[k]
        tr = np.clip(x["err_tr"], 1e-4, None); va = np.clip(x["err_va"], 1e-4, None)
        ax.hist(tr, bins=bins, alpha=0.55, color=ps.COLORS["blue"], label="train", density=True)
        ax.hist(va, bins=bins, alpha=0.5, color=ps.COLORS["red"], hatch="///", label="val", density=True)
        ax.axvline(NR_FLOOR, ls="--", color=ps.COLORS["black"], lw=0.8)
        ax.set_xscale("log"); ax.set_title(_short(x["scorecard"]["approach"]), fontsize=6)
        ax.set_xlabel("FD mismatch", fontsize=7)
        if k % 3 == 0:
            ax.set_ylabel("density", fontsize=7)
        if k == 0:
            ax.legend(frameon=False, fontsize=6)
    for k in range(len(top), len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(COMP / f"error_histograms.{e}", dpi=200)
    ps.plt.close(fig)


def _leg(ax):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="s", ls="", color=c, label=cat) for cat, c in CAT.items()]
    h.append(Line2D([0], [0], ls="--", color=ps.COLORS["black"], label="NR floor"))
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="best")
