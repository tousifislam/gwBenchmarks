"""Comparison artifacts for the Remnant Bench (opus48, original work)."""
import json
import numpy as np
from pathlib import Path
import gwbenchmarks.plot_settings as ps

HERE = Path(__file__).resolve().parent
COMP = HERE / "comparison"; COMP.mkdir(exist_ok=True)
NR_FLOOR = None  # vk has no single floor; delta_vf overlaid per-sample instead

CAT_COLOR = {"kernel_gp": ps.COLORS["blue"], "symbolic": ps.COLORS["red"],
             "interpolation": ps.COLORS["green"], "ml": ps.COLORS["orange"]}


def _short(n):
    return n.replace("_", " ")


def _vf(results):
    return [r for r in results if r["scorecard"]["target"] == "vf_mag"]


def update_progress(results):
    ps.apply()
    vf = sorted(_vf(results), key=lambda r: r["scorecard"]["approach_number"])
    nums = np.array([r["scorecard"]["approach_number"] for r in vf])
    losses = np.array([r["scorecard"]["loss"] for r in vf])
    best = np.minimum.accumulate(losses)
    fig, ax = ps.plt.subplots(figsize=(ps.SINGLE_COL, 2.6))
    ax.plot(nums, losses, "o", color=ps.COLORS["gray"], ms=4, label="per approach")
    ax.plot(nums, best, "-", color=ps.COLORS["blue"], lw=1.3, label="running best")
    ax.set_xlabel("approach number"); ax.set_ylabel("NRMSE(v_k)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(COMP / f"progress.{e}", dpi=200)
    ps.plt.close(fig)


def final_plots(results):
    ps.apply()
    vf = sorted(_vf(results), key=lambda r: r["scorecard"]["loss"])
    names = [_short(r["scorecard"]["approach"]) for r in vf]
    cats = [r["scorecard"]["category"] for r in vf]
    losses = np.array([r["scorecard"]["loss"] for r in vf])
    rts = np.array([r["scorecard"]["runtime_ms"] for r in vf])
    colors = [CAT_COLOR.get(c, ps.COLORS["gray"]) for c in cats]

    # loss_only
    fig, ax = ps.plt.subplots(figsize=(ps.DOUBLE_COL, 3.4))
    y = np.arange(len(names))
    ax.barh(y, losses, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("NRMSE(v_k) (raw loss)")
    _legend(ax)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(COMP / f"loss_only_comparison.{e}", dpi=200)
    ps.plt.close(fig)

    # pareto
    fig, ax = ps.plt.subplots(figsize=(ps.DOUBLE_COL, 4.2))
    for i in range(len(names)):
        ax.scatter(rts[i], losses[i], color=colors[i], s=28, zorder=3)
        ax.annotate(names[i], (rts[i], losses[i]), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("evaluation time per sample (ms)")
    ax.set_ylabel("NRMSE(v_k)")
    _legend(ax)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(COMP / f"pareto_accuracy_speed.{e}", dpi=200)
    ps.plt.close(fig)

    _hist(vf)

    summary = [{"rank": i + 1, "approach": r["scorecard"]["approach"],
                "category": r["scorecard"]["category"],
                "parameterization": r["scorecard"]["parameterization"],
                "loss": r["scorecard"]["loss"],
                "train_loss": r["scorecard"]["train_loss"],
                "runtime_ms": r["scorecard"]["runtime_ms"]}
               for i, r in enumerate(vf)]
    (COMP / "summary_table.json").write_text(json.dumps(summary, indent=2))
    (COMP / "best_model.json").write_text(json.dumps(vf[0]["scorecard"], indent=2))
    err = {r["scorecard"]["approach"]: {
        "train": list(map(float, r["err_tr"])),
        "val": list(map(float, r["err_va"]))} for r in results}
    (COMP / "error_data.json").write_text(json.dumps(err, indent=2))


def _hist(vf):
    top = vf[:min(6, len(vf))]
    nrow = int(np.ceil(len(top) / 3))
    fig, axes = ps.plt.subplots(nrow, 3, figsize=(ps.DOUBLE_COL, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    # NR floor for v_k (range-normalised delta_vf median)
    import rdata as RD
    _, yva, nr = RD.load("validation")
    floor = float(np.median(nr["vf_mag"]) / np.ptp(yva["vf_mag"]))
    bins = np.logspace(-5, 0, 30)
    for k, r in enumerate(top):
        ax = axes[k]
        tr = np.clip(r["err_tr"], 1e-6, None); va = np.clip(r["err_va"], 1e-6, None)
        ax.hist(tr, bins=bins, alpha=0.55, color=ps.COLORS["blue"], label="train", density=True)
        ax.hist(va, bins=bins, alpha=0.5, color=ps.COLORS["red"], hatch="///", label="val", density=True)
        ax.axvline(floor, ls="--", color=ps.COLORS["black"], lw=0.8)
        ax.set_xscale("log"); ax.set_title(_short(r["scorecard"]["approach"]), fontsize=6)
        ax.set_xlabel("|v_k err| / range", fontsize=7)
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


def _legend(ax):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="s", ls="", color=c, label=cat)
         for cat, c in CAT_COLOR.items()]
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="best")
