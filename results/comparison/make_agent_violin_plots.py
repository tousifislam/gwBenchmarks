"""
Violin plots comparing validation error distributions of the best model
from each agent across all 6 benchmarks.

Agents: haiku, opus46, opus47, sonnet46
Output: results/comparison/{benchmark}_violin.{pdf,png}
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── project root ──────────────────────────────────────────────────────────────
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJ)
from gwbenchmarks.plot_settings import apply, figsize, COLORS

apply()

RESULTS = os.path.join(PROJ, "results")
OUT_DIR = os.path.join(RESULTS, "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

AGENTS = ["haiku", "opus46", "opus47", "sonnet46"]
AGENT_LABELS = {
    "haiku":   "Haiku",
    "opus46":  "Opus 4.6",
    "opus47":  "Opus 4.7",
    "sonnet46": "Sonnet 4.6",
}
AGENT_COLORS = {
    "haiku":   COLORS["orange"],
    "opus46":  COLORS["blue"],
    "opus47":  COLORS["red"],
    "sonnet46": COLORS["green"],
}

BENCHMARKS = {
    "ringdown": {
        "title": "Ringdown (QNM)",
        "ylabel": "Relative error",
        "log": True,
    },
    "waveform": {
        "title": r"Waveform ($h_{22}$)",
        "ylabel": "Validation loss",
        "log": True,
    },
    "remnant": {
        "title": "Remnant properties",
        "ylabel": "NRMSE",
        "log": False,
    },
    "dynamics": {
        "title": "Orbital dynamics",
        "ylabel": "Loss per sample",
        "log": True,
    },
    "validity": {
        "title": "Surrogate validity",
        "ylabel": r"$|\Delta\log\,\mathrm{mismatch}|$",
        "log": False,
    },
    "analytic": {
        "title": "Analytic waveform",
        "ylabel": "Mismatch",
        "log": True,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _find_key(ed_dict, best_name):
    """Find the key in error_data dict that corresponds to the best model."""
    if best_name in ed_dict:
        return best_name
    # numbered prefix: e.g. best='cubic_spline_log' → '12_cubic_spline_log'
    match = next(
        (k for k in ed_dict if k.endswith("_" + best_name) or k == best_name),
        None
    )
    if match:
        return match
    # partial match
    match = next((k for k in ed_dict if best_name in k), None)
    return match


def get_val_errors(agent, bench):
    """
    Returns 1-D numpy array of non-negative per-sample validation errors
    for the best model of (agent, bench), or None if unavailable.
    """
    base = os.path.join(RESULTS, agent, bench, "comparison")
    bm_path = os.path.join(base, "best_model.json")
    ed_path = os.path.join(base, "error_data.json")

    if not os.path.exists(bm_path) or not os.path.exists(ed_path):
        return None

    bm = _load_json(bm_path)
    ed = _load_json(ed_path)

    # ── resolve best model name ───────────────────────────────────────────────
    best_name = bm.get("approach", bm.get("name", bm.get("best_model", "")))

    # ── dispatch by agent / benchmark ────────────────────────────────────────

    # haiku / ringdown: {omega_r: [...], omega_i: [...]} (no val/train split)
    if agent == "haiku" and bench == "ringdown":
        if not isinstance(ed, dict) or best_name not in ed:
            return None
        inner = ed[best_name]
        er = np.abs(inner.get("omega_r", []))
        ei = np.abs(inner.get("omega_i", []))
        if len(er) and len(ei):
            return 0.5 * (er + ei)
        return np.concatenate([er, ei]) if len(er) or len(ei) else None

    # haiku / other benchmarks: no per-sample data
    if agent == "haiku":
        return None

    # sonnet46 / analytic: only aggregate scores stored
    if agent == "sonnet46" and bench == "analytic":
        return None

    # sonnet46 / validity: raw predictions + true values in log space
    if agent == "sonnet46" and bench == "validity":
        key = _find_key(ed, best_name)
        if key is None:
            return None
        inner = ed[key]
        y_pred = np.array(inner.get("y_pred_v", []))
        y_true = np.array(inner.get("y_true_v", []))
        if len(y_pred) and len(y_true):
            return np.abs(y_pred - y_true)
        return None

    # sonnet46 / ringdown: val is a nested dict {er: [...], ei: [...]}
    if agent == "sonnet46" and bench == "ringdown":
        key = _find_key(ed, best_name)
        if key is None:
            return None
        inner = ed[key]
        val = inner.get("val", {})
        if isinstance(val, dict):
            er = np.abs(val.get("er", []))
            ei = np.abs(val.get("ei", []))
            if len(er) and len(ei):
                return 0.5 * (er + ei)
        return None

    # sonnet46 / waveform: val has mismatch_all list
    if agent == "sonnet46" and bench == "waveform":
        key = _find_key(ed, best_name)
        if key is None:
            return None
        inner = ed[key]
        arr = inner.get("mismatch_all", [])
        return np.array(arr) if arr else None

    # sonnet46 / dynamics: val_losses key
    if agent == "sonnet46" and bench == "dynamics":
        key = _find_key(ed, best_name)
        if key is None:
            return None
        inner = ed[key]
        arr = inner.get("val_losses", inner.get("val", []))
        return np.abs(arr) if len(arr) else None

    # sonnet46 / remnant: val key
    if agent == "sonnet46" and bench == "remnant":
        key = _find_key(ed, best_name)
        if key is None:
            return None
        inner = ed[key]
        arr = inner.get("val", inner.get("validation", []))
        return np.abs(arr) if len(arr) else None

    # opus46: validation key; opus47: val key
    val_key = "validation" if agent == "opus46" else "val"
    key = _find_key(ed, best_name)
    if key is None:
        return None
    inner = ed[key]
    arr = inner.get(val_key, inner.get("val", inner.get("validation", [])))
    if not arr:
        return None
    return np.abs(arr)


# ── collect data ──────────────────────────────────────────────────────────────

data = {}  # data[bench][agent] = np.array or None
for bench in BENCHMARKS:
    data[bench] = {}
    for agent in AGENTS:
        errors = get_val_errors(agent, bench)
        data[bench][agent] = errors
        n = len(errors) if errors is not None else 0
        print(f"  {agent:10s} / {bench:10s}: n={n}")


# ── plotting ──────────────────────────────────────────────────────────────────

def make_violin_plot(bench, cfg):
    import matplotlib.ticker as ticker

    # Only include agents that have data
    available = [(a, data[bench][a]) for a in AGENTS if data[bench][a] is not None]
    if not available:
        print(f"  [skip] {bench}: no data for any agent")
        return

    labels  = [AGENT_LABELS[a] for a, _ in available]
    colors  = [AGENT_COLORS[a] for a, _ in available]
    arrays  = [arr for _, arr in available]

    small = max(len(a) for a in arrays) < 50  # show all points for tiny datasets

    ncols = 1 if len(available) <= 3 else 2
    fig, ax = plt.subplots(figsize=figsize(cols=ncols, aspect=0.85))
    positions = np.arange(1, len(available) + 1)

    log_scale = cfg["log"]

    # For log-scale violins, matplotlib's violinplot needs linear data; we'll
    # transform to log10, draw violins, then relabel the y-axis manually using
    # a FixedLocator so there's no ambiguity.
    if log_scale:
        plot_arrays = [np.log10(np.clip(a, 1e-18, None)) for a in arrays]
    else:
        plot_arrays = list(arrays)

    # Draw violins
    parts = ax.violinplot(
        plot_arrays,
        positions=positions,
        widths=0.6,
        showmedians=True,
        showextrema=True,
    )

    # Colour each violin
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
        pc.set_edgecolor(col)
        pc.set_linewidth(0.8)

    for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
        if part_name in parts:
            parts[part_name].set_color("#1a1a1a")
            parts[part_name].set_linewidth(0.8)

    # Overlay individual points for small or tiny datasets
    rng = np.random.default_rng(42)
    for i, (arr, col) in enumerate(zip(plot_arrays, colors)):
        if small or len(arr) <= 30:
            jitter = rng.uniform(-0.08, 0.08, size=len(arr))
            ax.scatter(positions[i] + jitter, arr,
                       s=6, color=col, alpha=0.75, linewidths=0, zorder=5)

    # X axis
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(0.3, len(available) + 0.7)

    # Y axis
    if log_scale:
        # Place integer-decade ticks and label them as 10^n
        ymin = min(a.min() for a in plot_arrays)
        ymax = max(a.max() for a in plot_arrays)
        decade_min = int(np.floor(ymin))
        decade_max = int(np.ceil(ymax))
        # Use at most ~7 decade ticks; if range is too large, step by 2
        decades = np.arange(decade_min, decade_max + 1)
        step = max(1, int(np.ceil(len(decades) / 7)))
        decades = decades[::step]
        ax.yaxis.set_major_locator(ticker.FixedLocator(decades))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"$10^{{{int(v):d}}}$")
        )
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_ylabel(cfg["ylabel"], fontsize=10)
    else:
        ax.set_ylabel(cfg["ylabel"], fontsize=10)

    ax.set_title(cfg["title"], fontsize=10, pad=4)

    # Sample-size annotations below x labels
    for arr, pos in zip(arrays, positions):
        ax.text(pos, -0.12, f"n={len(arr)}",
                ha="center", va="top", fontsize=6, color="#555555",
                transform=ax.get_xaxis_transform())

    fig.savefig(os.path.join(OUT_DIR, f"{bench}_violin.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{bench}_violin.png"), dpi=300, bbox_inches="tight")
    for ext in ("pdf", "png"):
        print(f"  Saved: {os.path.join(OUT_DIR, f'{bench}_violin.{ext}')}")
    plt.close(fig)


print("\nGenerating violin plots…")
for bench, cfg in BENCHMARKS.items():
    print(f"\n{bench}:")
    make_violin_plot(bench, cfg)

print("\nDone.")
