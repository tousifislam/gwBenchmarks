"""
Waveform benchmark: per-sample validation mismatch distribution
for the best model of each agent that stored raw mismatch values.

Agents with mismatch data:
  opus47   → error_data['svd_random_forest_raw']['val']  (confirmed mismatch, n=250)
  sonnet46 → error_data['08_svd_pysr_eff']['mismatch_all']  (n=250)

opus46 stored only combined loss (mismatch + phase_rmse + log_amp_rmse) — excluded.
haiku has no per-sample waveform data — excluded.

Output: results/comparison/waveform_mismatch_violin.{pdf,png}
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJ)
from gwbenchmarks.plot_settings import apply, figsize, COLORS

apply()

RESULTS = os.path.join(PROJ, "results")
OUT_DIR = os.path.join(RESULTS, "comparison")


# ── load per-sample mismatches ────────────────────────────────────────────────

def _jload(path):
    with open(path) as f:
        return json.load(f)


opus47_ed  = _jload(os.path.join(RESULTS, "opus47/waveform/comparison/error_data.json"))
sonnet46_ed = _jload(os.path.join(RESULTS, "sonnet46/waveform/comparison/error_data.json"))

opus47_mismatch   = np.array(opus47_ed["svd_random_forest_raw"]["val"])
# sonnet46's agent used a mismatch formula without the [0,1] clip:
# overlap = Re(<h_pred, h_true>) / (‖h_pred‖ ‖h_true‖) can be negative for
# anti-aligned waveforms, making 1−overlap > 1. We clip here to match the
# correct physical definition used by gwbenchmarks.metrics.mismatch().
sonnet46_mismatch = np.clip(
    np.array(sonnet46_ed["08_svd_pysr_eff"]["mismatch_all"]), 0.0, 1.0
)

# Best model labels (from best_model.json)
opus47_bm   = _jload(os.path.join(RESULTS, "opus47/waveform/comparison/best_model.json"))
sonnet46_bm = _jload(os.path.join(RESULTS, "sonnet46/waveform/comparison/best_model.json"))

AGENTS = [
    ("Opus 4.7",   opus47_mismatch,   COLORS["red"],    opus47_bm["approach"]),
    ("Sonnet 4.6", sonnet46_mismatch, COLORS["green"],  sonnet46_bm["approach"]),
]

print("Data summary:")
for label, arr, _, approach in AGENTS:
    print(f"  {label} ({approach}): n={len(arr)}, "
          f"min={arr.min():.4f}, median={np.median(arr):.4f}, max={arr.max():.4f}")


# ── plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=figsize(cols=1, aspect=0.9))

labels   = [a[0] for a in AGENTS]
arrays   = [a[1] for a in AGENTS]
colors   = [a[2] for a in AGENTS]
positions = np.array([1, 2])

# work in log10 space so violin shapes are meaningful across decades
log_arrays = [np.log10(np.clip(arr, 1e-6, None)) for arr in arrays]

parts = ax.violinplot(
    log_arrays,
    positions=positions,
    widths=0.55,
    showmedians=True,
    showextrema=True,
)

for pc, col in zip(parts["bodies"], colors):
    pc.set_facecolor(col)
    pc.set_alpha(0.55)
    pc.set_edgecolor(col)
    pc.set_linewidth(0.8)

for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
    if part_name in parts:
        parts[part_name].set_color("#1a1a1a")
        parts[part_name].set_linewidth(0.9)

# ── y-axis: decade ticks with 10^n labels ────────────────────────────────────
ymin = min(a.min() for a in log_arrays)
ymax = max(a.max() for a in log_arrays)
decade_min = int(np.floor(ymin))
decade_max = int(np.ceil(ymax))
decades = np.arange(decade_min, decade_max + 1)
ax.yaxis.set_major_locator(ticker.FixedLocator(decades))
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda v, _: f"$10^{{{int(v):d}}}$")
)
ax.yaxis.set_minor_locator(ticker.NullLocator())

# ── x-axis ────────────────────────────────────────────────────────────────────
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlim(0.4, 2.6)

ax.set_ylabel("Mismatch", fontsize=11)
ax.set_title(r"Waveform ($h_{22}$) — validation mismatch", fontsize=10, pad=5)
ax.text(0.98, 0.02, "Sonnet 4.6 values clipped to [0,1]\n(agent bug: missing clip in mismatch fn.)",
        ha="right", va="bottom", fontsize=5.5, color="#888888",
        transform=ax.transAxes, style="italic")

# annotate best-model name and sample size below each violin
for (label, arr, col, approach), pos in zip(AGENTS, positions):
    ax.text(pos, -0.08, f"n={len(arr)}",
            ha="center", va="top", fontsize=6.5, color="#555555",
            transform=ax.get_xaxis_transform())
    ax.text(pos, -0.14, approach,
            ha="center", va="top", fontsize=5.5, color="#777777",
            style="italic", transform=ax.get_xaxis_transform())

for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"waveform_mismatch_violin.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
