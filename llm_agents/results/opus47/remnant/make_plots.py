"""Generate comparison plots and CHANGELOG for remnant benchmark."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
try:
    from gwbenchmarks.plot_settings import set_nature_style
    set_nature_style()
except Exception:
    pass

CMP = HERE / "comparison"
NR_FLOOR_VF = 1e-4  # delta_vf is order 1e-4

with open(CMP / "summary_table.json") as f:
    summary = json.load(f)
with open(CMP / "error_data.json") as f:
    error_data = json.load(f)

CAT_COLORS = {
    "kernel_gp": "#1f77b4",
    "symbolic": "#d62728",
    "interpolation": "#2ca02c",
    "ml": "#ff7f0e",
}

names = [s["approach"] for s in summary]
nums = [s["approach_number"] for s in summary]
losses = np.array([s["loss"] for s in summary])
runtimes = np.array([max(s["runtime_ms"], 1e-3) for s in summary])
cats = [s["category"] for s in summary]
params = [s["parameterization"] for s in summary]

# === Loss comparison bar chart ===
fig, ax = plt.subplots(figsize=(10, 6))
order = np.argsort(losses)
xs = np.arange(len(losses))
colors = [CAT_COLORS.get(cats[i], "gray") for i in order]
ax.barh(xs, losses[order], color=colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(xs)
ax.set_yticklabels([names[i] for i in order], fontsize=7)
ax.set_xlabel("NRMSE(v_k) — validation")
ax.axvline(NR_FLOOR_VF, color="gray", linestyle="--", lw=0.8, label=f"NR floor ({NR_FLOOR_VF:.0e})")
for c, col in CAT_COLORS.items():
    ax.bar([0], [0], color=col, label=c)
ax.legend(loc="lower right", fontsize=7)
ax.set_xscale("log")
plt.tight_layout()
plt.savefig(CMP / "loss_only_comparison.png", dpi=150)
plt.savefig(CMP / "loss_only_comparison.pdf")
plt.close()

# === Pareto plot ===
fig, ax = plt.subplots(figsize=(8, 6))
for i, (l, r) in enumerate(zip(losses, runtimes)):
    ax.scatter(r, l, s=70, color=CAT_COLORS.get(cats[i], "gray"), edgecolor="black",
               linewidth=0.5, alpha=0.9)
    ax.annotate(names[i], (r, l), fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Eval time (ms / sample)")
ax.set_ylabel("NRMSE(v_k)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.axhline(NR_FLOOR_VF, color="gray", linestyle="--", lw=0.8)
for c, col in CAT_COLORS.items():
    ax.scatter([], [], color=col, label=c, edgecolor="black", linewidth=0.5)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(CMP / "pareto_accuracy_speed.png", dpi=150)
plt.savefig(CMP / "pareto_accuracy_speed.pdf")
plt.close()

# === Progress plot ===
ord2 = np.argsort(nums)
running_best = np.minimum.accumulate(losses[ord2])
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.array(nums)[ord2], losses[ord2], "o-", color="C0", label="Per-approach loss")
ax.plot(np.array(nums)[ord2], running_best, "s--", color="C3", label="Running best")
ax.axhline(NR_FLOOR_VF, color="gray", linestyle=":", lw=0.8, label="NR floor")
for i, idx in enumerate(ord2):
    ax.annotate(names[idx], (nums[idx], losses[idx]), fontsize=5, rotation=45,
                xytext=(0, 4), textcoords="offset points")
ax.set_xlabel("Approach number")
ax.set_ylabel("NRMSE(v_k)")
ax.set_yscale("log")
ax.legend()
plt.tight_layout()
plt.savefig(CMP / "progress.png", dpi=150)
plt.savefig(CMP / "progress.pdf")
plt.close()

# === Error histograms ===
fig, axes = plt.subplots(6, 4, figsize=(15, 16))
for i, s in enumerate(summary):
    ax = axes.flat[i]
    nm = s["approach"]
    if nm not in error_data:
        ax.set_visible(False)
        continue
    ed = error_data[nm]
    tr = np.array(ed["train"])
    val = np.array(ed["val"])
    tr_pos = np.maximum(tr, 1e-12)
    val_pos = np.maximum(val, 1e-12)
    bins = np.geomspace(min(tr_pos.min(), val_pos.min()), max(tr_pos.max(), val_pos.max()), 25)
    ax.hist(tr_pos, bins=bins, alpha=0.5, color="C0", hatch="//", label="train", edgecolor="black", linewidth=0.3)
    ax.hist(val_pos, bins=bins, alpha=0.5, color="C3", label="val", edgecolor="black", linewidth=0.3)
    ax.axvline(NR_FLOOR_VF, color="gray", linestyle="--", lw=0.6)
    ax.set_xscale("log")
    ax.set_title(nm, fontsize=7)
    ax.tick_params(labelsize=6)
    if i == 0:
        ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(CMP / "error_histograms.png", dpi=120)
plt.savefig(CMP / "error_histograms.pdf")
plt.close()

best = sorted(summary, key=lambda s: s["loss"])[0]
with open(CMP / "best_model.json", "w") as f:
    json.dump(best, f, indent=2, default=str)

chl = HERE / "CHANGELOG.md"
lines = ["# Remnant Benchmark — CHANGELOG", "",
         "Each entry: observation -> hypothesis -> action -> outcome.", ""]
for s in summary:
    lines.append(f"## {s['approach_number']:02d} — {s['approach']}")
    lines.append(f"- Category: **{s['category']}**, parameterization: **{s['parameterization']}**")
    lines.append(f"- Loss (NRMSE v_k): **{s['loss']:.4e}**, runtime: {s['runtime_ms']:.3f} ms")
    lines.append(f"- Notes: {s['notes']}")
    if s['approach_number'] == 1:
        lines.append("- **Reasoning**: GPR baseline. Kicks have a sharp 'superkick' regime around q~1, antialigned spins. RBF kernel may smooth over this.")
    elif s['approach_number'] == 5:
        lines.append("- **Reasoning**: MLP can capture sharp features that kernel methods over-smooth. Standardized inputs help convergence.")
    elif s['approach_number'] == 9:
        lines.append("- **Reasoning**: PySR finds analytic forms. Top equation: see saved_model/expressions.json.")
    elif s['approach_number'] == 17:
        lines.append("- **Reasoning**: Lousto-Zlochower hand-crafted features encode the leading-order kick contributions: eta^2*delta_m, eta^2*chi_a, eta^2*chi_p.")
    elif s['approach_number'] == 20:
        lines.append("- **Reasoning**: Stacking GBM (captures bulk trend) + GPR-on-residuals (captures smooth correlations).")
    lines.append("")
lines.append(f"## Summary\n\n- Best: **{best['approach']}** with NRMSE {best['loss']:.4e}")
lines.append(f"- Total approaches: {len(summary)}")
lines.append(f"- Categories: {sorted(set(cats))}")
lines.append(f"- Parameterizations: {sorted(set(params))}")
chl.write_text("\n".join(lines))
print(f"Wrote CHANGELOG and plots. Best: {best['approach']} (loss={best['loss']:.4e})")
