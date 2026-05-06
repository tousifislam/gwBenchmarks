#!/usr/bin/env python3
"""Generate all comparison plots for the waveform benchmark."""

import sys, os
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../.."))
sys.path.insert(0, ROOT)

import gwbenchmarks.plot_settings as ps
ps.apply()

NR_ERROR_FLOOR = 1.4e-3
COMP_DIR = os.path.join(WORK_DIR, "comparison")
os.makedirs(COMP_DIR, exist_ok=True)

# Load all scorecards
results = []
models_dir = os.path.join(WORK_DIR, "models")
for d in sorted(os.listdir(models_dir)):
    sc_path = os.path.join(models_dir, d, "scorecard.json")
    if os.path.exists(sc_path):
        with open(sc_path) as f:
            sc = json.load(f)
            sc["dir_name"] = d
            results.append(sc)

results.sort(key=lambda x: x["approach_number"])
print(f"Found {len(results)} approaches")

names = [r["approach"].replace("_", " ") for r in results]
short_names = [r["dir_name"].split("_", 1)[1].replace("_", " ") for r in results]
losses = [r["loss"] for r in results]
runtimes = [r["runtime_ms"] for r in results]

CATEGORY_COLORS = {
    "svd": ps.COLORS["blue"],
    "decomp": ps.COLORS["blue"],
    "symbolic": ps.COLORS["green"],
    "interp": ps.COLORS["orange"],
    "ml": ps.COLORS["red"],
}

def get_category(name):
    name_l = name.lower()
    if any(x in name_l for x in ["pysr", "gplearn", "symbolic"]):
        return "symbolic"
    if any(x in name_l for x in ["rbf interp", "knn", "krr", "kernel"]):
        return "interp"
    if any(x in name_l for x in ["mlp", "rf ", "gbr", "gradient", "random", "ada", "extra", "svr", "forest", "boost", "tree"]):
        return "ml"
    return "svd"

categories = [get_category(n) for n in short_names]
colors = [CATEGORY_COLORS.get(c, ps.COLORS["gray"]) for c in categories]

# ─── 1. Progress plot ───
fig, ax = plt.subplots(figsize=ps.figsize(2, 0.5))
sorted_idx = np.argsort(losses)
y_pos = np.arange(len(results))
bars = ax.barh(y_pos, [losses[i] for i in sorted_idx], color=[colors[i] for i in sorted_idx], height=0.7)
ax.axvline(NR_ERROR_FLOOR, color='k', ls='--', lw=0.8, label=f'NR error floor ({NR_ERROR_FLOOR:.1e})')
ax.set_yticks(y_pos)
ax.set_yticklabels([short_names[i] for i in sorted_idx], fontsize=6)
ax.set_xlabel("Mean FD Mismatch")
ax.set_xscale("log")
ax.legend(fontsize=7)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(COMP_DIR, "progress.png"))
fig.savefig(os.path.join(COMP_DIR, "progress.pdf"))
plt.close()
print("Saved progress plot")

# ─── 2. Pareto accuracy vs speed ───
fig, ax = plt.subplots(figsize=ps.figsize(1, 0.9))
for cat in ["svd", "symbolic", "interp", "ml"]:
    idx = [i for i, c in enumerate(categories) if c == cat]
    if not idx:
        continue
    ax.scatter([losses[i] for i in idx], [runtimes[i] for i in idx],
               c=CATEGORY_COLORS[cat], label=cat.upper(), s=30, zorder=5)
    for i in idx:
        ax.annotate(short_names[i], (losses[i], runtimes[i]),
                     fontsize=4.5, ha='left', va='bottom', rotation=15)
ax.axvline(NR_ERROR_FLOOR, color='k', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel("Mean FD Mismatch (loss)")
ax.set_ylabel("Prediction time (ms/sample)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(COMP_DIR, "pareto_accuracy_speed.png"))
fig.savefig(os.path.join(COMP_DIR, "pareto_accuracy_speed.pdf"))
plt.close()
print("Saved Pareto plot")

# ─── 3. Loss-only comparison ───
fig, ax = plt.subplots(figsize=ps.figsize(2, 0.5))
sorted_idx2 = np.argsort(losses)
ax.barh(y_pos, [losses[i] for i in sorted_idx2],
        color=[colors[i] for i in sorted_idx2], height=0.7)
ax.axvline(NR_ERROR_FLOOR, color='k', ls='--', lw=0.8, label=f'NR floor')
ax.set_yticks(y_pos)
ax.set_yticklabels([short_names[i] for i in sorted_idx2], fontsize=6)
ax.set_xlabel("Mean FD Mismatch")
ax.set_xscale("log")
ax.legend(fontsize=7)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(COMP_DIR, "loss_only_comparison.png"))
fig.savefig(os.path.join(COMP_DIR, "loss_only_comparison.pdf"))
plt.close()
print("Saved loss comparison plot")

# ─── 4. Error histograms ───
error_data_path = os.path.join(COMP_DIR, "error_data.json")
if os.path.exists(error_data_path):
    with open(error_data_path) as f:
        error_data = json.load(f)

    fig, ax = plt.subplots(figsize=ps.figsize(2, 0.7))
    bins = np.logspace(-5, 0, 50)

    plot_items = sorted(error_data.items(), key=lambda x: np.median(x[1].get("val_losses", [1.0])))
    for idx, (name, data) in enumerate(plot_items[:12]):
        val_losses = data.get("val_losses", [])
        if val_losses:
            ax.hist(val_losses, bins=bins, alpha=0.4, label=name.replace("_", " "),
                    histtype='stepfilled', linewidth=0.5)

    ax.axvline(NR_ERROR_FLOOR, color='k', ls='--', lw=1.0, label='NR floor')
    ax.set_xlabel("Per-sample FD Mismatch")
    ax.set_ylabel("Count")
    ax.set_xscale("log")
    ax.legend(fontsize=5, ncol=2, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(COMP_DIR, "error_histograms.png"))
    fig.savefig(os.path.join(COMP_DIR, "error_histograms.pdf"))
    plt.close()
    print("Saved error histograms")
else:
    print("No error_data.json found - skipping histograms")

# ─── 5. Summary table ───
with open(os.path.join(COMP_DIR, "summary_table.json"), "w") as f:
    json.dump(sorted(results, key=lambda x: x["loss"]), f, indent=2)
print("Saved summary table")

# ─── 6. Best model ───
best = min(results, key=lambda x: x["loss"])
with open(os.path.join(COMP_DIR, "best_model.json"), "w") as f:
    json.dump(best, f, indent=2)
print(f"Best model: {best['approach']} with loss={best['loss']:.6f}")
