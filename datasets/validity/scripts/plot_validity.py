"""Generate Nature-style reference plots for the Validity Bench dataset.

Produces:
  - mm_histogram.{png,pdf}     — histogram of log10(mismatch) for train & val
  - mm_vs_q.{png,pdf}          — log10(mismatch) vs mass ratio
  - mm_vs_chi1z.{png,pdf}      — log10(mismatch) vs primary spin
  - mm_vs_chi2z.{png,pdf}      — log10(mismatch) vs secondary spin
  - mm_vs_chieff.{png,pdf}     — log10(mismatch) vs effective spin
  - mm_vs_params.{png,pdf}     — combined 2x2 panel

Usage:
    python plot_validity.py
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gwbenchmarks import plot_settings

plot_settings.apply()

DATASET_DIR = Path(__file__).parent.parent
PLOT_DIR = DATASET_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def load_split(path):
    with h5py.File(path, "r") as f:
        return {
            "q": f["q"][:],
            "chi1z": f["chi1z"][:],
            "chi2z": f["chi2z"][:],
            "omega0": f["omega0"][:],
            "mm_td": f["mm_td"][:],
        }


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"{name}.{ext}")
    plt.close(fig)


def chieff(q, chi1z, chi2z):
    return (q * chi1z + chi2z) / (1.0 + q)


def main():
    train = load_split(DATASET_DIR / "validity_training.h5")
    val = load_split(DATASET_DIR / "validity_validation.h5")

    log_mm_tr = np.log10(np.clip(train["mm_td"], 1e-12, None))
    log_mm_va = np.log10(np.clip(val["mm_td"], 1e-12, None))

    ce_tr = chieff(train["q"], train["chi1z"], train["chi2z"])
    ce_va = chieff(val["q"], val["chi1z"], val["chi2z"])

    c_tr = plot_settings.COLORS["blue"]
    c_va = plot_settings.COLORS["red"]

    # 1. Histogram of log10(mismatch)
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
    bins = np.linspace(-7, 0, 40)
    ax.hist(log_mm_tr, bins=bins, alpha=0.6, color=c_tr, label="Training")
    ax.hist(log_mm_va, bins=bins, alpha=0.6, color=c_va, label="Validation")
    ax.set_xlabel(r"$\log_{10}(\mathcal{MM})$")
    ax.set_ylabel("Count")
    ax.legend()
    save(fig, "mm_histogram")
    print("  mm_histogram done")

    # 2. Individual scatter plots
    scatter_params = [
        ("q", r"$q$", train["q"], val["q"]),
        ("chi1z", r"$\chi_{1z}$", train["chi1z"], val["chi1z"]),
        ("chi2z", r"$\chi_{2z}$", train["chi2z"], val["chi2z"]),
        ("chieff", r"$\chi_{\mathrm{eff}}$", ce_tr, ce_va),
    ]
    for pname, plabel, p_tr, p_va in scatter_params:
        fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
        ax.scatter(p_tr, log_mm_tr, s=8, alpha=0.5, color=c_tr,
                   label="Training", rasterized=True)
        ax.scatter(p_va, log_mm_va, s=8, alpha=0.5, marker="s", color=c_va,
                   label="Validation", rasterized=True)
        ax.set_xlabel(plabel)
        ax.set_ylabel(r"$\log_{10}(\mathcal{MM})$")
        ax.legend(markerscale=1.5)
        save(fig, f"mm_vs_{pname}")
        print(f"  mm_vs_{pname} done")

    # 3. Combined 2x2 panel
    fig, axes = plt.subplots(
        2, 2, figsize=plot_settings.figsize(cols=2, aspect=0.8)
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    for ax, (pname, plabel, p_tr, p_va) in zip(axes.flat, scatter_params):
        ax.scatter(p_tr, log_mm_tr, s=6, alpha=0.4, color=c_tr,
                   label="Train", rasterized=True)
        ax.scatter(p_va, log_mm_va, s=6, alpha=0.4, marker="s", color=c_va,
                   label="Val", rasterized=True)
        ax.set_xlabel(plabel)
        ax.set_ylabel(r"$\log_{10}(\mathcal{MM})$")
    axes[0, 1].legend(fontsize=7, markerscale=1.5)
    save(fig, "mm_vs_params")
    print("  mm_vs_params done")

    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
