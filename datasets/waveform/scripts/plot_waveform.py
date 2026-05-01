"""Generate Nature-style reference plots for the Waveform Bench dataset.

Produces:
  - param_space.{png,pdf}          — 2x3 parameter space scatter plots
  - param_q_vs_chieff.{png,pdf}    — q vs chi_eff
  - param_chieff_vs_chip.{png,pdf} — chi_eff vs chi_p
  - wf_length_vs_q.{png,pdf}       — waveform length vs q
  - example_h22.{png,pdf}          — example coprecessing h22 waveforms
  - nr_fd_mismatch.{png,pdf}       — NR FD mismatch histograms per total mass
  - nr_fd_mismatch_combined.{png,pdf} — combined NR FD mismatch histogram

Usage:
    python plot_waveform.py
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gwbenchmarks import plot_settings

plot_settings.apply()

DATASET_DIR = Path(__file__).parent.parent
PLOT_DIR = DATASET_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def load_split(path):
    data = {"sims": []}
    with h5py.File(path, "r") as f:
        meta = f["metadata"]
        data["q"] = meta["q"][:]
        data["chi1x"] = meta["chi1x"][:]
        data["chi1y"] = meta["chi1y"][:]
        data["chi1z"] = meta["chi1z"][:]
        data["chi2x"] = meta["chi2x"][:]
        data["chi2y"] = meta["chi2y"][:]
        data["chi2z"] = meta["chi2z"][:]
        data["omega0"] = meta["omega0"][:]
        data["chi_eff"] = meta["chi_eff"][:]
        data["chi_p"] = meta["chi_p"][:]
        data["wf_length"] = meta["wf_length"][:]
        data["n_sims"] = int(f.attrs["n_simulations"])

        nr = f["nr_errors"]
        data["nr_sxs_id"] = [s.decode() for s in nr["sxs_id"][:]] if "sxs_id" in nr else []
        for mtot in [40, 80, 120, 160, 200]:
            key = f"fd_mismatch_M{mtot}"
            data[f"nr_{key}"] = nr[key][:] if key in nr else np.array([])
        data["nr_fd_combined"] = nr["fd_mismatch_combined"][:] if "fd_mismatch_combined" in nr else np.array([])

        n_sims = data["n_sims"]
        for i in range(min(n_sims, 10)):
            grp = f[f"sim_{i:04d}"]
            data["sims"].append({
                "t": grp["t"][:],
                "h22_real": grp["h22_real"][:],
                "h22_imag": grp["h22_imag"][:],
                "q": float(grp.attrs["q"]),
                "chi_eff": float(grp.attrs.get("chi1z", 0) * grp.attrs["q"]
                                 + grp.attrs.get("chi2z", 0)) / (1.0 + grp.attrs["q"]),
                "sxs_id": grp.attrs["sxs_id"] if "sxs_id" in grp.attrs else f"sim_{i:04d}",
            })
    return data


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"{name}.{ext}")
    plt.close(fig)


def main():
    train = load_split(DATASET_DIR / "waveform_training.h5")
    val = load_split(DATASET_DIR / "waveform_validation.h5")

    c_tr = plot_settings.COLORS["blue"]
    c_va = plot_settings.COLORS["red"]

    # ── 1. Parameter space scatter (2x3) ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=plot_settings.figsize(cols=2, aspect=0.65))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    chi1_mag_tr = np.sqrt(train["chi1x"]**2 + train["chi1y"]**2 + train["chi1z"]**2)
    chi2_mag_tr = np.sqrt(train["chi2x"]**2 + train["chi2y"]**2 + train["chi2z"]**2)
    chi1_mag_va = np.sqrt(val["chi1x"]**2 + val["chi1y"]**2 + val["chi1z"]**2)
    chi2_mag_va = np.sqrt(val["chi2x"]**2 + val["chi2y"]**2 + val["chi2z"]**2)

    pairs = [
        (train["q"], train["chi_eff"], val["q"], val["chi_eff"],
         r"$q$", r"$\chi_{\mathrm{eff}}$"),
        (train["q"], train["chi_p"], val["q"], val["chi_p"],
         r"$q$", r"$\chi_p$"),
        (train["chi_eff"], train["chi_p"], val["chi_eff"], val["chi_p"],
         r"$\chi_{\mathrm{eff}}$", r"$\chi_p$"),
        (chi1_mag_tr, chi2_mag_tr, chi1_mag_va, chi2_mag_va,
         r"$|\chi_1|$", r"$|\chi_2|$"),
        (train["q"], train["chi1z"], val["q"], val["chi1z"],
         r"$q$", r"$\chi_{1z}$"),
        (train["q"], train["wf_length"], val["q"], val["wf_length"],
         r"$q$", r"Wf length [$M$]"),
    ]
    for ax, (x1, y1, x2, y2, xl, yl) in zip(axes.flat, pairs):
        ax.scatter(x1, y1, s=5, alpha=0.5, color=c_tr, label="Train", rasterized=True)
        ax.scatter(x2, y2, s=5, alpha=0.5, marker="s", color=c_va,
                   label="Val", rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0, 2].legend(fontsize=6, markerscale=1.5)
    save(fig, "param_space")
    print("  param_space done")

    # ── 2. q vs chi_eff ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
    ax.scatter(train["q"], train["chi_eff"], s=10, alpha=0.6, color=c_tr,
               label=f"Train ({train['n_sims']})", rasterized=True)
    ax.scatter(val["q"], val["chi_eff"], s=10, alpha=0.6, marker="s", color=c_va,
               label=f"Val ({val['n_sims']})", rasterized=True)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\chi_{\mathrm{eff}}$")
    ax.legend(fontsize=7, markerscale=1.5)
    save(fig, "param_q_vs_chieff")
    print("  param_q_vs_chieff done")

    # ── 3. chi_eff vs chi_p ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
    ax.scatter(train["chi_eff"], train["chi_p"], s=10, alpha=0.6, color=c_tr,
               label=f"Train ({train['n_sims']})", rasterized=True)
    ax.scatter(val["chi_eff"], val["chi_p"], s=10, alpha=0.6, marker="s", color=c_va,
               label=f"Val ({val['n_sims']})", rasterized=True)
    ax.set_xlabel(r"$\chi_{\mathrm{eff}}$")
    ax.set_ylabel(r"$\chi_p$")
    ax.legend(fontsize=7, markerscale=1.5)
    save(fig, "param_chieff_vs_chip")
    print("  param_chieff_vs_chip done")

    # ── 4. Waveform length vs q ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
    ax.scatter(train["q"], train["wf_length"], s=10, alpha=0.6, color=c_tr,
               label=f"Train ({train['n_sims']})", rasterized=True)
    ax.scatter(val["q"], val["wf_length"], s=10, alpha=0.6, marker="s", color=c_va,
               label=f"Val ({val['n_sims']})", rasterized=True)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"Waveform length [$M$]")
    ax.legend(fontsize=7, markerscale=1.5)
    save(fig, "wf_length_vs_q")
    print("  wf_length_vs_q done")

    # ── 5. Example coprecessing h22 waveforms ────────────────────────
    n_examples = min(6, len(train["sims"]))
    if n_examples > 0:
        fig, axes = plt.subplots(n_examples, 1,
                                 figsize=plot_settings.figsize(cols=2, aspect=0.25 * n_examples),
                                 sharex=False)
        if n_examples == 1:
            axes = [axes]
        cmap = plt.cm.viridis
        norm_q = Normalize(vmin=1.0, vmax=8.0)

        for ax, sim in zip(axes, train["sims"][:n_examples]):
            t = sim["t"]
            h_re = sim["h22_real"]
            h_im = sim["h22_imag"]
            amp = np.sqrt(h_re**2 + h_im**2)
            color = cmap(norm_q(sim["q"]))

            ax.plot(t, h_re, color=color, linewidth=0.5, alpha=0.8, label=r"$\mathrm{Re}(h_{22})$")
            ax.plot(t, amp, color="gray", linewidth=0.4, alpha=0.6, ls="--")
            ax.plot(t, -amp, color="gray", linewidth=0.4, alpha=0.6, ls="--")
            ax.set_ylabel(r"$h^{\mathrm{copr}}_{22}$", fontsize=7)
            ax.tick_params(labelsize=6)
            q_str = f"q={sim['q']:.1f}"
            ax.text(0.02, 0.85, q_str, transform=ax.transAxes, fontsize=6)

        axes[-1].set_xlabel(r"$t / M$")
        fig.subplots_adjust(hspace=0.3)
        save(fig, "example_h22")
        print("  example_h22 done")

    # ── 6. NR FD mismatch per total mass ─────────────────────────────
    all_nr = {}
    for mtot in [40, 80, 120, 160, 200]:
        key = f"nr_fd_mismatch_M{mtot}"
        vals = np.concatenate([train[key], val[key]]) if len(train[key]) > 0 and len(val[key]) > 0 \
               else (train[key] if len(train[key]) > 0 else val[key])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        all_nr[mtot] = vals

    if any(len(v) > 0 for v in all_nr.values()):
        fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
        colors = [plot_settings.COLORS["blue"], plot_settings.COLORS["red"],
                  plot_settings.COLORS["orange"], plot_settings.COLORS["green"],
                  plot_settings.COLORS["purple"]]
        for mtot, color in zip([40, 80, 120, 160, 200], colors):
            vals = all_nr[mtot]
            if len(vals) > 0:
                bins = np.logspace(np.log10(max(vals.min(), 1e-12)),
                                   np.log10(vals.max()), 25)
                ax.hist(vals, bins=bins, color=color, alpha=0.4,
                        label=rf"$M={mtot}\,M_\odot$",
                        edgecolor=color, linewidth=0.5, histtype="stepfilled")
        ax.set_xscale("log")
        ax.axvline(0.01, color="k", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlabel("FD mismatch (NR error)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6)
        save(fig, "nr_fd_mismatch")
        print("  nr_fd_mismatch done")

    # ── 7. Combined NR mismatch histogram ────────────────────────────
    combined = np.concatenate([train["nr_fd_combined"], val["nr_fd_combined"]]) \
               if len(train["nr_fd_combined"]) > 0 and len(val["nr_fd_combined"]) > 0 \
               else (train["nr_fd_combined"] if len(train["nr_fd_combined"]) > 0
                     else val["nr_fd_combined"])
    combined = combined[np.isfinite(combined) & (combined > 0)]

    if len(combined) > 0:
        fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
        bins = np.logspace(np.log10(max(combined.min(), 1e-12)),
                           np.log10(combined.max()), 30)
        ax.hist(combined, bins=bins, color=plot_settings.COLORS["blue"],
                alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.axvline(0.01, color="k", ls="--", lw=0.8, alpha=0.7)
        med = np.median(combined)
        p95 = np.percentile(combined, 95)
        ax.axvline(med, color=plot_settings.COLORS["red"], ls="-", lw=0.8,
                   label=f"Median: {med:.1e}")
        ax.axvline(p95, color=plot_settings.COLORS["orange"], ls="-.", lw=0.8,
                   label=f"95th: {p95:.1e}")
        ax.set_xlabel("Combined FD mismatch (NR error)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6)
        save(fig, "nr_fd_mismatch_combined")
        print("  nr_fd_mismatch_combined done")

    # ── 8. NR mismatch vs parameters ────────────────────────────────
    nr_combined_tr = train["nr_fd_combined"]
    nr_combined_va = val["nr_fd_combined"]
    if len(nr_combined_tr) > 5 and len(nr_combined_va) > 5:
        n_nr_tr = len(nr_combined_tr)
        n_nr_va = len(nr_combined_va)
        q_nr_tr = train["q"][:n_nr_tr]
        q_nr_va = val["q"][:n_nr_va]
        ce_nr_tr = train["chi_eff"][:n_nr_tr]
        ce_nr_va = val["chi_eff"][:n_nr_va]
        cp_nr_tr = train["chi_p"][:n_nr_tr]
        cp_nr_va = val["chi_p"][:n_nr_va]

        fig, axes = plt.subplots(1, 3, figsize=plot_settings.figsize(cols=2, aspect=0.35))
        for ax, (x_tr, x_va, xlabel) in zip(axes, [
            (q_nr_tr, q_nr_va, r"$q$"),
            (ce_nr_tr, ce_nr_va, r"$\chi_{\mathrm{eff}}$"),
            (cp_nr_tr, cp_nr_va, r"$\chi_p$"),
        ]):
            valid_tr = np.isfinite(nr_combined_tr) & (nr_combined_tr > 0)
            valid_va = np.isfinite(nr_combined_va) & (nr_combined_va > 0)
            ax.scatter(x_tr[valid_tr], nr_combined_tr[valid_tr], s=6, alpha=0.5,
                       color=c_tr, label="Train", rasterized=True)
            ax.scatter(x_va[valid_va], nr_combined_va[valid_va], s=6, alpha=0.5,
                       marker="s", color=c_va, label="Val", rasterized=True)
            ax.set_xlabel(xlabel)
            ax.set_yscale("log")
            ax.axhline(0.01, color="k", ls="--", lw=0.8, alpha=0.5)
        axes[0].set_ylabel("NR FD mismatch (combined)")
        axes[2].legend(fontsize=6, markerscale=1.5)
        fig.tight_layout()
        save(fig, "nr_fd_vs_params")
        print("  nr_fd_vs_params done")

    print(f"\nPlots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
