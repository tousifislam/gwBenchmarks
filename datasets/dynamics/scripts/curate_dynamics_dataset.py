"""Curate the Dynamics Bench dataset using SEOBNRv5EHM.

Generates eccentric BBH orbital dynamics — e(t), zeta(t), x(t) — for
Latin Hypercube-sampled parameters, split into 250 training + 250 validation.

Parameters:
  q      in [1, 6]
  chi1z  in [-0.6, 0.6]
  chi2z  in [-0.6, 0.6]
  e0     in [0.001, 0.5]
  zeta0  in [0, pi]
  omega0 in [0.0075, 0.0085]

Source: pyseobnr SEOBNRv5EHM model

Usage:
    python curate_dynamics_dataset.py
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.stats.qmc import LatinHypercube

# ── Configuration ──────────────────────────────────────────────────────────

N_TRAIN = 250
N_VAL = 250
N_TOTAL = N_TRAIN + N_VAL

PARAM_BOUNDS = {
    "q":     (1.0, 6.0),
    "chi1z": (-0.6, 0.6),
    "chi2z": (-0.6, 0.6),
    "e0":    (0.001, 0.5),
    "zeta0": (0.0, np.pi),
    "omega0": (0.0075, 0.0085),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
N_DIM = len(PARAM_NAMES)


# ── Latin Hypercube Sampling ─────────────────────────────────────────────


def generate_lhs_samples(n_samples, seed=42):
    """Generate Latin Hypercube samples in the parameter space."""
    sampler = LatinHypercube(d=N_DIM, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    samples = []
    for i in range(n_samples):
        params = {}
        for j, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[name]
            params[name] = lo + unit_samples[i, j] * (hi - lo)
        samples.append(params)

    return samples


# ── Dynamics generation ──────────────────────────────────────────────────


def generate_dynamics(q, chi1z, chi2z, e0, zeta0, omega0):
    """Generate eccentric orbital dynamics using SEOBNRv5EHM.

    Returns dict with t, e, zeta, x arrays, or raises on failure.
    """
    from pyseobnr.generate_waveform import generate_modes_opt

    t, modes, model = generate_modes_opt(
        q=q,
        chi1=chi1z,
        chi2=chi2z,
        omega_start=omega0,
        eccentricity=e0,
        rel_anomaly=zeta0,
        approximant="SEOBNRv5EHM",
        debug=True,
    )

    dyn = model.dynamics
    return {
        "t": dyn[:, 0].astype(np.float64),
        "e": dyn[:, 5].astype(np.float64),
        "zeta": dyn[:, 6].astype(np.float64),
        "x": dyn[:, 7].astype(np.float64),
        "n_pts": len(dyn),
    }


# ── HDF5 output ──────────────────────────────────────────────────────────


def write_split(records, output, split_name):
    """Write one split to HDF5."""
    gz = dict(compression="gzip", compression_opts=4)

    with h5py.File(output, "w") as h5:
        h5.attrs["description"] = (
            f"Dynamics Bench ({split_name}): eccentric orbital dynamics "
            f"e(t), zeta(t), x(t) from SEOBNRv5EHM."
        )
        h5.attrs["source"] = "pyseobnr SEOBNRv5EHM"
        h5.attrs["split"] = split_name
        h5.attrs["n_simulations"] = len(records)
        h5.attrs["sampling"] = "Latin Hypercube"

        for i, rec in enumerate(records):
            grp = h5.create_group(f"sim_{i:04d}")
            grp.attrs["q"] = rec["q"]
            grp.attrs["chi1z"] = rec["chi1z"]
            grp.attrs["chi2z"] = rec["chi2z"]
            grp.attrs["e0"] = rec["e0"]
            grp.attrs["zeta0"] = rec["zeta0"]
            grp.attrs["omega0"] = rec["omega0"]
            grp.attrs["n_pts"] = rec["n_pts"]

            grp.create_dataset("t", data=rec["t"], **gz)
            grp.create_dataset("e", data=rec["e"], **gz)
            grp.create_dataset("zeta", data=rec["zeta"], **gz)
            grp.create_dataset("x", data=rec["x"], **gz)

        # Metadata arrays for quick access
        meta = h5.create_group("metadata")
        meta.create_dataset("q", data=np.array([r["q"] for r in records]))
        meta.create_dataset("chi1z", data=np.array([r["chi1z"] for r in records]))
        meta.create_dataset("chi2z", data=np.array([r["chi2z"] for r in records]))
        meta.create_dataset("e0", data=np.array([r["e0"] for r in records]))
        meta.create_dataset("zeta0", data=np.array([r["zeta0"] for r in records]))
        meta.create_dataset("omega0", data=np.array([r["omega0"] for r in records]))

    print(f"  {split_name}: {len(records)} simulations -> {output}")


# ── Plots ────────────────────────────────────────────────────────────────


def make_plots(train, val, plot_dir):
    """Generate Nature-style reference plots."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from gwbenchmarks import plot_settings
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plot_settings.apply()

    c_tr = plot_settings.COLORS["blue"]
    c_va = plot_settings.COLORS["red"]

    def get_param(records, key):
        return np.array([r[key] for r in records])

    q_tr, q_va = get_param(train, "q"), get_param(val, "q")
    chi1z_tr, chi1z_va = get_param(train, "chi1z"), get_param(val, "chi1z")
    chi2z_tr, chi2z_va = get_param(train, "chi2z"), get_param(val, "chi2z")
    e0_tr, e0_va = get_param(train, "e0"), get_param(val, "e0")
    zeta0_tr, zeta0_va = get_param(train, "zeta0"), get_param(val, "zeta0")

    # Parameter space plots (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=plot_settings.figsize(cols=2, aspect=0.65))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    pairs = [
        (q_tr, e0_tr, q_va, e0_va, r"$q$", r"$e_0$"),
        (q_tr, chi1z_tr, q_va, chi1z_va, r"$q$", r"$\chi_{1z}$"),
        (chi1z_tr, chi2z_tr, chi1z_va, chi2z_va, r"$\chi_{1z}$", r"$\chi_{2z}$"),
        (q_tr, zeta0_tr, q_va, zeta0_va, r"$q$", r"$\zeta_0$"),
        (e0_tr, chi1z_tr, e0_va, chi1z_va, r"$e_0$", r"$\chi_{1z}$"),
        (e0_tr, zeta0_tr, e0_va, zeta0_va, r"$e_0$", r"$\zeta_0$"),
    ]
    for ax, (x1, y1, x2, y2, xl, yl) in zip(axes.flat, pairs):
        ax.scatter(x1, y1, s=5, alpha=0.5, color=c_tr, label="Train", rasterized=True)
        ax.scatter(x2, y2, s=5, alpha=0.5, marker="s", color=c_va,
                   label="Val", rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0, 2].legend(fontsize=6, markerscale=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"param_space.{ext}")
    plt.close(fig)
    print("  param_space done")

    # Eccentricity evolution e(t), colored by e0
    all_records = train + val
    e0_all = np.array([r["e0"] for r in all_records])
    norm = Normalize(vmin=e0_all.min(), vmax=e0_all.max())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.5))
    for rec in all_records:
        color = cmap(norm(rec["e0"]))
        ax.plot(rec["t"], rec["e"], color=color, alpha=0.3, linewidth=0.5)
    ax.set_xlabel(r"$t / M$")
    ax.set_ylabel(r"$e(t)$")
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r"$e_0$")
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"ecc_evolution.{ext}")
    plt.close(fig)
    print("  ecc_evolution done")

    # Anomaly evolution zeta(t), colored by e0
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.5))
    for rec in all_records:
        color = cmap(norm(rec["e0"]))
        ax.plot(rec["t"], rec["zeta"], color=color, alpha=0.3, linewidth=0.5)
    ax.set_xlabel(r"$t / M$")
    ax.set_ylabel(r"$\zeta(t)$")
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r"$e_0$")
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"anomaly_evolution.{ext}")
    plt.close(fig)
    print("  anomaly_evolution done")

    # PN frequency x(t), colored by q
    norm_q = Normalize(vmin=1.0, vmax=6.0)
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.5))
    for rec in all_records:
        color = cmap(norm_q(rec["q"]))
        ax.plot(rec["t"], rec["x"], color=color, alpha=0.3, linewidth=0.5)
    ax.set_xlabel(r"$t / M$")
    ax.set_ylabel(r"$x(t)$")
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm_q)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r"$q$")
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"x_evolution.{ext}")
    plt.close(fig)
    print("  x_evolution done")

    print(f"  Plots saved to {plot_dir}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Curate Dynamics Bench dataset using SEOBNRv5EHM"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent,
    )
    args = parser.parse_args()

    print("Generating Latin Hypercube samples...")
    train_samples = generate_lhs_samples(N_TRAIN, seed=42)
    val_samples = generate_lhs_samples(N_VAL, seed=137)
    print(f"  {N_TRAIN} training + {N_VAL} validation samples")

    # Process all samples
    for split_name, samples in [("training", train_samples), ("validation", val_samples)]:
        print(f"\nGenerating {split_name} dynamics...")
        records = []
        n_failed = 0
        t0 = time.time()

        for i, params in enumerate(samples):
            try:
                t_start = time.time()
                result = generate_dynamics(**params)
                elapsed = time.time() - t_start

                record = {**params, **result}
                records.append(record)

                if (i + 1) % 25 == 0 or i == 0:
                    rate = (i + 1) / (time.time() - t0)
                    eta = (len(samples) - i - 1) / rate / 60 if rate > 0 else 0
                    print(
                        f"  [{i+1}/{len(samples)}] q={params['q']:.2f} "
                        f"e0={params['e0']:.3f} chi1z={params['chi1z']:+.3f} "
                        f"n_pts={result['n_pts']} ({elapsed:.1f}s, ETA {eta:.0f}min)"
                    )
            except Exception as e:
                n_failed += 1
                print(f"  [FAIL {i+1}] q={params['q']:.2f} e0={params['e0']:.3f}: {e}")
                traceback.print_exc()

        elapsed_total = (time.time() - t0) / 60
        print(f"  {split_name}: {len(records)} done, {n_failed} failed, {elapsed_total:.1f}min")

        output_path = args.output_dir / f"dynamics_{split_name}.h5"
        write_split(records, output_path, split_name)

        if split_name == "training":
            train_records = records
        else:
            val_records = records

    # Summary
    all_records = train_records + val_records
    e0_all = np.array([r["e0"] for r in all_records])
    print(f"\ne0 range: [{e0_all.min():.4f}, {e0_all.max():.4f}]")
    npts = np.array([r["n_pts"] for r in all_records])
    print(f"n_pts: [{npts.min()}, {npts.max()}], median={np.median(npts):.0f}")

    # Plots
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    make_plots(train_records, val_records, plot_dir)


if __name__ == "__main__":
    main()
