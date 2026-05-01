"""Generate Nature-style reference plots for the Analytic Bench dataset.

Produces:
  - h22_amplitudes.{png,pdf}  — |h22|(t) for all waveforms, colored by q
  - h22_phases.{png,pdf}      — unwrapped phase of h22 for all waveforms
  - h22_merger_zoom.{png,pdf}  — amplitude & phase zoomed around merger

Usage:
    python plot_analytic.py
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gwbenchmarks import plot_settings

plot_settings.apply()

DATASET_DIR = Path(__file__).parent.parent
PLOT_DIR = DATASET_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def load_waveforms(*h5_paths):
    """Load all waveforms from one or more HDF5 files."""
    waveforms = []
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            for sid in sorted(f["sims"].keys()):
                g = f[f"sims/{sid}"]
                waveforms.append({
                    "sxs_id": sid,
                    "q": float(g.attrs["q"]),
                    "t": g["t"][:],
                    "h22": g["h22_real"][:] + 1j * g["h22_imag"][:],
                })
    waveforms.sort(key=lambda w: w["q"])
    return waveforms


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"{name}.{ext}")
    plt.close(fig)


def plot_amplitudes(waveforms):
    qs = np.array([w["q"] for w in waveforms])
    norm = Normalize(vmin=qs.min(), vmax=qs.max())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.5))
    for w in waveforms:
        color = cmap(norm(w["q"]))
        ax.plot(w["t"], np.abs(w["h22"]), color=color, alpha=0.8)

    ax.set_xlabel(r"$t / M$")
    ax.set_ylabel(r"$|h_{22}|$")
    ax.set_xlim(-6200, 120)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r"$q$")

    save(fig, "h22_amplitudes")


def plot_phases(waveforms):
    qs = np.array([w["q"] for w in waveforms])
    norm = Normalize(vmin=qs.min(), vmax=qs.max())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.5))
    for w in waveforms:
        color = cmap(norm(w["q"]))
        phase = np.unwrap(np.angle(w["h22"]))
        ax.plot(w["t"], phase, color=color, alpha=0.8)

    ax.set_xlabel(r"$t / M$")
    ax.set_ylabel(r"$\phi_{22}$ [rad]")
    ax.set_xlim(-6200, 120)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r"$q$")

    save(fig, "h22_phases")


def plot_merger_zoom(waveforms):
    qs = np.array([w["q"] for w in waveforms])
    norm = Normalize(vmin=qs.min(), vmax=qs.max())
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(
        2, 1, figsize=plot_settings.figsize(cols=2, aspect=0.7),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    for w in waveforms:
        color = cmap(norm(w["q"]))
        mask = (w["t"] >= -200) & (w["t"] <= 100)
        t = w["t"][mask]
        h = w["h22"][mask]
        axes[0].plot(t, np.abs(h), color=color, alpha=0.8)
        phase = np.unwrap(np.angle(h))
        axes[1].plot(t, phase, color=color, alpha=0.8)

    axes[0].set_ylabel(r"$|h_{22}|$")
    axes[1].set_ylabel(r"$\phi_{22}$ [rad]")
    axes[1].set_xlabel(r"$t / M$")
    axes[1].set_xlim(-200, 100)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r"$q$")

    save(fig, "h22_merger_zoom")


def main():
    train_path = DATASET_DIR / "analytic_training.h5"
    val_path = DATASET_DIR / "analytic_validation.h5"
    waveforms = load_waveforms(train_path, val_path)
    print(f"Loaded {len(waveforms)} waveforms")

    plot_amplitudes(waveforms)
    print("  h22_amplitudes done")
    plot_phases(waveforms)
    print("  h22_phases done")
    plot_merger_zoom(waveforms)
    print("  h22_merger_zoom done")
    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
