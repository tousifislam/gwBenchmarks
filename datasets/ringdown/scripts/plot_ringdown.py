"""Generate Nature-style reference plots for the Ringdown Bench dataset.

Produces:
  - qnm_freq_vs_spin.{png,pdf}     — omega_R(a/M) for l=2..5, m=l, n=0
  - qnm_damping_vs_spin.{png,pdf}   — |omega_I|(a/M) for l=2..5, m=l, n=0
  - qnm_overtones.{png,pdf}         — omega_R and omega_I vs spin for (2,2) n=0..7
  - qnm_complex_plane.{png,pdf}     — omega_R vs omega_I for select modes at all spins

Usage:
    python plot_ringdown.py
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

BENCHMARK_FILE = DATASET_DIR / "ringdown_benchmark.h5"


def load_mode(f, l, m, n):
    m_str = f"m+{abs(m)}" if m >= 0 else f"m-{abs(m)}"
    path = f"l{l}/{m_str}/n{n}"
    g = f[path]
    return g["spin"][:], g["omega_real"][:], g["omega_imag"][:]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"{name}.{ext}")
    plt.close(fig)


def plot_freq_vs_spin(f):
    """omega_R vs spin for fundamental (n=0) modes with m=l, l=2..6."""
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
    modes = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    for i, (l, m) in enumerate(modes):
        spin, wr, _ = load_mode(f, l, m, 0)
        ax.plot(spin, wr, color=plot_settings.COLOR_CYCLE[i],
                label=rf"$\ell={l},\, m={m}$")
    ax.set_xlabel(r"$a/M$")
    ax.set_ylabel(r"$M\omega_R$")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    save(fig, "qnm_freq_vs_spin")


def plot_damping_vs_spin(f):
    """Damping rate |omega_I| vs spin for fundamental modes."""
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.75))
    modes = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    for i, (l, m) in enumerate(modes):
        spin, _, wi = load_mode(f, l, m, 0)
        ax.semilogy(spin, np.abs(wi), color=plot_settings.COLOR_CYCLE[i],
                    label=rf"$\ell={l},\, m={m}$")
    ax.set_xlabel(r"$a/M$")
    ax.set_ylabel(r"$M|\omega_I|$")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    save(fig, "qnm_damping_vs_spin")


def plot_overtones(f):
    """omega_R and omega_I vs spin for the (2,2) mode, overtones n=0..7."""
    fig, axes = plt.subplots(
        2, 1, figsize=plot_settings.figsize(cols=1, aspect=1.2),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    for n in range(8):
        spin, wr, wi = load_mode(f, 2, 2, n)
        color = plot_settings.COLOR_CYCLE[n % len(plot_settings.COLOR_CYCLE)]
        axes[0].plot(spin, wr, color=color, label=rf"$n={n}$")
        axes[1].plot(spin, np.abs(wi), color=color, label=rf"$n={n}$")

    axes[0].set_ylabel(r"$M\omega_R$")
    axes[0].legend(loc="upper left", ncol=2, fontsize=7)
    axes[1].set_ylabel(r"$M|\omega_I|$")
    axes[1].set_xlabel(r"$a/M$")
    axes[1].set_xlim(0, 1)
    save(fig, "qnm_overtones")


def plot_complex_plane(f):
    """Trajectories in the complex omega plane as spin varies from 0 to ~1."""
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    modes = [
        (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3),
        (3, 3, 0), (4, 4, 0), (5, 5, 0),
    ]
    labels = {
        (2, 2, 0): r"$(2,2,0)$",
        (2, 2, 1): r"$(2,2,1)$",
        (2, 2, 2): r"$(2,2,2)$",
        (2, 2, 3): r"$(2,2,3)$",
        (3, 3, 0): r"$(3,3,0)$",
        (4, 4, 0): r"$(4,4,0)$",
        (5, 5, 0): r"$(5,5,0)$",
    }
    for i, (l, m, n) in enumerate(modes):
        spin, wr, wi = load_mode(f, l, m, n)
        color = plot_settings.COLOR_CYCLE[i % len(plot_settings.COLOR_CYCLE)]
        ax.plot(wr, -wi, color=color, label=labels[(l, m, n)], alpha=0.8)
        ax.plot(wr[0], -wi[0], "o", color=color, markersize=3)

    ax.set_xlabel(r"$M\omega_R$")
    ax.set_ylabel(r"$-M\omega_I$")
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    ax.set_xlim(0, 2.1)
    save(fig, "qnm_complex_plane")


def main():
    print(f"Reading {BENCHMARK_FILE}")
    with h5py.File(BENCHMARK_FILE, "r") as f:
        plot_freq_vs_spin(f)
        print("  qnm_freq_vs_spin done")
        plot_damping_vs_spin(f)
        print("  qnm_damping_vs_spin done")
        plot_overtones(f)
        print("  qnm_overtones done")
        plot_complex_plane(f)
        print("  qnm_complex_plane done")
    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
