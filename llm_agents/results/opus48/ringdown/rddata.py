"""Data + reparameterisations + scoring for Ringdown Bench (opus48, original).

Predict QNM (omega_R, omega_I) of the (l=2,m=2,n=0) Kerr mode vs spin a.
Loss = 0.5*(mean|dwR/wR| + mean|dwI/wI|).
"""
import sys
from pathlib import Path
import numpy as np
import h5py

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA = REPO / "datasets" / "ringdown"
MODE = "l2/m+2/n0"


def load(split, mode=MODE):
    with h5py.File(DATA / f"ringdown_{split}.h5", "r") as f:
        g = f[mode]
        a = g["spin"][:]
        wr = g["omega_real"][:]
        wi = g["omega_imag"][:]
    return a, np.column_stack([wr, wi])


def reparam(a, kind):
    a = np.asarray(a, float)
    if kind == "raw_a":
        x = a
    elif kind == "log_compact":
        x = -np.log(np.clip(1 - a, 1e-12, None))
    elif kind == "sqrt_irr":
        x = np.sqrt(np.clip(1 - a * a, 0, None))
    elif kind == "compact":
        x = a / np.clip(1 - a, 1e-12, None)
    elif kind == "cheby_map":
        x = 2 * a - 1
    else:
        raise ValueError(kind)
    return x.reshape(-1, 1)


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0); std = X.std(0); std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def loss(pred2, true2):
    """0.5*(mean rel err wR + mean rel err wI). pred/true: (n,2)."""
    rr = np.mean(np.abs(pred2[:, 0] - true2[:, 0]) / np.abs(true2[:, 0]))
    ri = np.mean(np.abs(pred2[:, 1] - true2[:, 1]) / np.abs(true2[:, 1]))
    return float(0.5 * (rr + ri)), float(rr), float(ri)


def per_sample_err(pred2, true2):
    rr = np.abs(pred2[:, 0] - true2[:, 0]) / np.abs(true2[:, 0])
    ri = np.abs(pred2[:, 1] - true2[:, 1]) / np.abs(true2[:, 1])
    return 0.5 * (rr + ri)
