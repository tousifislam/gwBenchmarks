"""Shared utilities for ringdown benchmark."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "datasets" / "ringdown"
RESULTS_DIR = Path(__file__).resolve().parent


def build_xy(fn: Path):
    X, Y = [], []
    with h5py.File(fn, "r") as f:
        for lk in f.keys():
            l = int(lk[1:])
            for mk in f[lk].keys():
                m = int(mk[1:])
                for nk in f[lk][mk].keys():
                    n = int(nk[1:])
                    g = f[lk][mk][nk]
                    spin = g["spin"][:]
                    or_ = g["omega_real"][:]
                    oi = g["omega_imag"][:]
                    Xi = np.column_stack([spin, np.full_like(spin, l), np.full_like(spin, m), np.full_like(spin, n)])
                    Yi = np.column_stack([or_, oi])
                    X.append(Xi); Y.append(Yi)
    return np.vstack(X), np.vstack(Y)


_cache = {}

def load_data():
    if "data" in _cache:
        return _cache["data"]
    Xt, Yt = build_xy(DATA_DIR / "ringdown_training.h5")
    Xv, Yv = build_xy(DATA_DIR / "ringdown_validation.h5")
    _cache["data"] = (Xt, Yt, Xv, Yv)
    return _cache["data"]


def reparam(X, mode: str):
    """X has columns [spin, l, m, n]."""
    a = X[:, 0]
    l = X[:, 1]
    m = X[:, 2]
    n = X[:, 3]
    if mode == "raw":
        return X.copy()
    if mode == "log_1ma":
        # log(1 - a) for spin extremality
        return np.column_stack([-np.log(1 - a + 1e-12), l, m, n])
    if mode == "compact":
        # x = a / (1 - a), maps [0, 1) to [0, inf)
        return np.column_stack([a / (1 - a + 1e-12), l, m, n])
    if mode == "chebyshev":
        # x = 2*a - 1 (still includes l,m,n)
        return np.column_stack([2 * a - 1, l, m, n])
    if mode == "lm_diff":
        # differential mode: l-|m|
        return np.column_stack([a, l, m, n, l - np.abs(m)])
    if mode == "all_normalized":
        return np.column_stack([a, l/16, m/16, n/8])
    raise ValueError(f"unknown mode {mode}")


def loss_fn(pred, true):
    """Mean of relative errors on Re(omega) and Im(omega)."""
    pred, true = np.asarray(pred), np.asarray(true)
    rel_re = np.abs(pred[:, 0] - true[:, 0]) / (np.abs(true[:, 0]) + 1e-12)
    rel_im = np.abs(pred[:, 1] - true[:, 1]) / (np.abs(true[:, 1]) + 1e-12)
    return float((rel_re.mean() + rel_im.mean()) / 2), {
        "rel_error_omega_real": float(rel_re.mean()),
        "rel_error_omega_imag": float(rel_im.mean()),
    }


def per_sample_err(pred, true):
    """Per-sample relative error (averaged over Re/Im)."""
    rel_re = np.abs(pred[:, 0] - true[:, 0]) / (np.abs(true[:, 0]) + 1e-12)
    rel_im = np.abs(pred[:, 1] - true[:, 1]) / (np.abs(true[:, 1]) + 1e-12)
    return (rel_re + rel_im) / 2


def save_scorecard(model_dir: Path, scorecard: dict):
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)


def write_train_predict(model_dir: Path, approach: str):
    model_dir.mkdir(parents=True, exist_ok=True)
    train = f'''"""Training script for {approach}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import *

def main():
    print("Run the master build_models.py for {approach}.")

if __name__ == "__main__":
    main()
'''
    predict = f'''"""Prediction module for {approach}."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import *

def predict(X):
    """X: (n, 4) array [spin, l, m, n]. Returns (n, 2) [omega_real, omega_imag]."""
    raise NotImplementedError("See saved_model/")
'''
    (model_dir / "train.py").write_text(train)
    (model_dir / "predict.py").write_text(predict)


def model_dir(approach_num: int, name: str) -> Path:
    p = RESULTS_DIR / "models" / f"{approach_num:02d}_{name}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "saved_model").mkdir(parents=True, exist_ok=True)
    return p
