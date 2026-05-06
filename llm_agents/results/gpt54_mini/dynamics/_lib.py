"""Shared utilities for dynamics benchmark."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "datasets" / "dynamics"
RESULTS_DIR = Path(__file__).resolve().parent

N_GRID = 256  # resample target
TAU_GRID = np.linspace(0, 1, N_GRID)


def load_split(split: str):
    """Returns (params, x_resampled, t_end)."""
    fn = DATA_DIR / f"dynamics_{split}.h5"
    with h5py.File(fn, "r") as f:
        keys = sorted([k for k in f.keys() if k.startswith("sim_")])
        n = len(keys)
        params = np.zeros((n, 6))
        t_end = np.zeros(n)
        x_grid = np.zeros((n, N_GRID))
        for i, k in enumerate(keys):
            g = f[k]
            params[i] = [g.attrs["q"], g.attrs["chi1z"], g.attrs["chi2z"],
                         g.attrs["e0"], g.attrs["zeta0"], g.attrs["omega0"]]
            t = g["t"][:]
            x = g["x"][:]
            tau = t / t[-1]
            x_grid[i] = np.interp(TAU_GRID, tau, x)
            t_end[i] = t[-1]
    return params, x_grid, t_end


_cache = {}

def load_data():
    if "data" in _cache:
        return _cache["data"]
    pt, xt, te_t = load_split("training")
    pv, xv, te_v = load_split("validation")
    _cache["data"] = (pt, xt, te_t, pv, xv, te_v)
    return _cache["data"]


def reparam(params, mode: str):
    q = params[:, 0]
    chi1z = params[:, 1]
    chi2z = params[:, 2]
    e0 = params[:, 3]
    zeta0 = params[:, 4]
    omega0 = params[:, 5]
    eta = q / (1 + q) ** 2
    chi_eff = (q * chi1z + chi2z) / (q + 1)
    chi_a = 0.5 * (chi1z - chi2z)
    if mode == "raw6":
        return params.copy()
    if mode == "eta_chieff":
        return np.column_stack([eta, chi_eff, chi_a, np.log(e0 + 1e-6), zeta0, omega0])
    if mode == "trig_zeta":
        return np.column_stack([eta, chi_eff, chi_a, e0, np.cos(zeta0), np.sin(zeta0), omega0])
    if mode == "log_omega":
        return np.column_stack([eta, chi_eff, chi_a, e0, zeta0, np.log(omega0)])
    raise ValueError(f"unknown mode {mode}")


def rms_rel_err(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    mask = true != 0
    if not np.any(mask):
        return float("inf")
    return float(np.sqrt(np.mean(((pred[mask] - true[mask]) / true[mask]) ** 2)))


def per_sample_rms_rel(pred, true):
    """Per-sample RMS relative error."""
    out = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        m = true[i] != 0
        if np.any(m):
            out[i] = np.sqrt(np.mean(((pred[i, m] - true[i, m]) / true[i, m]) ** 2))
        else:
            out[i] = np.inf
    return out


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
    print("Reproduce via build_models.py {approach}")

if __name__ == "__main__":
    main()
'''
    predict = f'''"""Prediction module for {approach}."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import *

def predict(X):
    """X: (n, n_features) array. Returns (n, N_GRID) x(tau)."""
    raise NotImplementedError("See saved_model/ for artifacts.")
'''
    (model_dir / "train.py").write_text(train)
    (model_dir / "predict.py").write_text(predict)


def model_dir(approach_num: int, name: str) -> Path:
    p = RESULTS_DIR / "models" / f"{approach_num:02d}_{name}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "saved_model").mkdir(parents=True, exist_ok=True)
    return p
