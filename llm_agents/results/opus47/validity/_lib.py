"""Shared utilities for validity benchmark."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "datasets" / "validity"
RESULTS_DIR = Path(__file__).resolve().parent


def load_split(split: str):
    fn = DATA_DIR / f"validity_{split}.h5"
    with h5py.File(fn, "r") as f:
        q = f["q"][:]; c1 = f["chi1z"][:]; c2 = f["chi2z"][:]
        om = f["omega0"][:]; mm = f["mm_td"][:]
    X = np.column_stack([q, c1, c2, om])
    y = np.log10(mm)  # Loss is RMSE in log10 space, so target is log10(mm)
    return X, y


_cache = {}

def load_data():
    if "data" in _cache:
        return _cache["data"]
    Xt, yt = load_split("training")
    Xv, yv = load_split("validation")
    _cache["data"] = (Xt, yt, Xv, yv)
    return _cache["data"]


def reparam(X, mode: str):
    q, c1, c2, om = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eta = q / (1 + q) ** 2
    chi_eff = (q * c1 + c2) / (q + 1)
    chi_a = 0.5 * (c1 - c2)
    if mode == "raw4":
        return X.copy()
    if mode == "eta_chieff":
        return np.column_stack([eta, chi_eff, chi_a, om])
    if mode == "log_q":
        return np.column_stack([np.log(q), chi_eff, chi_a, np.log(om + 1e-12)])
    if mode == "interaction":
        return np.column_stack([eta, chi_eff, chi_a, om, q * chi_eff, eta * chi_a])
    if mode == "boundary":
        # Distance from NRHybSur3dq8 valid region (q<=8, |chi|<=0.8)
        q_dist = np.maximum(q - 8, 0)
        chi_dist = np.maximum(np.maximum(np.abs(c1), np.abs(c2)) - 0.8, 0)
        return np.column_stack([q, c1, c2, om, q_dist, chi_dist])
    raise ValueError(f"unknown mode {mode}")


def loss_fn(pred, true):
    """RMSE in log10 space (pred and true are already log10(mm))."""
    pred, true = np.asarray(pred), np.asarray(true)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse, {"log_rmse": rmse}


def per_sample_err(pred, true):
    return np.abs(pred - true)


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
    print("Reproduce via build_models.py.")

if __name__ == "__main__":
    main()
'''
    predict = f'''"""Prediction module for {approach}."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import *

def predict(X):
    raise NotImplementedError("See saved_model/")
'''
    (model_dir / "train.py").write_text(train)
    (model_dir / "predict.py").write_text(predict)


def model_dir(approach_num: int, name: str) -> Path:
    p = RESULTS_DIR / "models" / f"{approach_num:02d}_{name}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "saved_model").mkdir(parents=True, exist_ok=True)
    return p
