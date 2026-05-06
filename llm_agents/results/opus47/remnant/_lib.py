"""Shared utilities for remnant benchmark approaches."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "datasets" / "remnant"
RESULTS_DIR = Path(__file__).resolve().parent


def load_split(split: str):
    fn = DATA_DIR / f"remnant_{split}.h5"
    with h5py.File(fn, "r") as f:
        params = np.column_stack([f["q"][:], f["chi1x"][:], f["chi1y"][:], f["chi1z"][:],
                                  f["chi2x"][:], f["chi2y"][:], f["chi2z"][:]])
        omega0 = f["omega0"][:]
        Mf = f["Mf"][:]
        chif = f["chif_mag"][:]
        vf = f["vf_mag"][:]
    return params, omega0, Mf, chif, vf


_cache = {}

def load_data():
    if "data" in _cache:
        return _cache["data"]
    pt, om_t, Mf_t, chif_t, vf_t = load_split("training")
    pv, om_v, Mf_v, chif_v, vf_v = load_split("validation")
    _cache["data"] = (pt, om_t, vf_t, Mf_t, chif_t, pv, om_v, vf_v, Mf_v, chif_v)
    return _cache["data"]


def reparam(params, mode: str, omega0=None):
    q = params[:, 0]
    chi1 = params[:, 1:4]
    chi2 = params[:, 4:7]
    eta = q / (1 + q) ** 2
    delta = (q - 1) / (q + 1)
    chi1_z, chi2_z = chi1[:, 2], chi2[:, 2]
    chi_eff = (q * chi1_z + chi2_z) / (q + 1)
    chi_a = 0.5 * (chi1_z - chi2_z)
    chi1_perp = np.sqrt(chi1[:, 0] ** 2 + chi1[:, 1] ** 2)
    chi2_perp = np.sqrt(chi2[:, 0] ** 2 + chi2[:, 1] ** 2)
    chi_p = np.maximum(chi1_perp, chi2_perp * (3 + 4 * q) / (4 + 3 * q) / q)
    chi1_mag = np.linalg.norm(chi1, axis=1)
    chi2_mag = np.linalg.norm(chi2, axis=1)
    eps = 1e-12
    cos_t1 = np.where(chi1_mag > eps, chi1[:, 2] / np.maximum(chi1_mag, eps), 0)
    cos_t2 = np.where(chi2_mag > eps, chi2[:, 2] / np.maximum(chi2_mag, eps), 0)
    theta1 = np.arccos(np.clip(cos_t1, -1, 1))
    theta2 = np.arccos(np.clip(cos_t2, -1, 1))
    phi1 = np.arctan2(chi1[:, 1], chi1[:, 0])
    phi2 = np.arctan2(chi2[:, 1], chi2[:, 0])

    if mode == "raw7":
        return params.copy()
    if mode == "eta_chieff":
        return np.column_stack([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])
    if mode == "delta_chia":
        return np.column_stack([delta, chi_eff, chi_a, chi1_mag, chi2_mag])
    if mode == "pn_products":
        return np.column_stack([eta, chi_eff, eta * chi_eff, delta * chi_a, chi_p])
    if mode == "spherical":
        return np.column_stack([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])
    raise ValueError(f"unknown mode {mode}")


def nrmse(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    rng = np.ptp(true)
    if rng == 0:
        return float(np.sqrt(np.mean((pred - true) ** 2)))
    return float(np.sqrt(np.mean((pred - true) ** 2)) / rng)


def per_sample_abs_err(pred, true):
    return np.abs(np.asarray(pred) - np.asarray(true))


def save_scorecard(model_dir: Path, scorecard: dict):
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)


def write_train_predict(model_dir: Path, approach: str, train_body: str = "def main():\n    pass\n",
                        predict_body: str = "def predict(X):\n    raise NotImplementedError\n"):
    model_dir.mkdir(parents=True, exist_ok=True)
    train = f'''"""Training script for {approach}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import *

{train_body}

if __name__ == "__main__":
    main()
'''
    predict = f'''"""Prediction module for {approach}."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import *

{predict_body}
'''
    (model_dir / "train.py").write_text(train)
    (model_dir / "predict.py").write_text(predict)


def model_dir(approach_num: int, name: str) -> Path:
    p = RESULTS_DIR / "models" / f"{approach_num:02d}_{name}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "saved_model").mkdir(parents=True, exist_ok=True)
    return p
