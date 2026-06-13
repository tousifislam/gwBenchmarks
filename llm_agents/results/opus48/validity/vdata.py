"""Data + reparameterisations + scoring for Validity Bench (opus48, original).

Predict NR-vs-NRHybSur3dq8 time-domain mismatch from (q,chi1z,chi2z,omega0).
The official scorer uses natural-log RMSE, so we model y = ln(mm) and return
mm_hat = exp(y_hat). NRHybSur3dq8 is valid for q<=8, |chi|<=0.8; beyond that
the mismatch saturates toward 1 (extrapolation region).
"""
import sys
from pathlib import Path
import numpy as np
import h5py

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA = REPO / "datasets" / "validity"
RAW = ["q", "chi1z", "chi2z", "omega0"]
QMAX, CHIMAX = 8.0, 0.8


def load(split):
    with h5py.File(DATA / f"validity_{split}.h5", "r") as f:
        P = np.column_stack([f[k][:] for k in RAW])
        mm = f["mm_td"][:]
    return P, np.log(np.clip(mm, 1e-30, None))     # target = ln(mm)


def reparam(P, kind):
    q = P[:, 0]; c1 = P[:, 1]; c2 = P[:, 2]; om = P[:, 3]
    eta = q / (1 + q) ** 2
    m1 = q / (1 + q); m2 = 1.0 / (1 + q)
    chi_eff = m1 * c1 + m2 * c2
    chi_a = 0.5 * (c1 - c2)
    dq = np.maximum(0.0, q - QMAX)             # how far past q=8
    dchi = np.maximum.reduce([np.abs(c1) - CHIMAX, np.abs(c2) - CHIMAX,
                              np.zeros_like(c1)])
    if kind == "raw_4d":
        return np.column_stack([q, c1, c2, om])
    if kind == "eff_spin":
        return np.column_stack([eta, chi_eff, chi_a, om])
    if kind == "log_q":
        return np.column_stack([np.log(q), chi_eff, chi_a, np.log(om)])
    if kind == "interactions":
        return np.column_stack([eta, chi_eff, chi_a, om, q * chi_eff, eta * chi_a])
    if kind == "boundary":
        return np.column_stack([eta, chi_eff, chi_a, om, dq, dchi,
                                np.maximum(np.abs(c1), np.abs(c2))])
    raise ValueError(kind)


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0); std = X.std(0); std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def log_rmse(pred_ln, true_ln):
    return float(np.sqrt(np.mean((np.asarray(pred_ln) - np.asarray(true_ln)) ** 2)))


def per_sample_err(pred_ln, true_ln):
    return np.abs(np.asarray(pred_ln) - np.asarray(true_ln))
