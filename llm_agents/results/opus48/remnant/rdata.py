"""Data + reparameterisations + scoring for the Remnant Bench (opus48, original).

Target: remnant kick-velocity magnitude vf_mag. Loss: NRMSE(v_k).
Auxiliary targets Mf, chif_mag are also exposed (symbolic models fit each).
"""
import sys
from pathlib import Path
import numpy as np
import h5py

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = REPO / "datasets" / "remnant"
RAW_KEYS = ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z"]
TARGETS = ["vf_mag", "Mf", "chif_mag"]


def load(split):
    with h5py.File(DATA / f"remnant_{split}.h5", "r") as f:
        P = np.column_stack([f[k][:] for k in RAW_KEYS])
        y = {t: f[t][:] for t in TARGETS}
        nr = {"vf_mag": f["delta_vf"][:], "Mf": f["delta_Mf"][:],
              "chif_mag": f["delta_chif"][:]}
    return P, y, nr


def reparam(P, kind):
    """Map raw params (n,7) -> feature matrix for a named reparameterisation."""
    q = P[:, 0]
    c1 = P[:, 1:4]; c2 = P[:, 4:7]
    eta = q / (1.0 + q) ** 2
    delta = (q - 1.0) / (q + 1.0)
    a1 = np.linalg.norm(c1, axis=1); a2 = np.linalg.norm(c2, axis=1)
    th1 = np.arctan2(np.linalg.norm(c1[:, :2], axis=1), c1[:, 2])
    th2 = np.arctan2(np.linalg.norm(c2[:, :2], axis=1), c2[:, 2])
    ph1 = np.arctan2(c1[:, 1], c1[:, 0]); ph2 = np.arctan2(c2[:, 1], c2[:, 0])
    m1 = q / (1 + q); m2 = 1.0 / (1 + q)
    chi_eff = (m1 * c1[:, 2] + m2 * c2[:, 2])           # along L (z)
    chi_a = 0.5 * (c1[:, 2] - c2[:, 2])
    chi_p = np.maximum(np.linalg.norm(c1[:, :2], axis=1),
                       np.linalg.norm(c2[:, :2], axis=1))
    # in-plane spin difference drives superkicks
    dperp = np.linalg.norm(c1[:, :2] - c2[:, :2], axis=1)

    if kind == "raw_7d":
        return P.copy()
    if kind == "eff_spin":
        return np.column_stack([eta, chi_eff, chi_p, a1, a2, th1, th2])
    if kind == "massdiff_antisym":
        return np.column_stack([delta, chi_eff, chi_a, a1, a2,
                                c1[:, 0], c1[:, 1], c2[:, 0], c2[:, 1]])
    if kind == "pn_products":
        return np.column_stack([eta, chi_eff, eta * chi_eff, delta * chi_a,
                                chi_p, dperp, eta * dperp])
    if kind == "spherical":
        return np.column_stack([eta, a1, th1, ph1, a2, th2, ph2])
    raise ValueError(kind)


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0); std = X.std(0); std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def nrmse_score(pred, true):
    from gwbenchmarks.metrics import nrmse
    return float(nrmse(pred, true))


# per-sample (absolute) errors normalised by the range, for histograms
def per_sample_err(pred, true):
    rng = np.ptp(true)
    return np.abs(np.asarray(pred) - np.asarray(true)) / (rng if rng else 1.0)
