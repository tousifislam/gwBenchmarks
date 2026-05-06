"""Shared utilities for waveform surrogate models."""
import h5py
import numpy as np
import json
import time
import os
from pathlib import Path
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_DIR = ROOT / ".." / ".." / ".." / ".." / "datasets" / "waveform"

MSUN_SEC = 4.925491025543576e-06
FD_MASSES = [40.0, 80.0, 120.0, 160.0, 200.0]


def _load_hdf5(path):
    f = h5py.File(str(path), "r")
    n = f.attrs["n_simulations"]
    dt = f.attrs["dt_geometric"]
    params_list = []
    h_list = []
    t_list = []
    wf_lengths = []
    for i in range(n):
        g = f[f"sim_{i:04d}"]
        q = g.attrs["q"]
        chi1 = np.array([g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]])
        chi2 = np.array([g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]])
        omega0 = g.attrs.get("omega0", 0.0)
        params_list.append(np.concatenate([np.array([q]), chi1, chi2, np.array([omega0])]))
        h_real = g["h22_real"][:]
        h_imag = g["h22_imag"][:]
        h_list.append(np.column_stack([h_real, h_imag]))
        t_list.append(g["t"][:])
        wf_lengths.append(len(h_real))
    f.close()
    params = np.array(params_list)
    return params, h_list, dt, t_list, wf_lengths


def load_data():
    train_path = DATA_DIR / "waveform_training.h5"
    val_path = DATA_DIR / "waveform_validation.h5"
    params_train, h_train, dt, t_train_list, wf_len_train = _load_hdf5(train_path)
    params_val, h_val, dt_val, t_val_list, wf_len_val = _load_hdf5(val_path)
    return params_train, h_train, params_val, h_val, dt, t_train_list, t_val_list


def raw_params(params):
    """Return (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) - 7D."""
    return params[:, :7]


def eta_chieff_params(params):
    """Return (eta, chi_eff, chi_p, |chi1|, |chi2|, theta1, theta2) - 7D."""
    q = params[:, 0]
    eta = q / (1 + q) ** 2
    chi1x, chi1y, chi1z = params[:, 1], params[:, 2], params[:, 3]
    chi2x, chi2y, chi2z = params[:, 4], params[:, 5], params[:, 6]

    chi1_mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2_mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)

    m1 = q / (1 + q)
    m2 = 1 / (1 + q)
    chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)

    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
    chi_p = np.maximum(chi1_perp, chi2_perp)

    theta1 = np.arctan2(chi1_perp, chi1z)
    theta2 = np.arctan2(chi2_perp, chi2z)

    out = np.column_stack([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])
    return out


def spherical_params(params):
    """Return (eta, |chi1|, theta1, phi1, |chi2|, theta2, phi2) - 7D."""
    q = params[:, 0]
    eta = q / (1 + q) ** 2
    chi1x, chi1y, chi1z = params[:, 1], params[:, 2], params[:, 3]
    chi2x, chi2y, chi2z = params[:, 4], params[:, 5], params[:, 6]

    chi1_mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2_mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)

    theta1 = np.arctan2(np.sqrt(chi1x**2 + chi1y**2), chi1z)
    phi1 = np.arctan2(chi1y, chi1x)
    theta2 = np.arctan2(np.sqrt(chi2x**2 + chi2y**2), chi2z)
    phi2 = np.arctan2(chi2y, chi2x)

    out = np.column_stack([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])
    return out


def mass_diff_params(params):
    """Return (delta_m, chi_eff, chi_p, |chi1|, |chi2|, phi1, phi2) - 7D."""
    q = params[:, 0]
    delta_m = (q - 1) / (q + 1)
    chi1x, chi1y, chi1z = params[:, 1], params[:, 2], params[:, 3]
    chi2x, chi2y, chi2z = params[:, 4], params[:, 5], params[:, 6]
    eta = q / (1 + q) ** 2

    chi1_mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2_mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)

    m1 = q / (1 + q)
    m2 = 1 / (1 + q)
    chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)

    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
    chi_p = np.maximum(chi1_perp, chi2_perp)

    phi1 = np.arctan2(chi1y, chi1x)
    phi2 = np.arctan2(chi2y, chi2x)

    out = np.column_stack([delta_m, chi_eff, chi_p, chi1_mag, chi2_mag, phi1, phi2])
    return out


def eval_mismatch(h_pred, h_ref, dt_geometric):
    """Evaluate mean FD mismatch across 5 masses."""
    from pycbc.filter import match
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import TimeSeries

    mismatches = []
    for mtot in FD_MASSES:
        dt_sec = dt_geometric * mtot * MSUN_SEC
        hp_pred = TimeSeries(np.real(np.asarray(h_pred, dtype=np.float64)), delta_t=dt_sec)
        hp_ref = TimeSeries(np.real(np.asarray(h_ref, dtype=np.float64)), delta_t=dt_sec)
        tlen = max(len(hp_pred), len(hp_ref))
        hp_pred.resize(tlen)
        hp_ref.resize(tlen)
        delta_f = 1.0 / hp_ref.duration
        flen = tlen // 2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, 15.0)
        m, _ = match(hp_ref, hp_pred, psd=psd, low_frequency_cutoff=15.0, high_frequency_cutoff=990.0)
        mismatches.append(float(1.0 - m))
    return np.mean(mismatches), mismatches


def compute_svd_basis(h_train, n_modes=50):
    """Compute SVD basis from training waveforms."""
    X = np.array([h.flatten() for h in h_train])
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    U, S, Vt = svd(X_centered, full_matrices=False)
    return X_mean, U[:, :n_modes], S[:n_modes], Vt[:n_modes, :], S


def reconstruct_waveform(coeffs, Vt, X_mean, orig_shape=None):
    """Reconstruct waveform from SVD coefficients."""
    recon_flat = X_mean + coeffs @ Vt
    if orig_shape is not None:
        return recon_flat.reshape(orig_shape)
    return recon_flat


def subsample_waveform(h, t, n_pts=5000, max_len=None):
    """Subsample waveform uniformly to n_pts."""
    n = len(h)
    if max_len is not None:
        n = min(n, max_len)
    idx = np.linspace(0, n - 1, n_pts, dtype=int)
    return h[idx]


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_param_func(name):
    mapping = {
        "raw": raw_params,
        "eta_chieff": eta_chieff_params,
        "spherical": spherical_params,
        "mass_diff": mass_diff_params,
    }
    return mapping[name]
