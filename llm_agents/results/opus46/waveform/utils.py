"""Shared utilities for waveform benchmark surrogate models."""

import sys, os
import numpy as np
import h5py
import json
import time
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from gwbenchmarks.metrics import mean_fd_mismatch, frequency_domain_mismatch, FD_MASSES_MSUN

TRAIN_PATH = os.path.join(ROOT, "datasets/waveform/waveform_training.h5")
VAL_PATH = os.path.join(ROOT, "datasets/waveform/waveform_validation.h5")
DT = 0.1
NR_ERROR_FLOOR = 1.4e-3

COMMON_T_START = -2847.0
COMMON_T_END = 100.0
COMMON_N = int((COMMON_T_END - COMMON_T_START) / DT) + 1


def load_dataset(path, common_grid=True):
    """Load waveform dataset, return params, waveforms, dt, metadata."""
    params_list = []
    waveforms = []
    metadata = []
    with h5py.File(path, "r") as f:
        n = f.attrs["n_simulations"]
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            q = g.attrs["q"]
            chi1x, chi1y, chi1z = g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]
            chi2x, chi2y, chi2z = g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]
            omega0 = g.attrs["omega0"]
            params_list.append([q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z])

            t = g["t"][:]
            h_real = g["h22_real"][:]
            h_imag = g["h22_imag"][:]
            h = h_real + 1j * h_imag

            if common_grid:
                t_common = np.arange(COMMON_T_START, COMMON_T_END + DT/2, DT)
                h_common = np.zeros(len(t_common), dtype=complex)
                t_start_idx = int((max(t[0], COMMON_T_START) - COMMON_T_START) / DT)
                t_end_idx = int((min(t[-1], COMMON_T_END) - COMMON_T_START) / DT) + 1
                t_end_idx = min(t_end_idx, len(t_common))
                src_start = int((max(COMMON_T_START, t[0]) - t[0]) / DT)
                n_copy = t_end_idx - t_start_idx
                src_end = src_start + n_copy
                if src_end <= len(h) and n_copy > 0:
                    h_common[t_start_idx:t_end_idx] = h[src_start:src_end]
                waveforms.append(h_common)
            else:
                waveforms.append(h)

            meta = {"omega0": omega0, "sxs_id": g.attrs.get("sxs_id", "")}
            if "nr_fd_mm_combined" in g.attrs:
                meta["nr_error"] = g.attrs["nr_fd_mm_combined"]
            metadata.append(meta)

    params = np.array(params_list)
    if common_grid:
        waveforms = np.array(waveforms)
    return params, waveforms, metadata


def reparameterize(params, scheme="raw"):
    """Convert raw (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) to different parameterizations."""
    q = params[:, 0]
    chi1x, chi1y, chi1z = params[:, 1], params[:, 2], params[:, 3]
    chi2x, chi2y, chi2z = params[:, 4], params[:, 5], params[:, 6]

    if scheme == "raw":
        return params

    eta = q / (1 + q)**2
    chi1_mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2_mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
    chi_eff = (q * chi1z + chi2z) / (1 + q)
    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
    chi_p = np.maximum(chi1_perp, (3 + 4*q) / (4 + 3*q) * q * chi2_perp)

    if scheme == "eta_chieff":
        theta1 = np.arccos(np.clip(chi1z / np.maximum(chi1_mag, 1e-15), -1, 1))
        theta2 = np.arccos(np.clip(chi2z / np.maximum(chi2_mag, 1e-15), -1, 1))
        return np.column_stack([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])

    if scheme == "spherical":
        theta1 = np.arccos(np.clip(chi1z / np.maximum(chi1_mag, 1e-15), -1, 1))
        phi1 = np.arctan2(chi1y, chi1x)
        theta2 = np.arccos(np.clip(chi2z / np.maximum(chi2_mag, 1e-15), -1, 1))
        phi2 = np.arctan2(chi2y, chi2x)
        return np.column_stack([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])

    if scheme == "mass_diff":
        delta_m = (q - 1) / (q + 1)
        phi1 = np.arctan2(chi1y, chi1x)
        phi2 = np.arctan2(chi2y, chi2x)
        return np.column_stack([delta_m, chi_eff, chi_p, chi1_mag, chi2_mag, phi1, phi2])

    if scheme == "raw_with_omega0":
        raise ValueError("Need metadata for omega0; use reparameterize_with_omega0")

    return params


def compute_svd(waveforms_real, n_basis=50):
    """Compute SVD decomposition: waveforms_real ≈ coeffs @ basis."""
    mean_wf = np.mean(waveforms_real, axis=0)
    centered = waveforms_real - mean_wf
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:n_basis]
    coeffs = U[:, :n_basis] * s[:n_basis]
    return coeffs, basis, mean_wf, s


def project_onto_basis(waveforms_real, basis, mean_wf):
    """Project waveforms onto SVD basis to get coefficients."""
    centered = waveforms_real - mean_wf
    coeffs = centered @ basis.T
    return coeffs


def reconstruct_from_basis(coeffs, basis, mean_wf):
    """Reconstruct waveforms from SVD coefficients."""
    return coeffs @ basis + mean_wf


def compute_loss_single(h_pred, h_ref, dt=DT):
    """Compute mean FD mismatch for a single waveform pair."""
    return mean_fd_mismatch(h_pred, h_ref, dt)


def compute_loss_batch(h_preds, h_refs, dt=DT):
    """Compute per-sample and mean FD mismatch for a batch."""
    losses = []
    for i in range(len(h_preds)):
        try:
            loss = mean_fd_mismatch(h_preds[i], h_refs[i], dt)
            losses.append(loss)
        except Exception:
            losses.append(1.0)
    return np.array(losses), float(np.mean(losses))


def compute_loss_per_mass(h_pred, h_ref, dt=DT):
    """Compute FD mismatch at each total mass."""
    result = {}
    for m in FD_MASSES_MSUN:
        mm = frequency_domain_mismatch(h_pred, h_ref, dt, m)
        result[f"mismatch_{int(m)}Msun"] = mm
    return result


def save_scorecard(model_dir, approach_name, approach_number, parameterization,
                   time_convention, loss, loss_components, runtime_ms,
                   n_train, n_val, n_params, notes=""):
    """Save scorecard.json for a model."""
    sc = {
        "approach": approach_name,
        "approach_number": approach_number,
        "benchmark": "waveform",
        "agent": "opus46",
        "parameterization": parameterization,
        "time_convention": time_convention,
        "loss": loss,
        "loss_components": loss_components,
        "runtime_ms": runtime_ms,
        "n_train": n_train,
        "n_val": n_val,
        "n_params": n_params,
        "notes": notes
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(sc, f, indent=2)
    return sc
