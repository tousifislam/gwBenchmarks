"""Shared utilities for waveform benchmark approaches."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "datasets" / "waveform"
RESULTS_DIR = Path(__file__).resolve().parent
DT = 0.1
T_GRID_MIN = -2500.0
T_GRID_MAX = 75.0
T_GRID_DT = 1.0
T_GRID = np.arange(T_GRID_MIN, T_GRID_MAX + T_GRID_DT, T_GRID_DT)
N_T = len(T_GRID)


def _resample(t, h, t_new):
    h_re = np.interp(t_new, t, h.real, left=0.0, right=0.0)
    h_im = np.interp(t_new, t, h.imag, left=0.0, right=0.0)
    return h_re + 1j * h_im


def load_split(split: str):
    """Load a dataset split. Returns (params, h_grid, sxs_ids, omega0)."""
    fn = DATA_DIR / f"waveform_{split}.h5"
    with h5py.File(fn, "r") as f:
        keys = sorted([k for k in f.keys() if k.startswith("sim_")])
        n = len(keys)
        params = np.zeros((n, 7))
        omega0 = np.zeros(n)
        ids = []
        h_grid = np.zeros((n, N_T), dtype=complex)
        for i, k in enumerate(keys):
            g = f[k]
            params[i] = [g.attrs["q"], g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"],
                         g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]]
            omega0[i] = g.attrs["omega0"]
            ids.append(g.attrs["sxs_id"])
            t = g["t"][:]
            h = g["h22_real"][:] + 1j * g["h22_imag"][:]
            h_grid[i] = _resample(t, h, T_GRID)
    return params, h_grid, ids, omega0


_cache = {}

def load_data():
    if "ptr" in _cache:
        return _cache["ptr"]
    pt, ht, idt, om_t = load_split("training")
    pv, hv, idv, om_v = load_split("validation")
    _cache["ptr"] = (pt, ht, om_t, pv, hv, om_v)
    return _cache["ptr"]


def reparam(params, mode: str, omega0=None):
    """Various reparameterizations.

    mode: 'raw7', 'eta_chieff', 'eta_chieff_chip', 'spherical', 'raw8' (with omega0), 'log_q'
    """
    q = params[:, 0]
    chi1 = params[:, 1:4]
    chi2 = params[:, 4:7]
    eta = q / (1 + q) ** 2
    delta = (q - 1) / (q + 1)
    chi1_z, chi2_z = chi1[:, 2], chi2[:, 2]
    chi_eff = (q * chi1_z + chi2_z) / (q + 1)
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
    if mode == "raw8":
        return np.column_stack([params, omega0])
    if mode == "log_q":
        return np.column_stack([np.log(q), chi1, chi2])
    if mode == "eta_chieff":
        return np.column_stack([eta, chi_eff, chi1_perp, chi2_perp, chi1_z, chi2_z, chi_p])
    if mode == "eta_chieff_chip":
        return np.column_stack([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])
    if mode == "spherical":
        return np.column_stack([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])
    if mode == "delta_chip":
        return np.column_stack([delta, chi_eff, chi_p, chi1_mag, chi2_mag, phi1, phi2])
    raise ValueError(f"Unknown mode {mode}")


def amp_phase(h):
    """Amplitude/phase decomposition with continuous unwrapped phase."""
    amp = np.abs(h)
    phase = np.unwrap(np.angle(h), axis=-1)
    return amp, phase


def amp_phase_to_h(amp, phase):
    return amp * np.exp(1j * phase)


def fd_mismatch_proxy(h_pred, h_true):
    """L2 mismatch (fast proxy, not the real FD mismatch).

    This is used during training/diagnostic for speed.
    fd_mismatch_real should be used for final scorecards.
    """
    num = np.sum(np.conjugate(h_pred) * h_true, axis=-1)
    den = np.sqrt(np.sum(np.abs(h_pred) ** 2, axis=-1) * np.sum(np.abs(h_true) ** 2, axis=-1))
    return 1.0 - np.abs(num) / (den + 1e-30)


def fd_mismatch_real(h_pred, h_true, n_subset=20, masses=(40, 80, 120, 160, 200)):
    """Real PyCBC FD mismatch over a subset of samples for cost.

    h_pred, h_true: (n_samples, N_T) complex arrays on T_GRID with dt=T_GRID_DT
    Returns mean over subset & masses, plus per-mass dict.
    """
    from gwbenchmarks.metrics import frequency_domain_mismatch
    n = h_pred.shape[0]
    idx = np.linspace(0, n - 1, min(n_subset, n)).astype(int)
    per_mass = {}
    all_per_sample = []  # list of (n_subset,) per mass
    for m in masses:
        vals = np.zeros(len(idx))
        for j, i in enumerate(idx):
            try:
                vals[j] = frequency_domain_mismatch(
                    h_pred[i], h_true[i], dt_geometric=T_GRID_DT, mtot_msun=float(m),
                )
            except Exception:
                vals[j] = 1.0
        per_mass[f"mismatch_{m}Msun"] = float(np.mean(vals))
        all_per_sample.append(vals)
    all_per_sample = np.stack(all_per_sample, axis=0)  # (n_masses, n_subset)
    per_sample_mean = all_per_sample.mean(axis=0)
    loss = float(np.mean(list(per_mass.values())))
    return loss, per_mass, per_sample_mean


def compute_per_sample_proxy(h_pred, h_true):
    return fd_mismatch_proxy(h_pred, h_true)


def save_scorecard(model_dir: Path, scorecard: dict):
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)


def write_train_predict(model_dir: Path, approach: str, train_body: str, predict_body: str):
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
import sys
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


def display_name(approach_num: int, name: str) -> str:
    return name.replace("_", " ")
