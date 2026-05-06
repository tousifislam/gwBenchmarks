"""Shared utilities for analytic benchmark (non-spinning q in [1,20]).

Closed-form models only — no SVD bases, no ML regressors that produce the
final waveform shape, no stored basis vectors.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "datasets" / "analytic"
RESULTS_DIR = Path(__file__).resolve().parent

T_GRID = np.arange(-2500.0, 75.0 + 1.0, 1.0)
N_T = len(T_GRID)
T_GRID_DT = 1.0


def _resample(t, h, t_new):
    re = np.interp(t_new, t, h.real, left=0.0, right=0.0)
    im = np.interp(t_new, t, h.imag, left=0.0, right=0.0)
    return re + 1j * im


def load_split(split: str):
    fn = DATA_DIR / f"analytic_{split}.h5"
    with h5py.File(fn, "r") as f:
        keys = sorted(list(f["sims"].keys()))
        n = len(keys)
        q = np.zeros(n)
        h_grid = np.zeros((n, N_T), dtype=complex)
        for i, k in enumerate(keys):
            s = f["sims"][k]
            q[i] = s.attrs["q"]
            t = s["t"][:]
            h = s["h22_real"][:] + 1j * s["h22_imag"][:]
            h_grid[i] = _resample(t, h, T_GRID)
    return q, h_grid


_cache = {}

def load_data():
    if "data" in _cache:
        return _cache["data"]
    qt, ht = load_split("training")
    qv, hv = load_split("validation")
    _cache["data"] = (qt, ht, qv, hv)
    return _cache["data"]


def amp_phase(h):
    """Return amplitude (>=0) and unwrapped phase phi where h = A * exp(-i*phi)."""
    A = np.abs(h)
    # h = A * exp(-i*phi) -> phi = -angle(h)
    phi = -np.unwrap(np.angle(h), axis=-1)
    return A, phi


def reparam_q(q, mode: str):
    q = np.asarray(q, dtype=float)
    if mode == "q":
        return q
    if mode == "eta":
        return q / (1 + q) ** 2
    if mode == "delta":
        return (q - 1) / (q + 1)
    if mode == "log_q":
        return np.log(q)
    if mode == "inv_q":
        return 1.0 / q
    raise ValueError(mode)


def fd_mismatch_proxy(h_pred, h_true):
    num = np.sum(np.conjugate(h_pred) * h_true, axis=-1)
    den = np.sqrt(np.sum(np.abs(h_pred) ** 2, axis=-1) * np.sum(np.abs(h_true) ** 2, axis=-1))
    return 1.0 - np.abs(num) / (den + 1e-30)


def fd_mismatch_real(h_pred, h_true, n_subset=None, masses=(40, 80, 120, 160, 200)):
    from gwbenchmarks.metrics import frequency_domain_mismatch
    n = h_pred.shape[0]
    if n_subset is None:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, min(n_subset, n)).astype(int)
    per_mass = {}
    all_per_sample = []
    for m in masses:
        vals = np.zeros(len(idx))
        for j, i in enumerate(idx):
            try:
                vals[j] = frequency_domain_mismatch(
                    h_pred[i], h_true[i], dt_geometric=T_GRID_DT, mtot_msun=float(m))
            except Exception:
                vals[j] = 1.0
        per_mass[f"mismatch_{m}Msun"] = float(np.mean(vals))
        all_per_sample.append(vals)
    all_per_sample = np.stack(all_per_sample, axis=0)
    per_sample_mean = all_per_sample.mean(axis=0)
    loss = float(np.mean(list(per_mass.values())))
    return loss, per_mass, per_sample_mean


def save_scorecard(model_dir: Path, scorecard: dict):
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)


def write_train_predict(model_dir: Path, approach: str, formula_summary: str = ""):
    model_dir.mkdir(parents=True, exist_ok=True)
    train = (
        f'"""Training script for {approach}.\n\n'
        f'{formula_summary}\n'
        f'"""\n'
        f'import sys\n'
        f'from pathlib import Path\n'
        f'sys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n'
        f'from _lib import *\n\n'
        f'def main():\n'
        f'    print("Run ../../build_models.py from the analytic/ directory.")\n\n'
        f'if __name__ == "__main__":\n'
        f'    main()\n'
    )
    predict = (
        f'"""Prediction module for {approach}.\n\n'
        f'Coefficients live in saved_model/coeffs.npz; the formula in expression.txt.\n'
        f'"""\n'
        f'import sys, numpy as np\n'
        f'from pathlib import Path\n'
        f'sys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n'
        f'from _lib import T_GRID\n\n'
        f'def predict(q):\n'
        f'    raise NotImplementedError("See build_models.py for the closed-form evaluator.")\n'
    )
    (model_dir / "train.py").write_text(train)
    (model_dir / "predict.py").write_text(predict)


def model_dir(approach_num: int, name: str) -> Path:
    p = RESULTS_DIR / "models" / f"{approach_num:02d}_{name}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "saved_model").mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------
# Closed-form helpers
# ----------------------------------------------------------------------

def poly_eval(coeffs, x):
    """Evaluate sum_k coeffs[k] * x**k. coeffs[0] is constant term."""
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    for k, c in enumerate(coeffs):
        out = out + c * x ** k
    return out


def poly_str(coeffs, var: str, fmt: str = ".6g") -> str:
    """Pretty-print sum_k coeffs[k]*var^k."""
    parts = []
    for k, c in enumerate(coeffs):
        c = float(c)
        if abs(c) < 1e-15:
            continue
        if k == 0:
            parts.append(f"({c:{fmt}})")
        elif k == 1:
            parts.append(f"({c:{fmt}})*{var}")
        else:
            parts.append(f"({c:{fmt}})*{var}^{k}")
    if not parts:
        return "0"
    return " + ".join(parts)


def fit_param_polynomial(x_q, y_q, deg, ridge=1e-8):
    """Fit y = sum c_k * x^k by ridge least squares."""
    x = np.asarray(x_q).reshape(-1)
    y = np.asarray(y_q).reshape(-1)
    X = np.vstack([x ** k for k in range(deg + 1)]).T
    A = X.T @ X + ridge * np.eye(deg + 1)
    b = X.T @ y
    return np.linalg.solve(A, b)


def smooth_blend(t, t0, width):
    """Smooth 0->1 transition centred at t0 with given width using tanh."""
    return 0.5 * (1.0 + np.tanh((t - t0) / max(width, 1e-3)))


def cumtrapz(y, dx):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out
