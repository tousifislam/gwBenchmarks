"""Shared utilities for the Analytic Bench clean-room implementation."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "analytic"


def _resample(t: np.ndarray, h: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    re = np.interp(t_new, t, h.real, left=0.0, right=0.0)
    im = np.interp(t_new, t, h.imag, left=0.0, right=0.0)
    return re + 1j * im


def _load_split(split: str) -> list[dict[str, Any]]:
    fn = DATA_DIR / f"analytic_{split}.h5"
    sims: list[dict[str, Any]] = []
    with h5py.File(fn, "r") as f:
        for sid in sorted(f["sims"].keys()):
            g = f[f"sims/{sid}"]
            t = g["t"][:].astype(np.float64)
            h = g["h22_real"][:] + 1j * g["h22_imag"][:]
            sims.append(
                {
                    "sid": sid,
                    "q": float(g.attrs["q"]),
                    "eccentricity": float(g.attrs["eccentricity"]),
                    "chi1_mag": float(g.attrs["chi1_mag"]),
                    "chi2_mag": float(g.attrs["chi2_mag"]),
                    "lev_used": int(g.attrs["lev_used"]),
                    "n_samples": int(g.attrs["n_samples"]),
                    "t": t,
                    "h": h.astype(np.complex128),
                }
            )
    return sims


@lru_cache(maxsize=1)
def load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _load_split("training"), _load_split("validation")


def reparam(q: np.ndarray | float, mode: str) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    flat = q.reshape(-1)
    eta = flat / (1.0 + flat) ** 2
    delta = (flat - 1.0) / (flat + 1.0)
    if mode == "raw":
        out = flat
    elif mode == "eta":
        out = eta
    elif mode == "delta":
        out = delta
    elif mode == "log_q":
        out = np.log(flat)
    elif mode == "sqrt_eta":
        out = np.sqrt(eta)
    else:
        raise ValueError(f"unknown reparameterization {mode!r}")
    return out.reshape(q.shape)


def time_scale(t: np.ndarray | float) -> np.ndarray:
    return np.asarray(t, dtype=np.float64) / 5000.0


def q_basis(q: np.ndarray | float, mode: str, degree: int) -> tuple[np.ndarray, list[str]]:
    r = reparam(q, mode)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    cols = [np.ones_like(r)]
    names = ["1"]
    for k in range(1, degree + 1):
        cols.append(r**k)
        names.append(f"{mode}^{k}")
    return np.column_stack(cols), names


def time_basis(u: np.ndarray | float, kind: str) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(u, dtype=np.float64).reshape(-1)
    if kind == "poly_exp":
        cols = [
            np.ones_like(x),
            x,
            x**2,
            np.exp(x),
            np.exp(-x),
        ]
        names = ["1", "u", "u^2", "exp(u)", "exp(-u)"]
    elif kind == "tanh_composite":
        cols = [
            np.ones_like(x),
            x,
            np.tanh(2.0 * x),
            np.exp(x),
            np.exp(-x),
        ]
        names = ["1", "u", "tanh(2u)", "exp(u)", "exp(-u)"]
    elif kind == "gauss_sum":
        cols = [
            np.ones_like(x),
            x,
            np.exp(-((x / 0.9) ** 2)),
            np.exp(-(((x - 0.6) / 0.45) ** 2)),
            np.exp(-(((x + 0.7) / 0.55) ** 2)),
        ]
        names = ["1", "u", "G0(u)", "G1(u)", "G2(u)"]
    elif kind == "lorentz_sum":
        cols = [
            np.ones_like(x),
            x,
            1.0 / (1.0 + (x / 0.9) ** 2),
            1.0 / (1.0 + ((x - 0.6) / 0.45) ** 2),
            1.0 / (1.0 + ((x + 0.7) / 0.55) ** 2),
        ]
        names = ["1", "u", "L0(u)", "L1(u)", "L2(u)"]
    elif kind == "damped_sin":
        cols = [
            np.ones_like(x),
            np.exp(x) * np.cos(4.0 * x),
            np.exp(x) * np.sin(4.0 * x),
            np.exp(2.0 * x) * np.cos(8.0 * x),
            np.exp(2.0 * x) * np.sin(8.0 * x),
        ]
        names = ["1", "e^u cos(4u)", "e^u sin(4u)", "e^{2u} cos(8u)", "e^{2u} sin(8u)"]
    elif kind == "poly_log":
        s = x + 2.5
        cols = [
            np.ones_like(x),
            x,
            x**2,
            np.log1p(s),
            s * np.log1p(s),
        ]
        names = ["1", "u", "u^2", "log(1+s)", "s log(1+s)"]
    else:
        raise ValueError(f"unknown time basis {kind!r}")
    return np.column_stack(cols), names


def design_matrix(q: np.ndarray, t: np.ndarray, q_mode: str, q_degree: int, time_kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    qb, q_names = q_basis(q, q_mode, q_degree)
    tb, t_names = time_basis(time_scale(t), time_kind)
    feats = np.einsum("ni,nj->nij", qb, tb).reshape(len(qb), -1)
    meta = {
        "q_mode": q_mode,
        "q_degree": q_degree,
        "time_kind": time_kind,
        "q_names": q_names,
        "time_names": t_names,
    }
    return feats, meta


def sample_points(sims: list[dict[str, Any]], n_per_sim: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qs, ts, hr, hi = [], [], [], []
    for sim in sims:
        t = sim["t"]
        h = sim["h"]
        idx = np.linspace(0, len(t) - 1, min(n_per_sim, len(t))).astype(int)
        qs.append(np.full(len(idx), sim["q"], dtype=np.float64))
        ts.append(t[idx])
        hr.append(h.real[idx])
        hi.append(h.imag[idx])
    return (
        np.concatenate(qs),
        np.concatenate(ts),
        np.concatenate(hr),
        np.concatenate(hi),
    )


def fit_linear_complex(q: np.ndarray, t: np.ndarray, h: np.ndarray, q_mode: str, q_degree: int, time_kind: str) -> dict[str, Any]:
    X, meta = design_matrix(q, t, q_mode, q_degree, time_kind)
    coef_re, *_ = np.linalg.lstsq(X, h.real, rcond=None)
    coef_im, *_ = np.linalg.lstsq(X, h.imag, rcond=None)
    return {
        "type": "linear_basis",
        "meta": meta,
        "coef_re": coef_re.astype(np.float64).tolist(),
        "coef_im": coef_im.astype(np.float64).tolist(),
    }


def evaluate_linear_model(model: dict[str, Any], q: np.ndarray, t: np.ndarray) -> np.ndarray:
    meta = model["meta"]
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.shape == () or q_arr.size == 1:
        q_arr = np.full(len(np.asarray(t)), float(q_arr.reshape(())), dtype=np.float64)
    X, _ = design_matrix(q_arr, t, meta["q_mode"], int(meta["q_degree"]), meta["time_kind"])
    re = X @ np.asarray(model["coef_re"], dtype=np.float64)
    im = X @ np.asarray(model["coef_im"], dtype=np.float64)
    return re + 1j * im


def _lambdify_expr(expr: str):
    import sympy as sp

    r = sp.symbols("r")
    funcs = {
        "sqrt": sp.sqrt,
        "log": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "abs": sp.Abs,
    }
    parsed = sp.sympify(expr, locals=funcs)
    return sp.lambdify(r, parsed, modules=["numpy"])


def evaluate_curve_model(curve: dict[str, Any], q: np.ndarray) -> np.ndarray:
    r = reparam(q, curve["q_mode"]).reshape(-1)
    family = curve["family"]
    if family == "poly":
        coeffs = np.asarray(curve["coeffs"], dtype=np.float64)
        out = np.zeros_like(r)
        for power, coeff in enumerate(coeffs):
            out = out + coeff * (r**power)
        return out
    if family == "pade":
        num = np.asarray(curve["num"], dtype=np.float64)
        den = np.asarray(curve["den"], dtype=np.float64)
        numv = np.zeros_like(r)
        denv = np.ones_like(r)
        for power, coeff in enumerate(num):
            numv = numv + coeff * (r**power)
        for power, coeff in enumerate(den, start=1):
            denv = denv + coeff * (r**power)
        return numv / denv
    if family == "pysr":
        if "_func" not in curve:
            curve["_func"] = _lambdify_expr(curve["expr"])
        return np.asarray(curve["_func"](r), dtype=np.float64)
    if family == "gplearn":
        est = curve["estimator"]
        return np.asarray(est.predict(r.reshape(-1, 1)), dtype=np.float64)
    if family == "poly_tanh":
        coeffs = np.asarray(curve["coeffs"], dtype=np.float64)
        r0 = float(curve["r0"])
        width = float(curve["width"])
        s = np.tanh((r - r0) / width)
        feats = [np.ones_like(r), r, r**2, s, r * s]
        return sum(c * f for c, f in zip(coeffs, feats))
    raise ValueError(f"unknown curve family {family!r}")


def evaluate_curve_basis_model(model: dict[str, Any], q: np.ndarray, t: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.shape == () or q_arr.size == 1:
        q_arr = np.full(len(np.asarray(t)), float(q_arr.reshape(())), dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    if q_arr.shape != t_arr.shape:
        raise ValueError("q and t must have matching shapes or q must be scalar")
    tb, _ = time_basis(time_scale(t_arr), model["time_kind"])
    re = np.zeros(len(t), dtype=np.float64)
    im = np.zeros(len(t), dtype=np.float64)
    for idx, basis_vec in enumerate(tb.T):
        curve_re = evaluate_curve_model(model["re_curves"][idx], q_arr)
        curve_im = evaluate_curve_model(model["im_curves"][idx], q_arr)
        re = re + curve_re * basis_vec
        im = im + curve_im * basis_vec
    return re + 1j * im


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_scorecard(model_dir: Path, scorecard: dict[str, Any]) -> None:
    save_json(model_dir / "scorecard.json", scorecard)


def model_dir(approach_num: int, name: str) -> Path:
    p = RESULTS_DIR / "models" / f"{approach_num:02d}_{name}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "saved_model").mkdir(parents=True, exist_ok=True)
    return p


def write_train_predict(model_dir: Path, approach: str) -> None:
    train = f'''"""Self-contained training stub for {approach}."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_models import retrain_named_model

if __name__ == "__main__":
    retrain_named_model("{approach}")
'''
    predict = f'''"""Prediction stub for {approach}."""
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import evaluate_linear_model

def predict(q, t):
    model_path = Path(__file__).resolve().parent / "saved_model" / "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if model["type"] == "linear_basis":
        return evaluate_linear_model(model, q, t)
    return evaluate_curve_basis_model(model, q, t)
'''
    (model_dir / "train.py").write_text(train)
    (model_dir / "predict.py").write_text(predict)


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
