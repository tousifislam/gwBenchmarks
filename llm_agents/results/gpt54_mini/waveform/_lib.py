"""Shared utilities for the Waveform Bench clean-room implementation."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "waveform"

DT_GEOMETRIC = 0.2
T_MIN = -2800.0
T_MAX = 100.0
N_T = int(round((T_MAX - T_MIN) / DT_GEOMETRIC)) + 1
T_GRID = np.round(np.linspace(T_MIN, T_MAX, N_T), 10)


def _resample(t: np.ndarray, y: np.ndarray, grid: np.ndarray = T_GRID) -> np.ndarray:
    re = np.interp(grid, t, y.real, left=0.0, right=0.0)
    im = np.interp(grid, t, y.imag, left=0.0, right=0.0)
    return re + 1j * im


def _load_split(split: str) -> list[dict[str, Any]]:
    fn = DATA_DIR / f"waveform_{split}.h5"
    sims: list[dict[str, Any]] = []
    with h5py.File(fn, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sim_"))
        for key in keys:
            g = f[key]
            t = g["t"][:].astype(np.float64)
            h = g["h22_real"][:] + 1j * g["h22_imag"][:]
            sims.append(
                {
                    "sid": key,
                    "q": float(g.attrs["q"]),
                    "chi1x": float(g.attrs["chi1x"]),
                    "chi1y": float(g.attrs["chi1y"]),
                    "chi1z": float(g.attrs["chi1z"]),
                    "chi2x": float(g.attrs["chi2x"]),
                    "chi2y": float(g.attrs["chi2y"]),
                    "chi2z": float(g.attrs["chi2z"]),
                    "chi_eff": float(g.attrs.get("chi_eff", (g.attrs["q"] * g.attrs["chi1z"] + g.attrs["chi2z"]) / (1.0 + g.attrs["q"]))),
                    "chi_p": float(g.attrs.get("chi_p", np.sqrt(g.attrs["chi1x"] ** 2 + g.attrs["chi1y"] ** 2) + np.sqrt(g.attrs["chi2x"] ** 2 + g.attrs["chi2y"] ** 2))),
                    "omega0": float(g.attrs["omega0"]),
                    "n_pts": int(g.attrs["n_pts"]),
                    "lev_used": int(g.attrs["lev_used"]),
                    "wf_length": float(g.attrs["wf_length"]),
                    "nr_fd_mm_combined": float(g.attrs.get("nr_fd_mm_combined", np.nan)),
                    "nr_fd_mm_M40": float(g.attrs.get("nr_fd_mm_M40", np.nan)),
                    "nr_fd_mm_M80": float(g.attrs.get("nr_fd_mm_M80", np.nan)),
                    "nr_fd_mm_M120": float(g.attrs.get("nr_fd_mm_M120", np.nan)),
                    "nr_fd_mm_M160": float(g.attrs.get("nr_fd_mm_M160", np.nan)),
                    "nr_fd_mm_M200": float(g.attrs.get("nr_fd_mm_M200", np.nan)),
                    "t": t,
                    "h": h.astype(np.complex128),
                }
            )
    return sims


@lru_cache(maxsize=1)
def load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _load_split("training"), _load_split("validation")


def resample_sims(sims: list[dict[str, Any]], grid: np.ndarray = T_GRID) -> np.ndarray:
    return np.stack([_resample(sim["t"], sim["h"], grid) for sim in sims], axis=0)


def resample_true_and_params(sims: list[dict[str, Any]], grid: np.ndarray = T_GRID) -> tuple[np.ndarray, np.ndarray]:
    X = feature_matrix(sims, "raw7")
    Y = resample_sims(sims, grid)
    return X, Y


def stack_complex(h: np.ndarray) -> np.ndarray:
    return np.concatenate([h.real, h.imag], axis=-1)


def unstack_complex(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1] // 2
    return x[..., :n] + 1j * x[..., n:]


def spin_spherical(chix: float, chiy: float, chiz: float) -> tuple[float, float, float]:
    mag = float(np.sqrt(chix * chix + chiy * chiy + chiz * chiz))
    if mag < 1e-12:
        return 0.0, 0.0, 0.0
    theta = float(np.arccos(np.clip(chiz / mag, -1.0, 1.0)))
    phi = float(np.arctan2(chiy, chix))
    return mag, theta, phi


def derived_quantities(sim: dict[str, Any]) -> dict[str, float]:
    q = sim["q"]
    eta = q / (1.0 + q) ** 2
    delta_m = (q - 1.0) / (q + 1.0)
    chi_eff = sim["chi_eff"]
    chi_p = sim["chi_p"]
    chi1_mag, theta1, phi1 = spin_spherical(sim["chi1x"], sim["chi1y"], sim["chi1z"])
    chi2_mag, theta2, phi2 = spin_spherical(sim["chi2x"], sim["chi2y"], sim["chi2z"])
    return {
        "q": q,
        "eta": eta,
        "delta_m": delta_m,
        "chi_eff": chi_eff,
        "chi_p": chi_p,
        "chi1_mag": chi1_mag,
        "chi2_mag": chi2_mag,
        "theta1": theta1,
        "phi1": phi1,
        "theta2": theta2,
        "phi2": phi2,
        "omega0": sim["omega0"],
    }


def feature_matrix(sims: list[dict[str, Any]], mode: str) -> np.ndarray:
    rows = []
    for sim in sims:
        q = sim["q"]
        eta = q / (1.0 + q) ** 2
        delta_m = (q - 1.0) / (q + 1.0)
        chi_eff = sim["chi_eff"]
        chi_p = sim["chi_p"]
        chi1_mag, theta1, phi1 = spin_spherical(sim["chi1x"], sim["chi1y"], sim["chi1z"])
        chi2_mag, theta2, phi2 = spin_spherical(sim["chi2x"], sim["chi2y"], sim["chi2z"])
        omega0 = sim["omega0"]
        if mode == "raw7":
            rows.append([q, sim["chi1x"], sim["chi1y"], sim["chi1z"], sim["chi2x"], sim["chi2y"], sim["chi2z"], omega0])
        elif mode == "raw7_noomega":
            rows.append([q, sim["chi1x"], sim["chi1y"], sim["chi1z"], sim["chi2x"], sim["chi2y"], sim["chi2z"]])
        elif mode == "eff":
            rows.append([eta, chi_eff, chi_p, chi1_mag, chi2_mag, q, omega0])
        elif mode == "delta":
            rows.append([delta_m, chi_eff, chi_p, sim["chi1z"], sim["chi2z"], q, omega0])
        elif mode == "spherical":
            rows.append([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2, omega0, q])
        elif mode == "omega":
            rows.append([omega0, q, chi_eff, chi_p, eta])
        else:
            raise ValueError(f"unknown feature mode {mode!r}")
    return np.asarray(rows, dtype=np.float64)


def feature_names(mode: str) -> list[str]:
    if mode == "raw7":
        return ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z", "omega0"]
    if mode == "raw7_noomega":
        return ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z"]
    if mode == "eff":
        return ["eta", "chi_eff", "chi_p", "|chi1|", "|chi2|", "q", "omega0"]
    if mode == "delta":
        return ["delta_m", "chi_eff", "chi_p", "chi1z", "chi2z", "q", "omega0"]
    if mode == "spherical":
        return ["eta", "|chi1|", "theta1", "phi1", "|chi2|", "theta2", "phi2", "omega0", "q"]
    if mode == "omega":
        return ["omega0", "q", "chi_eff", "chi_p", "eta"]
    raise ValueError(f"unknown feature mode {mode!r}")


def time_transform(grid: np.ndarray, mode: str) -> np.ndarray:
    t = np.asarray(grid, dtype=np.float64)
    span = float(t[-1] - t[0])
    if mode == "peak":
        return t / 100.0
    if mode == "start":
        return (t - t[0]) / span
    if mode == "reverse":
        return (t[-1] - t) / span
    raise ValueError(f"unknown time mode {mode!r}")


def time_basis(grid: np.ndarray, mode: str) -> tuple[np.ndarray, list[str]]:
    u = time_transform(grid, mode)
    if mode == "peak":
        basis = [np.ones_like(u), u, u**2, np.exp(u), np.exp(-u)]
        names = ["1", "u", "u^2", "exp(u)", "exp(-u)"]
    elif mode == "start":
        basis = [np.ones_like(u), u, np.tanh(2.0 * (u - 0.5)), np.exp(-((u - 0.75) / 0.2) ** 2), np.exp(-((u - 0.15) / 0.15) ** 2)]
        names = ["1", "u", "tanh(2(u-.5))", "G_peak", "G_insp"]
    elif mode == "reverse":
        basis = [np.ones_like(u), u, np.tanh(2.0 * (u - 0.5)), np.exp(-((u - 0.25) / 0.2) ** 2), np.exp(-((u - 0.85) / 0.15) ** 2)]
        names = ["1", "u", "tanh(2(u-.5))", "G_insp", "G_rd"]
    else:
        raise ValueError(f"unknown time mode {mode!r}")
    return np.column_stack(basis), names


def flatten_basis_pred(coeffs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return coeffs @ basis


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
    train = f'''"""Training stub for {approach}."""
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
from _lib import predict_from_model

def predict(params):
    model_path = Path(__file__).resolve().parent / "saved_model" / "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return predict_from_model(model, params)
'''
    (model_dir / "train.py").write_text(train)
    (model_dir / "predict.py").write_text(predict)


def params_to_vector(params: Any, mode: str) -> np.ndarray:
    if isinstance(params, dict):
        sim = params
    else:
        arr = np.asarray(params, dtype=np.float64).reshape(-1)
        if mode == "raw7":
            if arr.size == 8:
                return arr
            if arr.size == 7:
                return np.concatenate([arr, [0.0]])
        raise ValueError("array params supported only for raw7/raw7_noomega")
        # unreachable
    if mode == "raw7":
        return np.array([sim["q"], sim["chi1x"], sim["chi1y"], sim["chi1z"], sim["chi2x"], sim["chi2y"], sim["chi2z"], sim.get("omega0", 0.0)], dtype=np.float64)
    if mode == "raw7_noomega":
        return np.array([sim["q"], sim["chi1x"], sim["chi1y"], sim["chi1z"], sim["chi2x"], sim["chi2y"], sim["chi2z"]], dtype=np.float64)
    if mode == "eff":
        q = float(sim["q"])
        eta = q / (1.0 + q) ** 2
        chi_eff = float(sim.get("chi_eff", (q * sim["chi1z"] + sim["chi2z"]) / (1.0 + q)))
        chi_p = float(sim.get("chi_p", np.sqrt(sim["chi1x"] ** 2 + sim["chi1y"] ** 2) + np.sqrt(sim["chi2x"] ** 2 + sim["chi2y"] ** 2)))
        chi1_mag = float(np.sqrt(sim["chi1x"] ** 2 + sim["chi1y"] ** 2 + sim["chi1z"] ** 2))
        chi2_mag = float(np.sqrt(sim["chi2x"] ** 2 + sim["chi2y"] ** 2 + sim["chi2z"] ** 2))
        return np.array([eta, chi_eff, chi_p, chi1_mag, chi2_mag, q, sim.get("omega0", 0.0)], dtype=np.float64)
    if mode == "delta":
        q = float(sim["q"])
        delta_m = (q - 1.0) / (q + 1.0)
        chi_eff = float(sim.get("chi_eff", (q * sim["chi1z"] + sim["chi2z"]) / (1.0 + q)))
        chi_p = float(sim.get("chi_p", np.sqrt(sim["chi1x"] ** 2 + sim["chi1y"] ** 2) + np.sqrt(sim["chi2x"] ** 2 + sim["chi2y"] ** 2)))
        return np.array([delta_m, chi_eff, chi_p, sim["chi1z"], sim["chi2z"], q, sim.get("omega0", 0.0)], dtype=np.float64)
    if mode == "spherical":
        q = float(sim["q"])
        eta = q / (1.0 + q) ** 2
        chi1_mag, theta1, phi1 = spin_spherical(sim["chi1x"], sim["chi1y"], sim["chi1z"])
        chi2_mag, theta2, phi2 = spin_spherical(sim["chi2x"], sim["chi2y"], sim["chi2z"])
        return np.array([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2, sim.get("omega0", 0.0), q], dtype=np.float64)
    if mode == "omega":
        q = float(sim["q"])
        eta = q / (1.0 + q) ** 2
        chi_eff = float(sim.get("chi_eff", (q * sim["chi1z"] + sim["chi2z"]) / (1.0 + q)))
        chi_p = float(sim.get("chi_p", np.sqrt(sim["chi1x"] ** 2 + sim["chi1y"] ** 2) + np.sqrt(sim["chi2x"] ** 2 + sim["chi2y"] ** 2)))
        return np.array([sim.get("omega0", 0.0), q, chi_eff, chi_p, eta], dtype=np.float64)
    raise ValueError(f"unknown feature mode {mode!r}")


def feature_vector(params: Any, mode: str) -> np.ndarray:
    return params_to_vector(params, mode)


def symbolic_expr_func(expr: str):
    import sympy as sp
    import re

    expr = re.sub(r"\bX(\d+)\b", r"x\1", expr)
    symbols = sp.symbols(f"x0:{32}")
    locals_map = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "neg": lambda a: -a,
        "inv": lambda a: 1.0 / a,
        "max": sp.Max,
        "min": sp.Min,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
        "sin": sp.sin,
        "cos": sp.cos,
        "tanh": sp.tanh,
        "abs": sp.Abs,
        "x0": symbols[0],
        "x1": symbols[1],
        "x2": symbols[2],
        "x3": symbols[3],
        "x4": symbols[4],
        "x5": symbols[5],
        "x6": symbols[6],
        "x7": symbols[7],
        "x8": symbols[8],
        "x9": symbols[9],
    }
    parsed = sp.sympify(expr, locals=locals_map)
    return sp.lambdify(symbols, parsed, modules=["numpy"])


def predict_from_model(model: dict[str, Any], params: Any) -> np.ndarray:
    mode = model["feature_mode"]
    x = feature_vector(params, mode)
    single_input = x.ndim == 1
    if x.ndim == 1:
        x = x.reshape(1, -1)

    def _maybe_unreverse(arr: np.ndarray) -> np.ndarray:
        return arr[:, ::-1] if model.get("time_mode") == "reverse" else arr

    def _finalize_stack(stack: np.ndarray) -> np.ndarray:
        out = _maybe_unreverse(stack)
        if model.get("stacked_complex", False):
            out = unstack_complex(out)
        return out[0] if single_input else out

    def _predict_columns(regressors: Any, x_in: np.ndarray) -> np.ndarray:
        if isinstance(regressors, list):
            cols = []
            for reg in regressors:
                if hasattr(reg, "predict"):
                    pred = reg.predict(x_in)
                else:
                    pred = reg(x_in)
                cols.append(np.asarray(pred, dtype=np.float64).reshape(-1))
            return np.column_stack(cols)
        if hasattr(regressors, "predict"):
            pred = regressors.predict(x_in)
        else:
            pred = regressors(x_in)
            pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def _pad_row(row: np.ndarray) -> np.ndarray:
        if row.shape[0] >= 32:
            return row[:32]
        padded = np.zeros(32, dtype=np.float64)
        padded[: row.shape[0]] = row
        return padded

    if model["kind"] == "svd":
        coeffs = _predict_columns(model["regressor"], x)
        stack = coeffs @ model["basis"]
        if "mean_stack" in model:
            stack = stack + model["mean_stack"]
        return _finalize_stack(stack)
    if model["kind"] == "eim":
        node_vals = _predict_columns(model["regressor"], x)
        coeffs = node_vals @ model["eim_inverse"]
        stack = coeffs @ model["basis"]
        if "mean_stack" in model:
            stack = stack + model["mean_stack"]
        return _finalize_stack(stack)
    if model["kind"] == "ap_svd":
        amp_coeffs = _predict_columns(model["amp_regressor"], x)
        phase_coeffs = _predict_columns(model["phase_regressor"], x)
        amp = amp_coeffs @ model["amp_basis"]
        phase = phase_coeffs @ model["phase_basis"]
        if "amp_mean" in model:
            amp = amp + model["amp_mean"]
        if "phase_mean" in model:
            phase = phase + model["phase_mean"]
        amp = np.clip(amp, 0.0, None)
        out = _maybe_unreverse(amp * np.exp(1j * phase))
        return out[0] if single_input else out
    if model["kind"] == "symbolic_mix":
        top_cols = []
        if model["symbolic_tool"] in {"pysr", "gplearn"}:
            cache = model.setdefault("_compiled_exprs", {})
            scales = model.get("top_expr_scales")
            biases = model.get("top_expr_biases")
            expr_modes = model.get("top_expr_feature_modes", [mode] * len(model["top_exprs"]))
            for idx, expr in enumerate(model["top_exprs"]):
                fn = cache.get(expr)
                if fn is None:
                    fn = symbolic_expr_func(expr)
                    cache[expr] = fn
                x_expr = feature_vector(params, expr_modes[idx])
                if x_expr.ndim == 1:
                    x_expr = x_expr.reshape(1, -1)
                vals = np.array([fn(*_pad_row(row)) for row in x_expr], dtype=np.float64)
                if scales is not None:
                    vals = vals * float(scales[idx])
                if biases is not None:
                    vals = vals + float(biases[idx])
                top_cols.append(vals)
        else:
            raise ValueError(f"unknown symbolic tool {model['symbolic_tool']!r}")
        top_mat = np.column_stack(top_cols) if top_cols else np.zeros((x.shape[0], 0))
        rest = np.asarray(model["rest_regressor"].predict(x), dtype=np.float64)
        if rest.ndim == 1:
            rest = rest.reshape(-1, 1)
        coeffs = np.concatenate([top_mat, rest], axis=1) if rest.size else top_mat
        stack = coeffs @ model["basis"]
        if "mean_stack" in model:
            stack = stack + model["mean_stack"]
        return _finalize_stack(stack)
    raise ValueError(f"unknown model kind {model['kind']!r}")


def eval_fd_mismatch(h_pred: np.ndarray, h_true: np.ndarray, masses: Iterable[int] = (40, 80, 120, 160, 200)) -> tuple[float, dict[str, float]]:
    from gwbenchmarks.metrics import frequency_domain_mismatch

    vals = []
    comp = {}
    for m in masses:
        cur = float(frequency_domain_mismatch(h_pred, h_true, dt_geometric=DT_GEOMETRIC, mtot_msun=float(m)))
        comp[f"mismatch_{int(m)}Msun"] = cur
        vals.append(cur)
    return float(np.mean(vals)), comp


def sample_subset(sims: list[dict[str, Any]], n: int = 64) -> list[dict[str, Any]]:
    if len(sims) <= n:
        return sims
    idx = np.linspace(0, len(sims) - 1, n).astype(int)
    return [sims[i] for i in idx]
