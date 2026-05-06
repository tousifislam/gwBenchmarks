"""Clean-room waveform benchmark builder for gpt54_mini."""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.linalg import qr
from scipy.optimize import least_squares
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

ROOT = Path(__file__).resolve().parents[4]
WORK_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = WORK_DIR / "comparison"
MODELS_DIR = WORK_DIR / "models"
CHANGELOG = WORK_DIR / "CHANGELOG.md"
ENV_JULIA_PROJECT = "/private/tmp/sr_julia_project_926"
ENV_JULIA_EXE = "/private/tmp/pysr_julia_env/pyjuliapkg/install/bin/julia"
ENV_JULIA_DEPOT = "/private/tmp/gpt54_julia_depot2:/Users/tousifislam/.julia"
ENV_MPLCONFIG = "/private/tmp/gpt54_mplconfig"

os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", ENV_JULIA_PROJECT)
os.environ.setdefault("PYTHON_JULIAPKG_EXE", ENV_JULIA_EXE)
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
os.environ.setdefault("JULIA_PKG_SERVER", "")
os.environ.setdefault("JULIA_DEPOT_PATH", ENV_JULIA_DEPOT)
os.environ.setdefault("MPLCONFIGDIR", ENV_MPLCONFIG)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK_DIR))

from _lib import (  # noqa: E402
    RESULTS_DIR,
    T_GRID,
    eval_fd_mismatch,
    feature_matrix,
    feature_names,
    load_data,
    model_dir,
    resample_sims,
    sample_subset,
    save_json,
    save_scorecard,
    stack_complex,
    symbolic_expr_func,
    unstack_complex,
    write_train_predict,
)
from gwbenchmarks import plot_settings  # noqa: E402

plot_settings.apply()


@dataclass(frozen=True)
class ApproachSpec:
    number: int
    name: str
    category: str
    family: str
    feature_mode: str
    time_mode: str
    regressor: str
    n_components: int = 6
    notes: str = ""
    degree: int = 2
    top_expr_modes: tuple[str, ...] = ()
    symbolic_tool: str = ""
    symbolic_maxsizes: tuple[int, ...] = ()


SPECS: list[ApproachSpec] = [
    ApproachSpec(1, "svd_gpr_raw", "SVD/decomposition", "svd", "raw7", "peak", "gpr_rbf", 6, "Baseline complex SVD coefficients with RBF GPR."),
    ApproachSpec(2, "svd_gpr_eff", "SVD/decomposition", "svd", "eff", "peak", "gpr_matern", 6, "Effective-spin reparameterization with Matern GPR."),
    ApproachSpec(3, "svd_poly_delta", "SVD/decomposition", "svd", "delta", "peak", "poly_ridge", 6, "Low-order polynomial fit on mass-difference features."),
    ApproachSpec(4, "svd_mlp_spherical", "Machine learning", "svd", "spherical", "peak", "mlp_small", 6, "MLP on spherical-spin features."),
    ApproachSpec(5, "svd_rf_raw", "Machine learning", "svd", "raw7_noomega", "peak", "rf", 6, "Random forest on raw parameters."),
    ApproachSpec(6, "svd_gb_eff", "Machine learning", "svd", "eff", "peak", "hist_gb", 6, "Gradient boosting on effective-spin features."),
    ApproachSpec(7, "svd_huber_omega", "Machine learning", "svd", "omega", "peak", "huber", 6, "Robust linear model with omega0 included."),
    ApproachSpec(8, "svd_bayes_delta", "Machine learning", "svd", "delta", "peak", "bayes", 6, "Bayesian ridge regression on delta-mass features."),
    ApproachSpec(9, "svd_rbf_raw", "Interpolation/kernel", "svd", "raw7", "peak", "rbf_interp", 6, "RBF interpolation of SVD coefficients."),
    ApproachSpec(10, "svd_krr_eff", "Interpolation/kernel", "svd", "eff", "peak", "krr", 6, "Kernel ridge on effective-spin features."),
    ApproachSpec(11, "knn_correction_spherical", "Interpolation/kernel", "svd", "spherical", "peak", "knn", 6, "Nearest-neighbor interpolation with residual correction."),
    ApproachSpec(12, "eim_gpr_raw", "SVD/decomposition", "eim", "raw7", "peak", "gpr_rbf", 6, "EIM node values with GPR on raw parameters."),
    ApproachSpec(13, "eim_rf_eff", "SVD/decomposition", "eim", "eff", "peak", "rf", 6, "EIM node values with random forest on effective-spin features."),
    ApproachSpec(14, "ap_svd_gpr_peak", "SVD/decomposition", "ap_svd", "raw7", "peak", "gpr_rbf", 5, "Amplitude/phase SVD with peak-aligned time ordering."),
    ApproachSpec(15, "ap_svd_mlp_reverse", "SVD/decomposition", "ap_svd", "spherical", "reverse", "mlp_small", 5, "Amplitude/phase SVD with reversed time ordering."),
    ApproachSpec(16, "ap_svd_poly_peak", "SVD/decomposition", "ap_svd", "delta", "peak", "poly_ridge", 5, "Amplitude/phase SVD with polynomial coefficient fits."),
    ApproachSpec(17, "svd_svr_omega", "Machine learning", "svd", "omega", "peak", "svr", 6, "Support-vector regression on omega0-augmented features."),
    ApproachSpec(18, "svd_extra_trees_delta", "Machine learning", "svd", "delta", "peak", "extra_trees", 6, "Extra-trees ensemble on delta-mass features."),
    ApproachSpec(19, "pysr_svd_eff", "Symbolic/analytical", "symbolic", "eff", "peak", "ridge", 6, "PySR-discovered expressions calibrated on SVD coefficients.", top_expr_modes=("raw7", "eff", "spherical"), symbolic_tool="pysr", symbolic_maxsizes=(8, 12)),
    ApproachSpec(20, "gplearn_svd_spherical", "Symbolic/analytical", "symbolic", "spherical", "peak", "ridge", 6, "gplearn expressions calibrated on SVD coefficients.", top_expr_modes=("delta", "raw7_noomega", "omega"), symbolic_tool="gplearn", symbolic_maxsizes=(8,)),
]


def set_seed() -> None:
    np.random.seed(42)


def reverse_if_needed(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "reverse":
        return y[:, ::-1]
    return y


def build_feature_matrix(sims: list[dict[str, Any]], mode: str) -> np.ndarray:
    X = feature_matrix(sims, mode)
    return np.asarray(X, dtype=np.float64)


def build_physical_features(X: np.ndarray) -> np.ndarray:
    q = X[:, 0]
    chi1x, chi1y, chi1z = X[:, 1], X[:, 2], X[:, 3]
    chi2x, chi2y, chi2z = X[:, 4], X[:, 5], X[:, 6]
    omega0 = X[:, 7]
    eta = q / (1.0 + q) ** 2
    delta_m = (q - 1.0) / (q + 1.0)
    chi_eff = (q * chi1z + chi2z) / (1.0 + q)
    chi_p = np.sqrt(chi1x**2 + chi1y**2) + np.sqrt(chi2x**2 + chi2y**2)
    cols = [
        np.ones_like(q),
        q,
        eta,
        delta_m,
        chi_eff,
        chi_p,
        omega0,
        q * chi_eff,
        chi_eff**2,
        chi_p**2,
        eta * chi_eff,
        eta * chi_p,
    ]
    return np.column_stack(cols)


def fit_regressor(name: str, X: np.ndarray, y: np.ndarray, degree: int = 2) -> Any:
    if name == "gpr_rbf":
        est = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "gpr",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0, constant_value_bounds="fixed")
                        * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds="fixed")
                        + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed"),
                        alpha=1e-6,
                        normalize_y=True,
                        optimizer=None,
                        random_state=42,
                    ),
                ),
            ]
        )
        return est.fit(X, y)
    if name == "gpr_matern":
        est = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "gpr",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0, constant_value_bounds="fixed")
                        * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds="fixed", nu=1.5)
                        + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed"),
                        alpha=1e-6,
                        normalize_y=True,
                        optimizer=None,
                        random_state=42,
                    ),
                ),
            ]
        )
        return est.fit(X, y)
    if name == "poly_ridge":
        est = Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        return est.fit(X, y)
    if name == "mlp_small":
        est = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        activation="tanh",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=1200,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=42,
                    ),
                ),
            ]
        )
        return est.fit(X, y)
    if name == "rf":
        est = RandomForestRegressor(
            n_estimators=240,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        return est.fit(X, y)
    if name == "extra_trees":
        est = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=1,
        )
        return est.fit(X, y)
    if name == "hist_gb":
        est = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=250,
            l2_regularization=1e-4,
            random_state=42,
        )
        return est.fit(X, y)
    if name == "huber":
        est = Pipeline([("scale", StandardScaler()), ("huber", HuberRegressor(alpha=1e-4, max_iter=2000))])
        return est.fit(X, y)
    if name == "bayes":
        est = Pipeline([("scale", StandardScaler()), ("bayes", BayesianRidge())])
        return est.fit(X, y)
    if name == "krr":
        est = Pipeline([("scale", StandardScaler()), ("krr", KernelRidge(alpha=1e-3, kernel="rbf", gamma=0.8))])
        return est.fit(X, y)
    if name == "svr":
        est = Pipeline([("scale", StandardScaler()), ("svr", SVR(C=10.0, epsilon=1e-3, gamma="scale"))])
        return est.fit(X, y)
    if name == "knn":
        est = KNeighborsRegressor(n_neighbors=5, weights="distance", algorithm="auto")
        return est.fit(X, y)
    if name == "ridge":
        est = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1e-3))])
        return est.fit(X, y)
    raise ValueError(f"unknown regressor kind {name!r}")


def predict_regressor(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        pred = model.predict(X)
    else:
        pred = model(X)
    return np.asarray(pred, dtype=np.float64).reshape(-1)


def fit_column_models(X: np.ndarray, Y: np.ndarray, maker: Callable[[np.ndarray, np.ndarray], Any]) -> list[Any]:
    return [maker(X, Y[:, i]) for i in range(Y.shape[1])]


def predict_column_models(models: list[Any], X: np.ndarray) -> np.ndarray:
    cols = [predict_regressor(model, X) for model in models]
    return np.column_stack(cols)


def svd_decomposition(Y_stack: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_stack = Y_stack.mean(axis=0)
    centered = Y_stack - mean_stack
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components]
    coeffs = centered @ basis.T
    return mean_stack, basis, coeffs


def amp_phase_decomposition(Y: np.ndarray, n_components: int, time_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Y_work = reverse_if_needed(Y, time_mode)
    amp = np.log1p(np.abs(Y_work))
    phase = np.unwrap(np.angle(Y_work), axis=1)
    align_idx = np.argmin(np.abs(T_GRID))
    if time_mode == "reverse":
        align_idx = len(T_GRID) - 1 - align_idx
    phase -= phase[:, [align_idx]]
    amp_mean = amp.mean(axis=0)
    phase_mean = phase.mean(axis=0)
    amp_centered = amp - amp_mean
    phase_centered = phase - phase_mean
    _, _, amp_vt = np.linalg.svd(amp_centered, full_matrices=False)
    _, _, phase_vt = np.linalg.svd(phase_centered, full_matrices=False)
    amp_basis = amp_vt[:n_components]
    phase_basis = phase_vt[:n_components]
    amp_coeffs = amp_centered @ amp_basis.T
    phase_coeffs = phase_centered @ phase_basis.T
    return amp_mean, phase_mean, amp_basis, phase_basis, amp_coeffs, phase_coeffs


def eim_setup(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, _, piv = qr(basis, pivoting=True)
    nodes = np.asarray(piv[: basis.shape[0]], dtype=int)
    mat = basis[:, nodes]
    inv = np.linalg.inv(mat)
    return nodes, inv


def select_symbolic_expressions(
    tool: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_mode: str,
    maxsizes: tuple[int, ...],
    feature_labels: list[str],
) -> tuple[list[dict[str, Any]], str]:
    discovered: list[dict[str, Any]] = []
    best = None
    best_loss = np.inf

    if tool == "pysr":
        from pysr import PySRRegressor

        for maxsize in maxsizes:
            model = PySRRegressor(
                niterations=25,
                populations=6,
                population_size=80,
                maxsize=maxsize,
                procs=1,
                parallelism="serial",
                deterministic=True,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log", "exp", "sin", "cos", "tanh"],
                model_selection="best",
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y)
            table = model.equations_.copy()
            rows = []
            for _, row in table.iterrows():
                rows.append(
                    {
                        "equation": str(row["equation"]),
                        "complexity": int(row["complexity"]),
                        "loss": float(row["loss"]),
                        "feature_mode": feature_mode,
                        "maxsize": int(maxsize),
                    }
                )
            discovered.extend(rows)
            candidate = str(model.sympy())
            cand_vals = evaluate_expression(candidate, X)
            scale, bias, loss = calibrate_expression(cand_vals, y)
            if loss < best_loss and np.isfinite(loss):
                best_loss = loss
                best = candidate
    elif tool == "gplearn":
        from gplearn.genetic import SymbolicRegressor

        for maxsize in maxsizes:
            model = SymbolicRegressor(
                population_size=700,
                generations=12,
                tournament_size=20,
                function_set=["add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"],
                metric="mse",
                parsimony_coefficient=0.001,
                max_samples=0.9,
                verbose=0,
                random_state=42 + int(maxsize),
                n_jobs=1,
            )
            model.fit(X, y)
            programs = []
            for gen in getattr(model, "_programs", []):
                for prog in gen:
                    if prog is None:
                        continue
                    programs.append(str(prog))
            for prog in programs:
                discovered.append({"equation": prog, "complexity": None, "loss": None, "feature_mode": feature_mode, "maxsize": int(maxsize)})
            candidate = str(model._program)
            cand_vals = evaluate_expression(candidate, X)
            scale, bias, loss = calibrate_expression(cand_vals, y)
            if loss < best_loss and np.isfinite(loss):
                best_loss = loss
                best = candidate
    else:
        raise ValueError(f"unknown symbolic tool {tool!r}")

    if best is None:
        best = "x0"
    return discovered, best


def evaluate_expression(expr: str, X: np.ndarray) -> np.ndarray:
    fn = symbolic_expr_func(expr)
    vals = []
    for row in X:
        try:
            padded = np.zeros(32, dtype=np.float64)
            padded[: min(len(row), 32)] = row[:32]
            val = fn(*padded)
            if np.iscomplexobj(val):
                val = np.real(val)
        except Exception:
            val = np.nan
        vals.append(val)
    arr = np.asarray(vals, dtype=np.float64)
    return arr


def calibrate_expression(vals: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(vals) & np.isfinite(target)
    if mask.sum() < 4:
        return 1.0, 0.0, np.inf
    x = vals[mask]
    y = target[mask]
    A = np.column_stack([x, np.ones_like(x)])
    try:
        sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        scale, bias = float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        scale, bias = 1.0, 0.0
    pred = scale * x + bias
    loss = float(np.mean((pred - y) ** 2))
    return scale, bias, loss


def train_symbolic_mix(
    spec: ApproachSpec,
    train_sims: list[dict[str, Any]],
    X_train: np.ndarray,
    Y_train_stack: np.ndarray,
    basis: np.ndarray,
    mean_stack: np.ndarray,
    coeffs: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    top_targets = list(range(min(3, coeffs.shape[1])))
    rest_targets = list(range(3, coeffs.shape[1]))
    if not spec.top_expr_modes:
        raise ValueError("symbolic spec requires top_expr_modes")
    top_exprs: list[str] = []
    top_modes: list[str] = []
    top_scales: list[float] = []
    top_biases: list[float] = []
    all_discovered: list[dict[str, Any]] = []

    for idx, target_idx in enumerate(top_targets):
        mode = spec.top_expr_modes[idx % len(spec.top_expr_modes)]
        X_expr = build_feature_matrix(train_sims, mode)
        if spec.symbolic_tool == "pysr":
            feat_labels = feature_names(mode)
        else:
            feat_labels = feature_names(mode)
        discovered, best_expr = select_symbolic_expressions(
            spec.symbolic_tool,
            X_expr,
            coeffs[:, target_idx],
            mode,
            spec.symbolic_maxsizes,
            feat_labels,
        )
        all_discovered.extend(discovered)
        vals = evaluate_expression(best_expr, X_expr)
        scale, bias, _ = calibrate_expression(vals, coeffs[:, target_idx])
        top_exprs.append(best_expr)
        top_modes.append(mode)
        top_scales.append(scale)
        top_biases.append(bias)

    if rest_targets:
        rest_regressor = fit_regressor("ridge", X_train, coeffs[:, rest_targets], degree=2)
    else:
        rest_regressor = fit_regressor("ridge", X_train, coeffs[:, :0], degree=2)

    model = {
        "kind": "symbolic_mix",
        "symbolic_tool": spec.symbolic_tool,
        "feature_mode": spec.feature_mode,
        "time_mode": spec.time_mode,
        "stacked_complex": True,
        "basis": basis,
        "mean_stack": mean_stack,
        "top_exprs": top_exprs,
        "top_expr_feature_modes": top_modes,
        "top_expr_scales": top_scales,
        "top_expr_biases": top_biases,
        "rest_regressor": rest_regressor,
        "n_top": len(top_exprs),
        "n_features": X_train.shape[1],
    }
    meta = {"expressions": all_discovered, "target_indices": top_targets}
    return model, meta


def train_one(spec: ApproachSpec, train_sims: list[dict[str, Any]], val_sims: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    X_train = build_feature_matrix(train_sims, spec.feature_mode)
    X_val = build_feature_matrix(val_sims, spec.feature_mode)
    Y_train = reverse_if_needed(resample_sims(train_sims), spec.time_mode)
    Y_val = reverse_if_needed(resample_sims(val_sims), spec.time_mode)
    Y_train_stack = stack_complex(Y_train)
    Y_val_stack = stack_complex(Y_val)
    if spec.family == "svd":
        mean_stack, basis, coeffs = svd_decomposition(Y_train_stack, spec.n_components)
        if spec.name == "knn_correction_spherical":
            primary = fit_column_models(X_train, coeffs, lambda X, y: fit_regressor("knn", X, y))
            primary_train = predict_column_models(primary, X_train)
            residual = coeffs - primary_train
            residual_model = fit_regressor("ridge", X_train, residual, degree=2)
            model = {
                "kind": "knn_correction",
                "feature_mode": spec.feature_mode,
                "time_mode": spec.time_mode,
                "stacked_complex": True,
                "basis": basis,
                "mean_stack": mean_stack,
                "primary_models": primary,
                "residual_model": residual_model,
                "n_components": spec.n_components,
            }
            meta = {}
            model["runtime_ms_train"] = (time.perf_counter() - t0) * 1000.0
            return model, meta, {"X_train": X_train, "X_val": X_val, "Y_train_stack": Y_train_stack, "Y_val_stack": Y_val_stack, "coeffs": coeffs}
        else:
            if spec.regressor == "rbf_interp":
                models = [
                    RBFInterpolator(
                        X_train,
                        coeffs[:, j],
                        neighbors=min(32, len(X_train)),
                        smoothing=1e-6,
                        kernel="thin_plate_spline",
                    )
                    for j in range(coeffs.shape[1])
                ]
            else:
                maker = lambda X, y: fit_regressor(spec.regressor, X, y, degree=spec.degree)
                models = fit_column_models(X_train, coeffs, maker)
        model = {
            "kind": "svd",
            "feature_mode": spec.feature_mode,
            "time_mode": spec.time_mode,
            "stacked_complex": True,
            "basis": basis,
            "mean_stack": mean_stack,
            "regressor": models,
            "n_components": spec.n_components,
        }
        meta = {}
    elif spec.family == "eim":
        mean_stack, basis, coeffs = svd_decomposition(Y_train_stack, spec.n_components)
        nodes, inv = eim_setup(basis)
        node_vals = Y_train_stack[:, nodes] - mean_stack[nodes]
        models = fit_column_models(X_train, node_vals, lambda X, y: fit_regressor(spec.regressor, X, y, degree=spec.degree))
        model = {
            "kind": "eim",
            "feature_mode": spec.feature_mode,
            "time_mode": spec.time_mode,
            "stacked_complex": True,
            "basis": basis,
            "mean_stack": mean_stack,
            "regressor": models,
            "eim_nodes": nodes,
            "eim_inverse": inv,
            "n_components": spec.n_components,
        }
        meta = {}
    elif spec.family == "ap_svd":
        amp_mean, phase_mean, amp_basis, phase_basis, amp_coeffs, phase_coeffs = amp_phase_decomposition(Y_train, spec.n_components, spec.time_mode)
        amp_models = fit_column_models(X_train, amp_coeffs, lambda X, y: fit_regressor(spec.regressor, X, y, degree=spec.degree))
        phase_models = fit_column_models(X_train, phase_coeffs, lambda X, y: fit_regressor(spec.regressor, X, y, degree=spec.degree))
        model = {
            "kind": "ap_svd",
            "feature_mode": spec.feature_mode,
            "time_mode": spec.time_mode,
            "stacked_complex": False,
            "amp_mean": amp_mean,
            "phase_mean": phase_mean,
            "amp_basis": amp_basis,
            "phase_basis": phase_basis,
            "amp_regressor": amp_models,
            "phase_regressor": phase_models,
            "n_components": spec.n_components,
        }
        meta = {}
    elif spec.family == "symbolic":
        mean_stack, basis, coeffs = svd_decomposition(Y_train_stack, spec.n_components)
        model, meta = train_symbolic_mix(spec, train_sims, X_train, Y_train_stack, basis, mean_stack, coeffs)
    else:
        raise ValueError(f"unknown family {spec.family!r}")

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    model["runtime_ms_train"] = runtime_ms
    return model, meta, {"X_train": X_train, "X_val": X_val, "Y_train_stack": Y_train_stack, "Y_val_stack": Y_val_stack, "coeffs": coeffs if "coeffs" in locals() else None}


def predict_many(model: dict[str, Any], sims: list[dict[str, Any]]) -> np.ndarray:
    if model["kind"] == "knn_correction":
        X = build_feature_matrix(sims, model["feature_mode"])
        primary = predict_column_models(model["primary_models"], X)
        residual = np.asarray(model["residual_model"].predict(X))
        if residual.ndim == 1:
            residual = residual[:, None]
        coeffs = primary + residual
        stack = coeffs @ model["basis"] + model["mean_stack"]
        stack = stack[:, ::-1] if model.get("time_mode") == "reverse" else stack
        return unstack_complex(stack)
    preds = []
    from _lib import predict_from_model

    for sim in sims:
        preds.append(np.asarray(predict_from_model(model, sim)))
    return np.stack(preds, axis=0)


def evaluate_split(model: dict[str, Any], sims: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, float]]:
    preds = predict_many(model, sims)
    truths = resample_sims(sims)
    mean_losses = []
    per_mass = {40: [], 80: [], 120: [], 160: [], 200: []}
    for pred, truth in zip(preds, truths, strict=True):
        loss, comp = eval_fd_mismatch(pred, truth)
        mean_losses.append(loss)
        for mass in per_mass:
            per_mass[mass].append(comp[f"mismatch_{mass}Msun"])
    mass_means = {f"mismatch_{mass}Msun": float(np.mean(vals)) for mass, vals in per_mass.items()}
    return np.asarray(mean_losses, dtype=np.float64), mass_means


def count_numeric_entries(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return int(obj.size)
    if isinstance(obj, (list, tuple)):
        return sum(count_numeric_entries(x) for x in obj)
    if isinstance(obj, dict):
        return sum(count_numeric_entries(v) for v in obj.values())
    if isinstance(obj, (float, int, np.floating, np.integer)):
        return 1
    return 0


def summarize_model(model: dict[str, Any]) -> int:
    try:
        return int(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))
    except Exception:
        total = count_numeric_entries(model)
        for key in ("regressor", "amp_regressor", "phase_regressor", "primary_models"):
            if key in model:
                total += count_numeric_entries(model[key])
        return int(total)


def write_model_artifacts(spec: ApproachSpec, model: dict[str, Any], scorecard: dict[str, Any], meta: dict[str, Any]) -> Path:
    mdir = model_dir(spec.number, spec.name)
    write_train_predict(mdir, spec.name)
    model_to_save = dict(model)
    model_to_save.pop("_compiled_exprs", None)
    with open(mdir / "saved_model" / "model.pkl", "wb") as f:
        pickle.dump(model_to_save, f)
    if meta.get("expressions") is not None:
        save_json(mdir / "saved_model" / "expressions.json", meta["expressions"])
    save_scorecard(mdir, scorecard)
    return mdir


def append_changelog(entry: str) -> None:
    if not CHANGELOG.exists():
        CHANGELOG.write_text("# Waveform Bench Changelog\n\n")
    with open(CHANGELOG, "a") as f:
        f.write(entry)
        if not entry.endswith("\n"):
            f.write("\n")
        f.write("\n")


def plot_progress(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    import matplotlib.pyplot as plt

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1, aspect=0.9))
    xs = np.arange(1, len(records) + 1)
    ys = np.array([rec["loss"] for rec in records], dtype=float)
    best = np.minimum.accumulate(ys)
    colors = [plot_settings.COLORS["blue"], plot_settings.COLORS["red"], plot_settings.COLORS["orange"], plot_settings.COLORS["green"], plot_settings.COLORS["purple"]]
    cat_map = {}
    for rec in records:
        cat_map.setdefault(rec["category"], colors[len(cat_map) % len(colors)])
    for x, y, rec in zip(xs, ys, records, strict=True):
        ax.scatter(x, y, color=cat_map[rec["category"]], s=24, zorder=3)
        ax.text(x + 0.03, y, rec["short_name"], fontsize=5, rotation=25)
    ax.plot(xs, best, color="black", lw=1.0, label="Best so far")
    ax.set_xlabel("Completed approach")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(COMPARISON_DIR / f"progress.{ext}")
    plt.close(fig)


def plot_loss_only(records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(records, key=lambda r: r["loss"])
    labels = [r["short_name"] for r in ordered]
    vals = [r["loss"] for r in ordered]
    colors = [plot_settings.COLORS["blue"] if r["category"] == "SVD/decomposition" else plot_settings.COLORS["red"] if r["category"] == "Symbolic/analytical" else plot_settings.COLORS["orange"] if r["category"] == "Interpolation/kernel" else plot_settings.COLORS["green"] for r in ordered]
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(COMPARISON_DIR / f"loss_only_comparison.{ext}")
    plt.close(fig)


def plot_pareto(records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.65))
    colors = {
        "SVD/decomposition": plot_settings.COLORS["blue"],
        "Symbolic/analytical": plot_settings.COLORS["red"],
        "Interpolation/kernel": plot_settings.COLORS["orange"],
        "Machine learning": plot_settings.COLORS["green"],
    }
    for rec in records:
        ax.scatter(rec["runtime_s"], rec["loss"], color=colors[rec["category"]], s=28, alpha=0.9)
        ax.text(rec["runtime_s"] * 1.02, rec["loss"], rec["short_name"], fontsize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime [s]")
    ax.set_ylabel("Validation loss")
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=handles, fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(COMPARISON_DIR / f"pareto_accuracy_speed.{ext}")
    plt.close(fig)


def plot_error_histograms(error_data: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    records = error_data["records"]
    n = len(records)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=plot_settings.figsize(cols=2, aspect=0.22 * rows))
    axes = np.asarray(axes).reshape(-1)
    nr_floor = float(error_data["nr_floor"])
    for ax, rec in zip(axes, records, strict=False):
        train = np.asarray(rec["train_errors"], dtype=float)
        val = np.asarray(rec["val_errors"], dtype=float)
        lo = max(min(np.min(train), np.min(val)), 1e-6)
        hi = max(np.max(train), np.max(val))
        bins = np.logspace(np.log10(lo), np.log10(hi), 24) if hi > lo else 12
        ax.hist(train, bins=bins, alpha=0.55, color=plot_settings.COLORS["blue"], label="train", histtype="stepfilled", hatch="//", edgecolor=plot_settings.COLORS["blue"], linewidth=0.4)
        ax.hist(val, bins=bins, alpha=0.55, color=plot_settings.COLORS["red"], label="val", histtype="stepfilled", hatch="\\\\", edgecolor=plot_settings.COLORS["red"], linewidth=0.4)
        ax.axvline(nr_floor, color="black", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_title(rec["short_name"], fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(COMPARISON_DIR / f"error_histograms.{ext}")
    plt.close(fig)


def refresh_comparison_outputs(records: list[dict[str, Any]], error_records: list[dict[str, Any]]) -> None:
    ordered = sorted(records, key=lambda r: r["loss"])
    save_json(COMPARISON_DIR / "summary_table.json", ordered)
    if ordered:
        save_json(COMPARISON_DIR / "best_model.json", ordered[0])
    error_payload = {
        "nr_floor": 1.4e-3,
        "records": error_records,
    }
    save_json(COMPARISON_DIR / "error_data.json", error_payload)
    plot_progress(records)
    if len(records) >= 2:
        plot_loss_only(records)
        plot_pareto(records)
    if len(error_records) >= 1:
        plot_error_histograms(error_payload)


def build_scorecard(spec: ApproachSpec, model: dict[str, Any], train_stats: tuple[np.ndarray, dict[str, float]], val_stats: tuple[np.ndarray, dict[str, float]], runtime_ms: float) -> dict[str, Any]:
    train_errors, train_mass = train_stats
    val_errors, val_mass = val_stats
    loss = float(np.mean(val_errors))
    return {
        "approach": spec.name,
        "approach_number": spec.number,
        "benchmark": "waveform",
        "agent": "gpt54_mini",
        "category": spec.category,
        "parameterization": spec.feature_mode,
        "time_convention": spec.time_mode,
        "loss": loss,
        "loss_components": val_mass,
        "train_loss": float(np.mean(train_errors)),
        "runtime_ms": float(runtime_ms),
        "n_train": len(train_errors),
        "n_val": len(val_errors),
        "n_params": summarize_model(model),
        "notes": spec.notes,
    }


def retrain_named_model(name: str) -> dict[str, Any]:
    train_sims, val_sims = load_data()
    spec = next(s for s in SPECS if s.name == name)
    model, meta, _ = train_one(spec, train_sims, val_sims)
    train_stats = evaluate_split(model, train_sims)
    val_stats = evaluate_split(model, val_sims)
    scorecard = build_scorecard(spec, model, train_stats, val_stats, model["runtime_ms_train"])
    mdir = write_model_artifacts(spec, model, scorecard, meta)
    return {"scorecard": scorecard, "model_dir": str(mdir)}


def train_and_record(spec: ApproachSpec, train_sims: list[dict[str, Any]], val_sims: list[dict[str, Any]], records: list[dict[str, Any]], error_records: list[dict[str, Any]]) -> None:
    model, meta, _ = train_one(spec, train_sims, val_sims)
    train_errors, train_mass = evaluate_split(model, train_sims)
    val_errors, val_mass = evaluate_split(model, val_sims)
    runtime_ms = float(model["runtime_ms_train"])
    scorecard = build_scorecard(spec, model, (train_errors, train_mass), (val_errors, val_mass), runtime_ms)
    mdir = write_model_artifacts(spec, model, scorecard, meta)
    short_name = spec.name.replace("_", " ")
    record = {
        "approach": spec.name,
        "short_name": short_name,
        "category": spec.category,
        "parameterization": spec.feature_mode,
        "time_convention": spec.time_mode,
        "loss": scorecard["loss"],
        "train_loss": scorecard["train_loss"],
        "runtime_s": runtime_ms / 1000.0,
        "runtime_ms": runtime_ms,
        "model_dir": str(mdir),
    }
    records.append(record)
    error_records.append(
        {
            "approach": spec.name,
            "short_name": short_name,
            "train_errors": train_errors.tolist(),
            "val_errors": val_errors.tolist(),
            "train_components": {k: v for k, v in train_mass.items()},
            "val_components": {k: v for k, v in val_mass.items()},
        }
    )
    append_changelog(
        f"### {spec.number:02d}. {spec.name}\n"
        f"- Observed: train loss {scorecard['train_loss']:.4e}, val loss {scorecard['loss']:.4e}.\n"
        f"- Hypothesis: {spec.notes}\n"
        f"- Change: trained {spec.family} model with feature mode `{spec.feature_mode}` and time mode `{spec.time_mode}`.\n"
        f"- Result: saved to `{mdir}`."
    )
    refresh_comparison_outputs(records, error_records)


def verify_completion(records: list[dict[str, Any]], error_records: list[dict[str, Any]]) -> None:
    assert len(records) >= 20, "need at least 20 completed approaches"
    modes = {r["parameterization"] for r in records}
    assert len(modes) >= 3, "need at least 3 parameterizations"
    cats = {r["category"] for r in records}
    assert cats == {"SVD/decomposition", "Symbolic/analytical", "Interpolation/kernel", "Machine learning"}, cats
    for spec in SPECS:
        mdir = MODELS_DIR / f"{spec.number:02d}_{spec.name}"
        assert (mdir / "train.py").exists()
        assert (mdir / "predict.py").exists()
        assert (mdir / "saved_model").exists()
        assert (mdir / "scorecard.json").exists()
    pysr_dir = MODELS_DIR / "19_pysr_svd_eff" / "saved_model" / "expressions.json"
    gpl_dir = MODELS_DIR / "20_gplearn_svd_spherical" / "saved_model" / "expressions.json"
    assert pysr_dir.exists(), "PySR expressions missing"
    assert gpl_dir.exists(), "gplearn expressions missing"
    assert (COMPARISON_DIR / "progress.png").exists()
    assert (COMPARISON_DIR / "progress.pdf").exists()
    assert (COMPARISON_DIR / "error_histograms.png").exists()
    assert (COMPARISON_DIR / "error_histograms.pdf").exists()
    assert (COMPARISON_DIR / "loss_only_comparison.png").exists()
    assert (COMPARISON_DIR / "loss_only_comparison.pdf").exists()
    assert (COMPARISON_DIR / "pareto_accuracy_speed.png").exists()
    assert (COMPARISON_DIR / "pareto_accuracy_speed.pdf").exists()
    assert (COMPARISON_DIR / "summary_table.json").exists()
    assert (COMPARISON_DIR / "error_data.json").exists()
    assert (COMPARISON_DIR / "best_model.json").exists()
    assert len(error_records) == len(records)


def main() -> None:
    set_seed()
    train_sims, val_sims = load_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    CHANGELOG.write_text("# Waveform Bench Changelog\n\n")
    records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []

    for spec in SPECS:
        print(f"[waveform] training {spec.number:02d}/{len(SPECS)} {spec.name}")
        train_and_record(spec, train_sims, val_sims, records, error_records)
        print(f"[waveform] done {spec.name}")

    verify_completion(records, error_records)
    refresh_comparison_outputs(records, error_records)
    print("WAVEFORM_BENCH_COMPLETE")


if __name__ == "__main__":
    main()
