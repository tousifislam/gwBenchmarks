from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import h5py
import joblib
import numpy as np

# Ensure repo root (containing gwbenchmarks/) is importable when run from nested results.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_gpt52")

from gwbenchmarks import plot_settings
from gwbenchmarks.metrics import rms_relative_error


TimeConvention = Literal["normalized_time", "t0_at_end", "t0_at_start"]
Parameterization = Literal[
    "raw_6d",
    "eff_loge0_6d",
    "trig_anomaly_7d",
    "log_omega_6d",
    "fully_transformed_7d",
]


@dataclasses.dataclass(frozen=True)
class DynamicsSplit:
    params_raw: np.ndarray  # (N, 6): q, chi1z, chi2z, e0, zeta0, omega0
    x: np.ndarray  # (N, T)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _eta_from_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (1.0 + q) ** 2


def _chi_eff(q: np.ndarray, chi1z: np.ndarray, chi2z: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    m1 = q / (1.0 + q)
    m2 = 1.0 / (1.0 + q)
    return m1 * np.asarray(chi1z) + m2 * np.asarray(chi2z)


def _chi_a(chi1z: np.ndarray, chi2z: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(chi1z) - np.asarray(chi2z))


def parameterize(params_raw: np.ndarray, kind: Parameterization) -> np.ndarray:
    p = np.asarray(params_raw, dtype=np.float64)
    q, chi1z, chi2z, e0, zeta0, omega0 = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 5]
    if kind == "raw_6d":
        return p.copy()
    eta = _eta_from_q(q)
    ceff = _chi_eff(q, chi1z, chi2z)
    ca = _chi_a(chi1z, chi2z)
    if kind == "eff_loge0_6d":
        return np.stack([eta, ceff, ca, np.log(np.maximum(e0, 1e-6)), zeta0, omega0], axis=1)
    if kind == "trig_anomaly_7d":
        return np.stack([eta, ceff, ca, e0, np.cos(zeta0), np.sin(zeta0), omega0], axis=1)
    if kind == "log_omega_6d":
        return np.stack([eta, ceff, ca, e0, zeta0, np.log(np.maximum(omega0, 1e-12))], axis=1)
    if kind == "fully_transformed_7d":
        return np.stack([eta, ceff, ca, np.log(np.maximum(e0, 1e-6)), np.cos(zeta0), np.sin(zeta0), np.log(np.maximum(omega0, 1e-12))], axis=1)
    raise ValueError(f"Unknown parameterization: {kind}")


def _common_min_duration(path: Path) -> float:
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        durs = []
        for i in range(n):
            t = np.asarray(f[f"sim_{i:04d}"]["t"][:], dtype=np.float64)
            durs.append(float(t[-1] - t[0]))
    return float(np.min(durs))


def load_dynamics_split(
    path: str | Path,
    *,
    n_time: int = 512,
    time_convention: TimeConvention = "normalized_time",
    min_duration: float | None = None,
) -> DynamicsSplit:
    path = Path(path)
    if time_convention in ("t0_at_end", "t0_at_start") and min_duration is None:
        min_duration = _common_min_duration(path)

    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        params = np.zeros((n, 6), dtype=np.float64)
        x_out = np.zeros((n, n_time), dtype=np.float64)

        for i in range(n):
            g = f[f"sim_{i:04d}"]
            q = float(g.attrs["q"])
            chi1z = float(g.attrs["chi1z"])
            chi2z = float(g.attrs["chi2z"])
            e0 = float(g.attrs["e0"])
            zeta0 = float(g.attrs["zeta0"])
            omega0 = float(g.attrs["omega0"])
            params[i, :] = np.array([q, chi1z, chi2z, e0, zeta0, omega0], dtype=np.float64)

            t = np.asarray(g["t"][:], dtype=np.float64)
            x = np.asarray(g["x"][:], dtype=np.float64)

            if time_convention == "normalized_time":
                tau = (t - t[0]) / (t[-1] - t[0] + 1e-12)
                grid = np.linspace(0.0, 1.0, n_time)
                x_out[i, :] = np.interp(grid, tau, x)
            elif time_convention == "t0_at_end":
                assert min_duration is not None
                t_rel = t - t[-1]  # end at 0
                grid = np.linspace(-min_duration, 0.0, n_time)
                x_out[i, :] = np.interp(grid, t_rel, x)
            elif time_convention == "t0_at_start":
                assert min_duration is not None
                t_rel = t - t[0]
                grid = np.linspace(0.0, min_duration, n_time)
                x_out[i, :] = np.interp(grid, t_rel, x)
            else:
                raise ValueError(f"Unknown time_convention: {time_convention}")

    return DynamicsSplit(params_raw=params, x=x_out)


def randomized_svd_basis(W: np.ndarray, n_components: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.utils.extmath import randomized_svd

    W = np.asarray(W, dtype=np.float32)
    mean = W.mean(axis=0, keepdims=True)
    Wc = W - mean
    U, S, Vt = randomized_svd(Wc, n_components=n_components, random_state=seed)
    components = (S[:, None] * Vt).astype(np.float32, copy=False)
    return mean.astype(np.float32).reshape(-1), components


def project_svd(W: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    Wc = np.asarray(W, dtype=np.float32) - mean[None, :]
    gram = components @ components.T
    coeffs = (Wc @ components.T) @ np.linalg.pinv(gram)
    return coeffs.astype(np.float32, copy=False)


def reconstruct_svd(coeffs: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return coeffs @ components + mean[None, :]


def fit_regressor(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kind: str,
    seed: int = 0,
    hyperparams: dict[str, Any] | None = None,
) -> Any:
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    hyperparams = dict(hyperparams or {})

    if kind == "ridge":
        from sklearn.linear_model import Ridge

        base = Ridge(random_state=seed, alpha=float(hyperparams.get("alpha", 1e-2)))
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "lasso":
        from sklearn.linear_model import Lasso

        base = Lasso(random_state=seed, alpha=float(hyperparams.get("alpha", 1e-4)), max_iter=5000)
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "elasticnet":
        from sklearn.linear_model import ElasticNet

        base = ElasticNet(random_state=seed, alpha=float(hyperparams.get("alpha", 2e-4)), l1_ratio=float(hyperparams.get("l1_ratio", 0.3)), max_iter=5000)
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "poly_ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures

        deg = int(hyperparams.get("degree", 3))
        base = Ridge(random_state=seed, alpha=float(hyperparams.get("alpha", 1e-3)))
        est = Pipeline([("poly", PolynomialFeatures(degree=deg, include_bias=False)), ("scaler", StandardScaler(with_mean=False)), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "krr_rbf":
        from sklearn.kernel_ridge import KernelRidge

        base = KernelRidge(alpha=float(hyperparams.get("alpha", 1e-3)), kernel="rbf", gamma=hyperparams.get("gamma", None))
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "svr_rbf":
        from sklearn.svm import SVR

        base = SVR(C=float(hyperparams.get("C", 10.0)), epsilon=float(hyperparams.get("epsilon", 1e-3)), gamma=hyperparams.get("gamma", "scale"))
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "gpr_rbf":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

        length = float(hyperparams.get("length_scale", 1.0))
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        base = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed, n_restarts_optimizer=int(hyperparams.get("n_restarts", 0)))
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "gpr_matern":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

        length = float(hyperparams.get("length_scale", 1.0))
        nu = float(hyperparams.get("nu", 2.5))
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=length, nu=nu, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        base = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed, n_restarts_optimizer=int(hyperparams.get("n_restarts", 0)))
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "rbf_interp":
        from scipy.interpolate import RBFInterpolator

        return RBFInterpolator(X, Y, kernel=str(hyperparams.get("kernel", "thin_plate_spline")), neighbors=int(hyperparams.get("neighbors", min(60, len(X)))), smoothing=float(hyperparams.get("smoothing", 0.0)))

    if kind == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        base = KNeighborsRegressor(n_neighbors=int(hyperparams.get("n_neighbors", 7)), weights=str(hyperparams.get("weights", "distance")))
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=int(hyperparams.get("n_estimators", 500)), random_state=seed, n_jobs=-1)

    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(n_estimators=int(hyperparams.get("n_estimators", 800)), random_state=seed, n_jobs=-1)

    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        base = HistGradientBoostingRegressor(random_state=seed, max_depth=int(hyperparams.get("max_depth", 6)), learning_rate=float(hyperparams.get("learning_rate", 0.05)), max_iter=int(hyperparams.get("max_iter", 600)))
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor

        base = MLPRegressor(random_state=seed, hidden_layer_sizes=tuple(hyperparams.get("hidden_layer_sizes", (256, 256))), alpha=float(hyperparams.get("alpha", 1e-5)), learning_rate_init=float(hyperparams.get("learning_rate_init", 2e-3)), max_iter=int(hyperparams.get("max_iter", 1500)), early_stopping=True)
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    raise ValueError(f"Unknown regressor kind: {kind}")


def predict_regressor(est: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict"):
        return np.asarray(est.predict(X))
    if callable(est):
        return np.asarray(est(X))
    raise TypeError(f"Estimator has no predict(): {type(est)}")


def _eim_nodes(U: np.ndarray, k: int) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    Q, _ = np.linalg.qr(U[:, :k])
    nodes: list[int] = []
    for j in range(k):
        v = Q[:, j].copy()
        if nodes:
            A = Q[nodes, :j]
            b = v[nodes]
            if A.size:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
                v = v - Q[:, :j] @ coeff
        nodes.append(int(np.argmax(np.abs(v))))
    return np.asarray(nodes, dtype=np.int64)


def compute_per_sample_losses(x_pred: np.ndarray, x_true: np.ndarray) -> tuple[np.ndarray, float]:
    n = x_true.shape[0]
    losses = np.zeros(n, dtype=np.float64)
    for i in range(n):
        losses[i] = float(rms_relative_error(x_pred[i], x_true[i]))
    return losses, float(np.mean(losses))


def update_dynamics_comparison(work_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_settings.apply()
    models_dir = work_dir / "models"
    comp_dir = work_dir / "comparison"
    _ensure_dir(comp_dir)

    model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
    rows: list[dict[str, Any]] = []
    error_data: dict[str, Any] = {}
    for md in model_dirs:
        sc_path = md / "scorecard.json"
        if not sc_path.exists():
            continue
        sc = json.loads(sc_path.read_text())
        rows.append(sc)
        err_path = md / "saved_model" / "per_sample_errors.json"
        if err_path.exists():
            error_data[sc.get("display_name", md.name)] = json.loads(err_path.read_text())

    rows_sorted = sorted(rows, key=lambda r: float(r.get("loss", math.inf)))
    _json_dump(comp_dir / "summary_table.json", rows_sorted)
    _json_dump(comp_dir / "error_data.json", error_data)
    if rows_sorted:
        _json_dump(comp_dir / "best_model.json", rows_sorted[0])
    if not rows:
        return

    labels = [r.get("display_name", r.get("approach", "")) for r in rows]
    losses = np.asarray([float(r["loss"]) for r in rows], dtype=float)
    runtimes = np.asarray([float(r.get("runtime_ms", math.nan)) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    x = np.arange(len(losses))
    ax.scatter(x, losses, s=18, color=plot_settings.COLORS["blue"])
    ax.set_yscale("log")
    ax.set_xlabel("Approach #")
    ax.set_ylabel("Loss (RMS rel. err)")
    for i, lab in enumerate(labels):
        ax.text(i, losses[i], str(i + 1), fontsize=7, ha="center", va="bottom")
    fig.savefig(comp_dir / "loss_only_comparison.png")
    fig.savefig(comp_dir / "loss_only_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    ax.scatter(np.clip(runtimes, 1e-6, np.inf), losses, s=25, color=plot_settings.COLORS["red"], alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Eval time per sample (ms)")
    ax.set_ylabel("Loss (RMS rel. err)")
    for rt, lo, lab in zip(runtimes, losses, labels):
        ax.text(max(rt, 1e-6), lo, lab.split(" (")[0], fontsize=6, ha="left", va="bottom")
    fig.savefig(comp_dir / "pareto_accuracy_speed.png")
    fig.savefig(comp_dir / "pareto_accuracy_speed.pdf")
    plt.close(fig)

    order = sorted(rows, key=lambda r: int(r.get("approach_number", 9999)))
    loss_o = np.asarray([float(r["loss"]) for r in order], dtype=float)
    rt_o = np.asarray([float(r.get("runtime_ms", math.nan)) for r in order], dtype=float)
    xs = np.arange(1, len(order) + 1)
    fig, ax1 = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    ax1.plot(xs, loss_o, marker="o", ms=3, lw=1.0, color=plot_settings.COLORS["blue"])
    ax1.set_yscale("log")
    ax1.set_xlabel("Approach #")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(xs, np.clip(rt_o, 1e-6, np.inf), marker="s", ms=3, lw=1.0, color=plot_settings.COLORS["orange"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime (ms)")
    fig.savefig(comp_dir / "progress.png")
    fig.savefig(comp_dir / "progress.pdf")
    plt.close(fig)

    if error_data:
        names = list(error_data.keys())
        n = len(names)
        ncols = 4
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(plot_settings.DOUBLE_COL, 0.9 * nrows), squeeze=False)
        bins = np.logspace(-6, 0, 35)
        for j, name in enumerate(names):
            ax = axes[j // ncols][j % ncols]
            d = error_data[name]
            tr = np.asarray(d.get("train", []), dtype=float)
            va = np.asarray(d.get("val", []), dtype=float)
            if tr.size:
                ax.hist(tr, bins=bins, alpha=0.5, color=plot_settings.COLORS["blue"], label="train")
            if va.size:
                ax.hist(va, bins=bins, alpha=0.35, color=plot_settings.COLORS["red"], label="val", hatch="///")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Per-sample loss")
            ax.set_ylabel("Count")
            ax.text(0.02, 0.98, name, transform=ax.transAxes, ha="left", va="top", fontsize=6)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.savefig(comp_dir / "error_histograms.png")
        fig.savefig(comp_dir / "error_histograms.pdf")
        plt.close(fig)


def append_changelog_entry(agent_root: Path, entry_md: str) -> None:
    ch = agent_root / "CHANGELOG.md"
    _ensure_dir(ch.parent)
    with open(ch, "a") as f:
        if not entry_md.endswith("\n"):
            entry_md += "\n"
        f.write(entry_md)


def train_and_score_svd_regressor(
    *,
    work_dir: Path,
    agent_root: Path,
    model_dir: Path,
    approach_number: int,
    display_name: str,
    category: str,
    regressor_kind: str,
    regressor_hyperparams: dict[str, Any] | None,
    parameterization: Parameterization,
    time_convention: TimeConvention,
    n_svd: int,
    seed: int = 0,
    eim: bool = False,
) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    min_dur = _common_min_duration(Path("datasets/dynamics/dynamics_training.h5")) if time_convention != "normalized_time" else None
    train = load_dynamics_split("datasets/dynamics/dynamics_training.h5", time_convention=time_convention, min_duration=min_dur)
    val = load_dynamics_split("datasets/dynamics/dynamics_validation.h5", time_convention=time_convention, min_duration=min_dur)

    X_train = parameterize(train.params_raw, parameterization)
    X_val = parameterize(val.params_raw, parameterization)
    W_train = train.x.astype(np.float32)
    W_val = val.x.astype(np.float32)

    mean, components = randomized_svd_basis(W_train, n_components=n_svd, seed=seed)
    Y_train = project_svd(W_train, mean, components)

    if eim:
        U = components.T  # (T,K)
        nodes = _eim_nodes(U, n_svd)
        U_nodes_inv = np.linalg.pinv(U[nodes, :]).astype(np.float32)
        Wn_train = W_train[:, nodes]
        est = fit_regressor(X_train, Wn_train, kind=regressor_kind, seed=seed, hyperparams=regressor_hyperparams)
        if hasattr(est, "fit"):
            est.fit(X_train, Wn_train)
        # Reasoned tweak: regularization if overfitting.
        tr_pred0 = predict_regressor(est, X_train)[:, :]
        va_pred0 = predict_regressor(est, X_val)[:, :]
        # Reconstruct quick for loss estimates.
        coeff_tr0 = (tr_pred0 - mean[nodes][None, :]) @ U_nodes_inv.T
        coeff_va0 = (va_pred0 - mean[nodes][None, :]) @ U_nodes_inv.T
        Wtr0 = reconstruct_svd(coeff_tr0, mean, components)
        Wva0 = reconstruct_svd(coeff_va0, mean, components)
    else:
        est = fit_regressor(X_train, Y_train, kind=regressor_kind, seed=seed, hyperparams=regressor_hyperparams)
        if hasattr(est, "fit"):
            est.fit(X_train, Y_train)
        coeff_tr0 = predict_regressor(est, X_train).astype(np.float32)
        coeff_va0 = predict_regressor(est, X_val).astype(np.float32)
        Wtr0 = reconstruct_svd(coeff_tr0, mean, components)
        Wva0 = reconstruct_svd(coeff_va0, mean, components)

    tr_losses0, tr_mean0 = compute_per_sample_losses(Wtr0, W_train)
    va_losses0, va_mean0 = compute_per_sample_losses(Wva0, W_val)
    obs = f"train_loss={tr_mean0:.3e}, val_loss={va_mean0:.3e}"
    hypothesis = ""
    change = ""
    hp2 = dict(regressor_hyperparams or {})
    tuned = False
    if regressor_kind in {"ridge", "poly_ridge", "krr_rbf"} and va_mean0 > 1.25 * tr_mean0:
        hypothesis = "Validation loss suggests overfitting → increase regularization (alpha)."
        hp2["alpha"] = float(hp2.get("alpha", 1e-3)) * 10.0
        change = f"alpha -> {hp2['alpha']:.2e}"
        tuned = True
    if tuned:
        if eim:
            est = fit_regressor(X_train, Wn_train, kind=regressor_kind, seed=seed, hyperparams=hp2)
            if hasattr(est, "fit"):
                est.fit(X_train, Wn_train)
        else:
            est = fit_regressor(X_train, Y_train, kind=regressor_kind, seed=seed, hyperparams=hp2)
            if hasattr(est, "fit"):
                est.fit(X_train, Y_train)

    # Final eval and runtime.
    t0 = time.perf_counter()
    if eim:
        Wn_val = predict_regressor(est, X_val).astype(np.float32)
        coeff_val = (Wn_val - mean[nodes][None, :]) @ U_nodes_inv.T
        W_val_pred = reconstruct_svd(coeff_val, mean, components)
    else:
        coeff_val = predict_regressor(est, X_val).astype(np.float32)
        W_val_pred = reconstruct_svd(coeff_val, mean, components)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(X_val)

    if eim:
        Wn_tr = predict_regressor(est, X_train).astype(np.float32)
        coeff_tr = (Wn_tr - mean[nodes][None, :]) @ U_nodes_inv.T
        W_tr_pred = reconstruct_svd(coeff_tr, mean, components)
    else:
        coeff_tr = predict_regressor(est, X_train).astype(np.float32)
        W_tr_pred = reconstruct_svd(coeff_tr, mean, components)

    tr_losses, _ = compute_per_sample_losses(W_tr_pred, W_train)
    va_losses, loss = compute_per_sample_losses(W_val_pred, W_val)

    np.save(model_dir / "saved_model" / "svd_mean.npy", mean.astype(np.float32))
    np.save(model_dir / "saved_model" / "svd_components.npy", components.astype(np.float32))
    joblib.dump(est, model_dir / "saved_model" / "regressor.joblib")
    if eim:
        np.save(model_dir / "saved_model" / "eim_nodes.npy", nodes.astype(np.int64))
        np.save(model_dir / "saved_model" / "eim_U_nodes_inv.npy", U_nodes_inv.astype(np.float32))

    _json_dump(model_dir / "saved_model" / "config.json", {
        "benchmark": "dynamics",
        "display_name": display_name,
        "category": category,
        "parameterization": parameterization,
        "time_convention": time_convention,
        "n_svd": n_svd,
        "regressor_kind": regressor_kind,
        "regressor_hyperparams": hp2,
        "eim": bool(eim),
    })
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": tr_losses.tolist(), "val": va_losses.tolist()})

    scorecard = {
        "approach": regressor_kind if not eim else f"eim_{regressor_kind}",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "dynamics",
        "agent": "gpt52",
        "category": category,
        "parameterization": parameterization,
        "time_convention": time_convention,
        "loss": float(loss),
        "loss_components": {"rms_relative_error_x": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_params": 0,
        "notes": {"reasoned_optimization": {"observed": obs, "hypothesis": hypothesis, "change": change}},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    (model_dir / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    raise SystemExit('Use llm_agents/results/gpt52/dynamics/run_all.py or run_symbolic.py')\n"
    )
    (model_dir / "predict.py").write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import parameterize, reconstruct_svd  # noqa: E402


def predict(model_dir: str | Path, params_raw: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    mean = np.load(model_dir / "saved_model" / "svd_mean.npy")
    comps = np.load(model_dir / "saved_model" / "svd_components.npy")
    est = joblib.load(model_dir / "saved_model" / "regressor.joblib")
    eim = bool(cfg.get("eim", False))
    X = parameterize(params_raw, cfg["parameterization"])
    if eim:
        nodes = np.load(model_dir / "saved_model" / "eim_nodes.npy")
        inv = np.load(model_dir / "saved_model" / "eim_U_nodes_inv.npy")
        W_nodes = np.asarray(est.predict(X), dtype=np.float32)
        coeffs = (W_nodes - mean[nodes][None, :]) @ inv.T
    else:
        coeffs = np.asarray(est.predict(X), dtype=np.float32)
    x = reconstruct_svd(coeffs, mean, comps)
    return {"x": np.asarray(x, dtype=np.float64)}
"""
    )

    update_dynamics_comparison(work_dir)
    append_changelog_entry(
        agent_root,
        f"\n## [D-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_now_str()}\n"
        "- **Benchmark**: dynamics\n"
        f"- **Category**: {category}\n"
        f"- **Method**: {('EIM + ' if eim else 'SVD + ')}{regressor_kind}\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Time convention**: {time_convention}\n"
        f"- **Loss**: {loss:.4e}\n"
        f"- **Eval time**: {runtime_ms:.2f} ms\n"
        "- **Reasoned optimization**:\n"
        f"  - Observed: {obs}\n"
        f"  - Hypothesis: {hypothesis or 'N/A'}\n"
        f"  - Change: {change or 'N/A'}\n"
    )

