from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Tuple

import h5py
import joblib
import numpy as np

# Ensure repo root (containing gwbenchmarks/) is importable even when this file
# is imported from a nested results directory.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Ensure matplotlib cache dir is writable inside the sandbox.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_gpt52")

from gwbenchmarks.benchmarks.waveform import WaveformBench
from gwbenchmarks import plot_settings


TimeConvention = Literal["t0_at_peak", "t0_at_start", "reversed_time"]
Representation = Literal["real_imag", "amp_phase"]
Parameterization = Literal[
    "raw_7d",
    "effective_spins_7d",
    "spherical_spins_7d",
    "raw_plus_omega0_8d",
]


@dataclasses.dataclass(frozen=True)
class WaveformSplit:
    params_raw: np.ndarray  # (N, 8): q, chi1x..chi2z, omega0
    sxs_ids: list[str]
    dt: float
    t_min: float
    t_max: float
    h: np.ndarray  # (N, T) complex64/complex128


def _now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _count_params_sklearn(est: Any) -> int:
    # Best-effort: keep simple and deterministic.
    total = 0
    for attr in ("coef_", "intercept_", "dual_coef_", "support_vectors_"):
        if hasattr(est, attr):
            arr = getattr(est, attr)
            try:
                total += int(np.asarray(arr).size)
            except Exception:
                pass
    if hasattr(est, "estimators_"):
        try:
            total += sum(_count_params_sklearn(e) for e in est.estimators_)
        except Exception:
            pass
    if hasattr(est, "n_features_in_"):
        total += int(getattr(est, "n_features_in_", 0))
    return int(total)


def load_waveform_split(
    path: str | Path,
    *,
    t_min: float = -2847.0,
    t_max: float = 0.0,
    time_convention: TimeConvention = "t0_at_peak",
) -> WaveformSplit:
    """Load a split and crop all waveforms onto a common time window.

    Notes
    -----
    The curated dataset contains variable-length waveforms, but all simulations
    share the intersection interval [t_min, t_max] with uniform dt=0.1M.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        dt = float(f.attrs["dt_geometric"])
        n_t = int(round((t_max - t_min) / dt)) + 1

        params_raw = np.zeros((n, 8), dtype=np.float64)
        sxs_ids: list[str] = []
        h_out = np.zeros((n, n_t), dtype=np.complex64)

        for i in range(n):
            g = f[f"sim_{i:04d}"]
            sxs_ids.append(str(g.attrs["sxs_id"]))
            q = float(g.attrs["q"])
            chi1 = np.array([g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]], dtype=float)
            chi2 = np.array([g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]], dtype=float)
            omega0 = float(g.attrs.get("omega0", 0.0))
            params_raw[i, :] = np.array([q, *chi1.tolist(), *chi2.tolist(), omega0], dtype=float)

            t = np.asarray(g["t"][:], dtype=np.float64)
            h = np.asarray(g["h22_real"][:], dtype=np.float64) + 1j * np.asarray(g["h22_imag"][:], dtype=np.float64)

            idx0 = int(round((t_min - float(t[0])) / dt))
            idx1 = idx0 + n_t
            if idx0 < 0 or idx1 > len(t):
                raise ValueError(
                    f"{path.name}: sim_{i:04d} does not cover [{t_min}, {t_max}] "
                    f"(t0={t[0]:.1f}, t1={t[-1]:.1f}, needed n_t={n_t})."
                )
            seg = h[idx0:idx1].astype(np.complex64, copy=False)
            if time_convention == "reversed_time":
                seg = seg[::-1].copy()
            elif time_convention in ("t0_at_peak", "t0_at_start"):
                # Coordinate shift doesn't affect the stored arrays (the benchmark uses dt only).
                pass
            else:
                raise ValueError(f"Unknown time_convention: {time_convention}")
            h_out[i, :] = seg

    return WaveformSplit(
        params_raw=params_raw,
        sxs_ids=sxs_ids,
        dt=float(dt),
        t_min=float(t_min),
        t_max=float(t_max),
        h=h_out,
    )


def _eta_from_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (1.0 + q) ** 2


def _chi_eff(q: np.ndarray, chi1z: np.ndarray, chi2z: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    m1 = q / (1.0 + q)
    m2 = 1.0 / (1.0 + q)
    return m1 * np.asarray(chi1z) + m2 * np.asarray(chi2z)


def _chi_p(q: np.ndarray, chi1x: np.ndarray, chi1y: np.ndarray, chi2x: np.ndarray, chi2y: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    a1 = np.sqrt(np.asarray(chi1x) ** 2 + np.asarray(chi1y) ** 2)
    a2 = np.sqrt(np.asarray(chi2x) ** 2 + np.asarray(chi2y) ** 2)
    # Standard effective precession spin proxy (Schmidt et al.) up to a simple scaling.
    weight = (4.0 * q + 3.0) / (4.0 + 3.0 * q) * q
    return np.maximum(a1, weight * a2)


def parameterize(params_raw: np.ndarray, kind: Parameterization) -> np.ndarray:
    p = np.asarray(params_raw, dtype=np.float64)
    q = p[:, 0]
    chi1x, chi1y, chi1z = p[:, 1], p[:, 2], p[:, 3]
    chi2x, chi2y, chi2z = p[:, 4], p[:, 5], p[:, 6]
    omega0 = p[:, 7]

    if kind == "raw_7d":
        return p[:, :7].copy()

    if kind == "raw_plus_omega0_8d":
        return p[:, :8].copy()

    if kind == "effective_spins_7d":
        eta = _eta_from_q(q)
        ceff = _chi_eff(q, chi1z, chi2z)
        cp = _chi_p(q, chi1x, chi1y, chi2x, chi2y)
        chi1mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
        chi2mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
        theta1 = np.arccos(np.clip(np.where(chi1mag > 0, chi1z / chi1mag, 1.0), -1.0, 1.0))
        theta2 = np.arccos(np.clip(np.where(chi2mag > 0, chi2z / chi2mag, 1.0), -1.0, 1.0))
        return np.stack([eta, ceff, cp, chi1mag, chi2mag, theta1, theta2], axis=1)

    if kind == "spherical_spins_7d":
        eta = _eta_from_q(q)
        chi1mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
        chi2mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
        theta1 = np.arccos(np.clip(np.where(chi1mag > 0, chi1z / chi1mag, 1.0), -1.0, 1.0))
        theta2 = np.arccos(np.clip(np.where(chi2mag > 0, chi2z / chi2mag, 1.0), -1.0, 1.0))
        phi1 = np.arctan2(chi1y, chi1x)
        phi2 = np.arctan2(chi2y, chi2x)
        return np.stack([eta, chi1mag, theta1, phi1, chi2mag, theta2, phi2], axis=1)

    raise ValueError(f"Unknown parameterization: {kind}")


def _stack_real_imag(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h)
    return np.concatenate([h.real, h.imag], axis=1)


def _amp_phase(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h)
    amp = np.abs(h)
    phase = np.unwrap(np.angle(h), axis=1)
    # Stabilize amplitude dynamic range.
    log_amp = np.log(np.maximum(amp, 1e-12))
    return np.concatenate([log_amp, phase], axis=1)


def featurize_waveforms(h: np.ndarray, representation: Representation) -> np.ndarray:
    if representation == "real_imag":
        return _stack_real_imag(h)
    if representation == "amp_phase":
        return _amp_phase(h)
    raise ValueError(f"Unknown representation: {representation}")


def unfeaturize_waveforms(features: np.ndarray, representation: Representation) -> np.ndarray:
    features = np.asarray(features)
    n = features.shape[0]
    half = features.shape[1] // 2
    a = features[:, :half]
    b = features[:, half:]
    if representation == "real_imag":
        return a + 1j * b
    if representation == "amp_phase":
        amp = np.exp(a)
        return amp * (np.cos(b) + 1j * np.sin(b))
    raise ValueError(f"Unknown representation: {representation}")


def randomized_svd_basis(
    W: np.ndarray, n_components: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, components) for W ~ mean + coeffs @ components.

    components has shape (K, D).
    """
    from sklearn.utils.extmath import randomized_svd

    W = np.asarray(W, dtype=np.float32)
    mean = W.mean(axis=0, keepdims=True)
    Wc = W - mean
    # randomized_svd returns U (n,k), S(k,), Vt(k,d) so Wc ≈ U S Vt.
    U, S, Vt = randomized_svd(Wc, n_components=n_components, random_state=seed)
    components = (S[:, None] * Vt).astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False).reshape(-1), components


def project_svd(W: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    Wc = np.asarray(W, dtype=np.float32) - mean[None, :].astype(np.float32)
    # components = S * Vt (K,D), so coeffs = Wc @ components.T / (S^2)? but we baked S in.
    # Use least-squares against components rows (not orthonormal now). Compute pseudo-inverse once.
    C = components
    gram = C @ C.T
    coeffs = (Wc @ C.T) @ np.linalg.pinv(gram)
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
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "lasso":
        from sklearn.linear_model import Lasso

        base = Lasso(random_state=seed, alpha=float(hyperparams.get("alpha", 1e-4)), max_iter=5000)
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "elasticnet":
        from sklearn.linear_model import ElasticNet

        base = ElasticNet(
            random_state=seed,
            alpha=float(hyperparams.get("alpha", 1e-4)),
            l1_ratio=float(hyperparams.get("l1_ratio", 0.2)),
            max_iter=5000,
        )
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "poly_ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures

        deg = int(hyperparams.get("degree", 3))
        base = Ridge(random_state=seed, alpha=float(hyperparams.get("alpha", 1e-3)))
        est = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", base),
            ]
        )
        return MultiOutputRegressor(est)

    if kind == "krr_rbf":
        from sklearn.kernel_ridge import KernelRidge

        base = KernelRidge(alpha=float(hyperparams.get("alpha", 1e-3)), kernel="rbf", gamma=hyperparams.get("gamma", None))
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        base = KNeighborsRegressor(
            n_neighbors=int(hyperparams.get("n_neighbors", 7)),
            weights=str(hyperparams.get("weights", "distance")),
        )
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "rbf_interp":
        # SciPy RBFInterpolator supports multi-output directly.
        from scipy.interpolate import RBFInterpolator

        return RBFInterpolator(
            X,
            Y,
            kernel=str(hyperparams.get("kernel", "thin_plate_spline")),
            epsilon=hyperparams.get("epsilon", None),
            smoothing=float(hyperparams.get("smoothing", 0.0)),
            neighbors=int(hyperparams.get("neighbors", min(40, len(X)))),
        )

    if kind == "svr_rbf":
        from sklearn.svm import SVR

        base = SVR(C=float(hyperparams.get("C", 10.0)), epsilon=float(hyperparams.get("epsilon", 1e-3)), gamma=hyperparams.get("gamma", "scale"))
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "gpr_rbf":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

        length = float(hyperparams.get("length_scale", 1.0))
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        base = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed, n_restarts_optimizer=int(hyperparams.get("n_restarts", 0)))
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "gpr_matern":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

        length = float(hyperparams.get("length_scale", 1.0))
        nu = float(hyperparams.get("nu", 2.5))
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=length, nu=nu, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        base = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed, n_restarts_optimizer=int(hyperparams.get("n_restarts", 0)))
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])
        return MultiOutputRegressor(est)

    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        base = RandomForestRegressor(
            n_estimators=int(hyperparams.get("n_estimators", 400)),
            max_depth=hyperparams.get("max_depth", None),
            random_state=seed,
            n_jobs=-1,
        )
        return base

    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor

        base = ExtraTreesRegressor(
            n_estimators=int(hyperparams.get("n_estimators", 800)),
            max_depth=hyperparams.get("max_depth", None),
            random_state=seed,
            n_jobs=-1,
        )
        return base

    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        base = HistGradientBoostingRegressor(
            random_state=seed,
            max_depth=int(hyperparams.get("max_depth", 6)),
            learning_rate=float(hyperparams.get("learning_rate", 0.05)),
            max_iter=int(hyperparams.get("max_iter", 400)),
        )
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    if kind == "mlp_sklearn":
        from sklearn.neural_network import MLPRegressor

        base = MLPRegressor(
            random_state=seed,
            hidden_layer_sizes=tuple(hyperparams.get("hidden_layer_sizes", (256, 256))),
            alpha=float(hyperparams.get("alpha", 1e-5)),
            learning_rate_init=float(hyperparams.get("learning_rate_init", 2e-3)),
            max_iter=int(hyperparams.get("max_iter", 1500)),
            early_stopping=True,
        )
        return MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))

    raise ValueError(f"Unknown regressor kind: {kind}")


def predict_regressor(est: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict"):
        return np.asarray(est.predict(X))
    if callable(est):
        return np.asarray(est(X))
    raise TypeError(f"Estimator has no predict(): {type(est)}")


def compute_per_sample_losses(
    h_pred: np.ndarray,
    h_true: np.ndarray,
    *,
    dt: float,
) -> tuple[np.ndarray, dict[str, float]]:
    bench = WaveformBench()
    n = h_true.shape[0]
    losses = np.zeros(n, dtype=np.float64)
    comp_accum: dict[str, list[float]] = {}

    for i in range(n):
        loss_i, comps = bench.compute_loss(
            {
                "h22_real": np.asarray(h_pred[i].real, dtype=np.float64),
                "h22_imag": np.asarray(h_pred[i].imag, dtype=np.float64),
            },
            {
                "h22_real": np.asarray(h_true[i].real, dtype=np.float64),
                "h22_imag": np.asarray(h_true[i].imag, dtype=np.float64),
                "dt": np.array(dt, dtype=np.float64),
            },
        )
        losses[i] = float(loss_i)
        for k, v in comps.items():
            comp_accum.setdefault(k, []).append(float(v))

    comp_means = {k: float(np.mean(v)) for k, v in comp_accum.items()}
    return losses, comp_means


def update_waveform_comparison(work_dir: Path) -> None:
    """Regenerate comparison outputs from all model scorecards."""
    import matplotlib.pyplot as plt

    plot_settings.apply()
    models_dir = work_dir / "models"
    comp_dir = work_dir / "comparison"
    _ensure_dir(comp_dir)

    model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
    rows: list[dict[str, Any]] = []
    error_data: dict[str, Any] = {}

    for md in model_dirs:
        score_path = md / "scorecard.json"
        if not score_path.exists():
            continue
        with open(score_path) as f:
            sc = json.load(f)
        rows.append(sc)
        err_path = md / "saved_model" / "per_sample_errors.json"
        if err_path.exists():
            with open(err_path) as f:
                error_data[sc.get("display_name", md.name)] = json.load(f)

    rows_sorted = sorted(rows, key=lambda r: float(r.get("loss", math.inf)))
    _json_dump(comp_dir / "summary_table.json", rows_sorted)
    _json_dump(comp_dir / "error_data.json", error_data)
    if rows_sorted:
        best = rows_sorted[0]
        _json_dump(comp_dir / "best_model.json", best)

    # Loss-only plot
    if rows:
        labels = [r.get("display_name", r.get("approach", "")) for r in rows]
        losses = [float(r["loss"]) for r in rows]
        fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
        x = np.arange(len(losses))
        ax.scatter(x, losses, s=18, color=plot_settings.COLORS["blue"])
        ax.set_yscale("log")
        ax.set_xlabel("Approach #")
        ax.set_ylabel("Loss (mean FD mismatch)")
        for i, lab in enumerate(labels):
            ax.text(i, losses[i], str(i + 1), fontsize=7, ha="center", va="bottom")
        fig.savefig(comp_dir / "loss_only_comparison.png")
        fig.savefig(comp_dir / "loss_only_comparison.pdf")
        plt.close(fig)

        # Pareto: loss vs runtime
        runtimes = [float(r.get("runtime_ms", math.nan)) for r in rows]
        fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
        ax.scatter(runtimes, losses, s=25, color=plot_settings.COLORS["red"], alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Eval time per waveform (ms)")
        ax.set_ylabel("Loss (mean FD mismatch)")
        for rt, lo, lab in zip(runtimes, losses, labels):
            ax.text(rt, lo, lab.split(" (")[0], fontsize=6, ha="left", va="bottom")
        fig.savefig(comp_dir / "pareto_accuracy_speed.png")
        fig.savefig(comp_dir / "pareto_accuracy_speed.pdf")
        plt.close(fig)

        # Progress plot: loss + runtime vs approach number
        order = sorted(rows, key=lambda r: int(r.get("approach_number", 9999)))
        loss_o = [float(r["loss"]) for r in order]
        rt_o = [float(r.get("runtime_ms", math.nan)) for r in order]
        xs = np.arange(1, len(order) + 1)
        fig, ax1 = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
        ax1.plot(xs, loss_o, marker="o", ms=3, lw=1.0, color=plot_settings.COLORS["blue"], label="loss")
        ax1.set_yscale("log")
        ax1.set_xlabel("Approach #")
        ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        ax2.plot(xs, rt_o, marker="s", ms=3, lw=1.0, color=plot_settings.COLORS["orange"], label="runtime")
        ax2.set_yscale("log")
        ax2.set_ylabel("Runtime (ms)")
        fig.savefig(comp_dir / "progress.png")
        fig.savefig(comp_dir / "progress.pdf")
        plt.close(fig)

    # Error histogram grid (small multiples)
    if error_data:
        floor = 1.4e-3
        names = list(error_data.keys())
        n = len(names)
        ncols = 4
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(plot_settings.DOUBLE_COL, 0.9 * nrows),
            squeeze=False,
        )
        bins = np.logspace(-5, -1, 35)
        for j, name in enumerate(names):
            r = j // ncols
            c = j % ncols
            ax = axes[r][c]
            d = error_data[name]
            tr = np.asarray(d.get("train", []), dtype=float)
            va = np.asarray(d.get("val", []), dtype=float)
            if tr.size:
                ax.hist(tr, bins=bins, alpha=0.5, color=plot_settings.COLORS["blue"], label="train")
            if va.size:
                ax.hist(va, bins=bins, alpha=0.35, color=plot_settings.COLORS["red"], label="val", hatch="///")
            ax.axvline(floor, color=plot_settings.COLORS["black"], lw=0.8, ls="--")
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
    representation: Representation,
    n_svd: int,
    seed: int = 0,
    eim: bool = False,
) -> None:
    """Train an SVD->regressor surrogate and write required artifacts."""
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    train = load_waveform_split(
        "datasets/waveform/waveform_training.h5",
        t_min=-2847.0,
        t_max=0.0,
        time_convention=time_convention,
    )
    val = load_waveform_split(
        "datasets/waveform/waveform_validation.h5",
        t_min=-2847.0,
        t_max=0.0,
        time_convention=time_convention,
    )

    X_train = parameterize(train.params_raw, parameterization)
    X_val = parameterize(val.params_raw, parameterization)
    W_train = featurize_waveforms(train.h, representation)
    W_val = featurize_waveforms(val.h, representation)

    mean, components = randomized_svd_basis(W_train, n_components=n_svd, seed=seed)
    Y_train = project_svd(W_train, mean, components)
    Y_val_true = project_svd(W_val, mean, components)

    # Initial fit
    t0 = time.perf_counter()
    if eim:
        # Predict values at EIM nodes instead of coefficients.
        U = components.T  # (D,K)
        nodes = _eim_nodes(U, n_svd)
        U_nodes = U[nodes, :]  # (K,K)
        U_nodes_inv = np.linalg.pinv(U_nodes).astype(np.float32)
        Wn_train = W_train[:, nodes]
        est = fit_regressor(X_train, Wn_train, kind=regressor_kind, seed=seed, hyperparams=regressor_hyperparams)
        if hasattr(est, "fit"):
            est.fit(X_train, Wn_train)
        fit_time = time.perf_counter() - t0
    else:
        est = fit_regressor(X_train, Y_train, kind=regressor_kind, seed=seed, hyperparams=regressor_hyperparams)
        if hasattr(est, "fit"):
            est.fit(X_train, Y_train)
        fit_time = time.perf_counter() - t0

    # One round of reasoned adjustment:
    # - If val >> train, increase regularization (alpha); else if both high, increase basis size slightly.
    train_pred0 = _predict_waveforms_svd(
        est,
        X_train,
        mean=mean,
        components=components,
        representation=representation,
        eim=eim,
        eim_nodes_locals=None if not eim else (nodes, U_nodes_inv, mean[nodes]),
    )
    val_pred0 = _predict_waveforms_svd(
        est,
        X_val,
        mean=mean,
        components=components,
        representation=representation,
        eim=eim,
        eim_nodes_locals=None if not eim else (nodes, U_nodes_inv, mean[nodes]),
    )
    loss_tr0, _ = compute_per_sample_losses(train_pred0, train.h, dt=train.dt)
    loss_va0, _ = compute_per_sample_losses(val_pred0, val.h, dt=val.dt)
    mean_tr0 = float(np.mean(loss_tr0))
    mean_va0 = float(np.mean(loss_va0))

    obs = f"train_loss={mean_tr0:.3e}, val_loss={mean_va0:.3e}"
    hypothesis = ""
    change = ""

    # Only tune a subset of model families (those with alpha).
    tuned = False
    hp2 = dict(regressor_hyperparams or {})
    if regressor_kind in {"ridge", "poly_ridge", "krr_rbf"}:
        if mean_va0 > 1.25 * mean_tr0:
            hypothesis = "Validation loss suggests overfitting → increase regularization (alpha)."
            hp2["alpha"] = float(hp2.get("alpha", 1e-2)) * 10.0
            change = f"alpha -> {hp2['alpha']:.2e}"
            tuned = True
        elif mean_va0 > 3e-2:
            hypothesis = "Both losses high → increase basis size to capture more variance."
            n_svd2 = int(min(n_svd + 10, 80))
            if n_svd2 != n_svd:
                n_svd = n_svd2
                mean, components = randomized_svd_basis(W_train, n_components=n_svd, seed=seed)
                Y_train = project_svd(W_train, mean, components)
                tuned = True
                change = f"n_svd -> {n_svd}"
    if tuned and not eim:
        est = fit_regressor(X_train, Y_train, kind=regressor_kind, seed=seed, hyperparams=hp2)
        est.fit(X_train, Y_train)
    elif tuned and eim:
        U = components.T
        nodes = _eim_nodes(U, n_svd)
        U_nodes_inv = np.linalg.pinv(U[nodes, :]).astype(np.float32)
        Wn_train = W_train[:, nodes]
        est = fit_regressor(X_train, Wn_train, kind=regressor_kind, seed=seed, hyperparams=hp2)
        if hasattr(est, "fit"):
            est.fit(X_train, Wn_train)

    # Final eval + runtime
    t1 = time.perf_counter()
    val_pred = _predict_waveforms_svd(
        est,
        X_val,
        mean=mean,
        components=components,
        representation=representation,
        eim=eim,
        eim_nodes_locals=None if not eim else (nodes, U_nodes_inv, mean[nodes]),
    )
    runtime_s = time.perf_counter() - t1
    runtime_ms = 1000.0 * runtime_s / len(X_val)

    tr_pred = _predict_waveforms_svd(
        est,
        X_train,
        mean=mean,
        components=components,
        representation=representation,
        eim=eim,
        eim_nodes_locals=None if not eim else (nodes, U_nodes_inv, mean[nodes]),
    )
    tr_losses, _ = compute_per_sample_losses(tr_pred, train.h, dt=train.dt)
    va_losses, loss_components = compute_per_sample_losses(val_pred, val.h, dt=val.dt)
    loss = float(np.mean(va_losses))

    # Save artifacts
    np.save(model_dir / "saved_model" / "svd_mean.npy", mean.astype(np.float32))
    np.save(model_dir / "saved_model" / "svd_components.npy", components.astype(np.float32))
    joblib.dump(est, model_dir / "saved_model" / "regressor.joblib")

    if eim:
        np.save(model_dir / "saved_model" / "eim_nodes.npy", np.asarray(nodes, dtype=np.int64))
        np.save(model_dir / "saved_model" / "eim_U_nodes_inv.npy", np.asarray(U_nodes_inv, dtype=np.float32))

    _json_dump(
        model_dir / "saved_model" / "config.json",
        {
            "benchmark": "waveform",
            "display_name": display_name,
            "category": category,
            "parameterization": parameterization,
            "time_convention": time_convention,
            "representation": representation,
            "t_min": train.t_min,
            "t_max": train.t_max,
            "dt": train.dt,
            "n_svd": int(n_svd),
            "regressor_kind": regressor_kind,
            "regressor_hyperparams": hp2,
            "eim": bool(eim),
        },
    )

    _json_dump(
        model_dir / "saved_model" / "per_sample_errors.json",
        {"train": tr_losses.tolist(), "val": va_losses.tolist()},
    )

    scorecard = {
        "approach": regressor_kind if not eim else f"eim_{regressor_kind}",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "waveform",
        "agent": "gpt52",
        "category": category,
        "parameterization": parameterization,
        "time_convention": time_convention,
        "representation": representation,
        "loss": float(loss),
        "loss_components": loss_components,
        "runtime_ms": float(runtime_ms),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_params": int(_count_params_sklearn(est)),
        "notes": {
            "fit_time_s": float(fit_time),
            "reasoned_optimization": {
                "observed": obs,
                "hypothesis": hypothesis,
                "change": change,
            },
        },
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    # Minimal predict.py and train.py wrappers (kept in-place for reproducibility).
    _write_model_scripts(model_dir)

    # Update comparison artifacts + global changelog.
    update_waveform_comparison(work_dir)

    entry = (
        f"\n## [W-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_now_str()}\n"
        "- **Benchmark**: waveform\n"
        f"- **Category**: {category}\n"
        f"- **Method**: {('EIM + ' if eim else 'SVD + ')}{regressor_kind}\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Time convention**: {time_convention}\n"
        f"- **Representation**: {representation}\n"
        f"- **Loss**: {loss:.4e}\n"
        f"- **Eval time**: {runtime_ms:.2f} ms\n"
        "- **Reasoned optimization**:\n"
        f"  - Observed: {obs}\n"
        f"  - Hypothesis: {hypothesis or 'N/A'}\n"
        f"  - Change: {change or 'N/A'}\n"
    )
    append_changelog_entry(agent_root, entry)


def _predict_waveforms_svd(
    est: Any,
    X: np.ndarray,
    *,
    mean: np.ndarray,
    components: np.ndarray,
    representation: Representation,
    eim: bool,
    eim_nodes_locals: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if eim:
        if eim_nodes_locals is None:
            raise ValueError("eim_nodes_locals required for eim prediction")
        nodes, U_nodes_inv, mean_nodes = eim_nodes_locals
        W_nodes_pred = predict_regressor(est, X).astype(np.float32)
        coeffs = (W_nodes_pred - mean_nodes[None, :]) @ U_nodes_inv.T
        W_pred = reconstruct_svd(coeffs, mean, components)
    else:
        coeffs_pred = predict_regressor(est, X).astype(np.float32)
        W_pred = reconstruct_svd(coeffs_pred, mean, components)
    h_pred = unfeaturize_waveforms(W_pred, representation)
    return np.asarray(h_pred, dtype=np.complex64)


def _eim_nodes(U: np.ndarray, k: int) -> np.ndarray:
    """Greedy EIM node selection for basis columns U (D,K)."""
    U = np.asarray(U, dtype=np.float64)
    D = U.shape[0]
    nodes: list[int] = []
    # Orthonormalize-ish using QR for stability.
    Q, _ = np.linalg.qr(U[:, :k])
    for j in range(k):
        v = Q[:, j].copy()
        if nodes:
            A = Q[nodes, :j]
            b = v[nodes]
            if A.size:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
                v = v - Q[:, :j] @ coeff
        idx = int(np.argmax(np.abs(v)))
        nodes.append(idx)
    return np.asarray(nodes, dtype=np.int64)


def _write_model_scripts(model_dir: Path) -> None:
    train_py = model_dir / "train.py"
    predict_py = model_dir / "predict.py"
    if not train_py.exists():
        train_py.write_text(
            """# Wrapper for reproducibility: training is orchestrated by `run_all.py`.
# This script exists to satisfy the benchmark artifact contract.

if __name__ == "__main__":
    raise SystemExit(
        "Use llm_agents/results/gpt52/waveform/run_all.py to train this model reproducibly."
    )
"""
        )
    if not predict_py.exists():
        predict_py.write_text(
            """from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

# Make waveform/ importable when called from this model directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import parameterize, reconstruct_svd, unfeaturize_waveforms  # noqa: E402


def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    mean = np.load(model_dir / "saved_model" / "svd_mean.npy")
    comps = np.load(model_dir / "saved_model" / "svd_components.npy")
    est = joblib.load(model_dir / "saved_model" / "regressor.joblib")
    eim = bool(cfg.get("eim", False))
    nodes = inv = None
    if eim:
        nodes = np.load(model_dir / "saved_model" / "eim_nodes.npy")
        inv = np.load(model_dir / "saved_model" / "eim_U_nodes_inv.npy")
    return cfg, mean, comps, est, eim, nodes, inv


def predict(model_dir: str | Path, params_raw: np.ndarray) -> Dict[str, np.ndarray]:
    cfg, mean, comps, est, eim, nodes, inv = load_model(model_dir)
    X = parameterize(params_raw, cfg["parameterization"])
    if eim:
        W_nodes = np.asarray(est.predict(X), dtype=np.float32)
        mean_nodes = mean[nodes]
        coeffs = (W_nodes - mean_nodes[None, :]) @ inv.T
        W = reconstruct_svd(coeffs, mean, comps)
    else:
        coeffs = np.asarray(est.predict(X), dtype=np.float32)
        W = reconstruct_svd(coeffs, mean, comps)
    h = unfeaturize_waveforms(W, cfg["representation"])
    return {
        "h22_real": np.asarray(h.real, dtype=np.float64),
        "h22_imag": np.asarray(h.imag, dtype=np.float64),
    }
"""
        )
