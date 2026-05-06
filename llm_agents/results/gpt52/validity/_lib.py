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

from gwbenchmarks import plot_settings  # noqa: E402


Parameterization = Literal[
    "raw_4d",
    "effective_4d",
    "log_4d",
    "interaction_6d",
    "boundary_distance_6d",
]


@dataclasses.dataclass(frozen=True)
class ValiditySplit:
    params_raw: np.ndarray  # (N, 4): q, chi1z, chi2z, omega0
    sxs_ids: list[str]
    mm_td: np.ndarray  # (N,)
    y_log10: np.ndarray  # (N,) = log10(mm_td)


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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_validity_split(path: str | Path) -> ValiditySplit:
    path = Path(path)
    with h5py.File(path, "r") as f:
        q = np.asarray(f["q"][:], dtype=np.float64)
        chi1z = np.asarray(f["chi1z"][:], dtype=np.float64)
        chi2z = np.asarray(f["chi2z"][:], dtype=np.float64)
        omega0 = np.asarray(f["omega0"][:], dtype=np.float64)
        mm_td = np.asarray(f["mm_td"][:], dtype=np.float64)
        sxs_ids = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s) for s in f["sxs_id"][:]]

    mm_td = np.maximum(mm_td, 1e-300)
    y_log10 = np.log10(mm_td)
    params_raw = np.stack([q, chi1z, chi2z, omega0], axis=1)
    return ValiditySplit(params_raw=params_raw, sxs_ids=sxs_ids, mm_td=mm_td, y_log10=y_log10)


def _eta_from_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (1.0 + q) ** 2


def _chi_eff(q: np.ndarray, chi1z: np.ndarray, chi2z: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    m1 = q / (1.0 + q)
    m2 = 1.0 / (1.0 + q)
    return m1 * np.asarray(chi1z, dtype=np.float64) + m2 * np.asarray(chi2z, dtype=np.float64)


def parameterize(params_raw: np.ndarray, kind: Parameterization) -> np.ndarray:
    p = np.asarray(params_raw, dtype=np.float64)
    q = p[:, 0]
    chi1z = p[:, 1]
    chi2z = p[:, 2]
    omega0 = p[:, 3]

    if kind == "raw_4d":
        return p.copy()

    eta = _eta_from_q(q)
    ceff = _chi_eff(q, chi1z, chi2z)
    chi_a = 0.5 * (chi1z - chi2z)

    if kind == "effective_4d":
        return np.stack([eta, ceff, chi_a, omega0], axis=1)

    if kind == "log_4d":
        return np.stack([np.log(np.maximum(q, 1e-12)), ceff, chi_a, np.log(np.maximum(omega0, 1e-12))], axis=1)

    if kind == "interaction_6d":
        return np.stack([eta, ceff, chi_a, omega0, q * ceff, eta * chi_a], axis=1)

    if kind == "boundary_distance_6d":
        q_excess = np.maximum(q - 8.0, 0.0)
        spin_excess = np.maximum(np.maximum(np.abs(chi1z), np.abs(chi2z)) - 0.8, 0.0)
        return np.stack([eta, ceff, chi_a, omega0, q_excess, spin_excess], axis=1)

    raise ValueError(f"Unknown parameterization: {kind}")


def fit_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kind: str,
    seed: int = 0,
    hyperparams: dict[str, Any] | None = None,
) -> Any:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    hyperparams = dict(hyperparams or {})

    if kind == "ridge":
        from sklearn.linear_model import Ridge

        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=float(hyperparams.get("alpha", 1e-2)), random_state=seed))])

    if kind == "lasso":
        from sklearn.linear_model import Lasso

        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=float(hyperparams.get("alpha", 1e-3)), max_iter=20000, random_state=seed))])

    if kind == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=float(hyperparams.get("alpha", 1e-3)), l1_ratio=float(hyperparams.get("l1_ratio", 0.5)), max_iter=20000, random_state=seed))])

    if kind == "poly_ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures

        deg = int(hyperparams.get("degree", 3))
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", Ridge(alpha=float(hyperparams.get("alpha", 1e-3)), random_state=seed)),
            ]
        )

    if kind == "krr_rbf":
        from sklearn.kernel_ridge import KernelRidge

        base = KernelRidge(alpha=float(hyperparams.get("alpha", 1e-3)), kernel="rbf", gamma=hyperparams.get("gamma", None))
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "svr_rbf":
        from sklearn.svm import SVR

        base = SVR(C=float(hyperparams.get("C", 10.0)), epsilon=float(hyperparams.get("epsilon", 1e-3)), gamma=hyperparams.get("gamma", "scale"))
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "gpr_rbf":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

        length = float(hyperparams.get("length_scale", 1.0))
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        base = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed, n_restarts_optimizer=int(hyperparams.get("n_restarts", 0)))
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "gpr_matern":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

        length = float(hyperparams.get("length_scale", 1.0))
        nu = float(hyperparams.get("nu", 2.5))
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=length, nu=nu, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        base = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed, n_restarts_optimizer=int(hyperparams.get("n_restarts", 0)))
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "rbf_interp":
        from scipy.interpolate import RBFInterpolator

        return RBFInterpolator(
            X,
            y.reshape(-1, 1),
            kernel=str(hyperparams.get("kernel", "thin_plate_spline")),
            smoothing=float(hyperparams.get("smoothing", 0.0)),
            neighbors=int(hyperparams.get("neighbors", min(80, len(X)))),
        )

    if kind == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        base = KNeighborsRegressor(n_neighbors=int(hyperparams.get("n_neighbors", 7)), weights=str(hyperparams.get("weights", "distance")))
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(hyperparams.get("n_estimators", 600)),
            random_state=seed,
            n_jobs=-1,
            max_depth=hyperparams.get("max_depth", None),
        )

    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=int(hyperparams.get("n_estimators", 900)),
            random_state=seed,
            n_jobs=-1,
            max_depth=hyperparams.get("max_depth", None),
        )

    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            random_state=seed,
            max_depth=int(hyperparams.get("max_depth", 6)),
            learning_rate=float(hyperparams.get("learning_rate", 0.05)),
            max_iter=int(hyperparams.get("max_iter", 800)),
        )

    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        random_state=seed,
                        hidden_layer_sizes=tuple(hyperparams.get("hidden_layer_sizes", (64, 64))),
                        alpha=float(hyperparams.get("alpha", 1e-6)),
                        learning_rate_init=float(hyperparams.get("learning_rate_init", 1e-3)),
                        max_iter=int(hyperparams.get("max_iter", 5000)),
                        early_stopping=True,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown regressor kind: {kind}")


def predict_regressor(est: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict"):
        return np.asarray(est.predict(X), dtype=np.float64).reshape(-1)
    if callable(est):
        return np.asarray(est(X), dtype=np.float64).reshape(-1)
    # scipy RBFInterpolator: callable
    return np.asarray(est(X), dtype=np.float64).reshape(-1)


def append_changelog_entry(work_dir: Path, entry_md: str) -> None:
    ch = work_dir / "CHANGELOG.md"
    _ensure_dir(ch.parent)
    with open(ch, "a") as f:
        if not entry_md.endswith("\n"):
            entry_md += "\n"
        f.write(entry_md)


def update_validity_comparison(work_dir: Path, *, error_floor: float = 0.0) -> None:
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
        sc = json.loads(score_path.read_text())
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

    # Loss-only
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    x = np.arange(len(losses))
    ax.scatter(x, losses, s=18, color=plot_settings.COLORS["blue"])
    ax.set_xlabel("Approach #")
    ax.set_ylabel("Loss (RMSE in log10(mm))")
    for i, lo in enumerate(losses):
        ax.text(i, lo, str(i + 1), fontsize=7, ha="center", va="bottom")
    fig.savefig(comp_dir / "loss_only_comparison.png")
    fig.savefig(comp_dir / "loss_only_comparison.pdf")
    plt.close(fig)

    # Pareto
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    ax.scatter(runtimes, losses, s=25, color=plot_settings.COLORS["red"], alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Eval time per sample (ms)")
    ax.set_ylabel("Loss (RMSE in log10(mm))")
    for rt, lo, lab in zip(runtimes, losses, labels):
        ax.text(rt, lo, lab.split(" (")[0], fontsize=6, ha="left", va="bottom")
    fig.savefig(comp_dir / "pareto_accuracy_speed.png")
    fig.savefig(comp_dir / "pareto_accuracy_speed.pdf")
    plt.close(fig)

    # Progress
    order = sorted(rows, key=lambda r: int(r.get("approach_number", 9999)))
    loss_o = np.asarray([float(r["loss"]) for r in order], dtype=float)
    rt_o = np.asarray([float(r.get("runtime_ms", math.nan)) for r in order], dtype=float)
    xs = np.arange(1, len(order) + 1)
    fig, ax1 = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    ax1.plot(xs, loss_o, marker="o", ms=3, lw=1.0, color=plot_settings.COLORS["blue"])
    ax1.set_xlabel("Approach #")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(xs, np.clip(rt_o, 1e-6, np.inf), marker="s", ms=3, lw=1.0, color=plot_settings.COLORS["orange"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime (ms)")
    fig.savefig(comp_dir / "progress.png")
    fig.savefig(comp_dir / "progress.pdf")
    plt.close(fig)

    # Error histograms (small multiples)
    if error_data:
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
        for j, name in enumerate(names):
            ax = axes[j // ncols][j % ncols]
            d = error_data[name]
            tr = np.asarray(d.get("train", []), dtype=float)
            va = np.asarray(d.get("val", []), dtype=float)
            bins = 40
            if tr.size:
                ax.hist(tr, bins=bins, alpha=0.5, color=plot_settings.COLORS["blue"], label="train")
            if va.size:
                ax.hist(va, bins=bins, alpha=0.35, color=plot_settings.COLORS["red"], label="val", hatch="///")
            ax.axvline(float(error_floor), color=plot_settings.COLORS["black"], lw=0.8, ls="--")
            ax.set_yscale("log" if (tr.size + va.size) else "linear")
            ax.set_xlabel("|pred-true| (dex)")
            ax.set_ylabel("Count")
            ax.text(0.02, 0.98, name, transform=ax.transAxes, ha="left", va="top", fontsize=6)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.savefig(comp_dir / "error_histograms.png")
        fig.savefig(comp_dir / "error_histograms.pdf")
        plt.close(fig)


def train_and_score(
    *,
    work_dir: Path,
    model_dir: Path,
    approach_number: int,
    display_name: str,
    category: str,
    regressor_kind: str,
    regressor_hyperparams: dict[str, Any] | None,
    parameterization: Parameterization,
    seed: int = 0,
) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    train = load_validity_split("datasets/validity/validity_training.h5")
    val = load_validity_split("datasets/validity/validity_validation.h5")

    X_train = parameterize(train.params_raw, parameterization)
    X_val = parameterize(val.params_raw, parameterization)
    y_train = train.y_log10
    y_val = val.y_log10

    # Fit initial model
    est = fit_regressor(X_train, y_train, kind=regressor_kind, seed=seed, hyperparams=regressor_hyperparams)
    t_fit0 = time.perf_counter()
    if hasattr(est, "fit"):
        est.fit(X_train, y_train)
    fit_time = time.perf_counter() - t_fit0

    pred_tr0 = predict_regressor(est, X_train)
    pred_va0 = predict_regressor(est, X_val)
    loss_tr0 = rmse(pred_tr0, y_train)
    loss_va0 = rmse(pred_va0, y_val)

    observed = f"train_rmse={loss_tr0:.3e}, val_rmse={loss_va0:.3e}"
    hypothesis = ""
    change = ""

    hp2 = dict(regressor_hyperparams or {})
    tuned = False
    if regressor_kind in {"ridge", "lasso", "elasticnet", "poly_ridge", "krr_rbf"} and loss_va0 > 1.25 * loss_tr0:
        hypothesis = "Validation error suggests overfitting → increase regularization (alpha)."
        hp2["alpha"] = float(hp2.get("alpha", 1e-3)) * 10.0
        change = f"alpha -> {hp2['alpha']:.2e}"
        tuned = True
    if regressor_kind in {"rf", "extratrees"} and loss_va0 > 1.25 * loss_tr0 and hp2.get("max_depth", None) is None:
        hypothesis = "Trees overfitting → cap max_depth."
        hp2["max_depth"] = 12
        change = "max_depth -> 12"
        tuned = True

    if tuned:
        est = fit_regressor(X_train, y_train, kind=regressor_kind, seed=seed, hyperparams=hp2)
        t_fit1 = time.perf_counter()
        if hasattr(est, "fit"):
            est.fit(X_train, y_train)
        fit_time += time.perf_counter() - t_fit1

        pred_va1 = predict_regressor(est, X_val)
        loss_va1 = rmse(pred_va1, y_val)
        observed = f"{observed} -> val_rmse={loss_va1:.3e}"

    t0 = time.perf_counter()
    pred_val = predict_regressor(est, X_val)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(X_val)
    pred_tr = predict_regressor(est, X_train)

    loss = rmse(pred_val, y_val)
    abs_err_tr = np.abs(pred_tr - y_train)
    abs_err_va = np.abs(pred_val - y_val)

    joblib.dump(est, model_dir / "saved_model" / "model.joblib")
    _json_dump(
        model_dir / "saved_model" / "config.json",
        {
            "benchmark": "validity",
            "display_name": display_name,
            "category": category,
            "parameterization": parameterization,
            "regressor_kind": regressor_kind,
            "regressor_hyperparams": hp2 if tuned else dict(regressor_hyperparams or {}),
        },
    )
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": abs_err_tr.tolist(), "val": abs_err_va.tolist()})

    scorecard = {
        "approach": regressor_kind,
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "validity",
        "agent": "gpt52",
        "category": category,
        "parameterization": parameterization,
        "loss": float(loss),
        "loss_components": {"log_rmse": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_params": 0,
        "notes": {"fit_time_s": float(fit_time), "reasoned_optimization": {"observed": observed, "hypothesis": hypothesis, "change": change}},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    (model_dir / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    raise SystemExit('Use llm_agents/results/gpt52/validity/run_all.py or run_symbolic.py')\n"
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
from _lib import parameterize  # noqa: E402


def predict(model_dir: str | Path, params_raw: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    est = joblib.load(model_dir / "saved_model" / "model.joblib")
    X = parameterize(np.asarray(params_raw, dtype=np.float64), cfg["parameterization"])
    y_log = np.asarray(est.predict(X), dtype=np.float64).reshape(-1)
    y = np.power(10.0, y_log)
    return {"mm_td": y, "log10_mm_td": y_log}
"""
    )

    update_validity_comparison(work_dir, error_floor=0.0)
    append_changelog_entry(
        work_dir,
        f"\n## [V-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_now_str()}\n"
        "- **Benchmark**: validity\n"
        f"- **Category**: {category}\n"
        f"- **Method**: {regressor_kind}\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Loss (RMSE log10(mm))**: {loss:.4e}\n"
        f"- **Eval time**: {runtime_ms:.4f} ms\n"
        "- **Reasoned optimization**:\n"
        f"  - Observed: {observed}\n"
        f"  - Hypothesis: {hypothesis or 'N/A'}\n"
        f"  - Change: {change or 'N/A'}\n"
    )

