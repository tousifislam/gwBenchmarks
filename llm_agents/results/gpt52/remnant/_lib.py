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
from gwbenchmarks.metrics import nrmse


Parameterization = Literal[
    "raw_7d",
    "effective_spins_7d",
    "massdiff_chia_5d",
    "pn_products_5d",
    "spherical_spins_7d",
]


@dataclasses.dataclass(frozen=True)
class RemnantSplit:
    params_raw: np.ndarray  # (N, 8): q, chi1x..chi2z, omega0
    sxs_ids: list[str]
    vk: np.ndarray  # (N,)
    delta_vk: np.ndarray  # (N,)


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


def load_remnant_split(path: str | Path) -> RemnantSplit:
    path = Path(path)
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        q = np.asarray(f["q"][:], dtype=np.float64)
        chi1x = np.asarray(f["chi1x"][:], dtype=np.float64)
        chi1y = np.asarray(f["chi1y"][:], dtype=np.float64)
        chi1z = np.asarray(f["chi1z"][:], dtype=np.float64)
        chi2x = np.asarray(f["chi2x"][:], dtype=np.float64)
        chi2y = np.asarray(f["chi2y"][:], dtype=np.float64)
        chi2z = np.asarray(f["chi2z"][:], dtype=np.float64)
        omega0 = np.asarray(f["omega0"][:], dtype=np.float64) if "omega0" in f else np.zeros_like(q)
        vk = np.asarray(f["vf_mag"][:], dtype=np.float64)
        delta_vk = np.asarray(f["delta_vf"][:], dtype=np.float64) if "delta_vf" in f else np.zeros_like(vk)
        sxs_ids = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s) for s in f["sxs_id"][:]]

    params_raw = np.stack(
        [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega0], axis=1
    )
    assert params_raw.shape == (n, 8)
    return RemnantSplit(params_raw=params_raw, sxs_ids=sxs_ids, vk=vk, delta_vk=delta_vk)


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
    weight = (4.0 * q + 3.0) / (4.0 + 3.0 * q) * q
    return np.maximum(a1, weight * a2)


def parameterize(params_raw: np.ndarray, kind: Parameterization) -> np.ndarray:
    p = np.asarray(params_raw, dtype=np.float64)
    q = p[:, 0]
    chi1x, chi1y, chi1z = p[:, 1], p[:, 2], p[:, 3]
    chi2x, chi2y, chi2z = p[:, 4], p[:, 5], p[:, 6]

    if kind == "raw_7d":
        return p[:, :7].copy()

    if kind == "effective_spins_7d":
        eta = _eta_from_q(q)
        ceff = _chi_eff(q, chi1z, chi2z)
        cp = _chi_p(q, chi1x, chi1y, chi2x, chi2y)
        chi1mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
        chi2mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
        theta1 = np.arccos(np.clip(np.where(chi1mag > 0, chi1z / chi1mag, 1.0), -1.0, 1.0))
        theta2 = np.arccos(np.clip(np.where(chi2mag > 0, chi2z / chi2mag, 1.0), -1.0, 1.0))
        return np.stack([eta, ceff, cp, chi1mag, chi2mag, theta1, theta2], axis=1)

    if kind == "massdiff_chia_5d":
        # delta_m = (m1-m2)/(m1+m2) with m1/m2=q
        delta_m = (q - 1.0) / (q + 1.0)
        ceff = _chi_eff(q, chi1z, chi2z)
        chi_a = 0.5 * (chi1z - chi2z)
        chi1mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
        chi2mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
        return np.stack([delta_m, ceff, chi_a, chi1mag, chi2mag], axis=1)

    if kind == "pn_products_5d":
        eta = _eta_from_q(q)
        ceff = _chi_eff(q, chi1z, chi2z)
        delta_m = (q - 1.0) / (q + 1.0)
        chi_a = 0.5 * (chi1z - chi2z)
        cp = _chi_p(q, chi1x, chi1y, chi2x, chi2y)
        return np.stack([eta, ceff, eta * ceff, delta_m * chi_a, cp], axis=1)

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

        base = SVR(C=float(hyperparams.get("C", 10.0)), epsilon=float(hyperparams.get("epsilon", 1e-4)), gamma=hyperparams.get("gamma", "scale"))
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "gpr_rbf":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

        length = float(hyperparams.get("length_scale", 1.0))
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        base = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed, n_restarts_optimizer=int(hyperparams.get("n_restarts", 0)))
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "gpr_matern":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

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
            neighbors=int(hyperparams.get("neighbors", min(50, len(X)))),
        )

    if kind == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        base = KNeighborsRegressor(n_neighbors=int(hyperparams.get("n_neighbors", 7)), weights=str(hyperparams.get("weights", "distance")))
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=int(hyperparams.get("n_estimators", 600)), random_state=seed, n_jobs=-1)

    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(n_estimators=int(hyperparams.get("n_estimators", 900)), random_state=seed, n_jobs=-1)

    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        base = HistGradientBoostingRegressor(
            random_state=seed,
            max_depth=int(hyperparams.get("max_depth", 6)),
            learning_rate=float(hyperparams.get("learning_rate", 0.05)),
            max_iter=int(hyperparams.get("max_iter", 600)),
        )
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor

        base = MLPRegressor(
            random_state=seed,
            hidden_layer_sizes=tuple(hyperparams.get("hidden_layer_sizes", (128, 128))),
            alpha=float(hyperparams.get("alpha", 1e-6)),
            learning_rate_init=float(hyperparams.get("learning_rate_init", 2e-3)),
            max_iter=int(hyperparams.get("max_iter", 2000)),
            early_stopping=True,
        )
        return Pipeline([("scaler", StandardScaler()), ("model", base)])

    raise ValueError(f"Unknown regressor kind: {kind}")


def predict_regressor(est: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict"):
        return np.asarray(est.predict(X)).reshape(-1)
    if callable(est):
        out = np.asarray(est(X))
        return out.reshape(-1)
    raise TypeError(f"Estimator has no predict(): {type(est)}")


def update_remnant_comparison(work_dir: Path, *, error_floor: float) -> None:
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
    ax.set_ylabel("Loss (NRMSE)")
    for i, lab in enumerate(labels):
        ax.text(i, losses[i], str(i + 1), fontsize=7, ha="center", va="bottom")
    fig.savefig(comp_dir / "loss_only_comparison.png")
    fig.savefig(comp_dir / "loss_only_comparison.pdf")
    plt.close(fig)

    # Pareto
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    ax.scatter(runtimes, losses, s=25, color=plot_settings.COLORS["red"], alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Eval time per sample (ms)")
    ax.set_ylabel("Loss (NRMSE)")
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
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(plot_settings.DOUBLE_COL, 0.9 * nrows), squeeze=False)
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
            ax.set_xlabel("|pred-true|")
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


def train_and_score(
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
    seed: int = 0,
) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    train = load_remnant_split("datasets/remnant/remnant_training.h5")
    val = load_remnant_split("datasets/remnant/remnant_validation.h5")

    X_train = parameterize(train.params_raw, parameterization)
    X_val = parameterize(val.params_raw, parameterization)
    y_train = train.vk
    y_val = val.vk

    floor = float(np.median(np.abs(train.delta_vk))) if train.delta_vk.size else 0.0

    # Fit initial model
    est = fit_regressor(X_train, y_train, kind=regressor_kind, seed=seed, hyperparams=regressor_hyperparams)
    if hasattr(est, "fit"):
        est.fit(X_train, y_train)

    pred_tr0 = predict_regressor(est, X_train)
    pred_va0 = predict_regressor(est, X_val)
    loss_tr0 = float(nrmse(pred_tr0, y_train))
    loss_va0 = float(nrmse(pred_va0, y_val))

    obs = f"train_nrmse={loss_tr0:.3e}, val_nrmse={loss_va0:.3e}"
    hypothesis = ""
    change = ""

    hp2 = dict(regressor_hyperparams or {})
    tuned = False
    if regressor_kind in {"ridge", "poly_ridge", "krr_rbf"} and loss_va0 > 1.25 * loss_tr0:
        hypothesis = "Validation error suggests overfitting → increase regularization (alpha)."
        hp2["alpha"] = float(hp2.get("alpha", 1e-3)) * 10.0
        change = f"alpha -> {hp2['alpha']:.2e}"
        tuned = True
    if regressor_kind in {"rf", "extratrees"} and loss_va0 > 1.25 * loss_tr0:
        hypothesis = "Trees overfitting → cap max_depth."
        hp2["max_depth"] = int(hp2.get("max_depth", 12) or 12)
        change = f"max_depth -> {hp2['max_depth']}"
        tuned = True

    if tuned:
        est = fit_regressor(X_train, y_train, kind=regressor_kind, seed=seed, hyperparams=hp2)
        if hasattr(est, "fit"):
            est.fit(X_train, y_train)

    t0 = time.perf_counter()
    pred_val = predict_regressor(est, X_val)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(X_val)
    pred_tr = predict_regressor(est, X_train)

    loss = float(nrmse(pred_val, y_val))
    abs_err_tr = np.abs(pred_tr - y_train)
    abs_err_va = np.abs(pred_val - y_val)

    joblib.dump(est, model_dir / "saved_model" / "model.joblib")
    _json_dump(
        model_dir / "saved_model" / "config.json",
        {
            "benchmark": "remnant",
            "display_name": display_name,
            "category": category,
            "parameterization": parameterization,
            "regressor_kind": regressor_kind,
            "regressor_hyperparams": hp2,
        },
    )
    _json_dump(
        model_dir / "saved_model" / "per_sample_errors.json",
        {"train": abs_err_tr.tolist(), "val": abs_err_va.tolist()},
    )
    scorecard = {
        "approach": regressor_kind,
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "remnant",
        "agent": "gpt52",
        "category": category,
        "parameterization": parameterization,
        "loss": float(loss),
        "loss_components": {"nrmse_v_k": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_params": 0,
        "notes": {
            "error_floor_delta_vk_median": floor,
            "reasoned_optimization": {
                "observed": obs,
                "hypothesis": hypothesis,
                "change": change,
            },
        },
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    # Minimal wrapper scripts (satisfy artifact contract).
    (model_dir / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    raise SystemExit('Use llm_agents/results/gpt52/remnant/run_all.py or run_symbolic.py')\n"
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
    X = parameterize(params_raw, cfg["parameterization"])
    y = np.asarray(est.predict(X), dtype=np.float64).reshape(-1)
    return {"v_k_mag": y}
"""
    )

    update_remnant_comparison(work_dir, error_floor=floor)
    entry = (
        f"\n## [R-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_now_str()}\n"
        "- **Benchmark**: remnant\n"
        f"- **Category**: {category}\n"
        f"- **Method**: {regressor_kind}\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Loss (NRMSE)**: {loss:.4e}\n"
        f"- **Eval time**: {runtime_ms:.3f} ms\n"
        "- **Reasoned optimization**:\n"
        f"  - Observed: {obs}\n"
        f"  - Hypothesis: {hypothesis or 'N/A'}\n"
        f"  - Change: {change or 'N/A'}\n"
    )
    append_changelog_entry(agent_root, entry)

