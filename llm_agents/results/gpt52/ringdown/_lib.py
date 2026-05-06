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

# Ensure repo root is importable when run from nested results.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_gpt52")

from gwbenchmarks import plot_settings


Mode = Literal["l2/m+2/n0", "l2/m+2/n1", "l3/m+3/n0", "l4/m+4/n0"]
SpinParam = Literal["raw_a", "log_compact", "sqrt_irreducible", "compactified", "cheb_mapped"]


@dataclasses.dataclass(frozen=True)
class RingdownSplit:
    spin: np.ndarray  # (N,)
    omega_r: np.ndarray  # (N,)
    omega_i: np.ndarray  # (N,)


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


def load_ringdown_split(path: str | Path, *, mode: Mode = "l2/m+2/n0") -> RingdownSplit:
    path = Path(path)
    with h5py.File(path, "r") as f:
        g = f[mode]
        spin = np.asarray(g["spin"][:], dtype=np.float64)
        omega_r = np.asarray(g["omega_real"][:], dtype=np.float64)
        omega_i = np.asarray(g["omega_imag"][:], dtype=np.float64)
    return RingdownSplit(spin=spin, omega_r=omega_r, omega_i=omega_i)


def spin_feature(a: np.ndarray, kind: SpinParam) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if kind == "raw_a":
        return a
    if kind == "log_compact":
        return -np.log(np.maximum(1.0 - a, 1e-12))
    if kind == "sqrt_irreducible":
        return np.sqrt(np.maximum(1.0 - a**2, 0.0))
    if kind == "compactified":
        return a / np.maximum(1.0 - a, 1e-12)
    if kind == "cheb_mapped":
        return 2.0 * a - 1.0
    raise ValueError(f"Unknown spin parameterization: {kind}")


def per_sample_loss(
    omega_r_pred: np.ndarray,
    omega_i_pred: np.ndarray,
    omega_r_true: np.ndarray,
    omega_i_true: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], float]:
    omega_r_true = np.asarray(omega_r_true, dtype=np.float64)
    omega_i_true = np.asarray(omega_i_true, dtype=np.float64)
    omega_r_pred = np.asarray(omega_r_pred, dtype=np.float64)
    omega_i_pred = np.asarray(omega_i_pred, dtype=np.float64)

    rel_r = np.abs(omega_r_pred - omega_r_true) / np.maximum(np.abs(omega_r_true), 1e-15)
    rel_i = np.abs(omega_i_pred - omega_i_true) / np.maximum(np.abs(omega_i_true), 1e-15)
    per = 0.5 * (rel_r + rel_i)
    comps = {
        "rel_error_omega_real": float(np.mean(rel_r)),
        "rel_error_omega_imag": float(np.mean(rel_i)),
    }
    return per, comps, float(np.mean(per))


def update_ringdown_comparison(work_dir: Path, *, mode: Mode) -> None:
    import matplotlib.pyplot as plt

    plot_settings.apply()
    models_dir = work_dir / "models"
    comp_dir = work_dir / "comparison"
    _ensure_dir(comp_dir)

    rows: list[dict[str, Any]] = []
    error_data: dict[str, Any] = {}
    for md in sorted([p for p in models_dir.iterdir() if p.is_dir()]):
        sc_path = md / "scorecard.json"
        if not sc_path.exists():
            continue
        sc = json.loads(sc_path.read_text())
        if sc.get("mode") != mode:
            continue
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
    ax.set_ylabel("Loss (mean rel. err)")
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
    ax.set_ylabel("Loss (mean rel. err)")
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
        bins = np.logspace(-12, -1, 35)
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


def train_and_score(
    *,
    work_dir: Path,
    agent_root: Path,
    model_dir: Path,
    approach_number: int,
    display_name: str,
    category: str,
    parameterization: SpinParam,
    mode: Mode,
    method: str,
    method_hyperparams: dict[str, Any] | None,
    seed: int = 0,
) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    tr = load_ringdown_split("datasets/ringdown/ringdown_training.h5", mode=mode)
    va = load_ringdown_split("datasets/ringdown/ringdown_validation.h5", mode=mode)

    x_tr = spin_feature(tr.spin, parameterization)
    x_va = spin_feature(va.spin, parameterization)
    X_tr = x_tr.reshape(-1, 1)
    X_va = x_va.reshape(-1, 1)

    y_tr = np.stack([tr.omega_r, tr.omega_i], axis=1)
    y_va = np.stack([va.omega_r, va.omega_i], axis=1)

    hp = dict(method_hyperparams or {})

    # Fit
    t_fit0 = time.perf_counter()
    model_obj: Any
    if method == "poly":
        deg = int(hp.get("degree", 10))
        # np.polyfit expects x, y
        cr = np.polyfit(x_tr, tr.omega_r, deg=deg)
        ci = np.polyfit(x_tr, tr.omega_i, deg=deg)
        model_obj = {"coef_r": cr, "coef_i": ci}
        fit_time = time.perf_counter() - t_fit0
        def _predict(x):
            pr = np.polyval(cr, x)
            pi = np.polyval(ci, x)
            return np.stack([pr, pi], axis=1)
    elif method == "chebyshev":
        from numpy.polynomial.chebyshev import Chebyshev

        deg = int(hp.get("degree", 20))
        fr = Chebyshev.fit(x_tr, tr.omega_r, deg=deg)
        fi = Chebyshev.fit(x_tr, tr.omega_i, deg=deg)
        model_obj = {"cheb_r": fr, "cheb_i": fi}
        fit_time = time.perf_counter() - t_fit0
        def _predict(x):
            return np.stack([fr(x), fi(x)], axis=1)
    elif method == "rational":
        # Fit y = p(x)/q(x) with q0=1 by linear least squares.
        m = int(hp.get("m", 7))
        n = int(hp.get("n", 7))
        def fit_one(y):
            # Build design: [x^0..x^m, -y*x^1..-y*x^n]
            A_left = np.stack([x_tr**k for k in range(m + 1)], axis=1)
            A_right = np.stack([-(y * (x_tr**k)) for k in range(1, n + 1)], axis=1)
            A = np.concatenate([A_left, A_right], axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            p = coef[: m + 1]
            q = np.concatenate([[1.0], coef[m + 1 :]])
            return p, q
        pr, qr = fit_one(tr.omega_r)
        pi, qi = fit_one(tr.omega_i)
        model_obj = {"p_r": pr, "q_r": qr, "p_i": pi, "q_i": qi}
        fit_time = time.perf_counter() - t_fit0
        def _predict(x):
            def eval_rat(p, q):
                num = sum(p[k] * x**k for k in range(len(p)))
                den = sum(q[k] * x**k for k in range(len(q)))
                return num / np.maximum(den, 1e-18)
            return np.stack([eval_rat(pr, qr), eval_rat(pi, qi)], axis=1)
    elif method == "spline":
        from scipy.interpolate import CubicSpline

        order = np.argsort(x_tr)
        xs = x_tr[order]
        yr = tr.omega_r[order]
        yi = tr.omega_i[order]
        # Ensure strictly increasing (guard against rare duplicate mapped spins)
        xs_u, uidx = np.unique(xs, return_index=True)
        yr_u = yr[uidx]
        yi_u = yi[uidx]

        fr = CubicSpline(xs_u, yr_u, bc_type="natural")
        fi = CubicSpline(xs_u, yi_u, bc_type="natural")
        model_obj = {"spline_r": fr, "spline_i": fi}
        fit_time = time.perf_counter() - t_fit0
        def _predict(x):
            return np.stack([fr(x), fi(x)], axis=1)
    elif method == "rbf_interp":
        from scipy.interpolate import RBFInterpolator

        model_obj = RBFInterpolator(X_tr, y_tr, kernel=str(hp.get("kernel", "thin_plate_spline")), neighbors=int(hp.get("neighbors", min(200, len(X_tr)))), smoothing=float(hp.get("smoothing", 0.0)))
        fit_time = time.perf_counter() - t_fit0
        def _predict(x):
            return np.asarray(model_obj(x.reshape(-1, 1)))
    else:
        # sklearn regressors
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if method == "gpr_rbf":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
            length = float(hp.get("length_scale", 1.0))
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-12, 1e-3))
            base = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed, n_restarts_optimizer=int(hp.get("n_restarts", 0)))
            est = MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))
        elif method == "krr_rbf":
            from sklearn.kernel_ridge import KernelRidge
            base = KernelRidge(alpha=float(hp.get("alpha", 1e-8)), kernel="rbf", gamma=hp.get("gamma", None))
            est = MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))
        elif method == "svr_rbf":
            from sklearn.svm import SVR
            base = SVR(C=float(hp.get("C", 100.0)), epsilon=float(hp.get("epsilon", 1e-10)), gamma=hp.get("gamma", "scale"))
            est = MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))
        elif method == "mlp":
            from sklearn.neural_network import MLPRegressor
            base = MLPRegressor(random_state=seed, hidden_layer_sizes=tuple(hp.get("hidden_layer_sizes", (64, 64))), alpha=float(hp.get("alpha", 1e-6)), learning_rate_init=float(hp.get("learning_rate_init", 1e-3)), max_iter=int(hp.get("max_iter", 4000)), early_stopping=True)
            est = MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))
        elif method == "rf":
            from sklearn.ensemble import RandomForestRegressor
            est = RandomForestRegressor(n_estimators=int(hp.get("n_estimators", 600)), random_state=seed, n_jobs=-1)
        elif method == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            base = HistGradientBoostingRegressor(random_state=seed, max_depth=int(hp.get("max_depth", 6)), learning_rate=float(hp.get("learning_rate", 0.05)), max_iter=int(hp.get("max_iter", 800)))
            est = MultiOutputRegressor(Pipeline([("scaler", StandardScaler()), ("model", base)]))
        else:
            raise ValueError(f"Unknown method: {method}")
        est.fit(X_tr, y_tr)
        model_obj = est
        fit_time = time.perf_counter() - t_fit0
        def _predict(x):
            return np.asarray(est.predict(x.reshape(-1, 1)))

    # One round of reasoned tuning for some methods
    yva0 = _predict(x_va)
    _, _, loss0 = per_sample_loss(yva0[:, 0], yva0[:, 1], va.omega_r, va.omega_i)
    observed = f"val_loss={loss0:.3e}"
    hypothesis = ""
    change = ""
    if method in {"poly"} and loss0 > 2e-6:
        hypothesis = "High-degree polynomial likely oscillating near a→1 → reduce degree."
        cur_deg = int(hp.get("degree", 10))
        new_deg = max(6, cur_deg - 4)
        if new_deg < cur_deg:
            change = f"degree -> {new_deg}"
            hp["degree"] = new_deg

            # Refit in-place (avoid recursion if degree hits its lower bound)
            t_fit1 = time.perf_counter()
            cr = np.polyfit(x_tr, tr.omega_r, deg=new_deg)
            ci = np.polyfit(x_tr, tr.omega_i, deg=new_deg)
            model_obj = {"coef_r": cr, "coef_i": ci}
            fit_time += time.perf_counter() - t_fit1

            def _predict(x):
                pr = np.polyval(cr, x)
                pi = np.polyval(ci, x)
                return np.stack([pr, pi], axis=1)

            yva1 = _predict(x_va)
            _, _, loss1 = per_sample_loss(yva1[:, 0], yva1[:, 1], va.omega_r, va.omega_i)
            observed = f"val_loss={loss0:.3e} -> {loss1:.3e}"
        else:
            # Already at the lower bound; proceed without further tuning.
            change = "degree unchanged (already at lower bound)"

    t0 = time.perf_counter()
    yva = _predict(x_va)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(x_va)
    ytr = _predict(x_tr)

    per_tr, _, _ = per_sample_loss(ytr[:, 0], ytr[:, 1], tr.omega_r, tr.omega_i)
    per_va, comps, loss = per_sample_loss(yva[:, 0], yva[:, 1], va.omega_r, va.omega_i)

    # Save artifacts
    joblib.dump(model_obj, model_dir / "saved_model" / "model.joblib")
    _json_dump(model_dir / "saved_model" / "config.json", {
        "benchmark": "ringdown",
        "display_name": display_name,
        "category": category,
        "parameterization": parameterization,
        "mode": mode,
        "method": method,
        "method_hyperparams": hp,
    })
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": per_tr.tolist(), "val": per_va.tolist()})

    scorecard = {
        "approach": method,
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "ringdown",
        "agent": "gpt52",
        "parameterization": parameterization,
        "mode": mode,
        "loss": float(loss),
        "loss_components": comps,
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
        "n_params": 0,
        "notes": {"fit_time_s": float(fit_time), "reasoned_optimization": {"observed": observed, "hypothesis": hypothesis, "change": change}},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    (model_dir / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    raise SystemExit('Use llm_agents/results/gpt52/ringdown/run_all.py or run_symbolic.py')\n"
    )
    (model_dir / "predict.py").write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import spin_feature  # noqa: E402


def predict(model_dir: str | Path, spin_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    model_obj = joblib.load(model_dir / "saved_model" / "model.joblib")
    x = spin_feature(np.asarray(spin_array, dtype=np.float64), cfg["parameterization"])
    X = x.reshape(-1, 1)
    if hasattr(model_obj, "predict"):
        y = np.asarray(model_obj.predict(X), dtype=np.float64)
    elif callable(model_obj):
        y = np.asarray(model_obj(X), dtype=np.float64)
    elif isinstance(model_obj, dict) and "coef_r" in model_obj:
        y = np.stack([np.polyval(model_obj["coef_r"], x), np.polyval(model_obj["coef_i"], x)], axis=1)
    else:
        raise TypeError(f"Unsupported model object: {type(model_obj)}")
    return y[:, 0], y[:, 1]
"""
    )

    update_ringdown_comparison(work_dir, mode=mode)
    append_changelog_entry(
        agent_root,
        f"\n## [Q-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_now_str()}\n"
        "- **Benchmark**: ringdown\n"
        f"- **Category**: {category}\n"
        f"- **Method**: {method}\n"
        f"- **Spin parameterization**: {parameterization}\n"
        f"- **Mode**: {mode}\n"
        f"- **Loss**: {loss:.4e}\n"
        f"- **Eval time**: {runtime_ms:.4f} ms\n"
        "- **Reasoned optimization**:\n"
        f"  - Observed: {observed}\n"
        f"  - Hypothesis: {hypothesis or 'N/A'}\n"
        f"  - Change: {change or 'N/A'}\n"
    )
