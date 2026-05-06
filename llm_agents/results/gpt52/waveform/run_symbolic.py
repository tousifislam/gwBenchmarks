from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    append_changelog_entry,
    compute_per_sample_losses,
    featurize_waveforms,
    load_waveform_split,
    parameterize,
    randomized_svd_basis,
    reconstruct_svd,
    unfeaturize_waveforms,
    update_waveform_comparison,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _bootstrap_pysr_venv(venv_dir: Path) -> Path:
    py = _venv_python(venv_dir)
    if py.exists():
        return py
    _ensure_dir(venv_dir.parent)
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    subprocess.check_call([str(py), "-m", "pip", "install", "-q", "pysr"])
    return py


def _select_best_sympy(expressions: list[dict]) -> str:
    # Choose lowest loss among finite-loss entries; fall back to first.
    best = None
    for e in expressions:
        try:
            loss = float(e.get("loss", float("inf")))
        except Exception:
            continue
        if not np.isfinite(loss):
            continue
        if best is None or loss < best[0]:
            best = (loss, e)
    if best is None:
        return str(expressions[0].get("sympy_format") or expressions[0].get("equation") or "0")
    e = best[1]
    return str(e.get("sympy_format") or e.get("equation") or "0")


def _sympy_callable(expr: str, n_features: int):
    xs = sp.symbols(" ".join([f"x{i}" for i in range(n_features)]))
    parsed = sp.sympify(expr.replace("^", "**"))
    return sp.lambdify(xs, parsed, "numpy")


def run_pysr(
    *,
    work_dir: Path,
    agent_root: Path,
    model_dir: Path,
    approach_number: int,
) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    train = load_waveform_split("datasets/waveform/waveform_training.h5", t_min=-2847.0, t_max=0.0, time_convention="t0_at_peak")
    val = load_waveform_split("datasets/waveform/waveform_validation.h5", t_min=-2847.0, t_max=0.0, time_convention="t0_at_peak")

    parameterization = "effective_spins_7d"
    representation = "real_imag"
    n_svd = 3
    display_name = "PySR (SVD coeffs, eff)"

    X_train = parameterize(train.params_raw, parameterization)
    X_val = parameterize(val.params_raw, parameterization)
    W_train = featurize_waveforms(train.h, representation)
    W_val = featurize_waveforms(val.h, representation)

    mean, components = randomized_svd_basis(W_train, n_components=n_svd, seed=0)
    coeff_train = (W_train - mean[None, :]) @ components.T @ np.linalg.pinv(components @ components.T)
    coeff_train = coeff_train.astype(np.float32)

    # Fit PySR for each coefficient in an isolated venv to avoid julia dependency clashes.
    venv_dir = Path("/private/tmp/pysr_venv_gpt52")
    pysr_py = _bootstrap_pysr_venv(venv_dir)

    julia_exe = Path("/private/tmp/julia_env_gpt52/pyjuliapkg/install/bin/julia")
    env = os.environ.copy()
    if julia_exe.exists():
        env["PYTHON_JULIAPKG_EXE"] = str(julia_exe)
    env["PYTHON_JULIAPKG_PROJECT"] = "/private/tmp/pysr_julia_project_gpt52"
    env["JULIA_DEPOT_PATH"] = "/private/tmp/pysr_julia_depot_gpt52"
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_gpt52")

    all_exprs: list[dict] = []
    best_exprs: list[str] = []
    for j in range(n_svd):
        x_path = model_dir / "saved_model" / f"X_train.npy"
        y_path = model_dir / "saved_model" / f"y_train_c{j}.npy"
        np.save(x_path, X_train.astype(np.float32))
        np.save(y_path, coeff_train[:, j].astype(np.float32))
        expressions_all: list[dict] = []
        # Two complexity regimes.
        for maxsize, niter in [(12, 60), (20, 80)]:
            out_json = model_dir / "saved_model" / f"expressions_c{j}_max{maxsize}.json"
            subprocess.check_call(
                [
                    str(pysr_py),
                    str(work_dir / "_pysr_fit.py"),
                    "--x",
                    str(x_path),
                    "--y",
                    str(y_path),
                    "--out",
                    str(out_json),
                    "--maxsize",
                    str(maxsize),
                    "--niterations",
                    str(niter),
                    "--procs",
                    "2",
                    "--seed",
                    "0",
                ],
                env=env,
            )
            expressions_all.extend(json.loads(out_json.read_text()))
        expressions = expressions_all
        for e in expressions:
            e2 = dict(e)
            e2["coefficient_index"] = j
            all_exprs.append(e2)
        best_expr = _select_best_sympy(expressions)
        best_exprs.append(best_expr)

    _json_dump(model_dir / "saved_model" / "expressions.json", all_exprs)
    _json_dump(model_dir / "saved_model" / "best_equations.json", {"sympy": best_exprs})

    # Build fast predictor from the selected expressions.
    fns = [_sympy_callable(expr, X_train.shape[1]) for expr in best_exprs]

    def predict_coeffs(X: np.ndarray) -> np.ndarray:
        cols = []
        for fn in fns:
            cols.append(np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), dtype=np.float32).reshape(-1))
        return np.stack(cols, axis=1)

    t0 = time.perf_counter()
    coeff_val = predict_coeffs(X_val)
    W_val_pred = coeff_val @ components + mean[None, :]
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(X_val)
    h_val_pred = unfeaturize_waveforms(W_val_pred, representation)

    # Train predictions (for histograms)
    coeff_tr_pred = predict_coeffs(X_train)
    W_tr_pred = coeff_tr_pred @ components + mean[None, :]
    h_tr_pred = unfeaturize_waveforms(W_tr_pred, representation)

    tr_err, _ = compute_per_sample_losses(h_tr_pred, train.h, dt=train.dt)
    va_err, comps = compute_per_sample_losses(h_val_pred, val.h, dt=val.dt)
    loss = float(np.mean(va_err))

    np.save(model_dir / "saved_model" / "svd_mean.npy", mean.astype(np.float32))
    np.save(model_dir / "saved_model" / "svd_components.npy", components.astype(np.float32))
    _json_dump(model_dir / "saved_model" / "config.json", {
        "benchmark": "waveform",
        "display_name": display_name,
        "category": "symbolic/analytical",
        "parameterization": parameterization,
        "time_convention": "t0_at_peak",
        "representation": representation,
        "t_min": train.t_min,
        "t_max": train.t_max,
        "dt": train.dt,
        "n_svd": n_svd,
        "model": "pysr",
    })
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": tr_err.tolist(), "val": va_err.tolist()})

    scorecard = {
        "approach": "pysr",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "waveform",
        "agent": "gpt52",
        "category": "symbolic/analytical",
        "parameterization": parameterization,
        "time_convention": "t0_at_peak",
        "representation": representation,
        "loss": float(loss),
        "loss_components": comps,
        "runtime_ms": float(runtime_ms),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_params": 0,
        "notes": {"equations": best_exprs},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    # Write predict.py for this symbolic model.
    (model_dir / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    raise SystemExit('Run llm_agents/results/gpt52/waveform/run_symbolic.py')\n"
    )
    (model_dir / "predict.py").write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import parameterize, unfeaturize_waveforms  # noqa: E402


def _sympy_callable(expr: str, n_features: int):
    xs = sp.symbols(" ".join([f"x{i}" for i in range(n_features)]))
    parsed = sp.sympify(expr.replace("^", "**"))
    return sp.lambdify(xs, parsed, "numpy")


def predict(model_dir: str | Path, params_raw: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    mean = np.load(model_dir / "saved_model" / "svd_mean.npy")
    comps = np.load(model_dir / "saved_model" / "svd_components.npy")
    eqs = json.loads((model_dir / "saved_model" / "best_equations.json").read_text())["sympy"]

    X = parameterize(params_raw, cfg["parameterization"]).astype(np.float32)
    fns = [_sympy_callable(e, X.shape[1]) for e in eqs]
    cols = []
    for fn in fns:
        cols.append(np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), dtype=np.float32).reshape(-1))
    coeffs = np.stack(cols, axis=1)
    W = coeffs @ comps + mean[None, :]
    h = unfeaturize_waveforms(W, cfg["representation"])
    return {"h22_real": np.asarray(h.real, dtype=np.float64), "h22_imag": np.asarray(h.imag, dtype=np.float64)}
"""
    )

    update_waveform_comparison(work_dir)
    append_changelog_entry(
        agent_root,
        f"\n## [W-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_time_str()}\n"
        "- **Benchmark**: waveform\n"
        "- **Category**: symbolic/analytical\n"
        "- **Method**: PySR on first 3 SVD coefficients\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Loss**: {loss:.4e}\n"
        f"- **Eval time**: {runtime_ms:.2f} ms\n",
    )


def _time_str() -> str:
    import datetime as dt

    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def run_gplearn(
    *,
    work_dir: Path,
    agent_root: Path,
    model_dir: Path,
    approach_number: int,
) -> None:
    from gplearn.genetic import SymbolicRegressor
    # gplearn expects BaseEstimator._validate_data which was removed in newer sklearn.
    if not hasattr(SymbolicRegressor, "_validate_data"):
        from sklearn.utils.validation import check_X_y, check_array

        def _validate_data(self, X, y=None, y_numeric=False, **kwargs):
            if y is None:
                return check_array(X)
            Xv, yv = check_X_y(X, y, y_numeric=y_numeric)
            return Xv, yv

        SymbolicRegressor._validate_data = _validate_data  # type: ignore[attr-defined]

    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    train = load_waveform_split("datasets/waveform/waveform_training.h5", t_min=-2847.0, t_max=0.0, time_convention="t0_at_peak")
    val = load_waveform_split("datasets/waveform/waveform_validation.h5", t_min=-2847.0, t_max=0.0, time_convention="t0_at_peak")

    parameterization = "raw_7d"
    representation = "real_imag"
    n_svd = 6
    display_name = "gplearn (SVD coeffs, raw)"

    X_train = parameterize(train.params_raw, parameterization).astype(np.float32)
    X_val = parameterize(val.params_raw, parameterization).astype(np.float32)
    W_train = featurize_waveforms(train.h, representation)
    W_val = featurize_waveforms(val.h, representation)

    mean, components = randomized_svd_basis(W_train, n_components=n_svd, seed=0)
    coeff_train = (W_train - mean[None, :]) @ components.T @ np.linalg.pinv(components @ components.T)
    coeff_train = coeff_train.astype(np.float32)

    models = []
    exprs = []
    for j in range(n_svd):
        est = SymbolicRegressor(
            population_size=3000,
            generations=30,
            tournament_size=20,
            function_set=["add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"],
            metric="mse",
            parsimony_coefficient=0.001,
            max_samples=1.0,
            verbose=0,
            random_state=42 + j,
        )
        est.fit(X_train, coeff_train[:, j])
        # Newer sklearn conventions expected by gplearn.predict.
        est.n_features_in_ = X_train.shape[1]
        models.append(est)
        exprs.append(str(est._program))

    joblib.dump(models, model_dir / "saved_model" / "gplearn_models.joblib")
    _json_dump(
        model_dir / "saved_model" / "expressions.json",
        [{"coefficient_index": j, "expression": exprs[j]} for j in range(n_svd)],
    )

    def predict_coeffs(X: np.ndarray) -> np.ndarray:
        cols = [m.predict(X).astype(np.float32) for m in models]
        return np.stack(cols, axis=1)

    t0 = time.perf_counter()
    coeff_val = predict_coeffs(X_val)
    W_val_pred = coeff_val @ components + mean[None, :]
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(X_val)
    h_val_pred = unfeaturize_waveforms(W_val_pred, representation)

    coeff_tr = predict_coeffs(X_train)
    h_tr_pred = unfeaturize_waveforms(coeff_tr @ components + mean[None, :], representation)

    tr_err, _ = compute_per_sample_losses(h_tr_pred, train.h, dt=train.dt)
    va_err, comps = compute_per_sample_losses(h_val_pred, val.h, dt=val.dt)
    loss = float(np.mean(va_err))

    np.save(model_dir / "saved_model" / "svd_mean.npy", mean.astype(np.float32))
    np.save(model_dir / "saved_model" / "svd_components.npy", components.astype(np.float32))
    _json_dump(model_dir / "saved_model" / "config.json", {
        "benchmark": "waveform",
        "display_name": display_name,
        "category": "symbolic/analytical",
        "parameterization": parameterization,
        "time_convention": "t0_at_peak",
        "representation": representation,
        "t_min": train.t_min,
        "t_max": train.t_max,
        "dt": train.dt,
        "n_svd": n_svd,
        "model": "gplearn",
    })
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": tr_err.tolist(), "val": va_err.tolist()})

    scorecard = {
        "approach": "gplearn",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "waveform",
        "agent": "gpt52",
        "category": "symbolic/analytical",
        "parameterization": parameterization,
        "time_convention": "t0_at_peak",
        "representation": representation,
        "loss": float(loss),
        "loss_components": comps,
        "runtime_ms": float(runtime_ms),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_params": 0,
        "notes": {"expressions": exprs},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    (model_dir / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    raise SystemExit('Run llm_agents/results/gpt52/waveform/run_symbolic.py')\n"
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
from _lib import parameterize, unfeaturize_waveforms  # noqa: E402


def predict(model_dir: str | Path, params_raw: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    mean = np.load(model_dir / "saved_model" / "svd_mean.npy")
    comps = np.load(model_dir / "saved_model" / "svd_components.npy")
    models = joblib.load(model_dir / "saved_model" / "gplearn_models.joblib")

    X = parameterize(params_raw, cfg["parameterization"]).astype(np.float32)
    coeffs = np.stack([m.predict(X).astype(np.float32) for m in models], axis=1)
    W = coeffs @ comps + mean[None, :]
    h = unfeaturize_waveforms(W, cfg["representation"])
    return {"h22_real": np.asarray(h.real, dtype=np.float64), "h22_imag": np.asarray(h.imag, dtype=np.float64)}
"""
    )

    update_waveform_comparison(work_dir)
    append_changelog_entry(
        agent_root,
        f"\n## [W-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_time_str()}\n"
        "- **Benchmark**: waveform\n"
        "- **Category**: symbolic/analytical\n"
        "- **Method**: gplearn SymbolicRegressor on 6 SVD coefficients\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Loss**: {loss:.4e}\n"
        f"- **Eval time**: {runtime_ms:.2f} ms\n",
    )


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    agent_root = work_dir.parent
    models_dir = work_dir / "models"

    pysr_dir = models_dir / "19_pysr_svd_coeffs_eff"
    if not (pysr_dir / "scorecard.json").exists():
        run_pysr(work_dir=work_dir, agent_root=agent_root, model_dir=pysr_dir, approach_number=19)

    gpl_dir = models_dir / "20_gplearn_svd_coeffs_raw"
    if not (gpl_dir / "scorecard.json").exists():
        run_gplearn(work_dir=work_dir, agent_root=agent_root, model_dir=gpl_dir, approach_number=20)


if __name__ == "__main__":
    main()
