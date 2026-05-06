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
    load_dynamics_split,
    parameterize,
    project_svd,
    randomized_svd_basis,
    reconstruct_svd,
    update_dynamics_comparison,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _select_best_sympy(expressions: list[dict]) -> str:
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


def run_pysr(*, work_dir: Path, agent_root: Path, model_dir: Path, approach_number: int) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    train = load_dynamics_split("datasets/dynamics/dynamics_training.h5", time_convention="normalized_time")
    val = load_dynamics_split("datasets/dynamics/dynamics_validation.h5", time_convention="normalized_time")

    display_name = "PySR (SVD coeffs, eff)"
    parameterization = "eff_loge0_6d"
    n_svd = 3

    X_train = parameterize(train.params_raw, parameterization).astype(np.float32)
    X_val = parameterize(val.params_raw, parameterization).astype(np.float32)
    W_train = train.x.astype(np.float32)
    W_val = val.x.astype(np.float32)

    mean, components = randomized_svd_basis(W_train, n_components=n_svd, seed=0)
    coeff_train = project_svd(W_train, mean, components)

    pysr_py = Path("/private/tmp/pysr_venv_gpt52/bin/python")
    if not pysr_py.exists():
        raise RuntimeError("Expected PySR venv at /private/tmp/pysr_venv_gpt52")
    env = os.environ.copy()
    julia_exe = Path("/private/tmp/julia_env_gpt52/pyjuliapkg/install/bin/julia")
    if julia_exe.exists():
        env["PYTHON_JULIAPKG_EXE"] = str(julia_exe)
    env["PYTHON_JULIAPKG_PROJECT"] = "/private/tmp/pysr_julia_project_gpt52"
    env["JULIA_DEPOT_PATH"] = "/private/tmp/pysr_julia_depot_gpt52"
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_gpt52")

    all_exprs: list[dict] = []
    best_exprs: list[str] = []
    for j in range(n_svd):
        x_path = model_dir / "saved_model" / "X_train.npy"
        y_path = model_dir / "saved_model" / f"y_train_c{j}.npy"
        np.save(x_path, X_train)
        np.save(y_path, coeff_train[:, j])
        exprs_all: list[dict] = []
        for maxsize, niter in [(12, 80), (24, 120)]:
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
            exprs_all.extend(json.loads(out_json.read_text()))
        for e in exprs_all:
            e2 = dict(e)
            e2["coefficient_index"] = j
            all_exprs.append(e2)
        best_exprs.append(_select_best_sympy(exprs_all))

    _json_dump(model_dir / "saved_model" / "expressions.json", all_exprs)
    _json_dump(model_dir / "saved_model" / "best_equations.json", {"sympy": best_exprs})

    fns = [_sympy_callable(e, X_train.shape[1]) for e in best_exprs]
    def pred_coeffs(X):
        cols=[]
        for fn in fns:
            cols.append(np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), dtype=np.float32).reshape(-1))
        return np.stack(cols, axis=1)

    t0 = time.perf_counter()
    coeff_val = pred_coeffs(X_val)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(X_val)
    W_val_pred = reconstruct_svd(coeff_val, mean, components)
    W_tr_pred = reconstruct_svd(pred_coeffs(X_train), mean, components)

    from _lib import compute_per_sample_losses  # local import to keep file small

    tr_losses, _ = compute_per_sample_losses(W_tr_pred, W_train)
    va_losses, loss = compute_per_sample_losses(W_val_pred, W_val)

    np.save(model_dir / "saved_model" / "svd_mean.npy", mean.astype(np.float32))
    np.save(model_dir / "saved_model" / "svd_components.npy", components.astype(np.float32))
    _json_dump(model_dir / "saved_model" / "config.json", {"parameterization": parameterization, "display_name": display_name})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": tr_losses.tolist(), "val": va_losses.tolist()})
    scorecard = {
        "approach": "pysr",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "dynamics",
        "agent": "gpt52",
        "category": "symbolic/physics-informed",
        "parameterization": parameterization,
        "time_convention": "normalized_time",
        "loss": float(loss),
        "loss_components": {"rms_relative_error_x": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_params": 0,
        "notes": {"equations": best_exprs},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    (model_dir / "train.py").write_text("if __name__ == '__main__':\n    raise SystemExit('Run run_symbolic.py')\n")
    (model_dir / "predict.py").write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import parameterize, reconstruct_svd  # noqa: E402


def _sympy_callable(expr: str, n_features: int):
    xs = sp.symbols(" ".join([f"x{i}" for i in range(n_features)]))
    parsed = sp.sympify(expr.replace("^", "**"))
    return sp.lambdify(xs, parsed, "numpy")


def predict(model_dir: str | Path, params_raw: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    mean = np.load(model_dir / "saved_model" / "svd_mean.npy")
    comps = np.load(model_dir / "saved_model" / "svd_components.npy")
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    eqs = json.loads((model_dir / "saved_model" / "best_equations.json").read_text())["sympy"]
    X = parameterize(params_raw, cfg["parameterization"]).astype(np.float32)
    fns = [_sympy_callable(e, X.shape[1]) for e in eqs]
    cols = [np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), dtype=np.float32).reshape(-1) for fn in fns]
    coeffs = np.stack(cols, axis=1)
    x = reconstruct_svd(coeffs, mean, comps)
    return {"x": np.asarray(x, dtype=np.float64)}
"""
    )

    update_dynamics_comparison(work_dir)
    append_changelog_entry(
        agent_root,
        f"\n## [D-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_dt_now()}\n"
        "- **Benchmark**: dynamics\n"
        "- **Category**: symbolic/physics-informed\n"
        "- **Method**: PySR on first 3 SVD coefficients\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Loss**: {loss:.4e}\n",
    )


def _dt_now() -> str:
    import datetime as dt

    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def run_gplearn(*, work_dir: Path, agent_root: Path, model_dir: Path, approach_number: int) -> None:
    from gplearn.genetic import SymbolicRegressor

    # Compatibility shim for newer sklearn.
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

    train = load_dynamics_split("datasets/dynamics/dynamics_training.h5", time_convention="normalized_time")
    val = load_dynamics_split("datasets/dynamics/dynamics_validation.h5", time_convention="normalized_time")

    display_name = "gplearn (SVD coeffs, raw)"
    parameterization = "raw_6d"
    n_svd = 4

    X_train = parameterize(train.params_raw, parameterization).astype(np.float32)
    X_val = parameterize(val.params_raw, parameterization).astype(np.float32)
    W_train = train.x.astype(np.float32)
    W_val = val.x.astype(np.float32)

    mean, components = randomized_svd_basis(W_train, n_components=n_svd, seed=0)
    coeff_train = project_svd(W_train, mean, components)

    models = []
    exprs = []
    for j in range(n_svd):
        est = SymbolicRegressor(
            population_size=4000,
            generations=35,
            tournament_size=20,
            function_set=["add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"],
            metric="mse",
            parsimony_coefficient=0.001,
            max_samples=1.0,
            verbose=0,
            random_state=42 + j,
        )
        est.fit(X_train, coeff_train[:, j])
        est.n_features_in_ = X_train.shape[1]
        models.append(est)
        exprs.append(str(est._program))

    joblib.dump(models, model_dir / "saved_model" / "gplearn_models.joblib")
    _json_dump(model_dir / "saved_model" / "expressions.json", [{"coefficient_index": j, "expression": exprs[j]} for j in range(n_svd)])

    coeff_val = np.stack([m.predict(X_val).astype(np.float32) for m in models], axis=1)
    coeff_tr = np.stack([m.predict(X_train).astype(np.float32) for m in models], axis=1)
    W_val_pred = reconstruct_svd(coeff_val, mean, components)
    W_tr_pred = reconstruct_svd(coeff_tr, mean, components)

    from _lib import compute_per_sample_losses

    tr_losses, _ = compute_per_sample_losses(W_tr_pred, W_train)
    va_losses, loss = compute_per_sample_losses(W_val_pred, W_val)

    np.save(model_dir / "saved_model" / "svd_mean.npy", mean.astype(np.float32))
    np.save(model_dir / "saved_model" / "svd_components.npy", components.astype(np.float32))
    _json_dump(model_dir / "saved_model" / "config.json", {"parameterization": parameterization, "display_name": display_name})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": tr_losses.tolist(), "val": va_losses.tolist()})
    scorecard = {
        "approach": "gplearn",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "dynamics",
        "agent": "gpt52",
        "category": "symbolic/physics-informed",
        "parameterization": parameterization,
        "time_convention": "normalized_time",
        "loss": float(loss),
        "loss_components": {"rms_relative_error_x": float(loss)},
        "runtime_ms": 0.0,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_params": 0,
        "notes": {"expressions": exprs},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    (model_dir / "train.py").write_text("if __name__ == '__main__':\n    raise SystemExit('Run run_symbolic.py')\n")
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
    models = joblib.load(model_dir / "saved_model" / "gplearn_models.joblib")
    X = parameterize(params_raw, cfg["parameterization"]).astype(np.float32)
    coeffs = np.stack([m.predict(X).astype(np.float32) for m in models], axis=1)
    x = reconstruct_svd(coeffs, mean, comps)
    return {"x": np.asarray(x, dtype=np.float64)}
"""
    )

    update_dynamics_comparison(work_dir)
    append_changelog_entry(
        agent_root,
        f"\n## [D-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_dt_now()}\n"
        "- **Benchmark**: dynamics\n"
        "- **Category**: symbolic/physics-informed\n"
        "- **Method**: gplearn SymbolicRegressor on 4 SVD coefficients\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Loss**: {loss:.4e}\n",
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

