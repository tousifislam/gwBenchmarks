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
    load_validity_split,
    parameterize,
    rmse,
    update_validity_comparison,
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


def _dt() -> str:
    import datetime as dt

    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def run_pysr(*, work_dir: Path, model_dir: Path, approach_number: int) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    train = load_validity_split("datasets/validity/validity_training.h5")
    val = load_validity_split("datasets/validity/validity_validation.h5")

    display_name = "PySR (log features)"
    parameterization = "log_4d"
    X_train = parameterize(train.params_raw, parameterization).astype(np.float32)
    X_val = parameterize(val.params_raw, parameterization).astype(np.float32)
    y_train = train.y_log10.astype(np.float32)
    y_val = val.y_log10.astype(np.float32)

    pysr_py = Path("/private/tmp/pysr_venv_gpt52/bin/python")
    if not pysr_py.exists():
        raise RuntimeError("Expected PySR venv at /private/tmp/pysr_venv_gpt52; run waveform/remnant symbolic setup first.")

    env = os.environ.copy()
    julia_exe = Path("/private/tmp/julia_env_gpt52/pyjuliapkg/install/bin/julia")
    if julia_exe.exists():
        env["PYTHON_JULIAPKG_EXE"] = str(julia_exe)
    env["PYTHON_JULIAPKG_PROJECT"] = "/private/tmp/pysr_julia_project_gpt52"
    env["JULIA_DEPOT_PATH"] = "/private/tmp/pysr_julia_depot_gpt52"
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_gpt52")

    x_path = model_dir / "saved_model" / "X_train.npy"
    y_path = model_dir / "saved_model" / "y_train.npy"
    np.save(x_path, X_train)
    np.save(y_path, y_train)

    exprs_all: list[dict] = []
    for maxsize, niter in [(12, 100), (24, 160)]:
        out_json = model_dir / "saved_model" / f"expressions_max{maxsize}.json"
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

    _json_dump(model_dir / "saved_model" / "expressions.json", exprs_all)
    best_expr = _select_best_sympy(exprs_all)
    _json_dump(model_dir / "saved_model" / "best_equation.json", {"sympy": best_expr})

    fn = _sympy_callable(best_expr, X_train.shape[1])
    t0 = time.perf_counter()
    pred_val = np.asarray(fn(*[X_val[:, i] for i in range(X_val.shape[1])]), dtype=np.float64).reshape(-1)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(X_val)
    pred_tr = np.asarray(fn(*[X_train[:, i] for i in range(X_train.shape[1])]), dtype=np.float64).reshape(-1)

    loss = float(rmse(pred_val, y_val))
    abs_err_tr = np.abs(pred_tr - y_train)
    abs_err_va = np.abs(pred_val - y_val)

    _json_dump(model_dir / "saved_model" / "config.json", {"parameterization": parameterization, "display_name": display_name})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": abs_err_tr.tolist(), "val": abs_err_va.tolist()})
    scorecard = {
        "approach": "pysr",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "validity",
        "agent": "gpt52",
        "category": "symbolic/analytical",
        "parameterization": parameterization,
        "loss": float(loss),
        "loss_components": {"log_rmse": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_params": 0,
        "notes": {"equation": best_expr},
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
from _lib import parameterize  # noqa: E402


def _sympy_callable(expr: str, n_features: int):
    xs = sp.symbols(" ".join([f"x{i}" for i in range(n_features)]))
    parsed = sp.sympify(expr.replace("^", "**"))
    return sp.lambdify(xs, parsed, "numpy")


def predict(model_dir: str | Path, params_raw: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    best = json.loads((model_dir / "saved_model" / "best_equation.json").read_text())["sympy"]
    X = parameterize(np.asarray(params_raw, dtype=np.float64), cfg["parameterization"]).astype(np.float32)
    fn = _sympy_callable(best, X.shape[1])
    y_log = np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), dtype=np.float64).reshape(-1)
    y = np.power(10.0, y_log)
    return {"mm_td": y, "log10_mm_td": y_log}
"""
    )

    update_validity_comparison(work_dir, error_floor=0.0)
    append_changelog_entry(
        work_dir,
        f"\n## [V-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_dt()}\n"
        "- **Benchmark**: validity\n"
        "- **Category**: symbolic/analytical\n"
        "- **Method**: PySR\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Loss (RMSE log10(mm))**: {loss:.4e}\n",
    )


def run_gplearn(*, work_dir: Path, model_dir: Path, approach_number: int) -> None:
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

    train = load_validity_split("datasets/validity/validity_training.h5")
    val = load_validity_split("datasets/validity/validity_validation.h5")

    display_name = "gplearn (raw)"
    parameterization = "raw_4d"
    X_train = parameterize(train.params_raw, parameterization).astype(np.float32)
    X_val = parameterize(val.params_raw, parameterization).astype(np.float32)
    y_train = train.y_log10.astype(np.float32)
    y_val = val.y_log10.astype(np.float32)

    est = SymbolicRegressor(
        population_size=5000,
        generations=40,
        tournament_size=20,
        function_set=["add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"],
        metric="mse",
        parsimony_coefficient=0.001,
        max_samples=1.0,
        verbose=0,
        random_state=42,
    )
    est.fit(X_train, y_train)
    est.n_features_in_ = X_train.shape[1]

    t0 = time.perf_counter()
    pred_val = est.predict(X_val).astype(np.float64)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(X_val)
    pred_tr = est.predict(X_train).astype(np.float64)

    loss = float(rmse(pred_val, y_val))
    abs_err_tr = np.abs(pred_tr - y_train)
    abs_err_va = np.abs(pred_val - y_val)

    joblib.dump(est, model_dir / "saved_model" / "gplearn_model.joblib")
    _json_dump(model_dir / "saved_model" / "expressions.json", [{"expression": str(est._program)}])
    _json_dump(model_dir / "saved_model" / "config.json", {"parameterization": parameterization, "display_name": display_name})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": abs_err_tr.tolist(), "val": abs_err_va.tolist()})

    scorecard = {
        "approach": "gplearn",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "validity",
        "agent": "gpt52",
        "category": "symbolic/analytical",
        "parameterization": parameterization,
        "loss": float(loss),
        "loss_components": {"log_rmse": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_params": 0,
        "notes": {"expression": str(est._program)},
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
from _lib import parameterize  # noqa: E402


def predict(model_dir: str | Path, params_raw: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    est = joblib.load(model_dir / "saved_model" / "gplearn_model.joblib")
    X = parameterize(np.asarray(params_raw, dtype=np.float64), cfg["parameterization"]).astype(np.float32)
    y_log = np.asarray(est.predict(X), dtype=np.float64).reshape(-1)
    y = np.power(10.0, y_log)
    return {"mm_td": y, "log10_mm_td": y_log}
"""
    )

    update_validity_comparison(work_dir, error_floor=0.0)
    append_changelog_entry(
        work_dir,
        f"\n## [V-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_dt()}\n"
        "- **Benchmark**: validity\n"
        "- **Category**: symbolic/analytical\n"
        "- **Method**: gplearn\n"
        f"- **Parameterization**: {parameterization}\n"
        f"- **Loss (RMSE log10(mm))**: {loss:.4e}\n",
    )


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    models_dir = work_dir / "models"

    pysr_dir = models_dir / "19_pysr_log"
    if not (pysr_dir / "predict.py").exists():
        run_pysr(work_dir=work_dir, model_dir=pysr_dir, approach_number=19)

    gpl_dir = models_dir / "20_gplearn_raw"
    if not (gpl_dir / "predict.py").exists():
        run_gplearn(work_dir=work_dir, model_dir=gpl_dir, approach_number=20)


if __name__ == "__main__":
    main()

