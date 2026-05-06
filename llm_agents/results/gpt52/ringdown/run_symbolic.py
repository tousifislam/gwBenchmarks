from __future__ import annotations

import datetime as _dt
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
    load_ringdown_split,
    per_sample_loss,
    spin_feature,
    update_ringdown_comparison,
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


def _sympy_callable(expr: str):
    x0 = sp.Symbol("x0")
    parsed = sp.sympify(expr.replace("^", "**"))
    return sp.lambdify((x0,), parsed, "numpy")


def run_pysr(*, work_dir: Path, agent_root: Path, model_dir: Path, approach_number: int) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    mode = "l2/m+2/n0"
    # If a previous run crashed after writing scorecard/expressions, finalize without rerunning PySR.
    if (model_dir / "scorecard.json").exists() and (model_dir / "saved_model" / "best_equations.json").exists() and not (model_dir / "predict.py").exists():
        scorecard = json.loads((model_dir / "scorecard.json").read_text())
        loss = float(scorecard.get("loss", float("nan")))
        (model_dir / "train.py").write_text("if __name__ == '__main__':\n    raise SystemExit('Run run_symbolic.py')\n")
        if not (model_dir / "predict.py").exists():
            (model_dir / "predict.py").write_text(
                """from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import spin_feature  # noqa: E402


def _sympy_callable(expr: str):
    x0 = sp.Symbol("x0")
    parsed = sp.sympify(expr.replace("^", "**"))
    return sp.lambdify((x0,), parsed, "numpy")


def predict(model_dir: str | Path, spin_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model_dir = Path(model_dir)
    best = json.loads((model_dir / "saved_model" / "best_equations.json").read_text())
    x = spin_feature(np.asarray(spin_array, dtype=np.float64), best["spin_parameterization"])
    fr = _sympy_callable(best["omega_r"])
    fi = _sympy_callable(best["omega_i"])
    return np.asarray(fr(x), dtype=np.float64).reshape(-1), np.asarray(fi(x), dtype=np.float64).reshape(-1)
"""
            )

        update_ringdown_comparison(work_dir, mode=mode)
        append_changelog_entry(
            agent_root,
            f"\n## [Q-{approach_number:02d}] PySR (log)\n"
            f"- **Time**: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            "- **Benchmark**: ringdown\n"
            "- **Category**: symbolic regression\n"
            f"- **Mode**: {mode}\n"
            f"- **Loss**: {loss:.4e}\n",
        )
        return

    tr = load_ringdown_split("datasets/ringdown/ringdown_training.h5", mode=mode)
    va = load_ringdown_split("datasets/ringdown/ringdown_validation.h5", mode=mode)
    param = "log_compact"

    x_tr = spin_feature(tr.spin, param).astype(np.float32)
    x_va = spin_feature(va.spin, param).astype(np.float32)
    X_tr = x_tr.reshape(-1, 1)

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

    x_path = model_dir / "saved_model" / "X_train.npy"
    np.save(x_path, X_tr)

    exprs_r = []
    exprs_i = []
    for target_name, y in [("omega_r", tr.omega_r), ("omega_i", tr.omega_i)]:
        y_path = model_dir / "saved_model" / f"y_train_{target_name}.npy"
        np.save(y_path, np.asarray(y, dtype=np.float32))
        exprs_all: list[dict] = []
        for maxsize, niter in [(12, 120), (24, 200)]:
            out_json = model_dir / "saved_model" / f"expressions_{target_name}_max{maxsize}.json"
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
        best = _select_best_sympy(exprs_all)
        if target_name == "omega_r":
            exprs_r = exprs_all
            best_r = best
        else:
            exprs_i = exprs_all
            best_i = best

    _json_dump(model_dir / "saved_model" / "expressions.json", [{"target": "omega_r", "items": exprs_r}, {"target": "omega_i", "items": exprs_i}])
    _json_dump(model_dir / "saved_model" / "best_equations.json", {"omega_r": best_r, "omega_i": best_i, "spin_parameterization": param})

    fr = _sympy_callable(best_r)
    fi = _sympy_callable(best_i)
    t0 = time.perf_counter()
    pr = np.asarray(fr(x_va), dtype=np.float64).reshape(-1)
    pi = np.asarray(fi(x_va), dtype=np.float64).reshape(-1)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(x_va)
    ptr = np.asarray(fr(x_tr), dtype=np.float64).reshape(-1)
    pti = np.asarray(fi(x_tr), dtype=np.float64).reshape(-1)

    per_tr, _, _ = per_sample_loss(ptr, pti, tr.omega_r, tr.omega_i)
    per_va, comps, loss = per_sample_loss(pr, pi, va.omega_r, va.omega_i)

    _json_dump(model_dir / "saved_model" / "config.json", {"parameterization": param, "mode": mode})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": per_tr.tolist(), "val": per_va.tolist()})
    scorecard = {
        "approach": "pysr",
        "display_name": "PySR (log)",
        "approach_number": int(approach_number),
        "benchmark": "ringdown",
        "agent": "gpt52",
        "parameterization": param,
        "mode": mode,
        "loss": float(loss),
        "loss_components": comps,
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
        "n_params": 0,
        "notes": {"expression_omega_r": best_r, "expression_omega_i": best_i},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)
    (model_dir / "train.py").write_text("if __name__ == '__main__':\n    raise SystemExit('Run run_symbolic.py')\n")
    (model_dir / "predict.py").write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import spin_feature  # noqa: E402


def _sympy_callable(expr: str):
    x0 = sp.Symbol("x0")
    parsed = sp.sympify(expr.replace("^", "**"))
    return sp.lambdify((x0,), parsed, "numpy")


def predict(model_dir: str | Path, spin_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "saved_model" / "config.json").read_text())
    best = json.loads((model_dir / "saved_model" / "best_equations.json").read_text())
    x = spin_feature(np.asarray(spin_array, dtype=np.float64), best["spin_parameterization"])
    fr = _sympy_callable(best["omega_r"])
    fi = _sympy_callable(best["omega_i"])
    return np.asarray(fr(x), dtype=np.float64).reshape(-1), np.asarray(fi(x), dtype=np.float64).reshape(-1)
"""
    )

    update_ringdown_comparison(work_dir, mode=mode)
    append_changelog_entry(
        agent_root,
        f"\n## [Q-{approach_number:02d}] PySR (log)\n"
        f"- **Time**: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        "- **Benchmark**: ringdown\n"
        "- **Category**: symbolic regression\n"
        f"- **Mode**: {mode}\n"
        f"- **Loss**: {loss:.4e}\n",
    )


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

    mode = "l2/m+2/n0"
    tr = load_ringdown_split("datasets/ringdown/ringdown_training.h5", mode=mode)
    va = load_ringdown_split("datasets/ringdown/ringdown_validation.h5", mode=mode)
    param = "raw_a"

    x_tr = spin_feature(tr.spin, param).astype(np.float32)
    x_va = spin_feature(va.spin, param).astype(np.float32)
    X_tr = x_tr.reshape(-1, 1)
    X_va = x_va.reshape(-1, 1)

    def fit_one(y):
        est = SymbolicRegressor(
            population_size=5000,
            generations=50,
            tournament_size=20,
            function_set=["add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"],
            metric="mse",
            parsimony_coefficient=0.001,
            max_samples=1.0,
            verbose=0,
            random_state=42,
        )
        est.fit(X_tr, np.asarray(y, dtype=np.float32))
        est.n_features_in_ = 1
        return est

    est_r = fit_one(tr.omega_r)
    est_i = fit_one(tr.omega_i)

    t0 = time.perf_counter()
    pr = est_r.predict(X_va).astype(np.float64)
    pi = est_i.predict(X_va).astype(np.float64)
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / len(x_va)
    ptr = est_r.predict(X_tr).astype(np.float64)
    pti = est_i.predict(X_tr).astype(np.float64)

    per_tr, _, _ = per_sample_loss(ptr, pti, tr.omega_r, tr.omega_i)
    per_va, comps, loss = per_sample_loss(pr, pi, va.omega_r, va.omega_i)

    joblib.dump({"omega_r": est_r, "omega_i": est_i}, model_dir / "saved_model" / "gplearn_models.joblib")
    _json_dump(model_dir / "saved_model" / "expressions.json", [{"target": "omega_r", "expression": str(est_r._program)}, {"target": "omega_i", "expression": str(est_i._program)}])
    _json_dump(model_dir / "saved_model" / "config.json", {"parameterization": param, "mode": mode})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": per_tr.tolist(), "val": per_va.tolist()})
    scorecard = {
        "approach": "gplearn",
        "display_name": "gplearn (raw)",
        "approach_number": int(approach_number),
        "benchmark": "ringdown",
        "agent": "gpt52",
        "parameterization": param,
        "mode": mode,
        "loss": float(loss),
        "loss_components": comps,
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
        "n_params": 0,
        "notes": {"expression_omega_r": str(est_r._program), "expression_omega_i": str(est_i._program)},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)
    (model_dir / "train.py").write_text("if __name__ == '__main__':\n    raise SystemExit('Run run_symbolic.py')\n")
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
    models = joblib.load(model_dir / "saved_model" / "gplearn_models.joblib")
    x = spin_feature(np.asarray(spin_array, dtype=np.float64), cfg["parameterization"])
    X = x.reshape(-1, 1)
    return np.asarray(models["omega_r"].predict(X), dtype=np.float64).reshape(-1), np.asarray(models["omega_i"].predict(X), dtype=np.float64).reshape(-1)
"""
    )

    update_ringdown_comparison(work_dir, mode=mode)
    append_changelog_entry(
        agent_root,
        f"\n## [Q-{approach_number:02d}] gplearn (raw)\n"
        f"- **Time**: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        "- **Benchmark**: ringdown\n"
        "- **Category**: symbolic regression\n"
        f"- **Mode**: {mode}\n"
        f"- **Loss**: {loss:.4e}\n",
    )


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    agent_root = work_dir.parent
    models_dir = work_dir / "models"

    pysr_dir = models_dir / "19_pysr_log"
    if not (pysr_dir / "predict.py").exists():
        run_pysr(work_dir=work_dir, agent_root=agent_root, model_dir=pysr_dir, approach_number=19)

    gpl_dir = models_dir / "20_gplearn_raw"
    if not (gpl_dir / "predict.py").exists():
        run_gplearn(work_dir=work_dir, agent_root=agent_root, model_dir=gpl_dir, approach_number=20)


if __name__ == "__main__":
    main()
