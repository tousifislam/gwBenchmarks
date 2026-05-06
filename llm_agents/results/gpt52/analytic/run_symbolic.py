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
    AnalyticBench,
    append_changelog_entry,
    load_analytic_split,
    q_feature,
    update_analytic_comparison,
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


def _build_symbolic_training_data(split, *, n_per_sim: int = 1200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    Xs = []
    yr = []
    yi = []
    for sim in split.sims:
        t = sim.t
        h = sim.h
        n = len(t)
        if n == 0:
            continue
        # mixture: uniform + near peak
        iu = rng.integers(0, n, size=max(1, n_per_sim // 2))
        mask = (t >= -400.0) & (t <= 200.0)
        idx = np.where(mask)[0]
        if idx.size:
            ip = rng.integers(idx.min(), idx.max() + 1, size=n_per_sim - iu.size)
        else:
            ip = rng.integers(0, n, size=n_per_sim - iu.size)
        ii = np.unique(np.concatenate([iu, ip]))
        tt = t[ii].astype(np.float64)
        # features: x0=t/1000, x1=eta
        x0 = (tt / 1000.0).astype(np.float32)
        x1 = np.full_like(x0, float(q_feature(sim.q, "eta")), dtype=np.float32)
        Xs.append(np.stack([x0, x1], axis=1))
        yr.append(h[ii].real.astype(np.float32))
        yi.append(h[ii].imag.astype(np.float32))
    X = np.concatenate(Xs, axis=0)
    y_real = np.concatenate(yr, axis=0)
    y_imag = np.concatenate(yi, axis=0)
    return X, y_real, y_imag


def run_pysr(*, work_dir: Path, model_dir: Path, approach_number: int) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    tr = load_analytic_split("datasets/analytic/analytic_training.h5")
    va = load_analytic_split("datasets/analytic/analytic_validation.h5")
    bench = AnalyticBench()

    display_name = "PySR (t/1000, eta)"
    param_desc = "x0=t/1000, x1=eta=q/(1+q)^2"

    X, y_real, y_imag = _build_symbolic_training_data(tr, n_per_sim=1200)

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
    np.save(x_path, X)

    expressions_all: list[dict] = []
    best_eq: dict[str, str] = {}
    for target_name, y in [("h_real", y_real), ("h_imag", y_imag)]:
        y_path = model_dir / "saved_model" / f"y_train_{target_name}.npy"
        np.save(y_path, y)
        exprs_t: list[dict] = []
        for maxsize, niter in [(12, 120), (20, 180)]:
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
            exprs = json.loads(out_json.read_text())
            for e in exprs:
                e2 = dict(e)
                e2["target"] = target_name
                exprs_t.append(e2)
                expressions_all.append(e2)
        best_eq[target_name] = _select_best_sympy(exprs_t)

    _json_dump(model_dir / "saved_model" / "expressions.json", expressions_all)
    _json_dump(model_dir / "saved_model" / "best_equations.json", {"inputs": param_desc, **best_eq})

    (model_dir / "expression.txt").write_text(
        "# PySR closed form\n\n"
        f"Inputs: {param_desc}\n\n"
        f"h_real(t,q) = {best_eq['h_real']}\n"
        f"h_imag(t,q) = {best_eq['h_imag']}\n"
        "h22(t,q) = h_real + i*h_imag\n"
    )

    fr = _sympy_callable(best_eq["h_real"], 2)
    fi = _sympy_callable(best_eq["h_imag"], 2)

    def pred_for_sim(sim):
        x0 = (sim.t / 1000.0).astype(np.float64)
        x1 = np.full_like(x0, float(q_feature(sim.q, "eta")), dtype=np.float64)
        hr = np.asarray(fr(x0, x1), dtype=np.float64).reshape(-1)
        hi = np.asarray(fi(x0, x1), dtype=np.float64).reshape(-1)
        return hr + 1j * hi

    def eval_split(split):
        per = []
        comps_sum = {}
        for sim in split.sims:
            hp = pred_for_sim(sim)
            loss, comps = bench.compute_loss({"waveform": hp}, {"waveform": sim.h, "dt": split.dt_geometric})
            per.append(float(loss))
            for k, v in comps.items():
                comps_sum[k] = comps_sum.get(k, 0.0) + float(v)
        n = max(1, len(per))
        return per, {k: v / n for k, v in comps_sum.items()}

    t0 = time.perf_counter()
    n_pts = 0
    for sim in va.sims:
        _ = pred_for_sim(sim)
        n_pts += int(len(sim.t))
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / max(1, n_pts)

    per_tr, _ = eval_split(tr)
    per_va, comps_mean = eval_split(va)
    loss = float(np.mean(per_va)) if per_va else float("inf")

    _json_dump(model_dir / "saved_model" / "config.json", {"display_name": display_name, "category": "symbolic regression"})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": per_tr, "val": per_va})
    scorecard = {
        "approach": "pysr",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "analytic",
        "agent": "gpt52",
        "category": "symbolic regression",
        "parameterization": "eta + t/1000",
        "loss": float(loss),
        "loss_components": comps_mean,
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(tr.sims)),
        "n_val": int(len(va.sims)),
        "n_params": 0,
        "n_terms": 0,
        "expression_file": "expression.txt",
        "notes": {"best_equations_file": "saved_model/best_equations.json"},
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
from _lib import q_feature  # noqa: E402


def _sympy_callable(expr: str):
    x0, x1 = sp.Symbol("x0"), sp.Symbol("x1")
    parsed = sp.sympify(expr.replace("^", "**"))
    return sp.lambdify((x0, x1), parsed, "numpy")


def predict(model_dir: str | Path, q: float, t: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    best = json.loads((model_dir / "saved_model" / "best_equations.json").read_text())
    fr = _sympy_callable(best["h_real"])
    fi = _sympy_callable(best["h_imag"])
    x0 = (np.asarray(t, dtype=np.float64) / 1000.0).reshape(-1)
    x1 = np.full_like(x0, float(q_feature(float(q), "eta")), dtype=np.float64)
    hr = np.asarray(fr(x0, x1), dtype=np.float64).reshape(-1)
    hi = np.asarray(fi(x0, x1), dtype=np.float64).reshape(-1)
    return {"waveform": hr + 1j * hi}
"""
    )

    update_analytic_comparison(work_dir)
    append_changelog_entry(
        work_dir,
        f"\n## [A-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {time.strftime('%Y-%m-%d %H:%M')}\n"
        "- **Benchmark**: analytic\n"
        "- **Category**: symbolic regression\n"
        f"- **Loss (mean FD mismatch)**: {loss:.4e}\n",
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

    tr = load_analytic_split("datasets/analytic/analytic_training.h5")
    va = load_analytic_split("datasets/analytic/analytic_validation.h5")
    bench = AnalyticBench()

    display_name = "gplearn (t/1000, eta)"
    param_desc = "x0=t/1000, x1=eta=q/(1+q)^2"

    X, y_real, y_imag = _build_symbolic_training_data(tr, n_per_sim=900)

    def fit_one(y: np.ndarray) -> SymbolicRegressor:
        est = SymbolicRegressor(
            population_size=4000,
            generations=30,
            tournament_size=20,
            function_set=["add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"],
            metric="mse",
            parsimony_coefficient=0.001,
            max_samples=1.0,
            verbose=0,
            random_state=42,
        )
        est.fit(X, y)
        est.n_features_in_ = 2
        return est

    est_r = fit_one(y_real)
    est_i = fit_one(y_imag)

    (model_dir / "expression.txt").write_text(
        "# gplearn closed form\n\n"
        f"Inputs: {param_desc}\n\n"
        f"h_real(t,q) = {str(est_r._program)}\n"
        f"h_imag(t,q) = {str(est_i._program)}\n"
        "h22(t,q) = h_real + i*h_imag\n"
    )

    def pred_for_sim(sim):
        x0 = (sim.t / 1000.0).astype(np.float64)
        x1 = np.full_like(x0, float(q_feature(sim.q, "eta")), dtype=np.float64)
        Xs = np.stack([x0.astype(np.float32), x1.astype(np.float32)], axis=1)
        hr = est_r.predict(Xs).astype(np.float64)
        hi = est_i.predict(Xs).astype(np.float64)
        return hr + 1j * hi

    def eval_split(split):
        per = []
        comps_sum = {}
        for sim in split.sims:
            hp = pred_for_sim(sim)
            loss, comps = bench.compute_loss({"waveform": hp}, {"waveform": sim.h, "dt": split.dt_geometric})
            per.append(float(loss))
            for k, v in comps.items():
                comps_sum[k] = comps_sum.get(k, 0.0) + float(v)
        n = max(1, len(per))
        return per, {k: v / n for k, v in comps_sum.items()}

    t0 = time.perf_counter()
    n_pts = 0
    for sim in va.sims:
        _ = pred_for_sim(sim)
        n_pts += int(len(sim.t))
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / max(1, n_pts)

    per_tr, _ = eval_split(tr)
    per_va, comps_mean = eval_split(va)
    loss = float(np.mean(per_va)) if per_va else float("inf")

    joblib.dump({"real": est_r, "imag": est_i}, model_dir / "saved_model" / "gplearn_models.joblib")
    _json_dump(
        model_dir / "saved_model" / "expressions.json",
        [{"target": "h_real", "expression": str(est_r._program)}, {"target": "h_imag", "expression": str(est_i._program)}],
    )
    _json_dump(model_dir / "saved_model" / "config.json", {"display_name": display_name, "category": "symbolic regression"})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": per_tr, "val": per_va})
    scorecard = {
        "approach": "gplearn",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "analytic",
        "agent": "gpt52",
        "category": "symbolic regression",
        "parameterization": "eta + t/1000",
        "loss": float(loss),
        "loss_components": comps_mean,
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(tr.sims)),
        "n_val": int(len(va.sims)),
        "n_params": 0,
        "n_terms": 0,
        "expression_file": "expression.txt",
        "notes": {"models_file": "saved_model/gplearn_models.joblib"},
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    (model_dir / "train.py").write_text("if __name__ == '__main__':\n    raise SystemExit('Run run_symbolic.py')\n")
    (model_dir / "predict.py").write_text(
        """from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import q_feature  # noqa: E402


def predict(model_dir: str | Path, q: float, t: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    models = joblib.load(model_dir / "saved_model" / "gplearn_models.joblib")
    x0 = (np.asarray(t, dtype=np.float64) / 1000.0).reshape(-1)
    x1 = np.full_like(x0, float(q_feature(float(q), "eta")), dtype=np.float64)
    X = np.stack([x0.astype(np.float32), x1.astype(np.float32)], axis=1)
    hr = np.asarray(models["real"].predict(X), dtype=np.float64).reshape(-1)
    hi = np.asarray(models["imag"].predict(X), dtype=np.float64).reshape(-1)
    return {"waveform": hr + 1j * hi}
"""
    )

    update_analytic_comparison(work_dir)
    append_changelog_entry(
        work_dir,
        f"\n## [A-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {time.strftime('%Y-%m-%d %H:%M')}\n"
        "- **Benchmark**: analytic\n"
        "- **Category**: symbolic regression\n"
        f"- **Loss (mean FD mismatch)**: {loss:.4e}\n",
    )


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    models_dir = work_dir / "models"

    pysr_dir = models_dir / "19_pysr_t_eta"
    if not (pysr_dir / "scorecard.json").exists():
        run_pysr(work_dir=work_dir, model_dir=pysr_dir, approach_number=19)

    gpl_dir = models_dir / "20_gplearn_t_eta"
    if not (gpl_dir / "scorecard.json").exists():
        run_gplearn(work_dir=work_dir, model_dir=gpl_dir, approach_number=20)


if __name__ == "__main__":
    main()

