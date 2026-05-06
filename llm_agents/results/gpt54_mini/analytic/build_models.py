"""Build closed-form analytic waveform models for the Analytic Bench."""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwbenchmarks.metrics import frequency_domain_mismatch

from _lib import (
    RESULTS_DIR,
    append_text,
    evaluate_curve_basis_model,
    load_data,
    model_dir,
    q_basis,
    reparam,
    save_json,
    save_scorecard,
    sample_points,
    time_basis,
    time_scale,
    write_train_predict,
)

TRAIN_SIMS, VAL_SIMS = load_data()

OFFLINE_JULIA_PROJECT = Path(__file__).resolve().parent / "_julia_project_926"
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(OFFLINE_JULIA_PROJECT))
os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/private/tmp/pysr_julia_env/pyjuliapkg/install/bin/julia")
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
os.environ.setdefault("JULIA_PKG_SERVER", "")
os.environ.setdefault("JULIA_DEPOT_PATH", "/private/tmp/gpt54_julia_depot2:/Users/tousifislam/.julia")


Q_FORMULAS = {
    "raw": "q",
    "eta": "q / (1 + q)^2",
    "delta": "(q - 1) / (q + 1)",
    "log_q": "log(q)",
    "sqrt_eta": "sqrt(q / (1 + q)^2)",
}

TIME_FORMULAS = {
    "poly_exp": ["1", "u", "u**2", "exp(u)", "exp(-u)"],
    "tanh_composite": ["1", "u", "tanh(2u)", "exp(u)", "exp(-u)"],
    "gauss_sum": ["1", "u", "G0(u)", "G1(u)", "G2(u)"],
    "lorentz_sum": ["1", "u", "L0(u)", "L1(u)", "L2(u)"],
    "damped_sin": ["1", "e^u cos(4u)", "e^u sin(4u)", "e^{2u} cos(8u)", "e^{2u} sin(8u)"],
    "poly_log": ["1", "u", "u^2", "log(1+s)", "s log(1+s)"],
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    category: str
    parameterization: str
    time_kind: str
    curve_family: str
    curve_degree: int = 3
    special_tool: str | None = None
    special_part: str | None = None
    special_index: int | None = None
    notes: str = ""


SPECS = [
    ModelSpec("phys_poly_raw", "physics", "raw", "poly_exp", "poly", 3, notes="Polynomial q-curves on a PN-like time basis."),
    ModelSpec("phys_poly_eta", "physics", "eta", "poly_exp", "poly", 3, notes="Polynomial q-curves in eta on a PN-like time basis."),
    ModelSpec("phys_pade_delta", "physics", "delta", "poly_exp", "pade", 2, notes="Padé q-curves on a PN-like time basis."),
    ModelSpec("phys_logq", "physics", "log_q", "poly_log", "poly", 3, notes="Polynomial q-curves in log(q) on a log-augmented time basis."),
    ModelSpec("composite_tanh_raw", "matched_composite", "raw", "tanh_composite", "poly_tanh", 3, notes="Smooth inspiral/merger/ringdown blending."),
    ModelSpec("composite_tanh_eta", "matched_composite", "eta", "tanh_composite", "poly_tanh", 3, notes="Smooth blended model in eta."),
    ModelSpec("composite_tanh_delta", "matched_composite", "delta", "tanh_composite", "poly_tanh", 3, notes="Smooth blended model in delta."),
    ModelSpec("composite_tanh_sqrt_eta", "matched_composite", "sqrt_eta", "tanh_composite", "poly_tanh", 3, notes="Smooth blended model in sqrt(eta)."),
    ModelSpec("functional_gauss_raw", "functional_optimization", "raw", "gauss_sum", "poly", 2, notes="Gaussian basis on time with polynomial q-curves."),
    ModelSpec("functional_gauss_eta", "functional_optimization", "eta", "gauss_sum", "poly", 2, notes="Gaussian basis on time with eta q-curves."),
    ModelSpec("functional_lorentz_raw", "functional_optimization", "raw", "lorentz_sum", "poly", 2, notes="Lorentzian basis on time with polynomial q-curves."),
    ModelSpec("functional_lorentz_delta", "functional_optimization", "delta", "lorentz_sum", "poly", 2, notes="Lorentzian basis on time with delta q-curves."),
    ModelSpec("functional_damped_sin_raw", "functional_optimization", "raw", "damped_sin", "poly", 2, notes="Damped oscillatory time basis with polynomial q-curves."),
    ModelSpec("functional_damped_sin_eta", "functional_optimization", "eta", "damped_sin", "poly", 2, notes="Damped oscillatory time basis with eta q-curves."),
    ModelSpec("symbolic_pysr_amp_eta", "symbolic", "eta", "poly_exp", "poly", 3, "pysr", "re", 0, notes="PySR on the dominant real coefficient curve."),
    ModelSpec("symbolic_pysr_phase_raw", "symbolic", "raw", "tanh_composite", "poly", 3, "pysr", "im", 1, notes="PySR on a phase-sensitive imaginary coefficient curve."),
    ModelSpec("symbolic_pysr_freq_delta", "symbolic", "delta", "damped_sin", "poly", 2, "pysr", "im", 2, notes="PySR on a frequency-sensitive imaginary coefficient curve."),
    ModelSpec("symbolic_gplearn_amp_raw", "symbolic", "raw", "poly_exp", "poly", 3, "gplearn", "re", 0, notes="gplearn on the dominant real coefficient curve."),
    ModelSpec("symbolic_gplearn_phase_eta", "symbolic", "eta", "tanh_composite", "poly", 3, "gplearn", "im", 1, notes="gplearn on a phase-sensitive imaginary coefficient curve."),
    ModelSpec("symbolic_gplearn_freq_delta", "symbolic", "delta", "damped_sin", "poly", 2, "gplearn", "im", 2, notes="gplearn on a frequency-sensitive imaginary coefficient curve."),
]


RESULTS: list[dict[str, Any]] = []
ERROR_DATA: dict[str, Any] = {}
ALL_EXPRESSIONS: list[dict[str, Any]] = []


def _q_features(q: np.ndarray, mode: str) -> np.ndarray:
    r = reparam(q, mode).reshape(-1)
    return r


def _fit_poly_curve(r: np.ndarray, y: np.ndarray, degree: int) -> dict[str, Any]:
    v = np.vander(r, N=degree + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(v, y, rcond=None)
    return {"family": "poly", "q_mode": "raw", "coeffs": coeffs.tolist()}


def _fit_pade_curve(r: np.ndarray, y: np.ndarray, num_deg: int, den_deg: int) -> dict[str, Any]:
    cols = [r**k for k in range(num_deg + 1)]
    cols.extend([-(y * (r**k)) for k in range(1, den_deg + 1)])
    A = np.column_stack(cols)
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    num = theta[: num_deg + 1]
    den = theta[num_deg + 1 :]
    return {"family": "pade", "q_mode": "raw", "num": num.tolist(), "den": den.tolist()}


def _fit_poly_tanh_curve(r: np.ndarray, y: np.ndarray, q_mode: str) -> dict[str, Any]:
    if q_mode == "raw":
        r0, width = 8.0, 2.5
    elif q_mode == "eta":
        r0, width = 0.16, 0.06
    elif q_mode == "delta":
        r0, width = 0.35, 0.12
    elif q_mode == "sqrt_eta":
        r0, width = 0.4, 0.08
    else:
        r0, width = float(np.median(r)), max(0.1, 0.25 * float(np.std(r)))
    s = np.tanh((r - r0) / width)
    A = np.column_stack([np.ones_like(r), r, r**2, s, r * s])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"family": "poly_tanh", "q_mode": q_mode, "coeffs": coeffs.tolist(), "r0": r0, "width": width}


def _fit_pysr_curve(r: np.ndarray, y: np.ndarray, q_mode: str, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from pysr import PySRRegressor

    x = r.reshape(-1, 1)
    pm = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "exp", "sin", "cos", "tanh"],
        maxsize=18,
        populations=8,
        progress=False,
        random_state=0,
        deterministic=True,
        parallelism="serial",
        temp_equation_file=True,
    )
    pm.fit(x, y)
    eqs = []
    df = pm.equations_.sort_values("loss").reset_index(drop=True)
    for _, row in df.iterrows():
        eq = str(row["equation"]).replace("x0", "r")
        eqs.append(
            {
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "equation": eq,
            }
        )
    best = eqs[0]
    model = {"family": "pysr", "q_mode": q_mode, "expr": best["equation"], "label": label}
    return model, eqs


def _fit_gplearn_curve(r: np.ndarray, y: np.ndarray, q_mode: str, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from gplearn.genetic import SymbolicRegressor

    x = r.reshape(-1, 1)
    est = SymbolicRegressor(
        population_size=1000,
        generations=35,
        tournament_size=20,
        function_set=["add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"],
        metric="mse",
        parsimony_coefficient=0.001,
        max_samples=1.0,
        verbose=0,
        random_state=42,
        n_jobs=1,
    )
    est.fit(x, y)
    model = {"family": "gplearn", "q_mode": q_mode, "estimator": est, "expr": str(est._program), "label": label}
    eqs = [{"complexity": int(est._program.length_), "loss": float(est._program.fitness_), "equation": str(est._program)}]
    return model, eqs


def _fit_waveform_coeffs(sim: dict[str, Any], time_kind: str, n_sample: int = 700) -> tuple[np.ndarray, np.ndarray, list[str]]:
    t = sim["t"]
    h = sim["h"]
    idx = np.linspace(0, len(t) - 1, min(n_sample, len(t))).astype(int)
    tb, basis_names = time_basis(time_scale(t[idx]), time_kind)
    coef_re, *_ = np.linalg.lstsq(tb, h.real[idx], rcond=None)
    coef_im, *_ = np.linalg.lstsq(tb, h.imag[idx], rcond=None)
    return coef_re.astype(np.float64), coef_im.astype(np.float64), basis_names


def _fit_curve_family(
    qs: np.ndarray,
    values: np.ndarray,
    q_mode: str,
    family: str,
    degree: int,
    special_tool: str | None = None,
    label: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    r = _q_features(qs, q_mode)
    if family == "poly":
        model = _fit_poly_curve(r, values, degree)
        return model, []
    if family == "pade":
        model = _fit_pade_curve(r, values, 2, 2)
        return model, []
    if family == "poly_tanh":
        model = _fit_poly_tanh_curve(r, values, q_mode)
        return model, []
    if family == "symbolic":
        if special_tool == "pysr":
            return _fit_pysr_curve(r, values, q_mode, label)
        if special_tool == "gplearn":
            return _fit_gplearn_curve(r, values, q_mode, label)
    raise ValueError(f"unknown curve family {family!r}")


def _fit_model(spec: ModelSpec, train_sims: list[dict[str, Any]]) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    re_coeffs = []
    im_coeffs = []
    basis_names = None
    t0 = time.time()
    for sim in train_sims:
        coef_re, coef_im, basis_names = _fit_waveform_coeffs(sim, spec.time_kind)
        re_coeffs.append(coef_re)
        im_coeffs.append(coef_im)
    re_coeffs = np.vstack(re_coeffs)
    im_coeffs = np.vstack(im_coeffs)
    qs = np.array([sim["q"] for sim in train_sims], dtype=np.float64)

    re_curves = []
    im_curves = []
    expressions: list[dict[str, Any]] = []
    for idx in range(re_coeffs.shape[1]):
        fam = spec.curve_family
        tool = None
        if spec.special_part == "re" and spec.special_index == idx:
            fam = "symbolic"
            tool = spec.special_tool
        model, exprs = _fit_curve_family(qs, re_coeffs[:, idx], spec.parameterization, fam, spec.curve_degree, tool, f"{spec.name}:re:{idx}")
        re_curves.append(model)
        expressions.extend(exprs)
    for idx in range(im_coeffs.shape[1]):
        fam = spec.curve_family
        tool = None
        if spec.special_part == "im" and spec.special_index == idx:
            fam = "symbolic"
            tool = spec.special_tool
        model, exprs = _fit_curve_family(qs, im_coeffs[:, idx], spec.parameterization, fam, spec.curve_degree, tool, f"{spec.name}:im:{idx}")
        im_curves.append(model)
        expressions.extend(exprs)

    model = {
        "type": "curve_basis",
        "time_kind": spec.time_kind,
        "parameterization": spec.parameterization,
        "basis_names": basis_names,
        "re_curves": re_curves,
        "im_curves": im_curves,
    }
    expr_text = _render_expression(spec, model)
    model["fit_time_s"] = time.time() - t0
    return model, expr_text, expressions


def _curve_expr(curve: dict[str, Any]) -> str:
    fam = curve["family"]
    if fam == "poly":
        coeffs = np.asarray(curve["coeffs"], dtype=float)
        parts = []
        for i, c in enumerate(coeffs):
            if i == 0:
                parts.append(f"{c:.16e}")
            elif i == 1:
                parts.append(f"({c:.16e})*r")
            else:
                parts.append(f"({c:.16e})*r**{i}")
        return " + ".join(parts) if parts else "0"
    if fam == "pade":
        num = np.asarray(curve["num"], dtype=float)
        den = np.asarray(curve["den"], dtype=float)
        n_parts = []
        for i, c in enumerate(num):
            term = f"{c:.16e}" if i == 0 else f"({c:.16e})*r**{i}"
            n_parts.append(term)
        d_parts = ["1.0"]
        for i, c in enumerate(den, start=1):
            d_parts.append(f"({c:.16e})*r**{i}")
        return f"({ ' + '.join(n_parts) }) / ({ ' + '.join(d_parts) })"
    if fam == "poly_tanh":
        c = np.asarray(curve["coeffs"], dtype=float)
        r0 = curve["r0"]
        w = curve["width"]
        s = f"tanh((r - {r0:.16e}) / {w:.16e})"
        return (
            f"{c[0]:.16e} + ({c[1]:.16e})*r + ({c[2]:.16e})*r**2 + "
            f"({c[3]:.16e})*({s}) + ({c[4]:.16e})*r*({s})"
        )
    if fam == "pysr":
        return curve["expr"]
    if fam == "gplearn":
        return curve["expr"]
    raise ValueError(f"unknown family {fam!r}")


def _render_expression(spec: ModelSpec, model: dict[str, Any]) -> str:
    basis = TIME_FORMULAS[spec.time_kind]
    lines = []
    lines.append(f"Model: {spec.name}")
    lines.append(f"Category: {spec.category}")
    lines.append(f"Parameterization: {spec.parameterization}")
    lines.append("u = t / 5000")
    lines.append(f"r = {Q_FORMULAS[spec.parameterization]}")
    lines.append("")
    lines.append("Time basis:")
    for idx, b in enumerate(basis):
        lines.append(f"  B{idx}(u) = {b}")
    lines.append("")
    lines.append("Coefficient curves:")
    for idx, curve in enumerate(model["re_curves"]):
        lines.append(f"  a{idx}(r) = {_curve_expr(curve)}")
    for idx, curve in enumerate(model["im_curves"]):
        lines.append(f"  b{idx}(r) = {_curve_expr(curve)}")
    lines.append("")
    re_terms = [f"a{idx}(r) * B{idx}(u)" for idx in range(len(model["re_curves"]))]
    im_terms = [f"b{idx}(r) * B{idx}(u)" for idx in range(len(model["im_curves"]))]
    lines.append("Re[h](t,q) = " + " + ".join(re_terms))
    lines.append("Im[h](t,q) = " + " + ".join(im_terms))
    lines.append("h(t,q) = Re[h](t,q) + i * Im[h](t,q)")
    return "\n".join(lines) + "\n"


def _per_sample_proxy(pred: np.ndarray, true: np.ndarray) -> float:
    num = np.abs(np.vdot(pred, true))
    den = np.linalg.norm(pred) * np.linalg.norm(true) + 1e-30
    return float(1.0 - num / den)


def _evaluate_model(model: dict[str, Any], sims: list[dict[str, Any]]) -> tuple[list[float], list[float], float, dict[str, float]]:
    proxies = []
    fd_losses = []
    per_mass = {f"mismatch_{m}Msun": [] for m in [40, 80, 120, 160, 200]}
    for sim in sims:
        pred = evaluate_curve_basis_model(model, np.array([sim["q"]]), sim["t"])
        true = sim["h"]
        proxies.append(_per_sample_proxy(pred, true))
        for m in [40, 80, 120, 160, 200]:
            try:
                fd_losses.append(
                    frequency_domain_mismatch(pred, true, dt_geometric=0.1, mtot_msun=float(m))
                )
            except Exception:
                fd_losses.append(1.0)
            per_mass[f"mismatch_{m}Msun"].append(fd_losses[-1])
    per_mass_mean = {k: float(np.mean(v)) for k, v in per_mass.items()}
    loss = float(np.mean(list(per_mass_mean.values())))
    return proxies, fd_losses, loss, per_mass_mean


def _write_progress_plot() -> None:
    if not RESULTS:
        return
    cmp = RESULTS_DIR / "comparison"
    cmp.mkdir(parents=True, exist_ok=True)
    ordered = sorted(RESULTS, key=lambda r: r["approach_number"])
    nums = [r["approach_number"] for r in ordered]
    losses = [r["loss"] for r in ordered]
    labels = [r["approach"] for r in ordered]
    best = np.minimum.accumulate(losses)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nums, losses, "o-", label="per approach", color="C0")
    ax.plot(nums, best, "s--", label="running best", color="C3")
    for n, l, nm in zip(nums, losses, labels):
        ax.annotate(nm, (n, l), fontsize=5, xytext=(0, 4), textcoords="offset points", rotation=45)
    ax.set_xlabel("Approach number")
    ax.set_ylabel("Validation mismatch")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cmp / "progress.png", dpi=150)
    fig.savefig(cmp / "progress.pdf")
    plt.close(fig)


def _write_summary_artifacts() -> None:
    cmp = RESULTS_DIR / "comparison"
    cmp.mkdir(parents=True, exist_ok=True)
    ordered = sorted(RESULTS, key=lambda r: r["loss"])
    save_json(cmp / "summary_table.json", ordered)
    save_json(cmp / "error_data.json", ERROR_DATA)
    if ordered:
        save_json(cmp / "best_model.json", ordered[0])
    save_json(cmp / "all_expressions.json", ALL_EXPRESSIONS)
    _write_progress_plot()


def _append_changelog_entry(result: dict[str, Any]) -> None:
    text = (
        f"## {result['approach_number']:02d} - {result['approach']}\n"
        f"- Category: {result['category']}\n"
        f"- Parameterization: {result['parameterization']}\n"
        f"- Loss: {result['loss']:.6e}\n"
        f"- Notes: {result['notes']}\n"
        f"- Reasoning: coefficient curves were fit on sampled waveform coefficients, then promoted to analytic q-curves.\n"
    )
    append_text(RESULTS_DIR / "CHANGELOG.md", text)


def _save_model_dir(spec: ModelSpec, model: dict[str, Any], expression: str, result: dict[str, Any], expr_entries: list[dict[str, Any]]) -> None:
    md = model_dir(result["approach_number"], spec.name)
    with open(md / "saved_model" / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (md / "expression.txt").write_text(expression)
    save_json(md / "saved_model" / "expressions.json", expr_entries)
    write_train_predict(md, spec.name)


def _scorecard(spec: ModelSpec, result_num: int, loss: float, per_mass: dict[str, float], train_time: float, eval_ms: float, notes: str, n_terms: int, expression_file: str) -> dict[str, Any]:
    return {
        "approach": spec.name,
        "approach_number": result_num,
        "benchmark": "analytic",
        "agent": "gpt54_mini",
        "category": spec.category,
        "parameterization": spec.parameterization,
        "loss": float(loss),
        "loss_components": per_mass,
        "runtime_ms": float(eval_ms),
        "n_train": len(TRAIN_SIMS),
        "n_val": len(VAL_SIMS),
        "n_params": int(n_terms),
        "n_terms": int(n_terms),
        "train_time_s": float(train_time),
        "expression_file": expression_file,
        "notes": notes,
    }


def _train_single_spec(spec: ModelSpec, approach_num: int) -> dict[str, Any]:
    md = model_dir(approach_num, spec.name)
    start = time.time()
    model, expression_text, expr_entries = _fit_model(spec, TRAIN_SIMS)
    train_time = model.pop("fit_time_s", time.time() - start)
    _save_model_dir(spec, model, expression_text, {"approach_number": approach_num}, expr_entries)

    train_proxy, _, _, _ = _evaluate_model(model, TRAIN_SIMS)
    val_proxy, _, loss, per_mass = _evaluate_model(model, VAL_SIMS)
    eval_ms = 1000.0 * (time.time() - start) / max(1, len(TRAIN_SIMS) + len(VAL_SIMS))
    result = _scorecard(spec, approach_num, loss, per_mass, train_time, eval_ms, spec.notes, len(model["re_curves"]) * 2, "expression.txt")
    save_scorecard(md, result)
    ERROR_DATA[spec.name] = {"train_proxy": train_proxy, "val_proxy": val_proxy}
    RESULTS.append(result)
    ALL_EXPRESSIONS.append(
        {
            "approach": spec.name,
            "expression": expression_text,
            "complexity": len(model["re_curves"]) + len(model["im_curves"]),
            "loss": float(loss),
        }
    )
    _append_changelog_entry(result)
    _write_summary_artifacts()
    print(f"[{approach_num:02d}] {spec.name}: loss={loss:.6e}")
    return result


def retrain_named_model(name: str) -> None:
    for i, spec in enumerate(SPECS, start=1):
        if spec.name == name:
            _train_single_spec(spec, i)
            return
    raise KeyError(name)


def _ensure_changelog_header() -> None:
    path = RESULTS_DIR / "CHANGELOG.md"
    path.write_text("# Analytic Benchmark - CHANGELOG\n\n")


if __name__ == "__main__":
    _ensure_changelog_header()
    TRAIN_SIMS, VAL_SIMS = load_data()
    print(f"[init] train={len(TRAIN_SIMS)} val={len(VAL_SIMS)}")
    for i, spec in enumerate(SPECS, start=1):
        try:
            print(f"\n=== {i:02d} {spec.name} ===")
            _train_single_spec(spec, i)
        except Exception as exc:
            print(f"  [error] {spec.name}: {exc}")
            traceback.print_exc()
    _write_summary_artifacts()
    print(f"\n[done] {len(RESULTS)} approaches")
