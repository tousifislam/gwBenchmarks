"""Build the symbolic-regression and pure-IMR models (models 20-24) using
cached per-q decompositions written by build_models.py.

This script exists because PySR's Julia backend triggers multi-minute
precompilation on first import; running the gplearn+IMR add-ons in a separate
script avoids re-doing the slow Gaussian/Lorentzian/damped-sin fits."""
from __future__ import annotations
import os, sys, json, time, traceback, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (
    load_data, amp_phase, reparam_q,
    fd_mismatch_proxy, fd_mismatch_real,
    save_scorecard, write_train_predict, model_dir, RESULTS_DIR,
    T_GRID, N_T, T_GRID_DT,
    poly_eval, poly_str, fit_param_polynomial,
)
from numpy.polynomial import chebyshev as nc

qt, ht, qv, hv = load_data()
N_TRAIN, N_VAL = len(qt), len(qv)
T_MIN, T_MAX = T_GRID[0], T_GRID[-1]
T_NORM = 2 * (T_GRID - T_MIN) / (T_MAX - T_MIN) - 1.0


# ---------------------------------------------------------------------------
# IMR ansatz (must match build_models.py)
# ---------------------------------------------------------------------------
def imr_amp(t, log_Apk, T_pn, p_pre, tau_RD, p_post):
    safe_T_pn = np.maximum(T_pn, 1.0); safe_tau = np.maximum(tau_RD, 1.0); safe_pp = np.maximum(p_post, 0.5)
    s_post = 0.5 * (1 + np.tanh(t / 5.0)); s_pre = 1 - s_post
    pre = -p_pre * np.log1p(np.clip(-t / safe_T_pn, -0.99, None))
    pos = np.maximum(t, 0.0)
    post = -np.power(pos / safe_tau, safe_pp)
    return log_Apk + pre * s_pre + post * s_post


def imr_phi(t, phi_pk, omega_RD, omega_pn, T_pn_phi, T_b):
    safe_Tpn = np.maximum(T_pn_phi, 1.0); safe_Tb = np.maximum(T_b, 1.0)
    arg = np.clip(1.0 - t / safe_Tpn, 1e-6, None)
    pn_term = -(8.0 / 5.0) * omega_pn * safe_Tpn * (arg ** (5.0 / 8.0) - 1.0)
    post_term = omega_RD * t
    s_post = 0.5 * (1 + np.tanh(t / safe_Tb)); s_pre = 1 - s_post
    return phi_pk + pn_term * s_pre + post_term * s_post


# ---------------------------------------------------------------------------
# Load cached decomposition
# ---------------------------------------------------------------------------
cache_path = RESULTS_DIR / "_data" / "perq_decomp.npz"
print(f"[load] reading {cache_path}")
cache = np.load(cache_path)
amp_base_t = cache["amp_base"]
phi_base_t = cache["phi_base"]
cheb_amp_t = {d: cache[f"cheb_amp_d{d}"] for d in [10, 12, 14, 16, 18]}
cheb_phi_t = {d: cache[f"cheb_phi_d{d}"] for d in [10, 12, 14, 16, 18]}
print(f"  amp_base shape {amp_base_t.shape}, phi_base shape {phi_base_t.shape}")


def fit_poly_table(x, M, deg):
    n_cols = M.shape[1]
    table = np.zeros((n_cols, deg + 1))
    for j in range(n_cols):
        table[j] = fit_param_polynomial(x, M[:, j], deg)
    return table


def eval_poly_table(table, x):
    n_cols, _ = table.shape
    out = np.zeros((len(x), n_cols))
    for j in range(n_cols):
        out[:, j] = poly_eval(table[j], x)
    return out


# ---------------------------------------------------------------------------
# Result registry (only adds new models; doesn't touch existing files)
# ---------------------------------------------------------------------------
ALL_RESULTS = []


def evaluate_and_save(approach_num, name, category, parameterization, expression_text,
                      h_pred_val, train_time_s, n_params, notes,
                      coeffs_savedict=None):
    md = model_dir(approach_num, name)
    proxy = fd_mismatch_proxy(h_pred_val, hv)
    proxy_l2 = float(np.mean(proxy))
    t0 = time.perf_counter()
    loss, comps, per_sample = fd_mismatch_real(h_pred_val, hv)
    eval_time = (time.perf_counter() - t0) / max(N_VAL, 1)
    sc = {
        "approach": name,
        "approach_number": approach_num,
        "benchmark": "analytic",
        "agent": "opus47",
        "category": category,
        "parameterization": parameterization,
        "time_convention": "t0_at_peak",
        "loss": float(loss),
        "loss_proxy_l2": float(proxy_l2),
        "loss_components": {k: float(v) for k, v in comps.items()},
        "per_sample_loss": [float(x) for x in per_sample.tolist()],
        "runtime_ms": float(eval_time * 1000.0),
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_params": int(n_params),
        "train_time_s": float(train_time_s),
        "notes": notes,
        "is_closed_form": True,
    }
    save_scorecard(md, sc)
    (md / "expression.txt").write_text(expression_text)
    write_train_predict(md, name, expression_text[:600])
    if coeffs_savedict is not None:
        np.savez(md / "saved_model" / "coeffs.npz", **coeffs_savedict)
    ALL_RESULTS.append({"dir": str(md), "scorecard": sc})
    print(f"[done {approach_num:02d}_{name}] loss={loss:.4e}, n_params={n_params}")


def mode_to_text(mode):
    return {"q": "q", "eta": "q / (1 + q)^2", "delta": "(q - 1) / (q + 1)",
            "log_q": "log(q)", "inv_q": "1 / q"}[mode]


# ---------------------------------------------------------------------------
# gplearn symbolic regression — replaces polynomial-in-x for selected params
# ---------------------------------------------------------------------------
def _symreg_basis_library():
    return {
        "1":          lambda v: np.ones_like(v),
        "x":          lambda v: v,
        "x^2":        lambda v: v ** 2,
        "x^3":        lambda v: v ** 3,
        "sqrt(x)":    lambda v: np.sqrt(np.maximum(v, 1e-30)),
        "1/x":        lambda v: 1.0 / np.where(np.abs(v) < 1e-6, 1e-6, v),
        "log(x)":     lambda v: np.log(np.maximum(v, 1e-30)),
        "x*log(x)":   lambda v: v * np.log(np.maximum(v, 1e-30)),
        "1/(1+x)":    lambda v: 1.0 / (1.0 + v),
    }


def omp_basis_scalar(x, y, n_terms=4):
    """Greedy basis-function regression: select up to n_terms from the analytic
    library that minimise residual sum-of-squares.  Closed-form output as a
    weighted sum of selected analytic basis functions."""
    x = np.asarray(x).reshape(-1).astype(float)
    y = np.asarray(y).reshape(-1).astype(float)
    lib = _symreg_basis_library()
    names = list(lib.keys())
    F = np.stack([lib[n](x) for n in names], axis=1)
    selected = []
    for _ in range(min(n_terms, F.shape[1])):
        best_j = -1; best_red = 0.0
        for j in range(F.shape[1]):
            if j in selected:
                continue
            cols = selected + [j]
            X = F[:, cols]
            try:
                c, *_ = np.linalg.lstsq(X, y, rcond=None)
            except Exception:
                continue
            r = y - X @ c
            ss = float(np.sum(r ** 2))
            if best_j < 0 or ss < best_red:
                best_red = ss; best_j = j
        if best_j < 0:
            break
        selected.append(best_j)
    X_sel = F[:, selected]
    coefs, *_ = np.linalg.lstsq(X_sel, y, rcond=None)
    parts = [f"({c:.6g})*{names[j]}" for c, j in zip(coefs, selected)]
    expr_str = " + ".join(parts) if parts else "0"

    def fn(xq):
        xq = np.asarray(xq).reshape(-1).astype(float)
        Fq = np.stack([list(_symreg_basis_library().values())[selected[i]](xq) for i in range(len(selected))], axis=1)
        return Fq @ coefs

    return fn, expr_str, len(coefs)


def _patch_sklearn_for_gplearn():
    """gplearn 0.4.2 expects an older sklearn API; patch BaseEstimator with
    backwards-compatible _validate_data and n_features_in_."""
    from sklearn.base import BaseEstimator
    from sklearn.utils.validation import check_X_y, check_array
    if hasattr(BaseEstimator, "_validate_data"):
        return
    def _validate_data(self, X, y=None, reset=True, **kwargs):
        if y is None:
            Xv = check_array(X)
            if reset:
                self.n_features_in_ = Xv.shape[1]
            return Xv
        for k in ("y_numeric", "multi_output", "ensure_min_samples", "ensure_min_features"):
            kwargs.pop(k, None)
        Xv, yv = check_X_y(X, y)
        if reset:
            self.n_features_in_ = Xv.shape[1]
        return Xv, yv
    BaseEstimator._validate_data = _validate_data


def gplearn_scalar(x, y, generations=30, parsimony=0.05, pop=400, seed=42, init_depth=(1, 4)):
    """Run gplearn SymbolicRegressor on a 1D regression problem and return a
    callable plus the human-readable program string.  Higher parsimony and
    shallow init_depth keep expressions simple."""
    _patch_sklearn_for_gplearn()
    from gplearn.genetic import SymbolicRegressor
    sr = SymbolicRegressor(
        population_size=pop, generations=generations,
        init_depth=init_depth, init_method="grow",
        function_set=("add", "sub", "mul", "div", "log", "sqrt"),
        metric="mse", parsimony_coefficient=parsimony,
        random_state=seed, verbose=0, n_jobs=1,
    )
    Xfit = np.asarray(x).reshape(-1, 1).astype(float)
    yfit = np.asarray(y).reshape(-1).astype(float)
    sr.fit(Xfit, yfit)
    fn = lambda xq: sr.predict(np.asarray(xq).reshape(-1, 1).astype(float))
    return fn, str(sr._program)


N_AMP_BASE = 5
N_PHI_BASE = 5


def build_gplearn_amp(approach_num, name, parameterization,
                      generations=30, deg_cheb=14, deg_poly=3):
    x = reparam_q(qt, parameterization)
    x_v = reparam_q(qv, parameterization)
    t0 = time.perf_counter()
    amp_eval_v = np.zeros((N_VAL, N_AMP_BASE))
    expr_strs = []
    for j in range(N_AMP_BASE):
        fn, s = gplearn_scalar(x, amp_base_t[:, j], generations=generations)
        amp_eval_v[:, j] = fn(x_v)
        expr_strs.append(s)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    cheb_a = fit_poly_table(x, cheb_amp_t[deg_cheb], deg_poly)
    cheb_p = fit_poly_table(x, cheb_phi_t[deg_cheb], deg_poly)
    train_time = time.perf_counter() - t0
    n = N_VAL
    h_pred = np.zeros((n, N_T), dtype=complex)
    phi_eval_v = eval_poly_table(phi_table, x_v)
    ca_v = eval_poly_table(cheb_a, x_v)
    cp_v = eval_poly_table(cheb_p, x_v)
    for i in range(n):
        log_A = imr_amp(T_GRID, *amp_eval_v[i]) + nc.chebval(T_NORM, ca_v[i])
        phi = imr_phi(T_GRID, *phi_eval_v[i]) + nc.chebval(T_NORM, cp_v[i])
        h_pred[i] = np.exp(log_A) * np.exp(-1j * phi)

    var = mode_to_text(parameterization)
    lines = [
        "CLOSED-FORM h22(t; q) — IMR base + Chebyshev residual; amp params via gplearn",
        "=" * 60,
        f"  x = {var}",
        "",
        "Amplitude per-q params from gplearn symbolic regression:",
    ]
    for j, n_ in enumerate(["log_Apk", "T_pn", "p_pre", "tau_RD", "p_post"]):
        lines.append(f"  {n_}(q) = {expr_strs[j]}")
    lines.append(f"Phase per-q params (degree-{deg_poly} polynomial in x):")
    for j, n_ in enumerate(["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]):
        lines.append(f"  {n_}(q) = {poly_str(phi_table[j], 'x')}")
    lines.append(f"Chebyshev residual coefficients (degree-{deg_poly} polynomial in x):")
    for n_ in range(cheb_a.shape[0]):
        lines.append(f"  a_{n_}(q) = {poly_str(cheb_a[n_], 'x')}")
    for n_ in range(cheb_p.shape[0]):
        lines.append(f"  b_{n_}(q) = {poly_str(cheb_p[n_], 'x')}")
    lines += [
        "",
        "Auxiliary functions, base ansatz, Chebyshev formula are identical to model 01.",
        "h22(t; q) = exp(log A_base + Σ a_n(q) T_n(τ)) · exp(-i (phi_base + Σ b_n(q) T_n(τ)))",
    ]
    expr = "\n".join(lines)
    import re
    # Count numeric constants (genuinely fitted floats) in each symbolic expression
    sym_token_count = 0
    for s in expr_strs:
        sym_token_count += len(re.findall(r"-?\d+\.\d+|-?\d+\.|-?\d+(?=[,)])", s))
    n_params = sym_token_count + phi_table.size + cheb_a.size + cheb_p.size
    evaluate_and_save(
        approach_num, name, "symbolic_regression", parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"gplearn symbolic regression on amplitude params, polynomial elsewhere.",
        coeffs_savedict={"phi_table": phi_table, "cheb_amp_table": cheb_a, "cheb_phi_table": cheb_p})


def build_gplearn_phi(approach_num, name, parameterization,
                      generations=30, deg_cheb=14, deg_poly=3, library="gplearn"):
    x = reparam_q(qt, parameterization)
    x_v = reparam_q(qv, parameterization)
    t0 = time.perf_counter()
    phi_eval_v = np.zeros((N_VAL, N_PHI_BASE))
    expr_strs = []
    n_const_count = 0
    for j in range(N_PHI_BASE):
        if library == "gplearn":
            fn, s = gplearn_scalar(x, phi_base_t[:, j], generations=generations)
        else:  # OMP-basis fallback (simpler, fewer constants)
            fn, s, n_c = omp_basis_scalar(x, phi_base_t[:, j], n_terms=4)
            n_const_count += n_c
        phi_eval_v[:, j] = fn(x_v)
        expr_strs.append(s)
    amp_table = fit_poly_table(x, amp_base_t, deg_poly)
    cheb_a = fit_poly_table(x, cheb_amp_t[deg_cheb], deg_poly)
    cheb_p = fit_poly_table(x, cheb_phi_t[deg_cheb], deg_poly)
    train_time = time.perf_counter() - t0
    n = N_VAL
    h_pred = np.zeros((n, N_T), dtype=complex)
    amp_eval_v = eval_poly_table(amp_table, x_v)
    ca_v = eval_poly_table(cheb_a, x_v)
    cp_v = eval_poly_table(cheb_p, x_v)
    for i in range(n):
        log_A = imr_amp(T_GRID, *amp_eval_v[i]) + nc.chebval(T_NORM, ca_v[i])
        phi = imr_phi(T_GRID, *phi_eval_v[i]) + nc.chebval(T_NORM, cp_v[i])
        h_pred[i] = np.exp(log_A) * np.exp(-1j * phi)

    var = mode_to_text(parameterization)
    label = "gplearn symbolic regression" if library == "gplearn" else "greedy basis-function (OMP) symbolic regression"
    lines = [
        f"CLOSED-FORM h22(t; q) — IMR base + Chebyshev residual; phase params via {label}",
        "=" * 60,
        f"  x = {var}",
        "",
        f"Phase per-q params from {label}:",
    ]
    for j, n_ in enumerate(["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]):
        lines.append(f"  {n_}(q) = {expr_strs[j]}")
    lines.append(f"Amplitude per-q params (poly deg {deg_poly}):")
    for j, n_ in enumerate(["log_Apk", "T_pn", "p_pre", "tau_RD", "p_post"]):
        lines.append(f"  {n_}(q) = {poly_str(amp_table[j], 'x')}")
    lines.append(f"Chebyshev residual coefficients (poly deg {deg_poly}):")
    for n_ in range(cheb_a.shape[0]):
        lines.append(f"  a_{n_}(q) = {poly_str(cheb_a[n_], 'x')}")
    for n_ in range(cheb_p.shape[0]):
        lines.append(f"  b_{n_}(q) = {poly_str(cheb_p[n_], 'x')}")
    lines += [
        "",
        "Auxiliary functions, base ansatz, Chebyshev formula are identical to model 01.",
        "h22(t; q) = exp(log A_base + Σ a_n T_n(τ)) · exp(-i (phi_base + Σ b_n T_n(τ)))",
    ]
    expr = "\n".join(lines)
    import re
    # Count numeric constants in each symbolic expression
    sym_token_count = n_const_count
    if library == "gplearn":
        sym_token_count = 0
        for s in expr_strs:
            sym_token_count += len(re.findall(r"-?\d+\.\d+|-?\d+\.|-?\d+(?=[,)])", s))
    n_params = sym_token_count + amp_table.size + cheb_a.size + cheb_p.size
    evaluate_and_save(
        approach_num, name, "symbolic_regression", parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"{label} on phase params, polynomial elsewhere.",
        coeffs_savedict={"amp_table": amp_table, "cheb_amp_table": cheb_a, "cheb_phi_table": cheb_p})


print("[symreg] gplearn variants ...")
try:
    build_gplearn_amp(20, "gplearn_amp_imr_eta",   "eta",   generations=20, deg_cheb=14, deg_poly=3)
except Exception as e:
    print("  20 gplearn amp eta failed:", e); traceback.print_exc()
try:
    build_gplearn_phi(21, "ompbasis_phase_imr_delta", "delta", deg_cheb=14, deg_poly=3, library="omp")
except Exception as e:
    print("  21 OMP phi delta failed:", e); traceback.print_exc()
try:
    build_gplearn_amp(22, "gplearn_amp_imr_logq",   "log_q", generations=20, deg_cheb=14, deg_poly=3)
except Exception as e:
    print("  22 gplearn amp logq failed:", e); traceback.print_exc()
try:
    build_gplearn_phi(23, "ompbasis_phase_imr_eta",  "eta",   deg_cheb=14, deg_poly=3, library="omp")
except Exception as e:
    print("  23 OMP phi eta failed:", e); traceback.print_exc()


# ---------------------------------------------------------------------------
# Pure IMR baseline (no Cheb residual)
# ---------------------------------------------------------------------------
def build_pure_imr(approach_num, name, parameterization, deg_poly):
    x = reparam_q(qt, parameterization)
    x_v = reparam_q(qv, parameterization)
    t0 = time.perf_counter()
    amp_table = fit_poly_table(x, amp_base_t, deg_poly)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    train_time = time.perf_counter() - t0
    amp_p = eval_poly_table(amp_table, x_v)
    phi_p = eval_poly_table(phi_table, x_v)
    n = N_VAL
    h_pred = np.zeros((n, N_T), dtype=complex)
    for i in range(n):
        h_pred[i] = np.exp(imr_amp(T_GRID, *amp_p[i])) * np.exp(-1j * imr_phi(T_GRID, *phi_p[i]))

    var = mode_to_text(parameterization)
    lines = [
        "CLOSED-FORM h22(t; q) — pure IMR base (no Chebyshev residual)",
        "=" * 60,
        f"  x = {var}",
        "",
        "Same IMR base ansatz as model 01, but residual corrections are zero.",
        "",
    ]
    for j, n_ in enumerate(["log_Apk", "T_pn", "p_pre", "tau_RD", "p_post"]):
        lines.append(f"  {n_}(q) = {poly_str(amp_table[j], 'x')}")
    for j, n_ in enumerate(["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]):
        lines.append(f"  {n_}(q) = {poly_str(phi_table[j], 'x')}")
    expr = "\n".join(lines)
    n_params = amp_table.size + phi_table.size
    evaluate_and_save(
        approach_num, name, "physics_imr", parameterization, expr,
        h_pred, train_time, n_params,
        notes="Pure IMR base — PN inspiral + power-law decay amp; PN+RD phase. No Cheb residual.",
        coeffs_savedict={"amp_table": amp_table, "phi_table": phi_table})


build_pure_imr(24, "phen_imr_pure_eta_p4", "eta", 4)


with open(RESULTS_DIR / "_data" / "extra_results.json", "w") as f:
    json.dump(ALL_RESULTS, f, indent=2)

print(f"\n[summary] built {len(ALL_RESULTS)} extra closed-form models.")
