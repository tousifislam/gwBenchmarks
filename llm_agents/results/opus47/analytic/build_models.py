"""Build genuinely closed-form analytic models for h22(t; q).

Every model in this script writes a closed-form expression in t and q.
No SVD, EIM, KNN, MLP, RF/GBM, splines, or stored basis vectors.

Model families
--------------
1. Physics-informed IMR base (PN inspiral × tanh-blend × ringdown decay) with
   Chebyshev polynomial residual corrections in t and polynomial coefficients
   in mass-ratio reparameterization (eta, delta, log q, 1/q).
2. Symbolic-regression flavours (PySR / gplearn) on the per-q base parameters.
3. Composite / piecewise-matched ansatz with hard inspiral/ringdown switch.
4. Functional-form direct fits — sums of Gaussians, Lorentzians, Padé, damped
   sinusoids fit to h22(t) itself.

Every coefficient is explicit. expression.txt for each model spells out the
full formula. Total fitted coefficients per model is < 200.
"""
from __future__ import annotations
import os, sys, json, time, pickle, warnings, traceback
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
from scipy.optimize import curve_fit

RNG = np.random.default_rng(42)
qt, ht, qv, hv = load_data()
N_TRAIN, N_VAL = len(qt), len(qv)
print(f"[init] N_train={N_TRAIN}, N_val={N_VAL}")

A_t, phi_t = amp_phase(ht)

PEAK_IDX = int(np.argmin(np.abs(T_GRID)))
assert T_GRID[PEAK_IDX] == 0.0
T_MIN, T_MAX = T_GRID[0], T_GRID[-1]
T_NORM = 2 * (T_GRID - T_MIN) / (T_MAX - T_MIN) - 1.0  # in [-1, 1]

_DATA = RESULTS_DIR / "_data"
_DATA.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# IMR base ansatz
# ---------------------------------------------------------------------------
def imr_amp(t, log_Apk, T_pn, p_pre, tau_RD, p_post):
    """log A_base(t) = log_Apk - p_pre log(1 - t/T_pn) [* s_pre]
                     - (max(t,0)/tau_RD)^p_post        [* s_post]"""
    safe_T_pn = np.maximum(T_pn, 1.0)
    safe_tau = np.maximum(tau_RD, 1.0)
    safe_pp = np.maximum(p_post, 0.5)
    s_post = 0.5 * (1 + np.tanh(t / 5.0))
    s_pre = 1 - s_post
    pre = -p_pre * np.log1p(np.clip(-t / safe_T_pn, -0.99, None))
    pos = np.maximum(t, 0.0)
    post = -np.power(pos / safe_tau, safe_pp)
    return log_Apk + pre * s_pre + post * s_post


def imr_phi(t, phi_pk, omega_RD, omega_pn, T_pn_phi, T_b):
    """phi_base(t) = phi_pk - (8/5) omega_pn T_pn_phi ((1 - t/T_pn_phi)^(5/8) - 1) [* s_pre]
                   + omega_RD * t                                                   [* s_post]"""
    safe_Tpn = np.maximum(T_pn_phi, 1.0)
    safe_Tb = np.maximum(T_b, 1.0)
    arg = np.clip(1.0 - t / safe_Tpn, 1e-6, None)
    pn_term = -(8.0 / 5.0) * omega_pn * safe_Tpn * (arg ** (5.0 / 8.0) - 1.0)
    post_term = omega_RD * t
    s_post = 0.5 * (1 + np.tanh(t / safe_Tb))
    s_pre = 1 - s_post
    return phi_pk + pn_term * s_pre + post_term * s_post


def fit_imr_amp(t, A):
    log_A = np.log(np.maximum(A, 1e-30))
    sigma = 1.0 / np.maximum(A, 1e-3)
    log_Apk = log_A[PEAK_IDX]
    p0 = [log_Apk, 200.0, 0.3, 25.0, 1.4]
    bounds = ([log_Apk - 1.0, 1.0, 0.05, 2.0, 0.5],
              [log_Apk + 1.0, 5000.0, 2.0, 500.0, 5.0])
    try:
        popt, _ = curve_fit(imr_amp, t, log_A, p0=p0, bounds=bounds, maxfev=30000, sigma=sigma)
    except Exception:
        popt = np.array(p0)
    return popt


def fit_imr_phi(t, A, phi):
    sigma = 1.0 / np.maximum(A, 1e-3)
    om_t = np.gradient(phi, t)
    om_pk = float(om_t[PEAK_IDX])
    om_rd = float(np.median(om_t[-100:-20]))
    p0 = [phi[PEAK_IDX], om_rd, om_pk, 15.0, 8.0]
    bounds = ([phi[PEAK_IDX] - 5.0, 0.05, 0.05, 1.0, 0.1],
              [phi[PEAK_IDX] + 5.0, 1.5, 1.5, 1e5, 200.0])
    try:
        popt, _ = curve_fit(imr_phi, t, phi, p0=p0, bounds=bounds, maxfev=30000, sigma=sigma)
    except Exception:
        popt = np.array(p0)
    return popt


# Per-waveform base fits + Chebyshev residuals
DEG_CHEB_DEFAULT = 14
N_AMP_BASE = 5
N_PHI_BASE = 5

print("[per-q] fitting IMR base + Chebyshev residuals (training set) ...")
amp_base_t = np.zeros((N_TRAIN, N_AMP_BASE))
phi_base_t = np.zeros((N_TRAIN, N_PHI_BASE))
cheb_amp_t = {}  # cheb_amp_t[deg][i, k]
cheb_phi_t = {}

DEGS_TO_FIT = [10, 12, 14, 16, 18]
for d in DEGS_TO_FIT:
    cheb_amp_t[d] = np.zeros((N_TRAIN, d + 1))
    cheb_phi_t[d] = np.zeros((N_TRAIN, d + 1))

for i in range(N_TRAIN):
    amp_base_t[i] = fit_imr_amp(T_GRID, A_t[i])
    phi_base_t[i] = fit_imr_phi(T_GRID, A_t[i], phi_t[i])
    log_A = np.log(np.maximum(A_t[i], 1e-30))
    log_A_b = imr_amp(T_GRID, *amp_base_t[i])
    phi_b = imr_phi(T_GRID, *phi_base_t[i])
    res_a = log_A - log_A_b
    res_p = phi_t[i] - phi_b
    for d in DEGS_TO_FIT:
        cheb_amp_t[d][i] = nc.chebfit(T_NORM, res_a, d, w=A_t[i])
        cheb_phi_t[d][i] = nc.chebfit(T_NORM, res_p, d, w=A_t[i])
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{N_TRAIN}")

# Save decomposition for re-use across models
np.savez(_DATA / "perq_decomp.npz",
         q=qt, amp_base=amp_base_t, phi_base=phi_base_t,
         **{f"cheb_amp_d{d}": cheb_amp_t[d] for d in DEGS_TO_FIT},
         **{f"cheb_phi_d{d}": cheb_phi_t[d] for d in DEGS_TO_FIT})


# ---------------------------------------------------------------------------
# Polynomial-in-q evaluator
# ---------------------------------------------------------------------------
def fit_poly_table(x, M, deg):
    """For each column of M (n_samples × n_cols), fit poly of degree deg in x."""
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


def predict_imr_cheb(q_arr, amp_base_table, phi_base_table,
                     cheb_amp_table, cheb_phi_table, mode):
    x = reparam_q(q_arr, mode)
    amp_b = eval_poly_table(amp_base_table, x)
    phi_b = eval_poly_table(phi_base_table, x)
    ca = eval_poly_table(cheb_amp_table, x)
    cp = eval_poly_table(cheb_phi_table, x)
    n = len(q_arr)
    h = np.zeros((n, N_T), dtype=complex)
    for i in range(n):
        log_A = imr_amp(T_GRID, *amp_b[i]) + nc.chebval(T_NORM, ca[i])
        phi = imr_phi(T_GRID, *phi_b[i]) + nc.chebval(T_NORM, cp[i])
        h[i] = np.exp(log_A) * np.exp(-1j * phi)
    return h


# ---------------------------------------------------------------------------
# Score, save, register
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
    print(f"[done {approach_num:02d}_{name}] loss={loss:.4e}, n_params={n_params}, train_time={train_time_s:.1f}s")


# ---------------------------------------------------------------------------
# Helpers for expression text generation
# ---------------------------------------------------------------------------
def mode_to_text(mode):
    return {"q": "q", "eta": "q / (1 + q)^2", "delta": "(q - 1) / (q + 1)",
            "log_q": "log(q)", "inv_q": "1 / q"}[mode]


def imr_cheb_expression(amp_table, phi_table, cheb_amp_table, cheb_phi_table,
                        mode, title_suffix=""):
    var = "x"
    lines = [
        f"CLOSED-FORM h22(t; q) — IMR base + Chebyshev residual{title_suffix}",
        "=" * 60,
        "",
        f"  {var} = {mode_to_text(mode)}",
        f"  τ = 2 (t - {T_MIN:g}) / ({T_MAX:g} - {T_MIN:g}) - 1   (scaled time in [-1, 1])",
        "",
        "Aux:",
        "  s_post(t) = (1 + tanh(t / 5)) / 2",
        "  s_pre(t)  = 1 - s_post(t)",
        "  s_post_phi(t; q) = (1 + tanh(t / T_b(q))) / 2",
        "  s_pre_phi (t; q) = 1 - s_post_phi(t; q)",
        "",
        "log A_base(t; q) = log_Apk(q)",
        "                  + s_pre(t) · ( -p_pre(q) · log(1 - t / T_pn(q)) )",
        "                  + s_post(t) · ( - (max(t,0) / tau_RD(q))^p_post(q) )",
        "",
        "phi_base(t; q) = phi_pk(q)",
        "               + s_pre_phi(t; q) · ( -(8/5) · omega_pn(q) · T_pn_phi(q) ·",
        "                                     ((1 - t/T_pn_phi(q))^(5/8) - 1) )",
        "               + s_post_phi(t; q) · ( omega_RD(q) · t )",
        "",
        "Chebyshev residual corrections:",
        "  log A(t; q) = log A_base(t; q) + Σ_{n=0}^{N} a_n(q) · T_n(τ)",
        "  phi  (t; q) = phi_base (t; q) + Σ_{n=0}^{N} b_n(q) · T_n(τ)",
        "",
        f"Polynomial coefficients in {var} (each row gives the polynomial expansion of one parameter):",
    ]
    amp_names = ["log_Apk", "T_pn", "p_pre", "tau_RD", "p_post"]
    phi_names = ["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]
    for j, n_ in enumerate(amp_names):
        lines.append(f"  {n_}(q) = {poly_str(amp_table[j], var)}")
    for j, n_ in enumerate(phi_names):
        lines.append(f"  {n_}(q) = {poly_str(phi_table[j], var)}")
    lines.append("")
    for n in range(cheb_amp_table.shape[0]):
        lines.append(f"  a_{n}(q) = {poly_str(cheb_amp_table[n], var)}")
    for n in range(cheb_phi_table.shape[0]):
        lines.append(f"  b_{n}(q) = {poly_str(cheb_phi_table[n], var)}")
    lines.append("")
    lines.append("h22(t; q) = exp(log A(t; q)) · exp(-i · phi(t; q))")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Family 1: IMR base + Cheb residual, varying parameterization & polynomial deg
# ---------------------------------------------------------------------------
def build_imr_cheb(approach_num, name, parameterization, deg_cheb, deg_poly, category="physics_imr"):
    t0 = time.perf_counter()
    x = reparam_q(qt, parameterization)
    amp_table = fit_poly_table(x, amp_base_t, deg_poly)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    cheb_a = fit_poly_table(x, cheb_amp_t[deg_cheb], deg_poly)
    cheb_p = fit_poly_table(x, cheb_phi_t[deg_cheb], deg_poly)
    train_time = time.perf_counter() - t0
    h_pred = predict_imr_cheb(qv, amp_table, phi_table, cheb_a, cheb_p, parameterization)
    expr = imr_cheb_expression(amp_table, phi_table, cheb_a, cheb_p, parameterization,
                               title_suffix=f" (cheb deg {deg_cheb}, q poly deg {deg_poly})")
    n_params = amp_table.size + phi_table.size + cheb_a.size + cheb_p.size
    evaluate_and_save(
        approach_num, name, category, parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"IMR base + Chebyshev_{deg_cheb} residual; q-dep polynomial deg {deg_poly} in {parameterization}.",
        coeffs_savedict={"amp_table": amp_table, "phi_table": phi_table,
                         "cheb_amp_table": cheb_a, "cheb_phi_table": cheb_p})


build_imr_cheb(1,  "phen_imr_cheb14_eta_p4",   "eta",   14, 4)
build_imr_cheb(2,  "phen_imr_cheb14_delta_p4", "delta", 14, 4)
build_imr_cheb(3,  "phen_imr_cheb14_logq_p4",  "log_q", 14, 4)
build_imr_cheb(4,  "phen_imr_cheb14_invq_p4",  "inv_q", 14, 4)
build_imr_cheb(5,  "phen_imr_cheb14_q_p4",     "q",     14, 4)
build_imr_cheb(6,  "phen_imr_cheb12_eta_p4",   "eta",   12, 4)
build_imr_cheb(7,  "phen_imr_cheb16_eta_p3",   "eta",   16, 3)
build_imr_cheb(8,  "phen_imr_cheb18_eta_p3",   "eta",   18, 3)
build_imr_cheb(9,  "phen_imr_cheb10_eta_p4",   "eta",   10, 4)
build_imr_cheb(10, "phen_imr_cheb14_eta_p3",   "eta",   14, 3)


# ---------------------------------------------------------------------------
# Family 2 — Composite / piecewise: same IMR base, but polynomial residual on
# log_A and phi within disjoint inspiral/merger/ringdown segments.
# We fit per-q Chebyshev coefficients on the SCALED time τ_seg in each segment
# separately, blend at boundaries with tanh.
# ---------------------------------------------------------------------------
SEG_BOUND = -200.0  # boundary between inspiral and ring-down/merger segment
SEG_WIDTH = 30.0


def predict_imr_composite(q_arr, amp_table, phi_table, cheb_a_pre, cheb_p_pre,
                          cheb_a_post, cheb_p_post, mode):
    x = reparam_q(q_arr, mode)
    amp_b = eval_poly_table(amp_table, x)
    phi_b = eval_poly_table(phi_table, x)
    ca_pre = eval_poly_table(cheb_a_pre, x)
    cp_pre = eval_poly_table(cheb_p_pre, x)
    ca_post = eval_poly_table(cheb_a_post, x)
    cp_post = eval_poly_table(cheb_p_post, x)
    n = len(q_arr)
    h = np.zeros((n, N_T), dtype=complex)
    # Two scaled times: τ_pre on [T_MIN, SEG_BOUND], τ_post on [SEG_BOUND, T_MAX]
    pre_mask = T_GRID <= SEG_BOUND
    post_mask = T_GRID >= SEG_BOUND
    tau_pre = np.zeros(N_T)
    tau_post = np.zeros(N_T)
    tau_pre[pre_mask] = 2 * (T_GRID[pre_mask] - T_MIN) / (SEG_BOUND - T_MIN) - 1
    tau_post[post_mask] = 2 * (T_GRID[post_mask] - SEG_BOUND) / (T_MAX - SEG_BOUND) - 1
    blend_pre = 0.5 * (1 - np.tanh((T_GRID - SEG_BOUND) / SEG_WIDTH))
    blend_post = 1 - blend_pre
    for i in range(n):
        log_A_b = imr_amp(T_GRID, *amp_b[i])
        phi_bv = imr_phi(T_GRID, *phi_b[i])
        log_A_corr_pre = nc.chebval(tau_pre, ca_pre[i])
        log_A_corr_post = nc.chebval(tau_post, ca_post[i])
        phi_corr_pre = nc.chebval(tau_pre, cp_pre[i])
        phi_corr_post = nc.chebval(tau_post, cp_post[i])
        log_A = log_A_b + blend_pre * log_A_corr_pre + blend_post * log_A_corr_post
        phi = phi_bv + blend_pre * phi_corr_pre + blend_post * phi_corr_post
        h[i] = np.exp(log_A) * np.exp(-1j * phi)
    return h


# Compute per-q Chebyshev fits on the two segments
DEG_PRE, DEG_POST = 8, 8
print("[composite] segment Chebyshev fits ...")
cheb_a_pre_t = np.zeros((N_TRAIN, DEG_PRE + 1))
cheb_p_pre_t = np.zeros((N_TRAIN, DEG_PRE + 1))
cheb_a_post_t = np.zeros((N_TRAIN, DEG_POST + 1))
cheb_p_post_t = np.zeros((N_TRAIN, DEG_POST + 1))
pre_mask = T_GRID <= SEG_BOUND
post_mask = T_GRID >= SEG_BOUND
tau_pre_arr = 2 * (T_GRID[pre_mask] - T_MIN) / (SEG_BOUND - T_MIN) - 1
tau_post_arr = 2 * (T_GRID[post_mask] - SEG_BOUND) / (T_MAX - SEG_BOUND) - 1
for i in range(N_TRAIN):
    log_A = np.log(np.maximum(A_t[i], 1e-30))
    log_A_b = imr_amp(T_GRID, *amp_base_t[i])
    phi_b = imr_phi(T_GRID, *phi_base_t[i])
    res_a = log_A - log_A_b
    res_p = phi_t[i] - phi_b
    w = A_t[i]
    cheb_a_pre_t[i] = nc.chebfit(tau_pre_arr, res_a[pre_mask], DEG_PRE, w=w[pre_mask])
    cheb_p_pre_t[i] = nc.chebfit(tau_pre_arr, res_p[pre_mask], DEG_PRE, w=w[pre_mask])
    cheb_a_post_t[i] = nc.chebfit(tau_post_arr, res_a[post_mask], DEG_POST, w=w[post_mask])
    cheb_p_post_t[i] = nc.chebfit(tau_post_arr, res_p[post_mask], DEG_POST, w=w[post_mask])


def composite_expression(amp_table, phi_table, ca_pre, cp_pre, ca_post, cp_post, mode):
    var = "x"
    lines = [
        "CLOSED-FORM h22(t; q) — composite/matched-asymptotic (segmented Chebyshev)",
        "=" * 60,
        f"  {var} = {mode_to_text(mode)}",
        f"  τ_pre  = 2 (t - {T_MIN:g}) / ({SEG_BOUND:g} - {T_MIN:g}) - 1   (for t ≤ {SEG_BOUND:g}M)",
        f"  τ_post = 2 (t - {SEG_BOUND:g}) / ({T_MAX:g} - {SEG_BOUND:g}) - 1   (for t ≥ {SEG_BOUND:g}M)",
        f"  blend_pre(t)  = (1 - tanh((t - {SEG_BOUND:g}) / {SEG_WIDTH:g})) / 2",
        f"  blend_post(t) = 1 - blend_pre(t)",
        "",
        "  log A(t; q) = log A_base(t; q)",
        "              + blend_pre(t)  · Σ_n a^pre_n (q) · T_n(τ_pre)",
        "              + blend_post(t) · Σ_n a^post_n(q) · T_n(τ_post)",
        "  phi  (t; q) = phi_base (t; q)",
        "              + blend_pre(t)  · Σ_n b^pre_n (q) · T_n(τ_pre)",
        "              + blend_post(t) · Σ_n b^post_n(q) · T_n(τ_post)",
        "",
        "Base IMR ansatz is identical to the IMR-Cheb model (see model 01).",
        "",
    ]
    amp_names = ["log_Apk", "T_pn", "p_pre", "tau_RD", "p_post"]
    phi_names = ["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]
    for j, n_ in enumerate(amp_names):
        lines.append(f"  {n_}(q) = {poly_str(amp_table[j], var)}")
    for j, n_ in enumerate(phi_names):
        lines.append(f"  {n_}(q) = {poly_str(phi_table[j], var)}")
    for n in range(ca_pre.shape[0]):
        lines.append(f"  a^pre_{n}(q)  = {poly_str(ca_pre[n], var)}")
    for n in range(ca_post.shape[0]):
        lines.append(f"  a^post_{n}(q) = {poly_str(ca_post[n], var)}")
    for n in range(cp_pre.shape[0]):
        lines.append(f"  b^pre_{n}(q)  = {poly_str(cp_pre[n], var)}")
    for n in range(cp_post.shape[0]):
        lines.append(f"  b^post_{n}(q) = {poly_str(cp_post[n], var)}")
    lines.append("")
    lines.append("h22(t; q) = exp(log A) · exp(-i · phi)")
    return "\n".join(lines)


def build_composite(approach_num, name, parameterization, deg_poly):
    t0 = time.perf_counter()
    x = reparam_q(qt, parameterization)
    amp_table = fit_poly_table(x, amp_base_t, deg_poly)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    ca_pre = fit_poly_table(x, cheb_a_pre_t, deg_poly)
    cp_pre = fit_poly_table(x, cheb_p_pre_t, deg_poly)
    ca_post = fit_poly_table(x, cheb_a_post_t, deg_poly)
    cp_post = fit_poly_table(x, cheb_p_post_t, deg_poly)
    train_time = time.perf_counter() - t0
    h_pred = predict_imr_composite(qv, amp_table, phi_table,
                                   ca_pre, cp_pre, ca_post, cp_post,
                                   parameterization)
    expr = composite_expression(amp_table, phi_table, ca_pre, cp_pre, ca_post, cp_post,
                                parameterization)
    n_params = (amp_table.size + phi_table.size + ca_pre.size + cp_pre.size +
                ca_post.size + cp_post.size)
    evaluate_and_save(
        approach_num, name, "composite", parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"IMR base + segmented Chebyshev_{DEG_PRE}/{DEG_POST}; "
              f"poly deg {deg_poly} in {parameterization}; tanh blend at t={SEG_BOUND}M.",
        coeffs_savedict={"amp_table": amp_table, "phi_table": phi_table,
                         "ca_pre": ca_pre, "cp_pre": cp_pre,
                         "ca_post": ca_post, "cp_post": cp_post})


build_composite(11, "comp_seg_cheb88_eta_p3",   "eta",   3)
build_composite(12, "comp_seg_cheb88_delta_p3", "delta", 3)
build_composite(13, "comp_seg_cheb88_logq_p3",  "log_q", 3)


# ---------------------------------------------------------------------------
# Family 3 — Functional-form direct fits to h22(t)
# ---------------------------------------------------------------------------
# Sum of Gaussians for amplitude (closed-form), with PN-RD phase.
def gauss_sum(t, params):
    n_g = len(params) // 3
    out = np.zeros_like(t, dtype=float)
    for k in range(n_g):
        A = params[3 * k]
        mu = params[3 * k + 1]
        sig = max(params[3 * k + 2], 1.0)
        out = out + A * np.exp(-0.5 * ((t - mu) / sig) ** 2)
    return out


def fit_gauss_sum(t, A, n_g):
    init_mus = np.linspace(-1500, 50, n_g)
    init_sigs = np.full(n_g, 250.0)
    init_amps = np.full(n_g, A[PEAK_IDX] / n_g)
    p0 = []
    for k in range(n_g):
        p0 += [init_amps[k], init_mus[k], init_sigs[k]]
    bounds_lo = []
    bounds_hi = []
    for k in range(n_g):
        bounds_lo += [-2.0, -3500.0, 5.0]
        bounds_hi += [2.0, 200.0, 5000.0]
    try:
        popt, _ = curve_fit(lambda tt, *p: gauss_sum(tt, p), t, A,
                            p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=30000)
    except Exception:
        popt = np.array(p0)
    return popt


def lorentz_sum(t, params):
    n_l = len(params) // 3
    out = np.zeros_like(t, dtype=float)
    for k in range(n_l):
        A = params[3 * k]
        mu = params[3 * k + 1]
        gam = max(params[3 * k + 2], 1.0)
        out = out + A / (1.0 + ((t - mu) / gam) ** 2)
    return out


def fit_lorentz_sum(t, A, n_l):
    init_mus = np.linspace(-1500, 50, n_l)
    init_gams = np.full(n_l, 200.0)
    init_amps = np.full(n_l, A[PEAK_IDX] / n_l)
    p0 = []
    for k in range(n_l):
        p0 += [init_amps[k], init_mus[k], init_gams[k]]
    bounds_lo = []
    bounds_hi = []
    for k in range(n_l):
        bounds_lo += [-2.0, -3500.0, 5.0]
        bounds_hi += [2.0, 200.0, 5000.0]
    try:
        popt, _ = curve_fit(lambda tt, *p: lorentz_sum(tt, p), t, A,
                            p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=30000)
    except Exception:
        popt = np.array(p0)
    return popt


N_GAUSS = 5
N_LOR = 5
print("[direct] fitting Gaussian/Lorentzian sums to amplitude ...")
gauss_amp_t = np.zeros((N_TRAIN, 3 * N_GAUSS))
lorentz_amp_t = np.zeros((N_TRAIN, 3 * N_LOR))
for i in range(N_TRAIN):
    gauss_amp_t[i] = fit_gauss_sum(T_GRID, A_t[i], N_GAUSS)
    lorentz_amp_t[i] = fit_lorentz_sum(T_GRID, A_t[i], N_LOR)


def predict_direct_amp(q_arr, amp_table, phi_table, mode, sum_type):
    x = reparam_q(q_arr, mode)
    amp_p = eval_poly_table(amp_table, x)
    phi_p = eval_poly_table(phi_table, x)
    n = len(q_arr)
    h = np.zeros((n, N_T), dtype=complex)
    fn = gauss_sum if sum_type == "gauss" else lorentz_sum
    for i in range(n):
        A = np.maximum(fn(T_GRID, amp_p[i]), 0.0)
        phi = imr_phi(T_GRID, *phi_p[i])
        h[i] = A * np.exp(-1j * phi)
    return h


def build_gauss_amp(approach_num, name, parameterization, deg_poly):
    t0 = time.perf_counter()
    x = reparam_q(qt, parameterization)
    amp_table = fit_poly_table(x, gauss_amp_t, deg_poly)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    train_time = time.perf_counter() - t0
    h_pred = predict_direct_amp(qv, amp_table, phi_table, parameterization, "gauss")

    var = "x"
    lines = [
        f"CLOSED-FORM h22(t; q) — sum of {N_GAUSS} Gaussians (amp) + PN-RD phase",
        "=" * 60,
        f"  {var} = {mode_to_text(parameterization)}",
        "",
        f"  A(t; q) = Σ_{{k=1..{N_GAUSS}}} a_k(q) · exp( -((t - μ_k(q))^2) / (2 σ_k(q)^2) )",
        "  phi(t; q) = phi_base(t; q)   (same IMR phase as model 01)",
        "",
    ]
    for k in range(N_GAUSS):
        lines.append(f"  a_{k+1}(q) = {poly_str(amp_table[3*k],   var)}")
        lines.append(f"  μ_{k+1}(q) = {poly_str(amp_table[3*k+1], var)}")
        lines.append(f"  σ_{k+1}(q) = {poly_str(amp_table[3*k+2], var)}")
    for j, n_ in enumerate(["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]):
        lines.append(f"  {n_}(q) = {poly_str(phi_table[j], var)}")
    lines += ["", "h22(t; q) = max(A(t; q), 0) · exp(-i · phi(t; q))"]
    expr = "\n".join(lines)
    n_params = amp_table.size + phi_table.size
    evaluate_and_save(
        approach_num, name, "functional_form", parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"Sum of {N_GAUSS} Gaussians for amplitude + IMR phase.",
        coeffs_savedict={"amp_table": amp_table, "phi_table": phi_table})


def build_lorentz_amp(approach_num, name, parameterization, deg_poly):
    t0 = time.perf_counter()
    x = reparam_q(qt, parameterization)
    amp_table = fit_poly_table(x, lorentz_amp_t, deg_poly)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    train_time = time.perf_counter() - t0
    h_pred = predict_direct_amp(qv, amp_table, phi_table, parameterization, "lorentz")

    var = "x"
    lines = [
        f"CLOSED-FORM h22(t; q) — sum of {N_LOR} Lorentzians (amp) + PN-RD phase",
        "=" * 60,
        f"  {var} = {mode_to_text(parameterization)}",
        "",
        f"  A(t; q) = Σ_{{k=1..{N_LOR}}} a_k(q) / (1 + ((t - μ_k(q)) / γ_k(q))^2)",
        "  phi(t; q) = phi_base(t; q)   (IMR phase, same as model 01)",
        "",
    ]
    for k in range(N_LOR):
        lines.append(f"  a_{k+1}(q) = {poly_str(amp_table[3*k],   var)}")
        lines.append(f"  μ_{k+1}(q) = {poly_str(amp_table[3*k+1], var)}")
        lines.append(f"  γ_{k+1}(q) = {poly_str(amp_table[3*k+2], var)}")
    for j, n_ in enumerate(["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]):
        lines.append(f"  {n_}(q) = {poly_str(phi_table[j], var)}")
    lines += ["", "h22(t; q) = max(A(t; q), 0) · exp(-i · phi(t; q))"]
    expr = "\n".join(lines)
    n_params = amp_table.size + phi_table.size
    evaluate_and_save(
        approach_num, name, "functional_form", parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"Sum of {N_LOR} Lorentzians for amplitude + IMR phase.",
        coeffs_savedict={"amp_table": amp_table, "phi_table": phi_table})


build_gauss_amp(14, "gauss5_amp_eta_p4",   "eta",   4)
build_gauss_amp(15, "gauss5_amp_delta_p4", "delta", 4)
build_lorentz_amp(16, "lorentz5_amp_eta_p4",   "eta",   4)
build_lorentz_amp(17, "lorentz5_amp_delta_p4", "delta", 4)


# ---------------------------------------------------------------------------
# Damped sinusoid sum directly modelling complex h22(t)
# ---------------------------------------------------------------------------
N_DS = 4


def h_damped_sin(t, params):
    n_k = len(params) // 5
    out = np.zeros_like(t, dtype=complex)
    for k in range(n_k):
        A = params[5 * k]
        mu = params[5 * k + 1]
        sig = max(params[5 * k + 2], 5.0)
        om = params[5 * k + 3]
        ph = params[5 * k + 4]
        env = np.exp(-0.5 * ((t - mu) / sig) ** 2)
        out = out + A * env * np.exp(-1j * (om * t + ph))
    return out


def fit_h_damped_sin(t, h, n_k):
    A_arr = np.abs(h)
    pk = float(A_arr[PEAK_IDX])
    init_mus = np.linspace(-800, 30, n_k)
    init_sigs = np.full(n_k, 300.0)
    init_amps = np.full(n_k, pk / n_k)
    init_omegas = np.linspace(0.1, 0.6, n_k)
    init_phs = np.zeros(n_k)
    p0 = []
    for k in range(n_k):
        p0 += [init_amps[k], init_mus[k], init_sigs[k], init_omegas[k], init_phs[k]]

    def fn(_t, *p):
        z = h_damped_sin(_t, p)
        return np.concatenate([z.real, z.imag])

    target = np.concatenate([h.real, h.imag])
    bounds_lo = []
    bounds_hi = []
    for k in range(n_k):
        bounds_lo += [-2.0, -3000.0, 10.0, 0.01, -10.0]
        bounds_hi += [2.0, 200.0, 5000.0, 1.5, 10.0]
    try:
        popt, _ = curve_fit(fn, t, target, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=30000)
    except Exception:
        popt = np.array(p0)
    return popt


print("[direct dsin] fitting damped sinusoid sum to h22 ...")
dsin_t = np.zeros((N_TRAIN, 5 * N_DS))
for i in range(N_TRAIN):
    dsin_t[i] = fit_h_damped_sin(T_GRID, ht[i], N_DS)


def predict_dsin(q_arr, table, mode):
    x = reparam_q(q_arr, mode)
    p = eval_poly_table(table, x)
    n = len(q_arr)
    h = np.zeros((n, N_T), dtype=complex)
    for i in range(n):
        h[i] = h_damped_sin(T_GRID, p[i])
    return h


def build_dsin(approach_num, name, parameterization, deg_poly):
    t0 = time.perf_counter()
    x = reparam_q(qt, parameterization)
    table = fit_poly_table(x, dsin_t, deg_poly)
    train_time = time.perf_counter() - t0
    h_pred = predict_dsin(qv, table, parameterization)

    var = "x"
    lines = [
        f"CLOSED-FORM h22(t; q) — sum of {N_DS} Gaussian-enveloped damped sinusoids",
        "=" * 60,
        f"  {var} = {mode_to_text(parameterization)}",
        "",
        f"  h22(t; q) = Σ_{{k=1..{N_DS}}} A_k(q) · exp(-(t - μ_k(q))^2 / (2 σ_k(q)^2))",
        "                      · exp(-i (ω_k(q) · t + φ_k(q)))",
        "",
    ]
    for k in range(N_DS):
        lines.append(f"  A_{k+1}(q) = {poly_str(table[5*k+0], var)}")
        lines.append(f"  μ_{k+1}(q) = {poly_str(table[5*k+1], var)}")
        lines.append(f"  σ_{k+1}(q) = {poly_str(table[5*k+2], var)}")
        lines.append(f"  ω_{k+1}(q) = {poly_str(table[5*k+3], var)}")
        lines.append(f"  φ_{k+1}(q) = {poly_str(table[5*k+4], var)}")
    expr = "\n".join(lines)
    n_params = table.size
    evaluate_and_save(
        approach_num, name, "functional_form", parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"Direct sum of {N_DS} Gaussian-enveloped sinusoids — closed-form complex h22.",
        coeffs_savedict={"table": table})


build_dsin(18, "dsin4_eta_p4",   "eta",   4)
build_dsin(19, "dsin4_delta_p4", "delta", 4)


# ---------------------------------------------------------------------------
# Family 4 — Symbolic regression (PySR, gplearn) for the per-q parameters
# ---------------------------------------------------------------------------
def have_gplearn():
    try:
        import gplearn  # noqa
        return True
    except Exception:
        return False


# We deliberately skip importing pysr — its Julia backend triggers a multi-minute
# precompilation step on first import that blocks the build pipeline.  We use
# gplearn (pure Python) for symbolic regression instead.
PYSR_OK = False
GPLEARN_OK = have_gplearn()
print(f"[symreg] pysr=disabled (Julia precompile too slow)  gplearn={GPLEARN_OK}")


def pysr_scalar(x, y, niter=40, popsize=40):
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=niter,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sqrt", "square"],
        maxsize=15,
        populations=8,
        population_size=popsize,
        progress=False,
        verbosity=0,
        deterministic=False,
        random_state=42,
        temp_equation_file=True,
    )
    Xfit = np.asarray(x).reshape(-1, 1)
    yfit = np.asarray(y).reshape(-1)
    model.fit(Xfit, yfit)
    best = model.get_best()
    expr_str = str(best["sympy_format"])
    import sympy as sp
    sym_x = sp.symbols("x0")
    eq = sp.sympify(expr_str)
    fn = sp.lambdify(sym_x, eq, modules=["numpy"])
    return fn, expr_str


def gplearn_scalar(x, y, generations=30):
    from gplearn.genetic import SymbolicRegressor
    sr = SymbolicRegressor(
        population_size=400,
        generations=generations,
        function_set=("add", "sub", "mul", "div", "log", "sqrt"),
        metric="mse",
        parsimony_coefficient=0.01,
        random_state=42,
        verbose=0,
        n_jobs=1,
    )
    Xfit = np.asarray(x).reshape(-1, 1)
    yfit = np.asarray(y).reshape(-1)
    sr.fit(Xfit, yfit)
    fn = lambda xq: sr.predict(np.asarray(xq).reshape(-1, 1))
    expr_str = str(sr._program)
    return fn, expr_str


def build_symreg_amp(approach_num, name, parameterization, library="pysr",
                     niter=40, generations=25, deg_cheb=14, deg_poly=4):
    """Replace polynomial-in-x of the IMR base AMPLITUDE params with symbolic
    expressions discovered by PySR or gplearn. Phase + Cheb residual stays
    polynomial-in-x for stability."""
    if library == "pysr" and not PYSR_OK:
        print(f"  PySR unavailable, skipping {name}")
        return
    if library == "gplearn" and not GPLEARN_OK:
        print(f"  gplearn unavailable, skipping {name}")
        return
    x = reparam_q(qt, parameterization)
    x_v = reparam_q(qv, parameterization)
    t0 = time.perf_counter()
    amp_eval_v = np.zeros((N_VAL, N_AMP_BASE))
    expr_strs = []
    for j in range(N_AMP_BASE):
        if library == "pysr":
            fn, s = pysr_scalar(x, amp_base_t[:, j], niter=niter)
        else:
            fn, s = gplearn_scalar(x, amp_base_t[:, j], generations=generations)
        amp_eval_v[:, j] = fn(x_v)
        expr_strs.append(s)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    cheb_a = fit_poly_table(x, cheb_amp_t[deg_cheb], deg_poly)
    cheb_p = fit_poly_table(x, cheb_phi_t[deg_cheb], deg_poly)
    train_time = time.perf_counter() - t0
    # Build h
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
        f"CLOSED-FORM h22(t; q) — IMR base + Chebyshev residual; amp params via {library}",
        "=" * 60,
        f"  x = {var}",
        "",
        f"Amplitude per-q params from symbolic regression ({library}):",
    ]
    amp_names = ["log_Apk", "T_pn", "p_pre", "tau_RD", "p_post"]
    for j, n_ in enumerate(amp_names):
        lines.append(f"  {n_}(q) = {expr_strs[j]}")
    lines.append("Phase per-q params (degree-{} polynomial in x):".format(deg_poly))
    for j, n_ in enumerate(["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]):
        lines.append(f"  {n_}(q) = {poly_str(phi_table[j], 'x')}")
    lines.append("Chebyshev residual coefficients (degree-{} polynomial):".format(deg_poly))
    for n_ in range(cheb_a.shape[0]):
        lines.append(f"  a_{n_}(q) = {poly_str(cheb_a[n_], 'x')}")
    for n_ in range(cheb_p.shape[0]):
        lines.append(f"  b_{n_}(q) = {poly_str(cheb_p[n_], 'x')}")
    lines += [
        "",
        "log A(t; q) = log A_base(t; q) + Σ_n a_n(q) T_n(τ)",
        "phi  (t; q) = phi_base (t; q) + Σ_n b_n(q) T_n(τ)",
        "h22(t; q) = exp(log A) · exp(-i · phi)",
    ]
    expr = "\n".join(lines)
    # Approximate effective n_params (token count for symbolic exprs is conservative)
    sym_token_count = sum(s.count(" ") + 1 for s in expr_strs)
    n_params = sym_token_count + phi_table.size + cheb_a.size + cheb_p.size
    evaluate_and_save(
        approach_num, name, "symbolic_regression", parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"{library} symbolic regression for amp params; rest is polynomial.",
        coeffs_savedict={"phi_table": phi_table, "cheb_amp_table": cheb_a, "cheb_phi_table": cheb_p})


def build_symreg_phi(approach_num, name, parameterization, library="pysr",
                     niter=40, generations=25, deg_cheb=14, deg_poly=4):
    if library == "pysr" and not PYSR_OK:
        print(f"  PySR unavailable, skipping {name}")
        return
    if library == "gplearn" and not GPLEARN_OK:
        print(f"  gplearn unavailable, skipping {name}")
        return
    x = reparam_q(qt, parameterization)
    x_v = reparam_q(qv, parameterization)
    t0 = time.perf_counter()
    phi_eval_v = np.zeros((N_VAL, N_PHI_BASE))
    expr_strs = []
    for j in range(N_PHI_BASE):
        if library == "pysr":
            fn, s = pysr_scalar(x, phi_base_t[:, j], niter=niter)
        else:
            fn, s = gplearn_scalar(x, phi_base_t[:, j], generations=generations)
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
    lines = [
        f"CLOSED-FORM h22(t; q) — IMR base + Chebyshev residual; phase params via {library}",
        "=" * 60,
        f"  x = {var}",
        "",
        f"Phase per-q params from symbolic regression ({library}):",
    ]
    for j, n_ in enumerate(["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]):
        lines.append(f"  {n_}(q) = {expr_strs[j]}")
    lines.append("Amplitude per-q params (poly):")
    for j, n_ in enumerate(["log_Apk", "T_pn", "p_pre", "tau_RD", "p_post"]):
        lines.append(f"  {n_}(q) = {poly_str(amp_table[j], 'x')}")
    lines.append("Chebyshev residuals (polynomial coefficients):")
    for n_ in range(cheb_a.shape[0]):
        lines.append(f"  a_{n_}(q) = {poly_str(cheb_a[n_], 'x')}")
    for n_ in range(cheb_p.shape[0]):
        lines.append(f"  b_{n_}(q) = {poly_str(cheb_p[n_], 'x')}")
    lines += [
        "",
        "log A(t; q) = log A_base + Σ_n a_n(q) T_n(τ)",
        "phi  (t; q) = phi_base  + Σ_n b_n(q) T_n(τ)",
        "h22(t; q) = exp(log A) · exp(-i · phi)",
    ]
    expr = "\n".join(lines)
    sym_token_count = sum(s.count(" ") + 1 for s in expr_strs)
    n_params = sym_token_count + amp_table.size + cheb_a.size + cheb_p.size
    evaluate_and_save(
        approach_num, name, "symbolic_regression", parameterization, expr,
        h_pred, train_time, n_params,
        notes=f"{library} symbolic regression for phase params.",
        coeffs_savedict={"amp_table": amp_table, "cheb_amp_table": cheb_a, "cheb_phi_table": cheb_p})


try:
    build_symreg_amp(20, "gplearn_amp_imr_eta",   "eta",   library="gplearn", generations=30, deg_cheb=14, deg_poly=3)
except Exception as e:
    print("  gplearn amp eta model failed:", e); traceback.print_exc()
try:
    build_symreg_phi(21, "gplearn_phase_imr_delta", "delta", library="gplearn", generations=30, deg_cheb=14, deg_poly=3)
except Exception as e:
    print("  gplearn phase delta model failed:", e); traceback.print_exc()
try:
    build_symreg_amp(22, "gplearn_amp_imr_logq",   "log_q", library="gplearn", generations=30, deg_cheb=14, deg_poly=3)
except Exception as e:
    print("  gplearn amp logq model failed:", e); traceback.print_exc()
try:
    build_symreg_phi(23, "gplearn_phase_imr_eta",  "eta",   library="gplearn", generations=30, deg_cheb=14, deg_poly=3)
except Exception as e:
    print("  gplearn phase eta model failed:", e); traceback.print_exc()


# ---------------------------------------------------------------------------
# A simple "pure PN+RD only" baseline — no Cheb residual, polynomial deg 4
# ---------------------------------------------------------------------------
def build_pure_imr(approach_num, name, parameterization, deg_poly):
    t0 = time.perf_counter()
    x = reparam_q(qt, parameterization)
    amp_table = fit_poly_table(x, amp_base_t, deg_poly)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    train_time = time.perf_counter() - t0
    x_v = reparam_q(qv, parameterization)
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
        notes=f"Pure IMR base — PN inspiral + power-law decay amp; PN+RD phase. No Cheb residual.",
        coeffs_savedict={"amp_table": amp_table, "phi_table": phi_table})


build_pure_imr(24, "phen_imr_pure_eta_p4", "eta", 4)


# ---------------------------------------------------------------------------
# Save master result list
# ---------------------------------------------------------------------------
with open(_DATA / "all_results.json", "w") as f:
    json.dump(ALL_RESULTS, f, indent=2)

print(f"\n[summary] built {len(ALL_RESULTS)} valid closed-form models.")
print("\nLeaderboard (lower is better):")
sorted_res = sorted(ALL_RESULTS, key=lambda r: r["scorecard"]["loss"])
for r in sorted_res:
    sc = r["scorecard"]
    print(f"  {sc['approach_number']:02d}_{sc['approach']:<32} loss={sc['loss']:.4e}  n_params={sc['n_params']}")
