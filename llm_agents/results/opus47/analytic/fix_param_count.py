"""Trim the cheb_deg=14 / poly_deg=4 models from n_params=200 to n_params<200
by dropping the highest Chebyshev term (cheb_deg=14 -> effective 13).
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (
    load_data, fd_mismatch_proxy, fd_mismatch_real,
    save_scorecard, write_train_predict, model_dir, RESULTS_DIR,
    T_GRID, N_T, T_GRID_DT,
    poly_eval, poly_str, fit_param_polynomial, reparam_q,
)
from numpy.polynomial import chebyshev as nc

qt, ht, qv, hv = load_data()
N_TRAIN, N_VAL = len(qt), len(qv)
T_MIN, T_MAX = T_GRID[0], T_GRID[-1]
T_NORM = 2 * (T_GRID - T_MIN) / (T_MAX - T_MIN) - 1.0


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


# Load decomposition
cache = np.load(RESULTS_DIR / "_data" / "perq_decomp.npz")
amp_base_t = cache["amp_base"]
phi_base_t = cache["phi_base"]


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


def mode_to_text(mode):
    return {"q": "q", "eta": "q / (1 + q)^2", "delta": "(q - 1) / (q + 1)",
            "log_q": "log(q)", "inv_q": "1 / q"}[mode]


def imr_cheb13_expression(amp_table, phi_table, ca, cp, mode, suffix=""):
    var = "x"
    lines = [
        f"CLOSED-FORM h22(t; q) — IMR base + Chebyshev residual{suffix}",
        "=" * 60, "",
        f"  {var} = {mode_to_text(mode)}",
        f"  τ = 2 (t - {T_MIN:g}) / ({T_MAX:g} - {T_MIN:g}) - 1   (scaled time in [-1, 1])",
        "",
        "Aux:  s_post(t) = (1+tanh(t/5))/2;  s_pre = 1-s_post",
        "      s_post_phi(t; q) = (1+tanh(t/T_b(q)))/2;  s_pre_phi = 1-s_post_phi",
        "",
        "log A_base(t; q) = log_Apk(q)",
        "                  + s_pre(t)·(-p_pre(q)·log(1 - t/T_pn(q)))",
        "                  + s_post(t)·(-(max(t,0)/tau_RD(q))^p_post(q))",
        "phi_base(t; q) = phi_pk(q)",
        "               + s_pre_phi(t; q)·(-(8/5)·omega_pn(q)·T_pn_phi(q)·((1-t/T_pn_phi(q))^(5/8)-1))",
        "               + s_post_phi(t; q)·(omega_RD(q)·t)",
        "",
        "Chebyshev residual:",
        "  log A(t; q) = log A_base + Σ_{n=0}^{N} a_n(q)·T_n(τ)",
        "  phi(t; q)  = phi_base  + Σ_{n=0}^{N} b_n(q)·T_n(τ)",
        "",
    ]
    amp_names = ["log_Apk", "T_pn", "p_pre", "tau_RD", "p_post"]
    phi_names = ["phi_pk", "omega_RD", "omega_pn", "T_pn_phi", "T_b"]
    for j, n_ in enumerate(amp_names):
        lines.append(f"  {n_}(q) = {poly_str(amp_table[j], var)}")
    for j, n_ in enumerate(phi_names):
        lines.append(f"  {n_}(q) = {poly_str(phi_table[j], var)}")
    for n in range(ca.shape[0]):
        lines.append(f"  a_{n}(q) = {poly_str(ca[n], var)}")
    for n in range(cp.shape[0]):
        lines.append(f"  b_{n}(q) = {poly_str(cp[n], var)}")
    lines.append("")
    lines.append("h22(t; q) = exp(log A(t; q)) · exp(-i · phi(t; q))")
    return "\n".join(lines)


def predict_imr_cheb(q_arr, amp_table, phi_table, ca, cp, mode):
    x = reparam_q(q_arr, mode)
    amp_b = eval_poly_table(amp_table, x)
    phi_b = eval_poly_table(phi_table, x)
    cae = eval_poly_table(ca, x)
    cpe = eval_poly_table(cp, x)
    n = len(q_arr)
    h = np.zeros((n, N_T), dtype=complex)
    for i in range(n):
        log_A = imr_amp(T_GRID, *amp_b[i]) + nc.chebval(T_NORM, cae[i])
        phi = imr_phi(T_GRID, *phi_b[i]) + nc.chebval(T_NORM, cpe[i])
        h[i] = np.exp(log_A) * np.exp(-1j * phi)
    return h


# Five models to retrim: cheb=14, poly=4 -> drop one Cheb term (use cheb=13 effective)
# cheb_amp_t[d=14] is shape (20, 15). Drop the last column -> (20, 14)
cheb_amp_d14 = cache["cheb_amp_d14"]
cheb_phi_d14 = cache["cheb_phi_d14"]


def retrain_and_save(approach_num, name, parameterization, deg_poly=4, drop_last=1):
    ca_t = cheb_amp_d14[:, :-drop_last]
    cp_t = cheb_phi_d14[:, :-drop_last]
    eff_cheb_deg = ca_t.shape[1] - 1
    x = reparam_q(qt, parameterization)
    t0 = time.perf_counter()
    amp_table = fit_poly_table(x, amp_base_t, deg_poly)
    phi_table = fit_poly_table(x, phi_base_t, deg_poly)
    ca = fit_poly_table(x, ca_t, deg_poly)
    cp = fit_poly_table(x, cp_t, deg_poly)
    train_time = time.perf_counter() - t0
    h_pred = predict_imr_cheb(qv, amp_table, phi_table, ca, cp, parameterization)
    proxy = fd_mismatch_proxy(h_pred, hv)
    eval_t0 = time.perf_counter()
    loss, comps, per_sample = fd_mismatch_real(h_pred, hv)
    eval_time = (time.perf_counter() - eval_t0) / max(N_VAL, 1)
    n_params = amp_table.size + phi_table.size + ca.size + cp.size
    md = model_dir(approach_num, name)
    expr = imr_cheb13_expression(amp_table, phi_table, ca, cp, parameterization,
                                 suffix=f" (cheb deg {eff_cheb_deg}, q poly deg {deg_poly})")
    sc = {
        "approach": name,
        "approach_number": approach_num,
        "benchmark": "analytic",
        "agent": "opus47",
        "category": "physics_imr",
        "parameterization": parameterization,
        "time_convention": "t0_at_peak",
        "loss": float(loss),
        "loss_proxy_l2": float(np.mean(proxy)),
        "loss_components": {k: float(v) for k, v in comps.items()},
        "per_sample_loss": [float(x) for x in per_sample.tolist()],
        "runtime_ms": float(eval_time * 1000.0),
        "n_train": N_TRAIN, "n_val": N_VAL,
        "n_params": int(n_params),
        "train_time_s": float(train_time),
        "notes": f"IMR base + Chebyshev_{eff_cheb_deg} residual; q-dep polynomial deg {deg_poly} in {parameterization}.",
        "is_closed_form": True,
    }
    save_scorecard(md, sc)
    (md / "expression.txt").write_text(expr)
    write_train_predict(md, name, expr[:600])
    np.savez(md / "saved_model" / "coeffs.npz", amp_table=amp_table, phi_table=phi_table,
             cheb_amp_table=ca, cheb_phi_table=cp)
    print(f"[fixed {approach_num:02d}_{name}] loss={loss:.4e}, n_params={n_params}")


# Same names as before, just trimmed
old_names = [
    (1,  "phen_imr_cheb14_eta_p4",   "eta"),
    (2,  "phen_imr_cheb14_delta_p4", "delta"),
    (3,  "phen_imr_cheb14_logq_p4",  "log_q"),
    (4,  "phen_imr_cheb14_invq_p4",  "inv_q"),
    (5,  "phen_imr_cheb14_q_p4",     "q"),
]
for n, name, mode in old_names:
    retrain_and_save(n, name, mode, deg_poly=4, drop_last=1)
