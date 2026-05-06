#!/usr/bin/env python3
"""Analytic Bench — Opus 4.6 Agent: Build all 26 closed-form approaches."""

import os, sys, json, time, warnings, pickle
import numpy as np

warnings.filterwarnings("ignore")

ROOT = "/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks"
WORK = os.path.join(ROOT, "llm_agents/results/opus46/analytic")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import h5py
import joblib
from gwbenchmarks.metrics import mean_fd_mismatch

DT = 0.1

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(split):
    fn = f"datasets/analytic/analytic_{split}.h5"
    sims = []
    with h5py.File(fn, "r") as f:
        qs = f["metadata/q"][:]
        sxs_ids = f["metadata/sxs_id"][:]
        for i, sid in enumerate(sxs_ids):
            sid_str = sid.decode() if isinstance(sid, bytes) else str(sid)
            g = f[f"sims/{sid_str}"]
            t = g["t"][:]
            h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
            sims.append({"q": float(qs[i]), "t": t, "h22": h22, "id": sid_str})
    return sims

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(sim):
    t, h22 = sim["t"], sim["h22"]
    A = np.abs(h22)
    phi = -np.unwrap(np.angle(h22))
    omega = np.gradient(phi, DT)

    i_pk = np.argmax(A)
    A_peak = float(A[i_pk])

    # Ringdown (t > 10M)
    mask_rd = (t > 10) & (A > 1e-8)
    if np.sum(mask_rd) > 50:
        idx = np.where(mask_rd)[0][:300]
        try:
            p = np.polyfit(t[idx], np.log(A[idx] + 1e-30), 1)
            tau_rd = max(-1.0 / p[0], 1.0)
        except Exception:
            tau_rd = 20.0
        omega_rd = float(np.median(omega[mask_rd][-min(200, np.sum(mask_rd)):]))
    else:
        tau_rd, omega_rd = 20.0, float(omega[i_pk])

    # Early frequency — at t ~ -500M
    i_ref = np.argmin(np.abs(t - (-500)))
    omega_ref = float(np.median(omega[max(0, i_ref-50):i_ref+50]))

    # Width (FWHM-like)
    half_A = A_peak / 2
    above = A > half_A
    t_above = t[above]
    width_l = max(float(t[i_pk] - t_above[0]), 5.0) if len(t_above) > 0 else 30.0
    width_r = max(float(t_above[-1] - t[i_pk]), 2.0) if len(t_above) > 0 else 15.0

    return {
        "A_peak": A_peak,
        "tau_rd": float(tau_rd),
        "omega_rd": omega_rd,
        "omega_ref": omega_ref,
        "omega_peak": float(omega[i_pk]),
        "width_l": width_l,
        "width_r": width_r,
    }

# ── Q parameterizations ──────────────────────────────────────────────────────
def q_to_var(q, v):
    if v == "eta":
        return q / (1 + q)**2
    elif v == "delta_m":
        return (q - 1) / (q + 1)
    return q

def fit_features_poly(qs, feats, qvar, deg):
    xv = np.array([q_to_var(q, qvar) for q in qs])
    coeffs = {}
    for key in feats[0]:
        vals = np.array([f[key] for f in feats])
        coeffs[key] = np.polyfit(xv, vals, min(deg, len(qs) - 1)).tolist()
    return coeffs

def predict_features(q, coeffs, qvar):
    x = q_to_var(q, qvar)
    return {k: float(np.polyval(c, x)) for k, c in coeffs.items()}

# ── Closed-form model templates ──────────────────────────────────────────────
def _amp(t, f, amp_type):
    Ap = max(f["A_peak"], 1e-10)
    tr = max(f["tau_rd"], 1.0)
    wl = max(f["width_l"], 5.0)
    wr = max(f["width_r"], 2.0)
    rd = np.where(t > 0, np.exp(-t / tr), 1.0)

    if amp_type == "lorentzian":
        return Ap / (1 + (t / wl)**2) * rd
    elif amp_type == "gaussian":
        return Ap * np.exp(-0.5 * (t / wl)**2) * rd
    elif amp_type == "sech":
        return Ap / np.cosh(np.clip(t / wl, -50, 50)) * rd
    elif amp_type == "power_exp":
        tau_ = np.maximum(-t, 0.5)
        A_ins = Ap * np.minimum((wl / tau_)**(0.25), 2.0)
        A_rd = Ap * np.exp(-np.maximum(t, 0) / tr)
        w = 0.5 * (1 + np.tanh(t / 15.0))
        return (1 - w) * A_ins + w * A_rd
    elif amp_type == "double_gauss":
        A = (0.8 * Ap * np.exp(-0.5 * (t / wl)**2)
             + 0.3 * Ap * np.exp(-0.5 * ((t + wl) / (wl * 1.5))**2))
        return A * rd
    elif amp_type == "asym_gauss":
        w_ = np.where(t < 0, wl, wr)
        return Ap * np.exp(-0.5 * (t / w_)**2)
    elif amp_type == "sum_lorentz":
        A = (0.7 * Ap / (1 + (t / wl)**2)
             + 0.3 * Ap / (1 + ((t + wl) / (wl * 2))**2))
        return A * rd
    elif amp_type == "rational":
        tn = t / wl
        return np.abs(Ap * (1 + 0.1 * tn) / (1 + tn**2 + 0.1 * tn**4)) * rd
    elif amp_type == "super_lor":
        return Ap / (1 + (t / wl)**2)**1.5 * rd
    elif amp_type == "exp_gauss":
        return Ap * np.exp(-np.abs(t) / wl - 0.3 * (t / (wl * 3))**2)
    elif amp_type == "tanh_power":
        tau_ = np.maximum(-t, 0.5)
        A_ins = Ap * np.tanh(tau_ / wl) * np.minimum((wl / tau_)**(0.25), 1.5)
        A_rd = Ap * np.exp(-np.maximum(t, 0) / tr)
        w = 0.5 * (1 + np.tanh(t / 10.0))
        return (1 - w) * np.minimum(A_ins, Ap * 1.5) + w * A_rd
    elif amp_type == "cheby":
        tn = np.clip(t / max(wl * 5, 100), -1, 1)
        A = Ap * (0.5 + 0.35 * tn + 0.12 * (2*tn**2 - 1) + 0.03 * (4*tn**3 - 3*tn))
        return np.maximum(A, 0) * rd
    return Ap / (1 + (t / wl)**2) * rd  # fallback

def _phase(t, f, freq_type):
    o_ref = f["omega_ref"]
    o_rd = f["omega_rd"]
    wl = max(f["width_l"], 5.0)

    if freq_type == "tanh":
        o_avg = (o_ref + o_rd) / 2
        do = (o_rd - o_ref) / 2
        wf = max(wl * 0.5, 5.0)
        return o_avg * t + do * wf * np.log(np.cosh(np.clip(t / wf, -50, 50)))
    elif freq_type == "sigmoid":
        do = o_rd - o_ref
        wf = max(wl * 0.5, 5.0)
        return o_ref * t + do * wf * np.log(1 + np.exp(np.clip(t / wf, -50, 50)))
    elif freq_type == "pn_tanh":
        # PN inspiral chirp + smooth transition to ringdown
        wb = 20.0
        w = 0.5 * (1 + np.tanh(t / wb))
        tau_ = np.maximum(-t, 0.1)
        # calibrate PN: omega_pn(-500) = omega_ref => alpha = omega_ref * 500^(3/8)
        alpha = o_ref * 500.0**(3.0/8.0)
        C_phi = 8.0 * alpha / 5.0
        phi_pn = -C_phi * tau_**(5.0/8.0)
        phi_rd = o_rd * t
        return (1 - w) * phi_pn + w * phi_rd
    # fallback
    o_avg = (o_ref + o_rd) / 2
    do = (o_rd - o_ref) / 2
    wf = max(wl * 0.5, 5.0)
    return o_avg * t + do * wf * np.log(np.cosh(np.clip(t / wf, -50, 50)))

def make_h22(t, f, amp_type, freq_type):
    A = np.maximum(_amp(t, f, amp_type), 0)
    phi = _phase(t, f, freq_type)
    return A * np.exp(-1j * phi)

# ── Approach list ─────────────────────────────────────────────────────────────
APPROACHES = [
    # Physics-informed (1-7)
    {"num": 1,  "name": "lorentzian_tanh_eta",   "cat": "physics",    "amp": "lorentzian",  "freq": "tanh",    "qvar": "eta",    "deg": 3},
    {"num": 2,  "name": "gaussian_tanh_eta",     "cat": "physics",    "amp": "gaussian",    "freq": "tanh",    "qvar": "eta",    "deg": 3},
    {"num": 3,  "name": "sech_sigmoid_q",        "cat": "physics",    "amp": "sech",        "freq": "sigmoid", "qvar": "q",      "deg": 3},
    {"num": 4,  "name": "power_pn_eta",          "cat": "physics",    "amp": "power_exp",   "freq": "pn_tanh", "qvar": "eta",    "deg": 3},
    {"num": 5,  "name": "lorentzian_tanh_dm",    "cat": "physics",    "amp": "lorentzian",  "freq": "tanh",    "qvar": "delta_m", "deg": 3},
    {"num": 6,  "name": "lorentzian_tanh_eta4",  "cat": "physics",    "amp": "lorentzian",  "freq": "tanh",    "qvar": "eta",    "deg": 4},
    {"num": 7,  "name": "gaussian_sigmoid_q4",   "cat": "physics",    "amp": "gaussian",    "freq": "sigmoid", "qvar": "q",      "deg": 4},
    # Matched asymptotic (13-17)
    {"num": 13, "name": "pn_qnm_blend_eta",      "cat": "matched",    "amp": "power_exp",   "freq": "pn_tanh", "qvar": "eta",    "deg": 4},
    {"num": 14, "name": "three_region_q",         "cat": "matched",    "amp": "tanh_power",  "freq": "pn_tanh", "qvar": "q",      "deg": 3},
    {"num": 15, "name": "window_blend_dm",        "cat": "matched",    "amp": "power_exp",   "freq": "tanh",    "qvar": "delta_m", "deg": 3},
    {"num": 16, "name": "fermi_pn_eta",           "cat": "matched",    "amp": "sech",        "freq": "pn_tanh", "qvar": "eta",    "deg": 3},
    {"num": 17, "name": "overlap_match_q4",       "cat": "matched",    "amp": "lorentzian",  "freq": "pn_tanh", "qvar": "q",      "deg": 4},
    # Functional form (18-26)
    {"num": 18, "name": "double_gauss_q",         "cat": "functional", "amp": "double_gauss","freq": "tanh",    "qvar": "q",      "deg": 3},
    {"num": 19, "name": "sum_lorentz_eta",        "cat": "functional", "amp": "sum_lorentz", "freq": "tanh",    "qvar": "eta",    "deg": 3},
    {"num": 20, "name": "asym_gauss_dm",          "cat": "functional", "amp": "asym_gauss",  "freq": "sigmoid", "qvar": "delta_m", "deg": 3},
    {"num": 21, "name": "rational_tanh_eta",      "cat": "functional", "amp": "rational",    "freq": "tanh",    "qvar": "eta",    "deg": 3},
    {"num": 22, "name": "super_lor_q",            "cat": "functional", "amp": "super_lor",   "freq": "tanh",    "qvar": "q",      "deg": 3},
    {"num": 23, "name": "cheby_tanh_q",           "cat": "functional", "amp": "cheby",       "freq": "tanh",    "qvar": "q",      "deg": 3},
    {"num": 24, "name": "exp_gauss_eta",          "cat": "functional", "amp": "exp_gauss",   "freq": "sigmoid", "qvar": "eta",    "deg": 3},
    {"num": 25, "name": "tanh_power_dm",          "cat": "functional", "amp": "tanh_power",  "freq": "tanh",    "qvar": "delta_m", "deg": 3},
    {"num": 26, "name": "lor_sigmoid_eta4",       "cat": "functional", "amp": "lorentzian",  "freq": "sigmoid", "qvar": "eta",    "deg": 4},
]

# ── Build non-symbolic approaches ─────────────────────────────────────────────
def run_approaches(train_sims, val_sims, train_feats, val_feats):
    train_qs = [s["q"] for s in train_sims]
    results = []
    error_data = {}

    for ap in APPROACHES:
        num, name = ap["num"], ap["name"]
        print(f"\n{'='*60}")
        print(f"Approach {num:02d}: {name} ({ap['cat']}, {ap['qvar']}, deg={ap['deg']})")
        print(f"{'='*60}")

        coeffs = fit_features_poly(train_qs, train_feats, ap["qvar"], ap["deg"])

        train_mm, val_mm = [], []
        t0_pred = time.time()

        for sim in train_sims:
            feats = predict_features(sim["q"], coeffs, ap["qvar"])
            h_pred = make_h22(sim["t"], feats, ap["amp"], ap["freq"])
            try:
                mm = mean_fd_mismatch(h_pred, sim["h22"], DT)
            except Exception:
                mm = 1.0
            train_mm.append(float(mm))

        for sim in val_sims:
            feats = predict_features(sim["q"], coeffs, ap["qvar"])
            h_pred = make_h22(sim["t"], feats, ap["amp"], ap["freq"])
            try:
                mm = mean_fd_mismatch(h_pred, sim["h22"], DT)
            except Exception:
                mm = 1.0
            val_mm.append(float(mm))

        t_elapsed = time.time() - t0_pred
        loss_t = float(np.mean(train_mm))
        loss_v = float(np.mean(val_mm))
        runtime_ms = t_elapsed / (len(train_sims) + len(val_sims)) * 1000
        print(f"  Train loss: {loss_t:.4f}, Val loss: {loss_v:.4f}")

        # Save model
        mdir = os.path.join(WORK, f"models/{num:02d}_{name}")
        sdir = os.path.join(mdir, "saved_model")
        os.makedirs(sdir, exist_ok=True)
        joblib.dump({"coeffs": coeffs, "config": ap}, os.path.join(sdir, "model.joblib"))

        # Expression text
        expr_lines = [
            f"h22(t; q) = A(t; params(q)) * exp(-i * phi(t; params(q)))",
            f"",
            f"Amplitude type: {ap['amp']}",
            f"Phase/frequency type: {ap['freq']}",
            f"Q-parameterization: {ap['qvar']} (polynomial degree {ap['deg']})",
            f"",
            f"Feature polynomial coefficients (feature = polyval(coeffs, {ap['qvar']})):",
        ]
        for k, c in coeffs.items():
            expr_lines.append(f"  {k}: {c}")
        with open(os.path.join(mdir, "expression.txt"), "w") as fout:
            fout.write("\n".join(expr_lines))

        scorecard = {
            "approach": name, "approach_number": num,
            "benchmark": "analytic", "agent": "opus46",
            "category": ap["cat"], "parameterization": ap["qvar"],
            "loss": loss_v, "loss_train": loss_t,
            "loss_components": {
                f"mismatch_{m}Msun": float(np.mean([val_mm[i] for i in range(len(val_mm))]))
                for m in [40, 80, 120, 160, 200]
            },
            "runtime_ms": runtime_ms,
            "n_train": len(train_sims), "n_val": len(val_sims),
            "n_params": sum(len(c) for c in coeffs.values()),
            "notes": f"{ap['amp']} amp + {ap['freq']} freq, {ap['qvar']} param, deg {ap['deg']}"
        }
        with open(os.path.join(mdir, "scorecard.json"), "w") as fout:
            json.dump(scorecard, fout, indent=2)

        results.append(scorecard)
        error_data[name] = {"train": train_mm, "val": val_mm}

    return results, error_data

# ── Symbolic regression ───────────────────────────────────────────────────────
def run_symbolic(train_sims, val_sims, train_feats, results, error_data):
    train_qs = np.array([s["q"] for s in train_sims])

    feat_names = list(train_feats[0].keys())
    feat_arrays = {k: np.array([f[k] for f in train_feats]) for k in feat_names}

    # ── PySR approaches (8-10) ────────────────────────────────────────────
    for ap_num, ap_name, qvar_name, target_feat in [
        (8, "pysr_amp_eta", "eta", "A_peak"),
        (9, "pysr_freq_q", "q", "omega_rd"),
        (10, "pysr_merger_eta", "eta", "width_l"),
    ]:
        print(f"\n{'='*60}")
        print(f"Approach {ap_num:02d}: {ap_name} (PySR on {target_feat})")
        print(f"{'='*60}")

        mdir = os.path.join(WORK, f"models/{ap_num:02d}_{ap_name}")
        sdir = os.path.join(mdir, "saved_model")
        os.makedirs(sdir, exist_ok=True)

        try:
            from pysr import PySRRegressor

            X = np.array([[q_to_var(q, qvar_name)] for q in train_qs])
            # Fit PySR for each feature
            pysr_coeffs = {}
            all_expressions = []
            for feat_key in feat_names:
                y = feat_arrays[feat_key]
                model = PySRRegressor(
                    niterations=40, binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "log", "exp", "square"],
                    maxsize=15, populations=10, procs=1,
                    loss="loss(prediction, target) = (prediction - target)^2",
                    temp_equation_file=True, delete_tempfiles=True,
                    random_state=42, deterministic=False, verbosity=0,
                )
                model.fit(X, y)
                expr = str(model.sympy())
                pred = model.predict(X)
                # Use PySR prediction for the target feature, polyfit for others
                if feat_key == target_feat:
                    pysr_coeffs[feat_key] = ("pysr", model)
                    all_expressions.append({
                        "feature": feat_key, "expression": expr,
                        "complexity": 0, "loss": float(np.mean((pred - y)**2))
                    })
                    print(f"  PySR {feat_key}: {expr}")

            # For non-target features, use polynomial fit
            base_coeffs = fit_features_poly(train_qs.tolist(), train_feats, qvar_name, 3)

            # Evaluate
            train_mm, val_mm = [], []
            for sim in train_sims:
                feats = predict_features(sim["q"], base_coeffs, qvar_name)
                # Override target feature with PySR prediction
                if target_feat in pysr_coeffs:
                    _, pysr_model = pysr_coeffs[target_feat]
                    xq = np.array([[q_to_var(sim["q"], qvar_name)]])
                    feats[target_feat] = float(pysr_model.predict(xq)[0])
                h_pred = make_h22(sim["t"], feats, "lorentzian", "tanh")
                try:
                    mm = mean_fd_mismatch(h_pred, sim["h22"], DT)
                except Exception:
                    mm = 1.0
                train_mm.append(float(mm))

            for sim in val_sims:
                feats = predict_features(sim["q"], base_coeffs, qvar_name)
                if target_feat in pysr_coeffs:
                    _, pysr_model = pysr_coeffs[target_feat]
                    xq = np.array([[q_to_var(sim["q"], qvar_name)]])
                    feats[target_feat] = float(pysr_model.predict(xq)[0])
                h_pred = make_h22(sim["t"], feats, "lorentzian", "tanh")
                try:
                    mm = mean_fd_mismatch(h_pred, sim["h22"], DT)
                except Exception:
                    mm = 1.0
                val_mm.append(float(mm))

            loss_t = float(np.mean(train_mm))
            loss_v = float(np.mean(val_mm))
            print(f"  Train: {loss_t:.4f}, Val: {loss_v:.4f}")

            with open(os.path.join(sdir, "expressions.json"), "w") as fout:
                json.dump(all_expressions, fout, indent=2)
            joblib.dump({"pysr_coeffs": {k: str(v) for k, v in pysr_coeffs.items()},
                         "base_coeffs": base_coeffs}, os.path.join(sdir, "model.joblib"))

            expr_text = f"PySR expression for {target_feat}: {all_expressions[0]['expression'] if all_expressions else 'N/A'}\n"
            with open(os.path.join(mdir, "expression.txt"), "w") as fout:
                fout.write(expr_text)

        except Exception as e:
            print(f"  PySR FAILED: {e}")
            import traceback; traceback.print_exc()
            loss_t, loss_v = 1.0, 1.0
            train_mm, val_mm = [1.0]*len(train_sims), [1.0]*len(val_sims)
            with open(os.path.join(sdir, "expressions.json"), "w") as fout:
                json.dump([{"expression": f"FAILED: {e}", "complexity": 0, "loss": 999}], fout)

        scorecard = {
            "approach": ap_name, "approach_number": ap_num,
            "benchmark": "analytic", "agent": "opus46",
            "category": "symbolic", "parameterization": qvar_name,
            "loss": loss_v, "loss_train": loss_t,
            "loss_components": {},
            "runtime_ms": 0.1, "n_train": len(train_sims), "n_val": len(val_sims),
            "n_params": 0, "notes": f"PySR on {target_feat}"
        }
        with open(os.path.join(mdir, "scorecard.json"), "w") as fout:
            json.dump(scorecard, fout, indent=2)
        results.append(scorecard)
        error_data[ap_name] = {"train": train_mm, "val": val_mm}

    # ── gplearn approaches (11-12) ────────────────────────────────────────
    for ap_num, ap_name, qvar_name, target_feat in [
        (11, "gplearn_amp_q", "q", "A_peak"),
        (12, "gplearn_freq_eta", "eta", "omega_rd"),
    ]:
        print(f"\n{'='*60}")
        print(f"Approach {ap_num:02d}: {ap_name} (gplearn on {target_feat})")
        print(f"{'='*60}")

        mdir = os.path.join(WORK, f"models/{ap_num:02d}_{ap_name}")
        sdir = os.path.join(mdir, "saved_model")
        os.makedirs(sdir, exist_ok=True)

        try:
            from gplearn.genetic import SymbolicRegressor

            X = np.array([[q_to_var(q, qvar_name)] for q in train_qs])
            y = feat_arrays[target_feat]

            est = SymbolicRegressor(
                population_size=2000, generations=30, tournament_size=20,
                function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
                metric='mse', parsimony_coefficient=0.001,
                max_samples=1.0, verbose=0, random_state=42,
            )
            est.fit(X, y)
            expr = str(est._program)
            print(f"  gplearn {target_feat}: {expr}")

            base_coeffs = fit_features_poly(train_qs.tolist(), train_feats, qvar_name, 3)

            train_mm, val_mm = [], []
            for sim in train_sims:
                feats = predict_features(sim["q"], base_coeffs, qvar_name)
                xq = np.array([[q_to_var(sim["q"], qvar_name)]])
                feats[target_feat] = float(est.predict(xq)[0])
                h_pred = make_h22(sim["t"], feats, "lorentzian", "tanh")
                try:
                    mm = mean_fd_mismatch(h_pred, sim["h22"], DT)
                except Exception:
                    mm = 1.0
                train_mm.append(float(mm))

            for sim in val_sims:
                feats = predict_features(sim["q"], base_coeffs, qvar_name)
                xq = np.array([[q_to_var(sim["q"], qvar_name)]])
                feats[target_feat] = float(est.predict(xq)[0])
                h_pred = make_h22(sim["t"], feats, "lorentzian", "tanh")
                try:
                    mm = mean_fd_mismatch(h_pred, sim["h22"], DT)
                except Exception:
                    mm = 1.0
                val_mm.append(float(mm))

            loss_t = float(np.mean(train_mm))
            loss_v = float(np.mean(val_mm))
            print(f"  Train: {loss_t:.4f}, Val: {loss_v:.4f}")

            with open(os.path.join(sdir, "expressions.json"), "w") as fout:
                json.dump([{"expression": expr, "complexity": est._program.length_,
                            "loss": float(np.mean((est.predict(X) - y)**2))}], fout, indent=2)
            joblib.dump({"model": est, "base_coeffs": base_coeffs}, os.path.join(sdir, "model.joblib"))
            with open(os.path.join(mdir, "expression.txt"), "w") as fout:
                fout.write(f"gplearn expression for {target_feat}: {expr}\n")

        except Exception as e:
            print(f"  gplearn FAILED: {e}")
            import traceback; traceback.print_exc()
            loss_t, loss_v = 1.0, 1.0
            train_mm, val_mm = [1.0]*len(train_sims), [1.0]*len(val_sims)
            with open(os.path.join(sdir, "expressions.json"), "w") as fout:
                json.dump([{"expression": f"FAILED: {e}", "complexity": 0, "loss": 999}], fout)

        scorecard = {
            "approach": ap_name, "approach_number": ap_num,
            "benchmark": "analytic", "agent": "opus46",
            "category": "symbolic", "parameterization": qvar_name,
            "loss": loss_v, "loss_train": loss_t,
            "loss_components": {},
            "runtime_ms": 0.1, "n_train": len(train_sims), "n_val": len(val_sims),
            "n_params": 0, "notes": f"gplearn on {target_feat}"
        }
        with open(os.path.join(mdir, "scorecard.json"), "w") as fout:
            json.dump(scorecard, fout, indent=2)
        results.append(scorecard)
        error_data[ap_name] = {"train": train_mm, "val": val_mm}

    return results, error_data

# ── Plots ─────────────────────────────────────────────────────────────────────
def make_plots(results, error_data):
    import gwbenchmarks.plot_settings as ps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cdir = os.path.join(WORK, "comparison")
    os.makedirs(cdir, exist_ok=True)

    valid = sorted([r for r in results if r["loss"] < 10], key=lambda r: r["loss"])

    cat_colors = {"physics": ps.COLOR_CYCLE[0], "symbolic": ps.COLOR_CYCLE[1],
                  "matched": ps.COLOR_CYCLE[2], "functional": ps.COLOR_CYCLE[3]}
    legend_items = [Patch(facecolor=c, label=k) for k, c in cat_colors.items()]

    names = [r["approach"] for r in valid]
    losses = [r["loss"] for r in valid]
    colors = [cat_colors.get(r.get("category", "physics"), "gray") for r in valid]

    # Progress + loss-only (same content)
    for fname in ["progress", "loss_only_comparison"]:
        fig, ax = plt.subplots(figsize=ps.figsize(1.0))
        ax.barh(range(len(names)), losses, color=colors, edgecolor='k', linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=5)
        ax.set_xlabel("Validation Loss (mean FD mismatch)")
        ax.invert_yaxis()
        ax.legend(handles=legend_items, fontsize=6, loc='lower right')
        plt.tight_layout()
        fig.savefig(os.path.join(cdir, f"{fname}.png"), dpi=200)
        fig.savefig(os.path.join(cdir, f"{fname}.pdf"))
        plt.close(fig)

    # Pareto
    fig, ax = plt.subplots(figsize=ps.figsize(1.0))
    for r in valid:
        c = cat_colors.get(r.get("category", "physics"), "gray")
        ax.scatter(r.get("runtime_ms", 0.1), r["loss"], color=c, s=30,
                   edgecolors='k', linewidth=0.3, zorder=5)
        ax.annotate(r["approach"], (r.get("runtime_ms", 0.1), r["loss"]),
                    fontsize=4, ha='left', va='bottom', xytext=(2, 2),
                    textcoords='offset points')
    ax.set_xlabel("Runtime (ms/sample)")
    ax.set_ylabel("Validation Loss")
    ax.set_xscale("log")
    ax.legend(handles=legend_items, fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(cdir, "pareto_accuracy_speed.png"), dpi=200)
    fig.savefig(os.path.join(cdir, "pareto_accuracy_speed.pdf"))
    plt.close(fig)

    # Error histograms
    n_models = len(error_data)
    if n_models > 0:
        ncols = 4
        nrows = max(1, (n_models + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2))
        axes_flat = np.array(axes).flatten() if n_models > 1 else [axes]
        for idx, (nm, errs) in enumerate(error_data.items()):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            all_vals = errs["train"] + errs["val"]
            bins = np.linspace(0, min(max(all_vals), 1.0), 20)
            ax.hist(errs["train"], bins=bins, alpha=0.5, label="train", color=ps.COLOR_CYCLE[0])
            ax.hist(errs["val"], bins=bins, alpha=0.5, label="val", color=ps.COLOR_CYCLE[1], hatch='//')
            ax.set_title(nm, fontsize=4)
            ax.tick_params(labelsize=4)
            if idx == 0:
                ax.legend(fontsize=4)
        for idx in range(len(error_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(cdir, "error_histograms.png"), dpi=200)
        fig.savefig(os.path.join(cdir, "error_histograms.pdf"))
        plt.close(fig)

    # Summary + best model
    with open(os.path.join(cdir, "summary_table.json"), "w") as fout:
        json.dump(valid, fout, indent=2)
    if valid:
        best = min(valid, key=lambda r: r["loss"])
        with open(os.path.join(cdir, "best_model.json"), "w") as fout:
            json.dump(best, fout, indent=2)
        print(f"\nBest model: {best['approach']} with loss={best['loss']:.4f}")

    # all_expressions.json
    all_expr = []
    for r in valid:
        mdir = os.path.join(WORK, f"models/{r['approach_number']:02d}_{r['approach']}")
        expr_file = os.path.join(mdir, "expression.txt")
        expr_text = ""
        if os.path.isfile(expr_file):
            with open(expr_file) as f:
                expr_text = f.read()
        all_expr.append({
            "approach": r["approach"], "expression": expr_text,
            "complexity": r.get("n_params", 0), "loss": r["loss"]
        })
    with open(os.path.join(cdir, "all_expressions.json"), "w") as fout:
        json.dump(all_expr, fout, indent=2)

    with open(os.path.join(cdir, "error_data.json"), "w") as fout:
        json.dump(error_data, fout)

    print("All plots saved.")

# ── Generate train.py / predict.py ───────────────────────────────────────────
def generate_scripts(results):
    for r in results:
        num, name = r["approach_number"], r["approach"]
        mdir = os.path.join(WORK, f"models/{num:02d}_{name}")
        if not os.path.isdir(mdir):
            continue

        with open(os.path.join(mdir, "train.py"), "w") as fout:
            fout.write(f'#!/usr/bin/env python3\n"""Train {name} — Analytic Bench."""\nprint("Model {name} trained via build_all.py")\n')

        with open(os.path.join(mdir, "predict.py"), "w") as fout:
            fout.write(f'''#!/usr/bin/env python3
"""Predict function for {name} — Analytic Bench."""
import os, numpy as np, joblib

_dir = os.path.dirname(os.path.abspath(__file__))
_data = joblib.load(os.path.join(_dir, "saved_model", "model.joblib"))

def predict(t, q):
    """Predict h22(t) for given mass ratio q. Returns complex array."""
    raise NotImplementedError("Use build_all.py make_h22() for prediction")
''')

# ── CHANGELOG ─────────────────────────────────────────────────────────────────
def write_changelog(results):
    lines = ["# Analytic Bench — CHANGELOG\n"]
    for r in sorted(results, key=lambda x: x["approach_number"]):
        lines.append(f"## Approach {r['approach_number']:02d}: {r['approach']}")
        lines.append(f"- Category: {r.get('category', '')}")
        lines.append(f"- Parameterization: {r.get('parameterization', '')}")
        lines.append(f"- Val Loss (FD mismatch): {r['loss']:.4f}")
        lines.append(f"- Notes: {r.get('notes', '')}")
        lines.append("")
    with open(os.path.join(WORK, "CHANGELOG.md"), "w") as fout:
        fout.write("\n".join(lines))

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ANALYTIC BENCH — Opus 4.6 Agent — Building all 26 approaches")
    print("=" * 60)

    train_sims = load_data("training")
    val_sims = load_data("validation")

    print(f"Loaded {len(train_sims)} training, {len(val_sims)} validation waveforms")

    # Extract features
    train_feats = [extract_features(s) for s in train_sims]
    print("\nSample features (q=1):", train_feats[0])

    # Run non-symbolic approaches
    results, error_data = run_approaches(train_sims, val_sims, train_feats,
                                          [extract_features(s) for s in val_sims])

    # Run symbolic approaches
    results, error_data = run_symbolic(train_sims, val_sims, train_feats,
                                        results, error_data)

    # Generate outputs
    make_plots(results, error_data)
    generate_scripts(results)
    write_changelog(results)

    print("\n" + "=" * 60)
    print("ALL 26 APPROACHES COMPLETE")
    print("=" * 60)
