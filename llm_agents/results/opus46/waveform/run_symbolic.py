#!/usr/bin/env python3
"""Run PySR and gplearn symbolic regression approaches."""

import sys, os
import numpy as np
import json
import time
import warnings
import joblib
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, WORK_DIR)

from utils import (load_dataset, compute_svd, project_onto_basis,
                   reconstruct_from_basis, reparameterize,
                   compute_loss_batch, save_scorecard, DT)
from gwbenchmarks.metrics import mean_fd_mismatch

print("Loading data...")
params_train, wf_train, meta_train = load_dataset(
    os.path.join(ROOT, "datasets/waveform/waveform_training.h5"))
params_val, wf_val, meta_val = load_dataset(
    os.path.join(ROOT, "datasets/waveform/waveform_validation.h5"))

N_BASIS = 40
wf_real_train = np.real(wf_train)
wf_imag_train = np.imag(wf_train)
coeffs_r, basis_r, mean_r, sv_r = compute_svd(wf_real_train, N_BASIS)
coeffs_i, basis_i, mean_i, sv_i = compute_svd(wf_imag_train, N_BASIS)

wf_real_val = np.real(wf_val)
wf_imag_val = np.imag(wf_val)
coeffs_r_val = project_onto_basis(wf_real_val, basis_r, mean_r)
coeffs_i_val = project_onto_basis(wf_imag_val, basis_i, mean_i)

y_train_all = np.hstack([coeffs_r, coeffs_i])
y_val_all = np.hstack([coeffs_r_val, coeffs_i_val])

param_schemes = {
    "raw": reparameterize(params_train, "raw"),
    "eta_chieff": reparameterize(params_train, "eta_chieff"),
}
param_schemes_val = {
    "raw": reparameterize(params_val, "raw"),
    "eta_chieff": reparameterize(params_val, "eta_chieff"),
}

from sklearn.preprocessing import StandardScaler

def evaluate_svd_predictions(y_pred, wf_ref):
    pred_r = y_pred[:, :N_BASIS]
    pred_i = y_pred[:, N_BASIS:]
    wf_pred_real = reconstruct_from_basis(pred_r, basis_r, mean_r)
    wf_pred_imag = reconstruct_from_basis(pred_i, basis_i, mean_i)
    wf_pred = wf_pred_real + 1j * wf_pred_imag
    losses, mean_loss = compute_loss_batch(wf_pred, wf_ref, DT)
    return mean_loss, losses

def make_model_dir(num, name):
    d = os.path.join(WORK_DIR, f"models/{num:02d}_{name}")
    os.makedirs(os.path.join(d, "saved_model"), exist_ok=True)
    return d


# ═══════════════════════════════════════════════════════════════
# APPROACH 23: PySR on SVD coefficients (raw params)
# ═══════════════════════════════════════════════════════════════
print("\n=== Approach 23: PySR (raw params) ===")
model_dir = make_model_dir(23, "pysr_raw")

from pysr import PySRRegressor

X_tr = param_schemes["raw"]
scaler_X = StandardScaler().fit(X_tr)
X_tr_s = scaler_X.transform(X_tr)
X_val_s = scaler_X.transform(param_schemes_val["raw"])

n_pysr_basis = 5
y_pysr = y_train_all[:, :n_pysr_basis]

all_expressions = []
pysr_predictions_train = np.zeros((len(X_tr), n_pysr_basis))
pysr_predictions_val = np.zeros((len(X_val_s), n_pysr_basis))

for k in range(n_pysr_basis):
    print(f"  PySR fitting coefficient {k}...")
    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "exp", "sin", "cos"],
        maxsize=25,
        populations=20,
        procs=4,
        loss="loss(prediction, target) = abs(prediction - target)",
        verbosity=0,
        progress=False,
        random_state=42,
        temp_equation_file=True,
    )
    model.fit(X_tr_s, y_pysr[:, k])

    pysr_predictions_train[:, k] = model.predict(X_tr_s)
    pysr_predictions_val[:, k] = model.predict(X_val_s)

    try:
        best_expr = str(model.sympy())
        equations = []
        for idx in range(len(model.equations_)):
            eq_row = model.equations_.iloc[idx]
            equations.append({
                "expression": str(eq_row.get("equation", "")),
                "complexity": int(eq_row.get("complexity", 0)),
                "loss": float(eq_row.get("loss", 0)),
            })
        all_expressions.append({"coefficient": k, "best": best_expr, "pareto_front": equations})
    except Exception as e:
        all_expressions.append({"coefficient": k, "best": f"error: {e}", "pareto_front": []})

    joblib.dump(model, os.path.join(model_dir, f"saved_model/pysr_coeff_{k}.joblib"))

with open(os.path.join(model_dir, "saved_model/expressions.json"), "w") as f:
    json.dump(all_expressions, f, indent=2, default=str)

y_full_train = np.zeros_like(y_train_all)
y_full_val = np.zeros_like(y_val_all)
y_full_train[:, :n_pysr_basis] = pysr_predictions_train
y_full_val[:, :n_pysr_basis] = pysr_predictions_val

t0 = time.time()
loss23_val, losses23_val = evaluate_svd_predictions(y_full_val, wf_val)
rt23 = (time.time() - t0) / len(params_val) * 1000
loss23_train, losses23_train = evaluate_svd_predictions(y_full_train, wf_train)

save_scorecard(model_dir, "pysr_raw", 23, "raw", "t0_at_peak",
               float(loss23_val), {"mean_mismatch": float(loss23_val)},
               rt23, 250, 250, n_pysr_basis * 25, "PySR on top-5 SVD coefficients, raw params")
print(f"  PySR (raw): loss={loss23_val:.6f}")


# ═══════════════════════════════════════════════════════════════
# APPROACH 24: PySR on SVD coefficients (eta_chieff params)
# ═══════════════════════════════════════════════════════════════
print("\n=== Approach 24: PySR (eta_chieff) ===")
model_dir = make_model_dir(24, "pysr_eta")

X_tr2 = param_schemes["eta_chieff"]
scaler_X2 = StandardScaler().fit(X_tr2)
X_tr2_s = scaler_X2.transform(X_tr2)
X_val2_s = scaler_X2.transform(param_schemes_val["eta_chieff"])

all_expressions2 = []
pysr_pred2_train = np.zeros((len(X_tr2), n_pysr_basis))
pysr_pred2_val = np.zeros((len(X_val2_s), n_pysr_basis))

for k in range(n_pysr_basis):
    print(f"  PySR (eta) fitting coefficient {k}...")
    model2 = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "exp", "sin", "cos"],
        maxsize=25,
        populations=20,
        procs=4,
        loss="loss(prediction, target) = abs(prediction - target)",
        verbosity=0,
        progress=False,
        random_state=42,
        temp_equation_file=True,
    )
    model2.fit(X_tr2_s, y_pysr[:, k])

    pysr_pred2_train[:, k] = model2.predict(X_tr2_s)
    pysr_pred2_val[:, k] = model2.predict(X_val2_s)

    try:
        best_expr = str(model2.sympy())
        equations = []
        for idx in range(len(model2.equations_)):
            eq_row = model2.equations_.iloc[idx]
            equations.append({
                "expression": str(eq_row.get("equation", "")),
                "complexity": int(eq_row.get("complexity", 0)),
                "loss": float(eq_row.get("loss", 0)),
            })
        all_expressions2.append({"coefficient": k, "best": best_expr, "pareto_front": equations})
    except Exception as e:
        all_expressions2.append({"coefficient": k, "best": f"error: {e}", "pareto_front": []})

    joblib.dump(model2, os.path.join(model_dir, f"saved_model/pysr_coeff_{k}.joblib"))

with open(os.path.join(model_dir, "saved_model/expressions.json"), "w") as f:
    json.dump(all_expressions2, f, indent=2, default=str)

y_full2_train = np.zeros_like(y_train_all)
y_full2_val = np.zeros_like(y_val_all)
y_full2_train[:, :n_pysr_basis] = pysr_pred2_train
y_full2_val[:, :n_pysr_basis] = pysr_pred2_val

loss24_val, losses24_val = evaluate_svd_predictions(y_full2_val, wf_val)
loss24_train, losses24_train = evaluate_svd_predictions(y_full2_train, wf_train)

save_scorecard(model_dir, "pysr_eta", 24, "eta_chieff", "t0_at_peak",
               float(loss24_val), {"mean_mismatch": float(loss24_val)},
               rt23, 250, 250, n_pysr_basis * 25, "PySR on top-5 SVD coefficients, eta+chieff params")
print(f"  PySR (eta): loss={loss24_val:.6f}")


# ═══════════════════════════════════════════════════════════════
# APPROACH 25: gplearn on SVD coefficients (raw params)
# ═══════════════════════════════════════════════════════════════
print("\n=== Approach 25: gplearn (raw) ===")
model_dir = make_model_dir(25, "gplearn_raw")

from gplearn.genetic import SymbolicRegressor

X_tr_gp = param_schemes["raw"]
scaler_Xgp = StandardScaler().fit(X_tr_gp)
X_tr_gp_s = scaler_Xgp.transform(X_tr_gp)
X_val_gp_s = scaler_Xgp.transform(param_schemes_val["raw"])

gp_expressions = []
gp_pred_train = np.zeros((len(X_tr_gp), n_pysr_basis))
gp_pred_val = np.zeros((len(X_val_gp_s), n_pysr_basis))

for k in range(n_pysr_basis):
    print(f"  gplearn fitting coefficient {k}...")
    est = SymbolicRegressor(
        population_size=2000,
        generations=30,
        tournament_size=20,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
        metric='mse',
        parsimony_coefficient=0.001,
        max_samples=1.0,
        verbose=0,
        random_state=42,
    )
    est.fit(X_tr_gp_s, y_pysr[:, k])

    gp_pred_train[:, k] = est.predict(X_tr_gp_s)
    gp_pred_val[:, k] = est.predict(X_val_gp_s)

    gp_expressions.append({
        "coefficient": k,
        "expression": str(est._program),
        "complexity": est._program.length_,
        "fitness": float(est._program.fitness_),
    })
    joblib.dump(est, os.path.join(model_dir, f"saved_model/gplearn_coeff_{k}.joblib"))

with open(os.path.join(model_dir, "saved_model/expressions.json"), "w") as f:
    json.dump(gp_expressions, f, indent=2, default=str)

y_gp_train = np.zeros_like(y_train_all)
y_gp_val = np.zeros_like(y_val_all)
y_gp_train[:, :n_pysr_basis] = gp_pred_train
y_gp_val[:, :n_pysr_basis] = gp_pred_val

t0 = time.time()
loss25_val, losses25_val = evaluate_svd_predictions(y_gp_val, wf_val)
rt25 = (time.time() - t0) / len(params_val) * 1000
loss25_train, losses25_train = evaluate_svd_predictions(y_gp_train, wf_train)

save_scorecard(model_dir, "gplearn_raw", 25, "raw", "t0_at_peak",
               float(loss25_val), {"mean_mismatch": float(loss25_val)},
               rt25, 250, 250, n_pysr_basis * 30, "gplearn on top-5 SVD coefficients, raw params")
print(f"  gplearn (raw): loss={loss25_val:.6f}")


# ═══════════════════════════════════════════════════════════════
# APPROACH 26: gplearn on SVD coefficients (eta_chieff)
# ═══════════════════════════════════════════════════════════════
print("\n=== Approach 26: gplearn (eta) ===")
model_dir = make_model_dir(26, "gplearn_eta")

X_tr_gp2 = param_schemes["eta_chieff"]
scaler_Xgp2 = StandardScaler().fit(X_tr_gp2)
X_tr_gp2_s = scaler_Xgp2.transform(X_tr_gp2)
X_val_gp2_s = scaler_Xgp2.transform(param_schemes_val["eta_chieff"])

gp_expressions2 = []
gp_pred2_train = np.zeros((len(X_tr_gp2), n_pysr_basis))
gp_pred2_val = np.zeros((len(X_val_gp2_s), n_pysr_basis))

for k in range(n_pysr_basis):
    print(f"  gplearn (eta) fitting coefficient {k}...")
    est2 = SymbolicRegressor(
        population_size=2000,
        generations=30,
        tournament_size=20,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
        metric='mse',
        parsimony_coefficient=0.001,
        max_samples=1.0,
        verbose=0,
        random_state=42,
    )
    est2.fit(X_tr_gp2_s, y_pysr[:, k])

    gp_pred2_train[:, k] = est2.predict(X_tr_gp2_s)
    gp_pred2_val[:, k] = est2.predict(X_val_gp2_s)

    gp_expressions2.append({
        "coefficient": k,
        "expression": str(est2._program),
        "complexity": est2._program.length_,
        "fitness": float(est2._program.fitness_),
    })
    joblib.dump(est2, os.path.join(model_dir, f"saved_model/gplearn_coeff_{k}.joblib"))

with open(os.path.join(model_dir, "saved_model/expressions.json"), "w") as f:
    json.dump(gp_expressions2, f, indent=2, default=str)

y_gp2_train = np.zeros_like(y_train_all)
y_gp2_val = np.zeros_like(y_val_all)
y_gp2_train[:, :n_pysr_basis] = gp_pred2_train
y_gp2_val[:, :n_pysr_basis] = gp_pred2_val

loss26_val, losses26_val = evaluate_svd_predictions(y_gp2_val, wf_val)
loss26_train, losses26_train = evaluate_svd_predictions(y_gp2_train, wf_train)

save_scorecard(model_dir, "gplearn_eta", 26, "eta_chieff", "t0_at_peak",
               float(loss26_val), {"mean_mismatch": float(loss26_val)},
               rt25, 250, 250, n_pysr_basis * 30, "gplearn on top-5 SVD coefficients, eta+chieff params")
print(f"  gplearn (eta): loss={loss26_val:.6f}")


print("\n=== All symbolic approaches done ===")
print(f"PySR (raw): {loss23_val:.6f}")
print(f"PySR (eta): {loss24_val:.6f}")
print(f"gplearn (raw): {loss25_val:.6f}")
print(f"gplearn (eta): {loss26_val:.6f}")
