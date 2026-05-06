#!/usr/bin/env python3
"""Run PySR and gplearn on SVD coefficients — streamlined version."""

import sys, os, numpy as np, json, time, warnings, joblib
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, WORK_DIR)

from utils import (load_dataset, compute_svd, project_onto_basis,
                   reconstruct_from_basis, reparameterize,
                   compute_loss_batch, save_scorecard, DT)
from sklearn.preprocessing import StandardScaler

print("Loading data...")
params_train, wf_train, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_training.h5"))
params_val, wf_val, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_validation.h5"))

N_BASIS = 40
cr, br, mr, _ = compute_svd(np.real(wf_train), N_BASIS)
ci, bi, mi, _ = compute_svd(np.imag(wf_train), N_BASIS)
cr_v = project_onto_basis(np.real(wf_val), br, mr)
ci_v = project_onto_basis(np.imag(wf_val), bi, mi)
Y = np.hstack([cr, ci])

ps = {k: reparameterize(params_train, k) for k in ["raw", "eta_chieff"]}
pv = {k: reparameterize(params_val, k) for k in ["raw", "eta_chieff"]}

N_COEFF = 5

# Load existing results
err_path = os.path.join(WORK_DIR, "comparison/error_data.json")
with open(err_path) as f:
    err_data = json.load(f)
st_path = os.path.join(WORK_DIR, "comparison/summary_table.json")
with open(st_path) as f:
    all_results = json.load(f)
cl_path = os.path.join(WORK_DIR, "CHANGELOG.md")
with open(cl_path) as f:
    changelog = f.read()

def md(n, name):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    os.makedirs(os.path.join(d, "saved_model"), exist_ok=True)
    return d

def ev(yp, ref):
    if yp.shape[1] < 2*N_BASIS:
        f = np.zeros((len(yp), 2*N_BASIS)); f[:, :yp.shape[1]] = yp; yp = f
    wf = reconstruct_from_basis(yp[:,:N_BASIS], br, mr) + 1j*reconstruct_from_basis(yp[:,N_BASIS:], bi, mi)
    l, m = compute_loss_batch(wf, ref, DT)
    return m, l

def rec(n, name, sch, loss, lv, lt, rt, np_, notes):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    save_scorecard(d, name, n, sch, "t0_at_peak", float(loss),
                   {"mean_mismatch": float(loss)}, rt, 250, 250, np_, notes)
    all_results.append({"approach":name,"approach_number":n,"loss":float(loss),"runtime_ms":rt,"parameterization":sch,"notes":notes})
    err_data[name] = {"val_losses":[float(x) for x in lv], "train_losses":[float(x) for x in lt]}
    global changelog
    changelog += f"\n## {n}: {name}\n- Param: {sch}, Loss: {loss:.6f}\n- {notes}\n"
    print(f"  [{n:02d}] {name}: loss={loss:.6f}")


# ═══════ 23: PySR (raw) ═══════
print("\n=== 23: PySR (raw) ===")
d = md(23, "pysr_raw")
from pysr import PySRRegressor

X_raw = ps["raw"]
sX = StandardScaler().fit(X_raw)
Xs = sX.transform(X_raw); Xvs = sX.transform(pv["raw"])

all_expr = []
pred_tr = np.zeros((250, N_COEFF))
pred_va = np.zeros((250, N_COEFF))

for k in range(N_COEFF):
    print(f"  PySR coeff {k}...")
    m = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "exp", "sin", "cos"],
        maxsize=20,
        populations=10,
        procs=1,
        loss="loss(prediction, target) = abs(prediction - target)",
        verbosity=0, progress=False, random_state=42,
        temp_equation_file=True,
    )
    m.fit(Xs, Y[:, k])
    pred_tr[:, k] = m.predict(Xs)
    pred_va[:, k] = m.predict(Xvs)
    try:
        expr = str(m.sympy())
        eqs = []
        for idx in range(len(m.equations_)):
            row = m.equations_.iloc[idx]
            eqs.append({"expression": str(row.get("equation","")),
                        "complexity": int(row.get("complexity",0)),
                        "loss": float(row.get("loss",0))})
        all_expr.append({"coefficient": k, "best": expr, "pareto_front": eqs})
    except Exception as e:
        all_expr.append({"coefficient": k, "best": f"error: {e}", "pareto_front": []})
    joblib.dump(m, os.path.join(d, f"saved_model/pysr_coeff_{k}.joblib"))

with open(os.path.join(d, "saved_model/expressions.json"), "w") as f:
    json.dump(all_expr, f, indent=2, default=str)

yf_va = np.zeros((250, 2*N_BASIS)); yf_va[:, :N_COEFF] = pred_va
yf_tr = np.zeros((250, 2*N_BASIS)); yf_tr[:, :N_COEFF] = pred_tr
t0 = time.time()
loss23, lv23 = ev(yf_va, wf_val)
rt23 = (time.time()-t0)/250*1000
_, lt23 = ev(yf_tr, wf_train)
rec(23, "pysr_raw", "raw", loss23, lv23, lt23, rt23, N_COEFF*20, "PySR on top-5 SVD coefficients, raw params")


# ═══════ 24: PySR (eta) ═══════
print("\n=== 24: PySR (eta) ===")
d = md(24, "pysr_eta")

X_eta = ps["eta_chieff"]
sX2 = StandardScaler().fit(X_eta)
X2s = sX2.transform(X_eta); Xv2s = sX2.transform(pv["eta_chieff"])

all_expr2 = []
pred2_tr = np.zeros((250, N_COEFF))
pred2_va = np.zeros((250, N_COEFF))

for k in range(N_COEFF):
    print(f"  PySR(eta) coeff {k}...")
    m2 = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "exp", "sin", "cos"],
        maxsize=20,
        populations=10,
        procs=1,
        loss="loss(prediction, target) = abs(prediction - target)",
        verbosity=0, progress=False, random_state=42,
        temp_equation_file=True,
    )
    m2.fit(X2s, Y[:, k])
    pred2_tr[:, k] = m2.predict(X2s)
    pred2_va[:, k] = m2.predict(Xv2s)
    try:
        expr = str(m2.sympy())
        eqs = []
        for idx in range(len(m2.equations_)):
            row = m2.equations_.iloc[idx]
            eqs.append({"expression": str(row.get("equation","")),
                        "complexity": int(row.get("complexity",0)),
                        "loss": float(row.get("loss",0))})
        all_expr2.append({"coefficient": k, "best": expr, "pareto_front": eqs})
    except Exception as e:
        all_expr2.append({"coefficient": k, "best": f"error: {e}", "pareto_front": []})
    joblib.dump(m2, os.path.join(d, f"saved_model/pysr_coeff_{k}.joblib"))

with open(os.path.join(d, "saved_model/expressions.json"), "w") as f:
    json.dump(all_expr2, f, indent=2, default=str)

yf2_va = np.zeros((250, 2*N_BASIS)); yf2_va[:, :N_COEFF] = pred2_va
yf2_tr = np.zeros((250, 2*N_BASIS)); yf2_tr[:, :N_COEFF] = pred2_tr
loss24, lv24 = ev(yf2_va, wf_val)
_, lt24 = ev(yf2_tr, wf_train)
rec(24, "pysr_eta", "eta_chieff", loss24, lv24, lt24, rt23, N_COEFF*20, "PySR on top-5 SVD coefficients, eta+chieff")


# ═══════ 25: gplearn (raw) ═══════
print("\n=== 25: gplearn (raw) ===")
d = md(25, "gplearn_raw")
from gplearn.genetic import SymbolicRegressor

gp_expr = []
gp_tr = np.zeros((250, N_COEFF))
gp_va = np.zeros((250, N_COEFF))

for k in range(N_COEFF):
    print(f"  gplearn coeff {k}...")
    est = SymbolicRegressor(
        population_size=2000, generations=30, tournament_size=20,
        function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
        metric='mse', parsimony_coefficient=0.001,
        max_samples=1.0, verbose=0, random_state=42,
    )
    est.fit(Xs, Y[:, k])
    gp_tr[:, k] = est.predict(Xs)
    gp_va[:, k] = est.predict(Xvs)
    gp_expr.append({"coefficient": k, "expression": str(est._program),
                    "complexity": est._program.length_, "fitness": float(est._program.fitness_)})
    joblib.dump(est, os.path.join(d, f"saved_model/gplearn_coeff_{k}.joblib"))

with open(os.path.join(d, "saved_model/expressions.json"), "w") as f:
    json.dump(gp_expr, f, indent=2, default=str)

ygp_va = np.zeros((250, 2*N_BASIS)); ygp_va[:, :N_COEFF] = gp_va
ygp_tr = np.zeros((250, 2*N_BASIS)); ygp_tr[:, :N_COEFF] = gp_tr
t0 = time.time()
loss25, lv25 = ev(ygp_va, wf_val)
rt25 = (time.time()-t0)/250*1000
_, lt25 = ev(ygp_tr, wf_train)
rec(25, "gplearn_raw", "raw", loss25, lv25, lt25, rt25, N_COEFF*30, "gplearn on top-5 SVD coefficients, raw params")


# ═══════ 26: gplearn (eta) ═══════
print("\n=== 26: gplearn (eta) ===")
d = md(26, "gplearn_eta")

gp_expr2 = []
gp2_tr = np.zeros((250, N_COEFF))
gp2_va = np.zeros((250, N_COEFF))

for k in range(N_COEFF):
    print(f"  gplearn(eta) coeff {k}...")
    est2 = SymbolicRegressor(
        population_size=2000, generations=30, tournament_size=20,
        function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
        metric='mse', parsimony_coefficient=0.001,
        max_samples=1.0, verbose=0, random_state=42,
    )
    est2.fit(X2s, Y[:, k])
    gp2_tr[:, k] = est2.predict(X2s)
    gp2_va[:, k] = est2.predict(Xv2s)
    gp_expr2.append({"coefficient": k, "expression": str(est2._program),
                     "complexity": est2._program.length_, "fitness": float(est2._program.fitness_)})
    joblib.dump(est2, os.path.join(d, f"saved_model/gplearn_coeff_{k}.joblib"))

with open(os.path.join(d, "saved_model/expressions.json"), "w") as f:
    json.dump(gp_expr2, f, indent=2, default=str)

ygp2_va = np.zeros((250, 2*N_BASIS)); ygp2_va[:, :N_COEFF] = gp2_va
ygp2_tr = np.zeros((250, 2*N_BASIS)); ygp2_tr[:, :N_COEFF] = gp2_tr
loss26, lv26 = ev(ygp2_va, wf_val)
_, lt26 = ev(ygp2_tr, wf_train)
rec(26, "gplearn_eta", "eta_chieff", loss26, lv26, lt26, rt25, N_COEFF*30, "gplearn on top-5 SVD coefficients, eta+chieff")


# Save all
all_results.sort(key=lambda x: x["loss"])
with open(err_path, "w") as f:
    json.dump(err_data, f)
with open(st_path, "w") as f:
    json.dump(all_results, f, indent=2)
with open(os.path.join(WORK_DIR, "comparison/best_model.json"), "w") as f:
    json.dump(all_results[0], f, indent=2)
with open(cl_path, "w") as f:
    f.write(changelog)

print(f"\n=== ALL SYMBOLIC DONE ===")
print(f"PySR (raw): {loss23:.6f}")
print(f"PySR (eta): {loss24:.6f}")
print(f"gplearn (raw): {loss25:.6f}")
print(f"gplearn (eta): {loss26:.6f}")
print(f"Total approaches: {len(all_results)}")
