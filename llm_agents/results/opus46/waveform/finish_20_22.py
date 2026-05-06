#!/usr/bin/env python3
"""Complete approaches 20-22 (fix RBF epsilon issue)."""

import sys, os, numpy as np, json, time, warnings, joblib
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, WORK_DIR)

from utils import (load_dataset, compute_svd, project_onto_basis,
                   reconstruct_from_basis, reparameterize,
                   compute_loss_batch, save_scorecard, DT)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from scipy.interpolate import RBFInterpolator

print("Loading data...")
params_train, wf_train, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_training.h5"))
params_val, wf_val, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_validation.h5"))
N_TRAIN, N_VAL = len(params_train), len(params_val)
N_BASIS = 40

wf_r_tr = np.real(wf_train); wf_i_tr = np.imag(wf_train)
cr, br, mr, _ = compute_svd(wf_r_tr, N_BASIS)
ci, bi, mi, _ = compute_svd(wf_i_tr, N_BASIS)
cr_v = project_onto_basis(np.real(wf_val), br, mr)
ci_v = project_onto_basis(np.imag(wf_val), bi, mi)
y_train = np.hstack([cr, ci])
y_val = np.hstack([cr_v, ci_v])

ps = {k: reparameterize(params_train, k) for k in ["raw","eta_chieff","spherical","mass_diff"]}
psv = {k: reparameterize(params_val, k) for k in ["raw","eta_chieff","spherical","mass_diff"]}

def make_dir(n, name):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    os.makedirs(os.path.join(d, "saved_model"), exist_ok=True)
    return d

def eval_svd(yp, wf_ref):
    pr = yp[:, :N_BASIS]; pi = yp[:, N_BASIS:2*N_BASIS] if yp.shape[1]>=2*N_BASIS else np.zeros((len(yp),N_BASIS))
    wf = reconstruct_from_basis(pr, br, mr) + 1j * reconstruct_from_basis(pi, bi, mi)
    losses, ml = compute_loss_batch(wf, wf_ref, DT)
    return ml, losses

# Load existing results
ed_path = os.path.join(WORK_DIR, "comparison/error_data.json")
with open(ed_path) as f:
    error_data = json.load(f)

st_path = os.path.join(WORK_DIR, "comparison/summary_table.json")
with open(st_path) as f:
    all_results = json.load(f)

cl_path = os.path.join(WORK_DIR, "CHANGELOG.md")
with open(cl_path) as f:
    changelog = f.read()

# 20. SVD + RBF (eta)
print("\n=== 20: SVD+RBF eta ===")
md = make_dir(20, "svd_rbf_interp_eta")
X, Xv = ps["eta_chieff"], psv["eta_chieff"]
sX = StandardScaler().fit(X); Xs, Xvs = sX.transform(X), sX.transform(Xv)
sy = StandardScaler().fit(y_train); ys = sy.transform(y_train)
rbf = RBFInterpolator(Xs, ys, kernel='thin_plate_spline', smoothing=0.1)
t0 = time.time()
yp = sy.inverse_transform(rbf(Xvs))
rt = (time.time()-t0)/N_VAL*1000
loss, lv = eval_svd(yp, wf_val)
_, lt = eval_svd(sy.inverse_transform(rbf(Xs)), wf_train)
joblib.dump({"rbf": rbf, "scaler_X": sX, "scaler_y": sy}, os.path.join(md, "saved_model/model.joblib"))
save_scorecard(md, "svd_rbf_interp_eta", 20, "eta_chieff", "t0_at_peak", float(loss),
               {"mean_mismatch": float(loss)}, rt, N_TRAIN, N_VAL, 250*80, "SVD(40)+RBF TPS eta")
error_data["svd_rbf_interp_eta"] = {"val_losses": [float(x) for x in lv], "train_losses": [float(x) for x in lt]}
all_results.append({"approach": "svd_rbf_interp_eta", "approach_number": 20, "loss": float(loss),
                     "runtime_ms": rt, "parameterization": "eta_chieff"})
changelog += "\n## Approach 20: svd_rbf_interp_eta\n- Param: eta_chieff\n- Loss: {:.6f}\n".format(loss)
print(f"  [20] svd_rbf_interp_eta: loss={loss:.6f}")

# 21. SVD + BayRidge (mass_diff)
print("\n=== 21: SVD+BayRidge mdiff ===")
md = make_dir(21, "svd_bayridge_mdiff")
X21, Xv21 = ps["mass_diff"], psv["mass_diff"]
sX21 = StandardScaler().fit(X21); X21s, Xv21s = sX21.transform(X21), sX21.transform(Xv21)
poly21 = PolynomialFeatures(degree=3); X21p = poly21.fit_transform(X21s); Xv21p = poly21.transform(Xv21s)
sy21 = StandardScaler().fit(y_train); yt21s = sy21.transform(y_train)
br21 = MultiOutputRegressor(BayesianRidge(max_iter=500), n_jobs=-1)
br21.fit(X21p, yt21s)
t0 = time.time()
yp21 = sy21.inverse_transform(br21.predict(Xv21p))
rt21 = (time.time()-t0)/N_VAL*1000
loss21, lv21 = eval_svd(yp21, wf_val)
_, lt21 = eval_svd(sy21.inverse_transform(br21.predict(X21p)), wf_train)
joblib.dump({"br": br21, "poly": poly21, "scaler_X": sX21, "scaler_y": sy21}, os.path.join(md, "saved_model/model.joblib"))
save_scorecard(md, "svd_bayridge_mdiff", 21, "mass_diff", "t0_at_peak", float(loss21),
               {"mean_mismatch": float(loss21)}, rt21, N_TRAIN, N_VAL, X21p.shape[1]*80, "SVD(40)+BayRidge poly-3 mass_diff")
error_data["svd_bayridge_mdiff"] = {"val_losses": [float(x) for x in lv21], "train_losses": [float(x) for x in lt21]}
all_results.append({"approach": "svd_bayridge_mdiff", "approach_number": 21, "loss": float(loss21),
                     "runtime_ms": rt21, "parameterization": "mass_diff"})
changelog += "\n## Approach 21: svd_bayridge_mdiff\n- Param: mass_diff\n- Loss: {:.6f}\n".format(loss21)
print(f"  [21] svd_bayridge_mdiff: loss={loss21:.6f}")

# 22. SVD + ElasticNet (eta)
print("\n=== 22: SVD+ElasticNet eta ===")
md = make_dir(22, "svd_elasticnet_eta")
X22, Xv22 = ps["eta_chieff"], psv["eta_chieff"]
sX22 = StandardScaler().fit(X22); X22s, Xv22s = sX22.transform(X22), sX22.transform(Xv22)
poly22 = PolynomialFeatures(degree=3); X22p = poly22.fit_transform(X22s); Xv22p = poly22.transform(Xv22s)
sy22 = StandardScaler().fit(y_train); yt22s = sy22.transform(y_train)
enet = MultiOutputRegressor(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000), n_jobs=-1)
enet.fit(X22p, yt22s)
t0 = time.time()
yp22 = sy22.inverse_transform(enet.predict(Xv22p))
rt22 = (time.time()-t0)/N_VAL*1000
loss22, lv22 = eval_svd(yp22, wf_val)
_, lt22 = eval_svd(sy22.inverse_transform(enet.predict(X22p)), wf_train)
joblib.dump({"enet": enet, "poly": poly22, "scaler_X": sX22, "scaler_y": sy22}, os.path.join(md, "saved_model/model.joblib"))
save_scorecard(md, "svd_elasticnet_eta", 22, "eta_chieff", "t0_at_peak", float(loss22),
               {"mean_mismatch": float(loss22)}, rt22, N_TRAIN, N_VAL, X22p.shape[1]*80, "SVD(40)+ElasticNet poly-3 eta")
error_data["svd_elasticnet_eta"] = {"val_losses": [float(x) for x in lv22], "train_losses": [float(x) for x in lt22]}
all_results.append({"approach": "svd_elasticnet_eta", "approach_number": 22, "loss": float(loss22),
                     "runtime_ms": rt22, "parameterization": "eta_chieff"})
changelog += "\n## Approach 22: svd_elasticnet_eta\n- Param: eta_chieff\n- Loss: {:.6f}\n".format(loss22)
print(f"  [22] svd_elasticnet_eta: loss={loss22:.6f}")

# Save
with open(ed_path, "w") as f:
    json.dump(error_data, f)
all_results.sort(key=lambda x: x["loss"])
with open(st_path, "w") as f:
    json.dump(all_results, f, indent=2)
best = min(all_results, key=lambda x: x["loss"])
with open(os.path.join(WORK_DIR, "comparison/best_model.json"), "w") as f:
    json.dump(best, f, indent=2)
with open(cl_path, "w") as f:
    f.write(changelog)

print(f"\nAll 22 non-symbolic approaches complete. Best: {best['approach']} ({best['loss']:.6f})")
