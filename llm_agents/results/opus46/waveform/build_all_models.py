#!/usr/bin/env python3
"""Master script to build all 22+ waveform surrogate models (non-symbolic)."""

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

print("Loading data...")
params_train, wf_train, meta_train = load_dataset(
    os.path.join(ROOT, "datasets/waveform/waveform_training.h5"))
params_val, wf_val, meta_val = load_dataset(
    os.path.join(ROOT, "datasets/waveform/waveform_validation.h5"))
N_TRAIN, N_VAL = len(params_train), len(params_val)
print(f"Loaded {N_TRAIN} train, {N_VAL} val")

N_BASIS = 40
print(f"Computing SVD ({N_BASIS} basis)...")
wf_real_train = np.real(wf_train)
wf_imag_train = np.imag(wf_train)
coeffs_r, basis_r, mean_r, sv_r = compute_svd(wf_real_train, N_BASIS)
coeffs_i, basis_i, mean_i, sv_i = compute_svd(wf_imag_train, N_BASIS)

wf_real_val = np.real(wf_val)
wf_imag_val = np.imag(wf_val)
coeffs_r_val = project_onto_basis(wf_real_val, basis_r, mean_r)
coeffs_i_val = project_onto_basis(wf_imag_val, basis_i, mean_i)

svd_dir = os.path.join(WORK_DIR, "shared_svd")
os.makedirs(svd_dir, exist_ok=True)
np.savez(os.path.join(svd_dir, "svd_basis.npz"),
         basis_r=basis_r, basis_i=basis_i,
         mean_r=mean_r, mean_i=mean_i,
         sv_r=sv_r[:N_BASIS], sv_i=sv_i[:N_BASIS])

param_schemes = {k: reparameterize(params_train, k) for k in ["raw", "eta_chieff", "spherical", "mass_diff"]}
param_schemes_val = {k: reparameterize(params_val, k) for k in ["raw", "eta_chieff", "spherical", "mass_diff"]}

y_train_all = np.hstack([coeffs_r, coeffs_i])
y_val_all = np.hstack([coeffs_r_val, coeffs_i_val])

from sklearn.preprocessing import StandardScaler

all_results = []
all_error_data = {}
changelog_entries = []


def make_model_dir(num, name):
    d = os.path.join(WORK_DIR, f"models/{num:02d}_{name}")
    os.makedirs(os.path.join(d, "saved_model"), exist_ok=True)
    return d


def eval_svd_predictions(y_pred, wf_ref):
    nb = min(y_pred.shape[1] // 2, N_BASIS) if y_pred.shape[1] > N_BASIS else y_pred.shape[1]
    if y_pred.shape[1] >= 2 * N_BASIS:
        pr, pi = y_pred[:, :N_BASIS], y_pred[:, N_BASIS:]
    else:
        pr = y_pred[:, :nb]
        pi = np.zeros((len(y_pred), N_BASIS))
        if y_pred.shape[1] > nb:
            pi[:, :y_pred.shape[1]-nb] = y_pred[:, nb:]
        full_r = np.zeros((len(y_pred), N_BASIS))
        full_r[:, :nb] = pr
        pr = full_r
    wf_pred = reconstruct_from_basis(pr, basis_r, mean_r) + 1j * reconstruct_from_basis(pi, basis_i, mean_i)
    losses, ml = compute_loss_batch(wf_pred, wf_ref, DT)
    return ml, losses


def record(num, name, scheme, loss, losses_val, losses_train, runtime_ms, n_params, notes):
    model_dir = os.path.join(WORK_DIR, f"models/{num:02d}_{name}")
    save_scorecard(model_dir, name, num, scheme, "t0_at_peak", float(loss),
                   {"mean_mismatch": float(loss)}, runtime_ms, N_TRAIN, N_VAL, n_params, notes)
    all_results.append({"approach": name, "approach_number": num, "loss": float(loss),
                        "runtime_ms": runtime_ms, "parameterization": scheme, "notes": notes})
    all_error_data[name] = {
        "val_losses": [float(x) for x in losses_val],
        "train_losses": [float(x) for x in losses_train] if losses_train is not None else [],
    }
    changelog_entries.append(f"## Approach {num}: {name}\n- Param: {scheme}\n- Loss: {loss:.6f}\n- Runtime: {runtime_ms:.1f}ms\n- {notes}\n")
    print(f"  [{num:02d}] {name}: loss={loss:.6f}, rt={runtime_ms:.1f}ms")


# ─────────────────────────────────────────────────────────────
# 1. SVD + GPR (RBF, raw) - use only 5 basis for speed
# ─────────────────────────────────────────────────────────────
print("\n=== 1: SVD+GPR RBF raw ===")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.multioutput import MultiOutputRegressor

md = make_model_dir(1, "svd_gpr_rbf_raw")
n_gpr = 5
X, Xv = param_schemes["raw"], param_schemes_val["raw"]
sX = StandardScaler().fit(X); Xs, Xvs = sX.transform(X), sX.transform(Xv)
yt = y_train_all[:, :n_gpr]
sy = StandardScaler().fit(yt); yts = sy.transform(yt)
k1 = ConstantKernel(1.0) * RBF(length_scale=np.ones(7)) + WhiteKernel(1e-3)
gpr = MultiOutputRegressor(GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=1, alpha=1e-6, normalize_y=True), n_jobs=-1)
gpr.fit(Xs, yts)
t0 = time.time()
yp = sy.inverse_transform(gpr.predict(Xvs))
rt = (time.time()-t0)/N_VAL*1000
yfull = np.zeros((N_VAL, 2*N_BASIS)); yfull[:, :n_gpr] = yp
loss, lv = eval_svd_predictions(yfull, wf_val)
# Train losses
yp_tr = sy.inverse_transform(gpr.predict(Xs))
yfull_tr = np.zeros((N_TRAIN, 2*N_BASIS)); yfull_tr[:, :n_gpr] = yp_tr
_, lt = eval_svd_predictions(yfull_tr, wf_train)
joblib.dump({"gpr": gpr, "scaler_X": sX, "scaler_y": sy, "n_basis": n_gpr}, os.path.join(md, "saved_model/model.joblib"))
record(1, "svd_gpr_rbf_raw", "raw", loss, lv, lt, rt, n_gpr*250, "SVD(5 real) + GPR RBF kernel")


# ─────────────────────────────────────────────────────────────
# 2. SVD + GPR (Matern, eta_chieff)
# ─────────────────────────────────────────────────────────────
print("\n=== 2: SVD+GPR Matern eta ===")
md = make_model_dir(2, "svd_gpr_matern_eta")
X2, Xv2 = param_schemes["eta_chieff"], param_schemes_val["eta_chieff"]
sX2 = StandardScaler().fit(X2); X2s, Xv2s = sX2.transform(X2), sX2.transform(Xv2)
yt2 = y_train_all[:, :n_gpr]; sy2 = StandardScaler().fit(yt2); yt2s = sy2.transform(yt2)
k2 = ConstantKernel(1.0) * Matern(length_scale=np.ones(7), nu=2.5) + WhiteKernel(1e-3)
gpr2 = MultiOutputRegressor(GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=1, alpha=1e-6, normalize_y=True), n_jobs=-1)
gpr2.fit(X2s, yt2s)
t0 = time.time()
yp2 = sy2.inverse_transform(gpr2.predict(Xv2s))
rt2 = (time.time()-t0)/N_VAL*1000
yf2 = np.zeros((N_VAL, 2*N_BASIS)); yf2[:, :n_gpr] = yp2
loss2, lv2 = eval_svd_predictions(yf2, wf_val)
yp2t = sy2.inverse_transform(gpr2.predict(X2s))
yf2t = np.zeros((N_TRAIN, 2*N_BASIS)); yf2t[:, :n_gpr] = yp2t
_, lt2 = eval_svd_predictions(yf2t, wf_train)
joblib.dump({"gpr": gpr2, "scaler_X": sX2, "scaler_y": sy2}, os.path.join(md, "saved_model/model.joblib"))
record(2, "svd_gpr_matern_eta", "eta_chieff", loss2, lv2, lt2, rt2, n_gpr*250, "SVD(5) + GPR Matern-5/2, eta+chieff")


# ─────────────────────────────────────────────────────────────
# 3. SVD + Polynomial (raw, deg 3)
# ─────────────────────────────────────────────────────────────
print("\n=== 3: SVD+Poly3 raw ===")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

md = make_model_dir(3, "svd_poly3_raw")
X3, Xv3 = param_schemes["raw"], param_schemes_val["raw"]
sX3 = StandardScaler().fit(X3); X3s, Xv3s = sX3.transform(X3), sX3.transform(Xv3)
poly3 = PolynomialFeatures(degree=3); X3p = poly3.fit_transform(X3s); Xv3p = poly3.transform(Xv3s)
sy3 = StandardScaler().fit(y_train_all); yt3s = sy3.transform(y_train_all)
ridge3 = Ridge(alpha=1.0).fit(X3p, yt3s)
t0 = time.time()
yp3 = sy3.inverse_transform(ridge3.predict(Xv3p))
rt3 = (time.time()-t0)/N_VAL*1000
loss3, lv3 = eval_svd_predictions(yp3, wf_val)
_, lt3 = eval_svd_predictions(sy3.inverse_transform(ridge3.predict(X3p)), wf_train)
joblib.dump({"ridge": ridge3, "poly": poly3, "scaler_X": sX3, "scaler_y": sy3}, os.path.join(md, "saved_model/model.joblib"))
record(3, "svd_poly3_raw", "raw", loss3, lv3, lt3, rt3, X3p.shape[1]*80, "SVD(40) + deg-3 poly + Ridge")


# ─────────────────────────────────────────────────────────────
# 4. SVD + MLP (raw)
# ─────────────────────────────────────────────────────────────
print("\n=== 4: SVD+MLP raw ===")
from sklearn.neural_network import MLPRegressor

md = make_model_dir(4, "svd_mlp_raw")
X4, Xv4 = param_schemes["raw"], param_schemes_val["raw"]
sX4 = StandardScaler().fit(X4); X4s, Xv4s = sX4.transform(X4), sX4.transform(Xv4)
sy4 = StandardScaler().fit(y_train_all); yt4s = sy4.transform(y_train_all)
mlp4 = MLPRegressor(hidden_layer_sizes=(256,128,64), max_iter=2000, early_stopping=True,
                    validation_fraction=0.15, random_state=42, learning_rate_init=0.001)
mlp4.fit(X4s, yt4s)
t0 = time.time()
yp4 = sy4.inverse_transform(mlp4.predict(Xv4s))
rt4 = (time.time()-t0)/N_VAL*1000
loss4, lv4 = eval_svd_predictions(yp4, wf_val)
_, lt4 = eval_svd_predictions(sy4.inverse_transform(mlp4.predict(X4s)), wf_train)
joblib.dump({"mlp": mlp4, "scaler_X": sX4, "scaler_y": sy4}, os.path.join(md, "saved_model/model.joblib"))
record(4, "svd_mlp_raw", "raw", loss4, lv4, lt4, rt4, sum(c.size for c in mlp4.coefs_), "SVD(40)+MLP [256,128,64]")


# ─────────────────────────────────────────────────────────────
# 5. SVD + RF (raw)
# ─────────────────────────────────────────────────────────────
print("\n=== 5: SVD+RF raw ===")
from sklearn.ensemble import RandomForestRegressor

md = make_model_dir(5, "svd_rf_raw")
X5 = param_schemes["raw"]; Xv5 = param_schemes_val["raw"]
rf5 = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf5.fit(X5, y_train_all)
t0 = time.time()
yp5 = rf5.predict(Xv5)
rt5 = (time.time()-t0)/N_VAL*1000
loss5, lv5 = eval_svd_predictions(yp5, wf_val)
_, lt5 = eval_svd_predictions(rf5.predict(X5), wf_train)
joblib.dump({"rf": rf5}, os.path.join(md, "saved_model/model.joblib"))
record(5, "svd_rf_raw", "raw", loss5, lv5, lt5, rt5, rf5.n_estimators*200, "SVD(40)+RF 200 trees")


# ─────────────────────────────────────────────────────────────
# 6. SVD + GBR (eta)
# ─────────────────────────────────────────────────────────────
print("\n=== 6: SVD+GBR eta ===")
from sklearn.ensemble import GradientBoostingRegressor

md = make_model_dir(6, "svd_gbr_eta")
X6 = param_schemes["eta_chieff"]; Xv6 = param_schemes_val["eta_chieff"]
gbr6 = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42), n_jobs=-1)
gbr6.fit(X6, y_train_all)
t0 = time.time()
yp6 = gbr6.predict(Xv6)
rt6 = (time.time()-t0)/N_VAL*1000
loss6, lv6 = eval_svd_predictions(yp6, wf_val)
_, lt6 = eval_svd_predictions(gbr6.predict(X6), wf_train)
joblib.dump({"gbr": gbr6}, os.path.join(md, "saved_model/model.joblib"))
record(6, "svd_gbr_eta", "eta_chieff", loss6, lv6, lt6, rt6, 200*80, "SVD(40)+GBR eta+chieff")


# ─────────────────────────────────────────────────────────────
# 7. SVD + KRR (raw)
# ─────────────────────────────────────────────────────────────
print("\n=== 7: SVD+KRR raw ===")
from sklearn.kernel_ridge import KernelRidge

md = make_model_dir(7, "svd_krr_raw")
X7 = param_schemes["raw"]; Xv7 = param_schemes_val["raw"]
sX7 = StandardScaler().fit(X7); X7s, Xv7s = sX7.transform(X7), sX7.transform(Xv7)
sy7 = StandardScaler().fit(y_train_all); yt7s = sy7.transform(y_train_all)
krr7 = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1).fit(X7s, yt7s)
t0 = time.time()
yp7 = sy7.inverse_transform(krr7.predict(Xv7s))
rt7 = (time.time()-t0)/N_VAL*1000
loss7, lv7 = eval_svd_predictions(yp7, wf_val)
_, lt7 = eval_svd_predictions(sy7.inverse_transform(krr7.predict(X7s)), wf_train)
joblib.dump({"krr": krr7, "scaler_X": sX7, "scaler_y": sy7}, os.path.join(md, "saved_model/model.joblib"))
record(7, "svd_krr_raw", "raw", loss7, lv7, lt7, rt7, 250*80, "SVD(40)+KRR RBF")


# ─────────────────────────────────────────────────────────────
# 8. SVD + RBF Interp (raw)
# ─────────────────────────────────────────────────────────────
print("\n=== 8: SVD+RBF raw ===")
from scipy.interpolate import RBFInterpolator

md = make_model_dir(8, "svd_rbf_interp_raw")
sX8 = StandardScaler().fit(X7); X8s, Xv8s = sX8.transform(X7), sX8.transform(Xv7)
sy8 = StandardScaler().fit(y_train_all); yt8s = sy8.transform(y_train_all)
rbf8 = RBFInterpolator(X8s, yt8s, kernel='thin_plate_spline', smoothing=0.1)
t0 = time.time()
yp8 = sy8.inverse_transform(rbf8(Xv8s))
rt8 = (time.time()-t0)/N_VAL*1000
loss8, lv8 = eval_svd_predictions(yp8, wf_val)
_, lt8 = eval_svd_predictions(sy8.inverse_transform(rbf8(X8s)), wf_train)
joblib.dump({"rbf": rbf8, "scaler_X": sX8, "scaler_y": sy8}, os.path.join(md, "saved_model/model.joblib"))
record(8, "svd_rbf_interp_raw", "raw", loss8, lv8, lt8, rt8, 250*80, "SVD(40)+RBF TPS interp")


# ─────────────────────────────────────────────────────────────
# 9. SVD + KNN (raw)
# ─────────────────────────────────────────────────────────────
print("\n=== 9: SVD+KNN raw ===")
from sklearn.neighbors import KNeighborsRegressor

md = make_model_dir(9, "svd_knn_raw")
sX9 = StandardScaler().fit(X7); X9s, Xv9s = sX9.transform(X7), sX9.transform(Xv7)
knn9 = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(X9s, y_train_all)
t0 = time.time()
yp9 = knn9.predict(Xv9s)
rt9 = (time.time()-t0)/N_VAL*1000
loss9, lv9 = eval_svd_predictions(yp9, wf_val)
_, lt9 = eval_svd_predictions(knn9.predict(X9s), wf_train)
joblib.dump({"knn": knn9, "scaler_X": sX9}, os.path.join(md, "saved_model/model.joblib"))
record(9, "svd_knn_raw", "raw", loss9, lv9, lt9, rt9, 0, "SVD(40)+5-NN distance-weighted")


# ─────────────────────────────────────────────────────────────
# 10. SVD + MLP (eta)
# ─────────────────────────────────────────────────────────────
print("\n=== 10: SVD+MLP eta ===")
md = make_model_dir(10, "svd_mlp_eta")
X10 = param_schemes["eta_chieff"]; Xv10 = param_schemes_val["eta_chieff"]
sX10 = StandardScaler().fit(X10); X10s, Xv10s = sX10.transform(X10), sX10.transform(Xv10)
sy10 = StandardScaler().fit(y_train_all); yt10s = sy10.transform(y_train_all)
mlp10 = MLPRegressor(hidden_layer_sizes=(512,256,128), max_iter=3000, early_stopping=True,
                     validation_fraction=0.15, random_state=42, learning_rate_init=0.0005)
mlp10.fit(X10s, yt10s)
t0 = time.time()
yp10 = sy10.inverse_transform(mlp10.predict(Xv10s))
rt10 = (time.time()-t0)/N_VAL*1000
loss10, lv10 = eval_svd_predictions(yp10, wf_val)
_, lt10 = eval_svd_predictions(sy10.inverse_transform(mlp10.predict(X10s)), wf_train)
joblib.dump({"mlp": mlp10, "scaler_X": sX10, "scaler_y": sy10}, os.path.join(md, "saved_model/model.joblib"))
record(10, "svd_mlp_eta", "eta_chieff", loss10, lv10, lt10, rt10, sum(c.size for c in mlp10.coefs_), "SVD(40)+MLP [512,256,128] eta")


# ─────────────────────────────────────────────────────────────
# 11. SVD + Poly4 (eta)
# ─────────────────────────────────────────────────────────────
print("\n=== 11: SVD+Poly4 eta ===")
md = make_model_dir(11, "svd_poly4_eta")
sX11 = StandardScaler().fit(X10); X11s, Xv11s = sX11.transform(X10), sX11.transform(Xv10)
poly11 = PolynomialFeatures(degree=4); X11p = poly11.fit_transform(X11s); Xv11p = poly11.transform(Xv11s)
sy11 = StandardScaler().fit(y_train_all); yt11s = sy11.transform(y_train_all)
ridge11 = Ridge(alpha=10.0).fit(X11p, yt11s)
t0 = time.time()
yp11 = sy11.inverse_transform(ridge11.predict(Xv11p))
rt11 = (time.time()-t0)/N_VAL*1000
loss11, lv11 = eval_svd_predictions(yp11, wf_val)
_, lt11 = eval_svd_predictions(sy11.inverse_transform(ridge11.predict(X11p)), wf_train)
joblib.dump({"ridge": ridge11, "poly": poly11, "scaler_X": sX11, "scaler_y": sy11}, os.path.join(md, "saved_model/model.joblib"))
record(11, "svd_poly4_eta", "eta_chieff", loss11, lv11, lt11, rt11, X11p.shape[1]*80, "SVD(40)+deg-4 poly+Ridge eta")


# ─────────────────────────────────────────────────────────────
# 12. SVD + GPR Matern (spherical)
# ─────────────────────────────────────────────────────────────
print("\n=== 12: SVD+GPR Matern sph ===")
md = make_model_dir(12, "svd_gpr_matern_sph")
X12 = param_schemes["spherical"]; Xv12 = param_schemes_val["spherical"]
sX12 = StandardScaler().fit(X12); X12s, Xv12s = sX12.transform(X12), sX12.transform(Xv12)
yt12 = y_train_all[:, :n_gpr]; sy12 = StandardScaler().fit(yt12); yt12s = sy12.transform(yt12)
k12 = ConstantKernel(1.0) * Matern(length_scale=np.ones(7), nu=1.5) + WhiteKernel(1e-3)
gpr12 = MultiOutputRegressor(GaussianProcessRegressor(kernel=k12, n_restarts_optimizer=1, alpha=1e-6, normalize_y=True), n_jobs=-1)
gpr12.fit(X12s, yt12s)
t0 = time.time()
yp12 = sy12.inverse_transform(gpr12.predict(Xv12s))
rt12 = (time.time()-t0)/N_VAL*1000
yf12 = np.zeros((N_VAL, 2*N_BASIS)); yf12[:, :n_gpr] = yp12
loss12, lv12 = eval_svd_predictions(yf12, wf_val)
yp12t = sy12.inverse_transform(gpr12.predict(X12s))
yf12t = np.zeros((N_TRAIN, 2*N_BASIS)); yf12t[:, :n_gpr] = yp12t
_, lt12 = eval_svd_predictions(yf12t, wf_train)
joblib.dump({"gpr": gpr12, "scaler_X": sX12, "scaler_y": sy12}, os.path.join(md, "saved_model/model.joblib"))
record(12, "svd_gpr_matern_sph", "spherical", loss12, lv12, lt12, rt12, n_gpr*250, "SVD(5)+GPR Matern-3/2 spherical")


# ─────────────────────────────────────────────────────────────
# 13. SVD + RF (eta)
# ─────────────────────────────────────────────────────────────
print("\n=== 13: SVD+RF eta ===")
md = make_model_dir(13, "svd_rf_eta")
rf13 = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf13.fit(X10, y_train_all)
t0 = time.time()
yp13 = rf13.predict(Xv10)
rt13 = (time.time()-t0)/N_VAL*1000
loss13, lv13 = eval_svd_predictions(yp13, wf_val)
_, lt13 = eval_svd_predictions(rf13.predict(X10), wf_train)
joblib.dump({"rf": rf13}, os.path.join(md, "saved_model/model.joblib"))
record(13, "svd_rf_eta", "eta_chieff", loss13, lv13, lt13, rt13, rf13.n_estimators*200, "SVD(40)+RF 500 trees eta")


# ─────────────────────────────────────────────────────────────
# 14. SVD + Extra Trees (spherical)
# ─────────────────────────────────────────────────────────────
print("\n=== 14: SVD+ET sph ===")
from sklearn.ensemble import ExtraTreesRegressor

md = make_model_dir(14, "svd_et_sph")
X14 = param_schemes["spherical"]; Xv14 = param_schemes_val["spherical"]
et14 = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
et14.fit(X14, y_train_all)
t0 = time.time()
yp14 = et14.predict(Xv14)
rt14 = (time.time()-t0)/N_VAL*1000
loss14, lv14 = eval_svd_predictions(yp14, wf_val)
_, lt14 = eval_svd_predictions(et14.predict(X14), wf_train)
joblib.dump({"et": et14}, os.path.join(md, "saved_model/model.joblib"))
record(14, "svd_et_sph", "spherical", loss14, lv14, lt14, rt14, et14.n_estimators*200, "SVD(40)+ExtraTrees 500 spherical")


# ─────────────────────────────────────────────────────────────
# 15. SVD + SVR (mass_diff)
# ─────────────────────────────────────────────────────────────
print("\n=== 15: SVD+SVR mdiff ===")
from sklearn.svm import SVR

md = make_model_dir(15, "svd_svr_mdiff")
X15 = param_schemes["mass_diff"]; Xv15 = param_schemes_val["mass_diff"]
sX15 = StandardScaler().fit(X15); X15s, Xv15s = sX15.transform(X15), sX15.transform(Xv15)
sy15 = StandardScaler().fit(y_train_all); yt15s = sy15.transform(y_train_all)
svr15 = MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.01), n_jobs=-1)
svr15.fit(X15s, yt15s)
t0 = time.time()
yp15 = sy15.inverse_transform(svr15.predict(Xv15s))
rt15 = (time.time()-t0)/N_VAL*1000
loss15, lv15 = eval_svd_predictions(yp15, wf_val)
_, lt15 = eval_svd_predictions(sy15.inverse_transform(svr15.predict(X15s)), wf_train)
joblib.dump({"svr": svr15, "scaler_X": sX15, "scaler_y": sy15}, os.path.join(md, "saved_model/model.joblib"))
record(15, "svd_svr_mdiff", "mass_diff", loss15, lv15, lt15, rt15, 250*80, "SVD(40)+SVR RBF mass_diff")


# ─────────────────────────────────────────────────────────────
# 16. SVD + Lasso poly3 (raw)
# ─────────────────────────────────────────────────────────────
print("\n=== 16: SVD+Lasso raw ===")
from sklearn.linear_model import Lasso

md = make_model_dir(16, "svd_lasso_raw")
sX16 = StandardScaler().fit(X7); X16s, Xv16s = sX16.transform(X7), sX16.transform(Xv7)
poly16 = PolynomialFeatures(degree=3); X16p = poly16.fit_transform(X16s); Xv16p = poly16.transform(Xv16s)
sy16 = StandardScaler().fit(y_train_all); yt16s = sy16.transform(y_train_all)
lasso16 = MultiOutputRegressor(Lasso(alpha=0.001, max_iter=5000), n_jobs=-1)
lasso16.fit(X16p, yt16s)
t0 = time.time()
yp16 = sy16.inverse_transform(lasso16.predict(Xv16p))
rt16 = (time.time()-t0)/N_VAL*1000
loss16, lv16 = eval_svd_predictions(yp16, wf_val)
_, lt16 = eval_svd_predictions(sy16.inverse_transform(lasso16.predict(X16p)), wf_train)
joblib.dump({"lasso": lasso16, "poly": poly16, "scaler_X": sX16, "scaler_y": sy16}, os.path.join(md, "saved_model/model.joblib"))
record(16, "svd_lasso_raw", "raw", loss16, lv16, lt16, rt16, X16p.shape[1]*80, "SVD(40)+Lasso poly-3")


# ─────────────────────────────────────────────────────────────
# 17. SVD + AdaBoost (eta)
# ─────────────────────────────────────────────────────────────
print("\n=== 17: SVD+AdaBoost eta ===")
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

md = make_model_dir(17, "svd_adaboost_eta")
ada17 = MultiOutputRegressor(
    AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=6),
                      n_estimators=100, learning_rate=0.1, random_state=42), n_jobs=-1)
ada17.fit(X10, y_train_all)
t0 = time.time()
yp17 = ada17.predict(Xv10)
rt17 = (time.time()-t0)/N_VAL*1000
loss17, lv17 = eval_svd_predictions(yp17, wf_val)
_, lt17 = eval_svd_predictions(ada17.predict(X10), wf_train)
joblib.dump({"ada": ada17}, os.path.join(md, "saved_model/model.joblib"))
record(17, "svd_adaboost_eta", "eta_chieff", loss17, lv17, lt17, rt17, 100*80, "SVD(40)+AdaBoost DT-6 eta")


# ─────────────────────────────────────────────────────────────
# 18. SVD + MLP large (spherical)
# ─────────────────────────────────────────────────────────────
print("\n=== 18: SVD+MLP-large sph ===")
md = make_model_dir(18, "svd_mlp_large_sph")
sX18 = StandardScaler().fit(X14); X18s, Xv18s = sX18.transform(X14), sX18.transform(Xv14)
sy18 = StandardScaler().fit(y_train_all); yt18s = sy18.transform(y_train_all)
mlp18 = MLPRegressor(hidden_layer_sizes=(512,512,256,128), max_iter=5000, early_stopping=True,
                     validation_fraction=0.15, random_state=42, learning_rate_init=0.0003, batch_size=32)
mlp18.fit(X18s, yt18s)
t0 = time.time()
yp18 = sy18.inverse_transform(mlp18.predict(Xv18s))
rt18 = (time.time()-t0)/N_VAL*1000
loss18, lv18 = eval_svd_predictions(yp18, wf_val)
_, lt18 = eval_svd_predictions(sy18.inverse_transform(mlp18.predict(X18s)), wf_train)
joblib.dump({"mlp": mlp18, "scaler_X": sX18, "scaler_y": sy18}, os.path.join(md, "saved_model/model.joblib"))
record(18, "svd_mlp_large_sph", "spherical", loss18, lv18, lt18, rt18, sum(c.size for c in mlp18.coefs_), "SVD(40)+MLP [512,512,256,128] spherical")


# ─────────────────────────────────────────────────────────────
# 19. Amp/Phase + RF (eta)
# ─────────────────────────────────────────────────────────────
print("\n=== 19: AmpPhase+RF eta ===")
md = make_model_dir(19, "ampphase_rf_eta")
amp_tr = np.abs(wf_train); phase_tr = np.unwrap(np.angle(wf_train), axis=1)
amp_v = np.abs(wf_val); phase_v = np.unwrap(np.angle(wf_val), axis=1)
ca, ba, ma, _ = compute_svd(amp_tr, N_BASIS)
cp, bp, mp, _ = compute_svd(phase_tr, N_BASIS)
ca_v = project_onto_basis(amp_v, ba, ma)
cp_v = project_onto_basis(phase_v, bp, mp)
y_ap = np.hstack([ca, cp])
rf_ap = RandomForestRegressor(n_estimators=300, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_ap.fit(X10, y_ap)
t0 = time.time()
y_ap_pred = rf_ap.predict(Xv10)
rt19 = (time.time()-t0)/N_VAL*1000
a_rec = reconstruct_from_basis(y_ap_pred[:, :N_BASIS], ba, ma)
p_rec = reconstruct_from_basis(y_ap_pred[:, N_BASIS:], bp, mp)
wf19 = a_rec * np.exp(1j * p_rec)
loss19, lv19 = compute_loss_batch(wf19, wf_val, DT)
y_ap_tr = rf_ap.predict(X10)
a_rec_tr = reconstruct_from_basis(y_ap_tr[:, :N_BASIS], ba, ma)
p_rec_tr = reconstruct_from_basis(y_ap_tr[:, N_BASIS:], bp, mp)
wf19_tr = a_rec_tr * np.exp(1j * p_rec_tr)
lt19, _ = compute_loss_batch(wf19_tr, wf_train, DT)
loss19 = lv19  # lv19 is actually the losses array from compute_loss_batch (losses, mean)
# Fix: compute_loss_batch returns (losses_array, mean_loss)
lv19_arr, loss19 = compute_loss_batch(wf19, wf_val, DT)
lt19_arr, _ = compute_loss_batch(wf19_tr, wf_train, DT)
joblib.dump({"rf": rf_ap, "basis_amp": ba, "mean_amp": ma, "basis_phase": bp, "mean_phase": mp},
            os.path.join(md, "saved_model/model.joblib"))
record(19, "ampphase_rf_eta", "eta_chieff", loss19, lv19_arr, lt19_arr, rt19, rf_ap.n_estimators*200, "Amp/Phase SVD(40)+RF 300 eta")


# ─────────────────────────────────────────────────────────────
# 20. SVD + RBF Interp (eta)
# ─────────────────────────────────────────────────────────────
print("\n=== 20: SVD+RBF eta ===")
md = make_model_dir(20, "svd_rbf_interp_eta")
sX20 = StandardScaler().fit(X10); X20s, Xv20s = sX20.transform(X10), sX20.transform(Xv10)
sy20 = StandardScaler().fit(y_train_all); yt20s = sy20.transform(y_train_all)
rbf20 = RBFInterpolator(X20s, yt20s, kernel='multiquadric', smoothing=0.05, epsilon=1.0)
t0 = time.time()
yp20 = sy20.inverse_transform(rbf20(Xv20s))
rt20 = (time.time()-t0)/N_VAL*1000
loss20, lv20 = eval_svd_predictions(yp20, wf_val)
_, lt20 = eval_svd_predictions(sy20.inverse_transform(rbf20(X20s)), wf_train)
joblib.dump({"rbf": rbf20, "scaler_X": sX20, "scaler_y": sy20}, os.path.join(md, "saved_model/model.joblib"))
record(20, "svd_rbf_interp_eta", "eta_chieff", loss20, lv20, lt20, rt20, 250*80, "SVD(40)+RBF MQ interp eta")


# ─────────────────────────────────────────────────────────────
# 21. SVD + BayesianRidge (mass_diff)
# ─────────────────────────────────────────────────────────────
print("\n=== 21: SVD+BayRidge mdiff ===")
from sklearn.linear_model import BayesianRidge

md = make_model_dir(21, "svd_bayridge_mdiff")
sX21 = StandardScaler().fit(X15); X21s, Xv21s = sX21.transform(X15), sX21.transform(Xv15)
poly21 = PolynomialFeatures(degree=3); X21p = poly21.fit_transform(X21s); Xv21p = poly21.transform(Xv21s)
sy21 = StandardScaler().fit(y_train_all); yt21s = sy21.transform(y_train_all)
br21 = MultiOutputRegressor(BayesianRidge(max_iter=500), n_jobs=-1)
br21.fit(X21p, yt21s)
t0 = time.time()
yp21 = sy21.inverse_transform(br21.predict(Xv21p))
rt21 = (time.time()-t0)/N_VAL*1000
loss21, lv21 = eval_svd_predictions(yp21, wf_val)
_, lt21 = eval_svd_predictions(sy21.inverse_transform(br21.predict(X21p)), wf_train)
joblib.dump({"br": br21, "poly": poly21, "scaler_X": sX21, "scaler_y": sy21}, os.path.join(md, "saved_model/model.joblib"))
record(21, "svd_bayridge_mdiff", "mass_diff", loss21, lv21, lt21, rt21, X21p.shape[1]*80, "SVD(40)+BayRidge poly-3 mass_diff")


# ─────────────────────────────────────────────────────────────
# 22. SVD + ElasticNet (eta)
# ─────────────────────────────────────────────────────────────
print("\n=== 22: SVD+ElasticNet eta ===")
from sklearn.linear_model import ElasticNet

md = make_model_dir(22, "svd_elasticnet_eta")
sX22 = StandardScaler().fit(X10); X22s, Xv22s = sX22.transform(X10), sX22.transform(Xv10)
poly22 = PolynomialFeatures(degree=3); X22p = poly22.fit_transform(X22s); Xv22p = poly22.transform(Xv22s)
sy22 = StandardScaler().fit(y_train_all); yt22s = sy22.transform(y_train_all)
enet22 = MultiOutputRegressor(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000), n_jobs=-1)
enet22.fit(X22p, yt22s)
t0 = time.time()
yp22 = sy22.inverse_transform(enet22.predict(Xv22p))
rt22 = (time.time()-t0)/N_VAL*1000
loss22, lv22 = eval_svd_predictions(yp22, wf_val)
_, lt22 = eval_svd_predictions(sy22.inverse_transform(enet22.predict(X22p)), wf_train)
joblib.dump({"enet": enet22, "poly": poly22, "scaler_X": sX22, "scaler_y": sy22}, os.path.join(md, "saved_model/model.joblib"))
record(22, "svd_elasticnet_eta", "eta_chieff", loss22, lv22, lt22, rt22, X22p.shape[1]*80, "SVD(40)+ElasticNet poly-3 eta")


# ═════════════════════════════════════════════════════════════
# Save all outputs
# ═════════════════════════════════════════════════════════════
print("\n=== Saving outputs ===")
os.makedirs(os.path.join(WORK_DIR, "comparison"), exist_ok=True)

with open(os.path.join(WORK_DIR, "comparison/error_data.json"), "w") as f:
    json.dump(all_error_data, f)

with open(os.path.join(WORK_DIR, "comparison/summary_table.json"), "w") as f:
    json.dump(sorted(all_results, key=lambda x: x["loss"]), f, indent=2)

best = min(all_results, key=lambda x: x["loss"])
with open(os.path.join(WORK_DIR, "comparison/best_model.json"), "w") as f:
    json.dump(best, f, indent=2)

with open(os.path.join(WORK_DIR, "CHANGELOG.md"), "w") as f:
    f.write("# Waveform Benchmark CHANGELOG — Opus 4.6\n\n")
    for entry in changelog_entries:
        f.write(entry + "\n")

print(f"\n=== SUMMARY ({len(all_results)} approaches) ===")
for r in sorted(all_results, key=lambda x: x["loss"]):
    print(f"  {r['approach_number']:02d} {r['approach']:<30s} loss={r['loss']:.6f} rt={r['runtime_ms']:.1f}ms")
print(f"\nBest: {best['approach']} with loss={best['loss']:.6f}")
print("Next: run symbolic regression (PySR + gplearn)")
