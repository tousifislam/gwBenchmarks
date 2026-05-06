#!/usr/bin/env python3
"""Build all 26 remnant models: predict kick velocity |v_k| from binary parameters."""

import sys, os, numpy as np, json, time, warnings, joblib
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../.."))
sys.path.insert(0, ROOT)
import h5py
from gwbenchmarks.metrics import nrmse

# ─── Load data ───
def load_remnant(path):
    with h5py.File(path, "r") as f:
        q = f["q"][:]
        chi1x, chi1y, chi1z = f["chi1x"][:], f["chi1y"][:], f["chi1z"][:]
        chi2x, chi2y, chi2z = f["chi2x"][:], f["chi2y"][:], f["chi2z"][:]
        vf = f["vf_mag"][:]
    params = np.column_stack([q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z])
    return params, vf

print("Loading data...")
X_train, y_train = load_remnant(os.path.join(ROOT, "datasets/remnant/remnant_training.h5"))
X_val, y_val = load_remnant(os.path.join(ROOT, "datasets/remnant/remnant_validation.h5"))
N_TR, N_VA = len(X_train), len(X_val)
print(f"Train: {N_TR}, Val: {N_VA}, y range: [{y_train.min():.6f}, {y_train.max():.6f}]")

# ─── Reparameterizations ───
def reparam(X, scheme="raw"):
    q = X[:, 0]
    chi1x, chi1y, chi1z = X[:, 1], X[:, 2], X[:, 3]
    chi2x, chi2y, chi2z = X[:, 4], X[:, 5], X[:, 6]
    if scheme == "raw":
        return X
    eta = q / (1+q)**2
    chi1_mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2_mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
    chi_eff = (q*chi1z + chi2z) / (1+q)
    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
    chi_p = np.maximum(chi1_perp, (3+4*q)/(4+3*q)*q*chi2_perp)
    if scheme == "eta_chieff":
        t1 = np.arccos(np.clip(chi1z/np.maximum(chi1_mag,1e-15), -1, 1))
        t2 = np.arccos(np.clip(chi2z/np.maximum(chi2_mag,1e-15), -1, 1))
        return np.column_stack([eta, chi_eff, chi_p, chi1_mag, chi2_mag, t1, t2])
    if scheme == "mass_diff":
        dm = (q-1)/(q+1)
        chi_a = (q*chi1z - chi2z)/(1+q)
        return np.column_stack([dm, chi_eff, chi_a, chi1_mag, chi2_mag])
    if scheme == "spherical":
        t1 = np.arccos(np.clip(chi1z/np.maximum(chi1_mag,1e-15), -1, 1))
        p1 = np.arctan2(chi1y, chi1x)
        t2 = np.arccos(np.clip(chi2z/np.maximum(chi2_mag,1e-15), -1, 1))
        p2 = np.arctan2(chi2y, chi2x)
        return np.column_stack([eta, chi1_mag, t1, p1, chi2_mag, t2, p2])
    if scheme == "pn_inspired":
        dm = (q-1)/(q+1)
        chi_a = (q*chi1z - chi2z)/(1+q)
        return np.column_stack([eta, chi_eff, eta*chi_eff, dm*chi_a, chi_p])
    return X

ps = {k: reparam(X_train, k) for k in ["raw","eta_chieff","mass_diff","spherical","pn_inspired"]}
pv = {k: reparam(X_val, k) for k in ["raw","eta_chieff","mass_diff","spherical","pn_inspired"]}

from sklearn.preprocessing import StandardScaler

# ─── Helpers ───
results = []; err_data = {}; cl = []

def md(n, name):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    os.makedirs(os.path.join(d, "saved_model"), exist_ok=True)
    return d

def rec(n, name, sch, loss_val, pred_val, pred_tr, rt, np_, notes):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    loss_tr = nrmse(pred_tr, y_train)
    sc = {"approach":name,"approach_number":n,"benchmark":"remnant","agent":"opus46",
          "parameterization":sch,"loss":float(loss_val),
          "loss_components":{"nrmse_v_k":float(loss_val)},
          "runtime_ms":rt,"n_train":N_TR,"n_val":N_VA,"n_params":np_,"notes":notes}
    with open(os.path.join(d, "scorecard.json"), "w") as f:
        json.dump(sc, f, indent=2)
    results.append(sc)
    err_data[name] = {
        "val_losses": [float(abs(pred_val[i]-y_val[i])) for i in range(N_VA)],
        "train_losses": [float(abs(pred_tr[i]-y_train[i])) for i in range(N_TR)],
    }
    cl.append(f"## {n}: {name}\n- Param: {sch}, NRMSE: {loss_val:.6f}, RT: {rt:.1f}ms\n- {notes}\n")
    print(f"  [{n:02d}] {name}: NRMSE={loss_val:.6f}")

# ═══════ BUILD ALL MODELS ═══════
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import RBFInterpolator

# 1. GPR RBF raw
print("\n=== 1: GPR RBF raw ===")
d = md(1, "gpr_rbf_raw")
sX = StandardScaler().fit(ps["raw"]); Xs = sX.transform(ps["raw"]); Xvs = sX.transform(pv["raw"])
sy = StandardScaler().fit(y_train.reshape(-1,1)); ys = sy.transform(y_train.reshape(-1,1)).ravel()
k1 = ConstantKernel(1.0)*RBF(length_scale=1.0)+WhiteKernel(1e-4)
gpr1 = GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=0, alpha=1e-6, normalize_y=True)
gpr1.fit(Xs, ys)
t0=time.time(); yp=sy.inverse_transform(gpr1.predict(Xvs).reshape(-1,1)).ravel(); rt=(time.time()-t0)/N_VA*1000
loss=nrmse(yp, y_val)
yp_tr=sy.inverse_transform(gpr1.predict(Xs).reshape(-1,1)).ravel()
joblib.dump({"gpr":gpr1,"scaler_X":sX,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(1,"gpr_rbf_raw","raw",loss,yp,yp_tr,rt,1000,"GPR RBF kernel, raw params")

# 2. GPR Matern eta
print("\n=== 2: GPR Matern eta ===")
d = md(2, "gpr_matern_eta")
sX2 = StandardScaler().fit(ps["eta_chieff"]); X2s = sX2.transform(ps["eta_chieff"]); Xv2s = sX2.transform(pv["eta_chieff"])
k2 = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5)+WhiteKernel(1e-4)
gpr2 = GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=0, alpha=1e-6, normalize_y=True)
gpr2.fit(X2s, ys)
t0=time.time(); yp2=sy.inverse_transform(gpr2.predict(Xv2s).reshape(-1,1)).ravel(); rt2=(time.time()-t0)/N_VA*1000
loss2=nrmse(yp2, y_val)
yp2_tr=sy.inverse_transform(gpr2.predict(X2s).reshape(-1,1)).ravel()
joblib.dump({"gpr":gpr2,"scaler_X":sX2,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(2,"gpr_matern_eta","eta_chieff",loss2,yp2,yp2_tr,rt2,1000,"GPR Matern-5/2, eta+chieff")

# 3. KRR raw
print("\n=== 3: KRR raw ===")
d = md(3, "krr_raw")
krr3 = KernelRidge(kernel='rbf', alpha=0.01, gamma=0.1).fit(Xs, ys)
t0=time.time(); yp3=sy.inverse_transform(krr3.predict(Xvs).reshape(-1,1)).ravel(); rt3=(time.time()-t0)/N_VA*1000
loss3=nrmse(yp3, y_val)
yp3_tr=sy.inverse_transform(krr3.predict(Xs).reshape(-1,1)).ravel()
joblib.dump({"krr":krr3,"scaler_X":sX,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(3,"krr_raw","raw",loss3,yp3,yp3_tr,rt3,1000,"KRR RBF")

# 4. SVR raw
print("\n=== 4: SVR raw ===")
d = md(4, "svr_raw")
svr4 = SVR(kernel='rbf', C=100.0, epsilon=0.01).fit(Xs, ys)
t0=time.time(); yp4=sy.inverse_transform(svr4.predict(Xvs).reshape(-1,1)).ravel(); rt4=(time.time()-t0)/N_VA*1000
loss4=nrmse(yp4, y_val)
yp4_tr=sy.inverse_transform(svr4.predict(Xs).reshape(-1,1)).ravel()
joblib.dump({"svr":svr4,"scaler_X":sX,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(4,"svr_raw","raw",loss4,yp4,yp4_tr,rt4,1000,"SVR RBF C=100")

# 5. MLP raw
print("\n=== 5: MLP raw ===")
d = md(5, "mlp_raw")
mlp5 = MLPRegressor(hidden_layer_sizes=(256,128,64), max_iter=3000, early_stopping=True,
                    validation_fraction=0.15, random_state=42, learning_rate_init=0.001)
mlp5.fit(Xs, ys)
t0=time.time(); yp5=sy.inverse_transform(mlp5.predict(Xvs).reshape(-1,1)).ravel(); rt5=(time.time()-t0)/N_VA*1000
loss5=nrmse(yp5, y_val)
yp5_tr=sy.inverse_transform(mlp5.predict(Xs).reshape(-1,1)).ravel()
joblib.dump({"mlp":mlp5,"scaler_X":sX,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(5,"mlp_raw","raw",loss5,yp5,yp5_tr,rt5,sum(c.size for c in mlp5.coefs_),"MLP [256,128,64]")

# 6. MLP eta
print("\n=== 6: MLP eta ===")
d = md(6, "mlp_eta")
mlp6 = MLPRegressor(hidden_layer_sizes=(512,256,128), max_iter=3000, early_stopping=True,
                    validation_fraction=0.15, random_state=42, learning_rate_init=0.0005)
mlp6.fit(X2s, ys)
t0=time.time(); yp6=sy.inverse_transform(mlp6.predict(Xv2s).reshape(-1,1)).ravel(); rt6=(time.time()-t0)/N_VA*1000
loss6=nrmse(yp6, y_val)
yp6_tr=sy.inverse_transform(mlp6.predict(X2s).reshape(-1,1)).ravel()
joblib.dump({"mlp":mlp6,"scaler_X":sX2,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(6,"mlp_eta","eta_chieff",loss6,yp6,yp6_tr,rt6,sum(c.size for c in mlp6.coefs_),"MLP [512,256,128] eta")

# 7. RF raw
print("\n=== 7: RF raw ===")
d = md(7, "rf_raw")
rf7 = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf7.fit(ps["raw"], y_train)
t0=time.time(); yp7=rf7.predict(pv["raw"]); rt7=(time.time()-t0)/N_VA*1000
loss7=nrmse(yp7, y_val)
yp7_tr=rf7.predict(ps["raw"])
joblib.dump({"rf":rf7}, os.path.join(d,"saved_model/model.joblib"))
rec(7,"rf_raw","raw",loss7,yp7,yp7_tr,rt7,500*200,"RF 500 trees")

# 8. RF eta
print("\n=== 8: RF eta ===")
d = md(8, "rf_eta")
rf8 = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf8.fit(ps["eta_chieff"], y_train)
t0=time.time(); yp8=rf8.predict(pv["eta_chieff"]); rt8=(time.time()-t0)/N_VA*1000
loss8=nrmse(yp8, y_val)
yp8_tr=rf8.predict(ps["eta_chieff"])
joblib.dump({"rf":rf8}, os.path.join(d,"saved_model/model.joblib"))
rec(8,"rf_eta","eta_chieff",loss8,yp8,yp8_tr,rt8,500*200,"RF 500 trees eta")

# 9. GBR eta
print("\n=== 9: GBR eta ===")
d = md(9, "gbr_eta")
gbr9 = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
gbr9.fit(ps["eta_chieff"], y_train)
t0=time.time(); yp9=gbr9.predict(pv["eta_chieff"]); rt9=(time.time()-t0)/N_VA*1000
loss9=nrmse(yp9, y_val)
yp9_tr=gbr9.predict(ps["eta_chieff"])
joblib.dump({"gbr":gbr9}, os.path.join(d,"saved_model/model.joblib"))
rec(9,"gbr_eta","eta_chieff",loss9,yp9,yp9_tr,rt9,500,"GBR 500 est eta")

# 10. Poly3 raw
print("\n=== 10: Poly3 raw ===")
d = md(10, "poly3_raw")
p10 = PolynomialFeatures(3); X10p = p10.fit_transform(Xs); Xv10p = p10.transform(Xvs)
r10 = Ridge(alpha=1.0).fit(X10p, ys)
t0=time.time(); yp10=sy.inverse_transform(r10.predict(Xv10p).reshape(-1,1)).ravel(); rt10=(time.time()-t0)/N_VA*1000
loss10=nrmse(yp10, y_val)
yp10_tr=sy.inverse_transform(r10.predict(X10p).reshape(-1,1)).ravel()
joblib.dump({"ridge":r10,"poly":p10,"scaler_X":sX,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(10,"poly3_raw","raw",loss10,yp10,yp10_tr,rt10,X10p.shape[1],"Poly-3 + Ridge")

# 11. Poly4 eta
print("\n=== 11: Poly4 eta ===")
d = md(11, "poly4_eta")
p11 = PolynomialFeatures(4); X11p = p11.fit_transform(X2s); Xv11p = p11.transform(Xv2s)
r11 = Ridge(alpha=10.0).fit(X11p, ys)
t0=time.time(); yp11=sy.inverse_transform(r11.predict(Xv11p).reshape(-1,1)).ravel(); rt11=(time.time()-t0)/N_VA*1000
loss11=nrmse(yp11, y_val)
yp11_tr=sy.inverse_transform(r11.predict(X11p).reshape(-1,1)).ravel()
joblib.dump({"ridge":r11,"poly":p11,"scaler_X":sX2,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(11,"poly4_eta","eta_chieff",loss11,yp11,yp11_tr,rt11,X11p.shape[1],"Poly-4 + Ridge eta")

# 12. KNN raw
print("\n=== 12: KNN raw ===")
d = md(12, "knn_raw")
knn12 = KNeighborsRegressor(n_neighbors=7, weights='distance').fit(Xs, y_train)
t0=time.time(); yp12=knn12.predict(Xvs); rt12=(time.time()-t0)/N_VA*1000
loss12=nrmse(yp12, y_val)
yp12_tr=knn12.predict(Xs)
joblib.dump({"knn":knn12,"scaler_X":sX}, os.path.join(d,"saved_model/model.joblib"))
rec(12,"knn_raw","raw",loss12,yp12,yp12_tr,rt12,0,"7-NN distance-weighted")

# 13. RBF interp raw
print("\n=== 13: RBF interp raw ===")
d = md(13, "rbf_interp_raw")
rb13 = RBFInterpolator(Xs, y_train, kernel='thin_plate_spline', smoothing=0.01)
t0=time.time(); yp13=rb13(Xvs); rt13=(time.time()-t0)/N_VA*1000
loss13=nrmse(yp13, y_val)
yp13_tr=rb13(Xs)
joblib.dump({"rbf":rb13,"scaler_X":sX}, os.path.join(d,"saved_model/model.joblib"))
rec(13,"rbf_interp_raw","raw",loss13,yp13,yp13_tr,rt13,1000,"RBF TPS interp")

# 14. RBF interp eta
print("\n=== 14: RBF interp eta ===")
d = md(14, "rbf_interp_eta")
rb14 = RBFInterpolator(X2s, y_train, kernel='thin_plate_spline', smoothing=0.01)
t0=time.time(); yp14=rb14(Xv2s); rt14=(time.time()-t0)/N_VA*1000
loss14=nrmse(yp14, y_val)
yp14_tr=rb14(X2s)
joblib.dump({"rbf":rb14,"scaler_X":sX2}, os.path.join(d,"saved_model/model.joblib"))
rec(14,"rbf_interp_eta","eta_chieff",loss14,yp14,yp14_tr,rt14,1000,"RBF TPS interp eta")

# 15. ET raw
print("\n=== 15: ET raw ===")
d = md(15, "et_raw")
et15 = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
et15.fit(ps["raw"], y_train)
t0=time.time(); yp15=et15.predict(pv["raw"]); rt15=(time.time()-t0)/N_VA*1000
loss15=nrmse(yp15, y_val)
yp15_tr=et15.predict(ps["raw"])
joblib.dump({"et":et15}, os.path.join(d,"saved_model/model.joblib"))
rec(15,"et_raw","raw",loss15,yp15,yp15_tr,rt15,500*200,"ExtraTrees 500")

# 16. Lasso poly3 eta
print("\n=== 16: Lasso poly3 eta ===")
d = md(16, "lasso_poly3_eta")
p16 = PolynomialFeatures(3); X16p = p16.fit_transform(X2s); Xv16p = p16.transform(Xv2s)
la16 = Lasso(alpha=0.0001, max_iter=5000).fit(X16p, ys)
t0=time.time(); yp16=sy.inverse_transform(la16.predict(Xv16p).reshape(-1,1)).ravel(); rt16=(time.time()-t0)/N_VA*1000
loss16=nrmse(yp16, y_val)
yp16_tr=sy.inverse_transform(la16.predict(X16p).reshape(-1,1)).ravel()
joblib.dump({"lasso":la16,"poly":p16,"scaler_X":sX2,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(16,"lasso_poly3_eta","eta_chieff",loss16,yp16,yp16_tr,rt16,X16p.shape[1],"Lasso poly-3 eta")

# 17. BayRidge mass_diff
print("\n=== 17: BayRidge mdiff ===")
d = md(17, "bayridge_mdiff")
sX17 = StandardScaler().fit(ps["mass_diff"]); X17s = sX17.transform(ps["mass_diff"]); Xv17s = sX17.transform(pv["mass_diff"])
p17 = PolynomialFeatures(3); X17p = p17.fit_transform(X17s); Xv17p = p17.transform(Xv17s)
br17 = BayesianRidge(max_iter=500).fit(X17p, ys)
t0=time.time(); yp17=sy.inverse_transform(br17.predict(Xv17p).reshape(-1,1)).ravel(); rt17=(time.time()-t0)/N_VA*1000
loss17=nrmse(yp17, y_val)
yp17_tr=sy.inverse_transform(br17.predict(X17p).reshape(-1,1)).ravel()
joblib.dump({"br":br17,"poly":p17,"scaler_X":sX17,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(17,"bayridge_mdiff","mass_diff",loss17,yp17,yp17_tr,rt17,X17p.shape[1],"BayRidge poly-3 mass_diff")

# 18. AdaBoost eta
print("\n=== 18: AdaBoost eta ===")
d = md(18, "adaboost_eta")
ab18 = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=8),
                         n_estimators=200, learning_rate=0.1, random_state=42)
ab18.fit(ps["eta_chieff"], y_train)
t0=time.time(); yp18=ab18.predict(pv["eta_chieff"]); rt18=(time.time()-t0)/N_VA*1000
loss18=nrmse(yp18, y_val)
yp18_tr=ab18.predict(ps["eta_chieff"])
joblib.dump({"ada":ab18}, os.path.join(d,"saved_model/model.joblib"))
rec(18,"adaboost_eta","eta_chieff",loss18,yp18,yp18_tr,rt18,200,"AdaBoost DT-8 eta")

# 19. MLP large spherical
print("\n=== 19: MLP sph ===")
d = md(19, "mlp_sph")
sX19 = StandardScaler().fit(ps["spherical"]); X19s = sX19.transform(ps["spherical"]); Xv19s = sX19.transform(pv["spherical"])
mlp19 = MLPRegressor(hidden_layer_sizes=(512,512,256,128), max_iter=5000, early_stopping=True,
                     validation_fraction=0.15, random_state=42, learning_rate_init=0.0003)
mlp19.fit(X19s, ys)
t0=time.time(); yp19=sy.inverse_transform(mlp19.predict(Xv19s).reshape(-1,1)).ravel(); rt19=(time.time()-t0)/N_VA*1000
loss19=nrmse(yp19, y_val)
yp19_tr=sy.inverse_transform(mlp19.predict(X19s).reshape(-1,1)).ravel()
joblib.dump({"mlp":mlp19,"scaler_X":sX19,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(19,"mlp_sph","spherical",loss19,yp19,yp19_tr,rt19,sum(c.size for c in mlp19.coefs_),"MLP [512,512,256,128] spherical")

# 20. Bagging RF eta
print("\n=== 20: Bagging RF ===")
d = md(20, "bagging_rf_eta")
bag20 = BaggingRegressor(estimator=RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
                         n_estimators=10, random_state=42, n_jobs=-1)
bag20.fit(ps["eta_chieff"], y_train)
t0=time.time(); yp20=bag20.predict(pv["eta_chieff"]); rt20=(time.time()-t0)/N_VA*1000
loss20=nrmse(yp20, y_val)
yp20_tr=bag20.predict(ps["eta_chieff"])
joblib.dump({"bag":bag20}, os.path.join(d,"saved_model/model.joblib"))
rec(20,"bagging_rf_eta","eta_chieff",loss20,yp20,yp20_tr,rt20,1000,"Bagging RF eta")

# 21. ElasticNet pn_inspired
print("\n=== 21: ElasticNet PN ===")
d = md(21, "enet_pn")
sX21 = StandardScaler().fit(ps["pn_inspired"]); X21s = sX21.transform(ps["pn_inspired"]); Xv21s = sX21.transform(pv["pn_inspired"])
p21 = PolynomialFeatures(4); X21p = p21.fit_transform(X21s); Xv21p = p21.transform(Xv21s)
en21 = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000).fit(X21p, ys)
t0=time.time(); yp21=sy.inverse_transform(en21.predict(Xv21p).reshape(-1,1)).ravel(); rt21=(time.time()-t0)/N_VA*1000
loss21=nrmse(yp21, y_val)
yp21_tr=sy.inverse_transform(en21.predict(X21p).reshape(-1,1)).ravel()
joblib.dump({"enet":en21,"poly":p21,"scaler_X":sX21,"scaler_y":sy}, os.path.join(d,"saved_model/model.joblib"))
rec(21,"enet_pn","pn_inspired",loss21,yp21,yp21_tr,rt21,X21p.shape[1],"ElasticNet poly-4 PN-inspired")

# 22. GBR mass_diff
print("\n=== 22: GBR mdiff ===")
d = md(22, "gbr_mdiff")
gbr22 = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, random_state=42)
gbr22.fit(ps["mass_diff"], y_train)
t0=time.time(); yp22=gbr22.predict(pv["mass_diff"]); rt22=(time.time()-t0)/N_VA*1000
loss22=nrmse(yp22, y_val)
yp22_tr=gbr22.predict(ps["mass_diff"])
joblib.dump({"gbr":gbr22}, os.path.join(d,"saved_model/model.joblib"))
rec(22,"gbr_mdiff","mass_diff",loss22,yp22,yp22_tr,rt22,500,"GBR 500 mass_diff")

# ═══════ SYMBOLIC ═══════
# 23. PySR raw
print("\n=== 23: PySR raw ===")
d = md(23, "pysr_raw")
from pysr import PySRRegressor
m23 = PySRRegressor(niterations=60, binary_operators=["+","-","*","/"],
                    unary_operators=["sqrt","exp","sin","cos"], maxsize=25, populations=15,
                    procs=1, verbosity=0, progress=False, random_state=42, temp_equation_file=True,
                    loss="loss(prediction, target) = abs(prediction - target)")
m23.fit(Xs, y_train)
t0=time.time(); yp23=m23.predict(Xvs); rt23=(time.time()-t0)/N_VA*1000
loss23=nrmse(yp23, y_val)
yp23_tr=m23.predict(Xs)
try:
    expr23 = str(m23.sympy())
    eqs23 = [{"expression":str(r.get("equation","")),"complexity":int(r.get("complexity",0)),
              "loss":float(r.get("loss",0))} for _,r in m23.equations_.iterrows()]
except: expr23 = "error"; eqs23 = []
with open(os.path.join(d,"saved_model/expressions.json"),"w") as f:
    json.dump({"best":expr23,"pareto_front":eqs23}, f, indent=2, default=str)
joblib.dump(m23, os.path.join(d,"saved_model/pysr_model.joblib"))
rec(23,"pysr_raw","raw",loss23,yp23,yp23_tr,rt23,25,"PySR symbolic regression raw")

# 24. PySR eta
print("\n=== 24: PySR eta ===")
d = md(24, "pysr_eta")
m24 = PySRRegressor(niterations=60, binary_operators=["+","-","*","/"],
                    unary_operators=["sqrt","exp","sin","cos"], maxsize=25, populations=15,
                    procs=1, verbosity=0, progress=False, random_state=42, temp_equation_file=True,
                    loss="loss(prediction, target) = abs(prediction - target)")
m24.fit(X2s, y_train)
t0=time.time(); yp24=m24.predict(Xv2s); rt24=(time.time()-t0)/N_VA*1000
loss24=nrmse(yp24, y_val)
yp24_tr=m24.predict(X2s)
try:
    expr24 = str(m24.sympy())
    eqs24 = [{"expression":str(r.get("equation","")),"complexity":int(r.get("complexity",0)),
              "loss":float(r.get("loss",0))} for _,r in m24.equations_.iterrows()]
except: expr24 = "error"; eqs24 = []
with open(os.path.join(d,"saved_model/expressions.json"),"w") as f:
    json.dump({"best":expr24,"pareto_front":eqs24}, f, indent=2, default=str)
joblib.dump(m24, os.path.join(d,"saved_model/pysr_model.joblib"))
rec(24,"pysr_eta","eta_chieff",loss24,yp24,yp24_tr,rt24,25,"PySR symbolic regression eta")

# 25. gplearn raw
print("\n=== 25: gplearn raw ===")
d = md(25, "gplearn_raw")
from gplearn.genetic import SymbolicRegressor
est25 = SymbolicRegressor(population_size=2000, generations=30, tournament_size=20,
                          function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                          metric='mse', parsimony_coefficient=0.001, verbose=0, random_state=42)
est25.fit(Xs, y_train)
t0=time.time(); yp25=est25.predict(Xvs); rt25=(time.time()-t0)/N_VA*1000
loss25=nrmse(yp25, y_val)
yp25_tr=est25.predict(Xs)
with open(os.path.join(d,"saved_model/expressions.json"),"w") as f:
    json.dump({"expression":str(est25._program),"complexity":est25._program.length_,
               "fitness":float(est25._program.fitness_)}, f, indent=2, default=str)
joblib.dump(est25, os.path.join(d,"saved_model/gplearn_model.joblib"))
rec(25,"gplearn_raw","raw",loss25,yp25,yp25_tr,rt25,30,"gplearn symbolic regression raw")

# 26. gplearn eta
print("\n=== 26: gplearn eta ===")
d = md(26, "gplearn_eta")
est26 = SymbolicRegressor(population_size=2000, generations=30, tournament_size=20,
                          function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                          metric='mse', parsimony_coefficient=0.001, verbose=0, random_state=42)
est26.fit(X2s, y_train)
t0=time.time(); yp26=est26.predict(Xv2s); rt26=(time.time()-t0)/N_VA*1000
loss26=nrmse(yp26, y_val)
yp26_tr=est26.predict(X2s)
with open(os.path.join(d,"saved_model/expressions.json"),"w") as f:
    json.dump({"expression":str(est26._program),"complexity":est26._program.length_,
               "fitness":float(est26._program.fitness_)}, f, indent=2, default=str)
joblib.dump(est26, os.path.join(d,"saved_model/gplearn_model.joblib"))
rec(26,"gplearn_eta","eta_chieff",loss26,yp26,yp26_tr,rt26,30,"gplearn symbolic regression eta")


# ═══════ SAVE & PLOTS ═══════
print("\n=== Saving ===")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ROOT)
import gwbenchmarks.plot_settings as pset; pset.apply()

os.makedirs(os.path.join(WORK_DIR, "comparison"), exist_ok=True)

with open(os.path.join(WORK_DIR, "comparison/error_data.json"), "w") as f:
    json.dump(err_data, f)
results.sort(key=lambda x: x["loss"])
with open(os.path.join(WORK_DIR, "comparison/summary_table.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(WORK_DIR, "comparison/best_model.json"), "w") as f:
    json.dump(results[0], f, indent=2)
with open(os.path.join(WORK_DIR, "CHANGELOG.md"), "w") as f:
    f.write("# Remnant Benchmark CHANGELOG — Opus 4.6\n\n")
    for c in cl: f.write(c + "\n")

# Generate train.py / predict.py
for r in results:
    n = r["approach_number"]; name = r["approach"]; sch = r["parameterization"]
    dpath = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    with open(os.path.join(dpath, "train.py"), "w") as f:
        f.write(f'#!/usr/bin/env python3\n"""Training script for {name}."""\n# See build_all.py approach {n}\n')
    with open(os.path.join(dpath, "predict.py"), "w") as f:
        f.write(f'''#!/usr/bin/env python3
"""Prediction function for {name}."""
import os, numpy as np, joblib
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

def predict(params_raw):
    """Predict kick velocity from raw 7D parameters (q, chi1x..chi2z)."""
    saved = joblib.load(os.path.join(WORK_DIR, "saved_model/model.joblib"))
    X = np.atleast_2d(params_raw)
    if "scaler_X" in saved:
        X = saved["scaler_X"].transform(X)
    if "poly" in saved:
        X = saved["poly"].transform(X)
    model_key = [k for k in saved if k not in ("scaler_X","scaler_y","poly")][0]
    model = saved[model_key]
    y = model(X) if callable(model) else model.predict(X)
    if "scaler_y" in saved:
        y = saved["scaler_y"].inverse_transform(y.reshape(-1,1)).ravel()
    return y
''')

# Plots
names = [r["approach"] for r in results]
losses = [r["loss"] for r in results]
rts = [r["runtime_ms"] for r in results]

CAT_COLORS = {"kernel":pset.COLORS["blue"],"symbolic":pset.COLORS["green"],
              "interp":pset.COLORS["orange"],"ml":pset.COLORS["red"]}
def cat(n):
    nl = n.lower()
    if any(x in nl for x in ["pysr","gplearn"]): return "symbolic"
    if any(x in nl for x in ["rbf_interp","knn"]): return "interp"
    if any(x in nl for x in ["gpr","krr","svr"]): return "kernel"
    return "ml"
cats = [cat(n) for n in names]
cols = [CAT_COLORS[c] for c in cats]

# Progress
fig, ax = plt.subplots(figsize=pset.figsize(2, 0.5))
y_pos = np.arange(len(results))
ax.barh(y_pos, losses, color=cols, height=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels([n.replace("_"," ") for n in names], fontsize=6)
ax.set_xlabel("NRMSE"); ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "comparison/progress.png"))
fig.savefig(os.path.join(WORK_DIR, "comparison/progress.pdf")); plt.close()

# Pareto
fig, ax = plt.subplots(figsize=pset.figsize(1, 0.9))
for c in ["kernel","symbolic","interp","ml"]:
    idx = [i for i,x in enumerate(cats) if x==c]
    if idx:
        ax.scatter([losses[i] for i in idx], [rts[i] for i in idx], c=CAT_COLORS[c], label=c.upper(), s=30, zorder=5)
        for i in idx:
            ax.annotate(names[i].replace("_"," "), (losses[i], rts[i]), fontsize=4.5, ha='left', rotation=10)
ax.set_xlabel("NRMSE"); ax.set_ylabel("Runtime (ms/sample)"); ax.set_xscale("log"); ax.set_yscale("log")
ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "comparison/pareto_accuracy_speed.png"))
fig.savefig(os.path.join(WORK_DIR, "comparison/pareto_accuracy_speed.pdf")); plt.close()

# Loss comparison
fig, ax = plt.subplots(figsize=pset.figsize(2, 0.5))
ax.barh(y_pos, losses, color=cols, height=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels([n.replace("_"," ") for n in names], fontsize=6)
ax.set_xlabel("NRMSE"); ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "comparison/loss_only_comparison.png"))
fig.savefig(os.path.join(WORK_DIR, "comparison/loss_only_comparison.pdf")); plt.close()

# Error histograms
fig, ax = plt.subplots(figsize=pset.figsize(2, 0.7))
bins = np.linspace(0, y_train.max()*0.5, 50)
for i, (name, data) in enumerate(sorted(err_data.items(), key=lambda x: np.mean(x[1]["val_losses"]))[:10]):
    ax.hist(data["val_losses"], bins=bins, alpha=0.4, label=name.replace("_"," "), histtype='stepfilled', linewidth=0.5)
ax.set_xlabel("Absolute error"); ax.set_ylabel("Count"); ax.legend(fontsize=5, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "comparison/error_histograms.png"))
fig.savefig(os.path.join(WORK_DIR, "comparison/error_histograms.pdf")); plt.close()

print(f"\n=== DONE ({len(results)} approaches) ===")
for r in results:
    print(f"  {r['approach_number']:02d} {r['approach']:<25s} NRMSE={r['loss']:.6f}")
print(f"Best: {results[0]['approach']} NRMSE={results[0]['loss']:.6f}")
