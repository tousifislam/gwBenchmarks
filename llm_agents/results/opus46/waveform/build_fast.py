#!/usr/bin/env python3
"""Build all 22 non-symbolic models. Fast approaches first, GPR last."""

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
from sklearn.multioutput import MultiOutputRegressor

print("Loading data...")
params_train, wf_train, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_training.h5"))
params_val, wf_val, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_validation.h5"))
N_TR, N_VA = len(params_train), len(params_val)

N_BASIS = 40
print(f"SVD ({N_BASIS} basis)...")
wr, wi = np.real(wf_train), np.imag(wf_train)
cr, br, mr, sr = compute_svd(wr, N_BASIS)
ci, bi, mi, si = compute_svd(wi, N_BASIS)
cr_v = project_onto_basis(np.real(wf_val), br, mr)
ci_v = project_onto_basis(np.imag(wf_val), bi, mi)

svd_dir = os.path.join(WORK_DIR, "shared_svd")
os.makedirs(svd_dir, exist_ok=True)
np.savez(os.path.join(svd_dir, "svd_basis.npz"),
         basis_r=br, basis_i=bi, mean_r=mr, mean_i=mi, sv_r=sr[:N_BASIS], sv_i=si[:N_BASIS])

Y = np.hstack([cr, ci])
Yv = np.hstack([cr_v, ci_v])

ps = {k: reparameterize(params_train, k) for k in ["raw","eta_chieff","spherical","mass_diff"]}
pv = {k: reparameterize(params_val, k) for k in ["raw","eta_chieff","spherical","mass_diff"]}

results = []
err_data = {}
cl = []

def md(n, name):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    os.makedirs(os.path.join(d, "saved_model"), exist_ok=True)
    return d

def ev(yp, ref):
    nb = N_BASIS
    if yp.shape[1] < 2*nb:
        f = np.zeros((len(yp), 2*nb)); f[:, :yp.shape[1]] = yp; yp = f
    wf = reconstruct_from_basis(yp[:,:nb], br, mr) + 1j*reconstruct_from_basis(yp[:,nb:], bi, mi)
    l, m = compute_loss_batch(wf, ref, DT)
    return m, l

def rec(n, name, sch, loss, lv, lt, rt, np_, notes):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    save_scorecard(d, name, n, sch, "t0_at_peak", float(loss),
                   {"mean_mismatch": float(loss)}, rt, N_TR, N_VA, np_, notes)
    results.append({"approach":name,"approach_number":n,"loss":float(loss),"runtime_ms":rt,"parameterization":sch,"notes":notes})
    err_data[name] = {"val_losses":[float(x) for x in lv], "train_losses":[float(x) for x in lt]}
    cl.append(f"## {n}: {name}\n- Param: {sch}, Loss: {loss:.6f}, RT: {rt:.1f}ms\n- {notes}\n")
    print(f"  [{n:02d}] {name}: loss={loss:.6f}")

# ═══════ FAST APPROACHES FIRST ═══════

# 3. Poly3 raw
print("\n=== 3: Poly3 raw ===")
d = md(3, "svd_poly3_raw")
from sklearn.linear_model import Ridge
sX = StandardScaler().fit(ps["raw"]); Xs, Xvs = sX.transform(ps["raw"]), sX.transform(pv["raw"])
p3 = PolynomialFeatures(3); X3 = p3.fit_transform(Xs); Xv3 = p3.transform(Xvs)
sY = StandardScaler().fit(Y); Ys = sY.transform(Y)
r3 = Ridge(alpha=1.0).fit(X3, Ys)
t0=time.time(); yp=sY.inverse_transform(r3.predict(Xv3)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY.inverse_transform(r3.predict(X3)), wf_train)
joblib.dump({"ridge":r3,"poly":p3,"scaler_X":sX,"scaler_y":sY}, os.path.join(d,"saved_model/model.joblib"))
rec(3,"svd_poly3_raw","raw",l,lv,lt,rt,X3.shape[1]*80,"SVD(40)+deg-3 poly+Ridge")

# 4. MLP raw
print("\n=== 4: MLP raw ===")
d = md(4, "svd_mlp_raw")
from sklearn.neural_network import MLPRegressor
sX4 = StandardScaler().fit(ps["raw"]); X4s, Xv4s = sX4.transform(ps["raw"]), sX4.transform(pv["raw"])
sY4 = StandardScaler().fit(Y); Y4s = sY4.transform(Y)
m4 = MLPRegressor(hidden_layer_sizes=(256,128,64), max_iter=2000, early_stopping=True,
                  validation_fraction=0.15, random_state=42, learning_rate_init=0.001)
m4.fit(X4s, Y4s)
t0=time.time(); yp=sY4.inverse_transform(m4.predict(Xv4s)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY4.inverse_transform(m4.predict(X4s)), wf_train)
joblib.dump({"mlp":m4,"scaler_X":sX4,"scaler_y":sY4}, os.path.join(d,"saved_model/model.joblib"))
rec(4,"svd_mlp_raw","raw",l,lv,lt,rt,sum(c.size for c in m4.coefs_),"SVD(40)+MLP[256,128,64]")

# 5. RF raw
print("\n=== 5: RF raw ===")
d = md(5, "svd_rf_raw")
from sklearn.ensemble import RandomForestRegressor
rf5 = RandomForestRegressor(200, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf5.fit(ps["raw"], Y)
t0=time.time(); yp=rf5.predict(pv["raw"]); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(rf5.predict(ps["raw"]), wf_train)
joblib.dump({"rf":rf5}, os.path.join(d,"saved_model/model.joblib"))
rec(5,"svd_rf_raw","raw",l,lv,lt,rt,rf5.n_estimators*200,"SVD(40)+RF200")

# 6. GBR eta
print("\n=== 6: GBR eta ===")
d = md(6, "svd_gbr_eta")
from sklearn.ensemble import GradientBoostingRegressor
g6 = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42), n_jobs=-1)
g6.fit(ps["eta_chieff"], Y)
t0=time.time(); yp=g6.predict(pv["eta_chieff"]); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(g6.predict(ps["eta_chieff"]), wf_train)
joblib.dump({"gbr":g6}, os.path.join(d,"saved_model/model.joblib"))
rec(6,"svd_gbr_eta","eta_chieff",l,lv,lt,rt,200*80,"SVD(40)+GBR eta")

# 7. KRR raw
print("\n=== 7: KRR raw ===")
d = md(7, "svd_krr_raw")
from sklearn.kernel_ridge import KernelRidge
sX7 = StandardScaler().fit(ps["raw"]); X7s, Xv7s = sX7.transform(ps["raw"]), sX7.transform(pv["raw"])
sY7 = StandardScaler().fit(Y); Y7s = sY7.transform(Y)
k7 = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1).fit(X7s, Y7s)
t0=time.time(); yp=sY7.inverse_transform(k7.predict(Xv7s)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY7.inverse_transform(k7.predict(X7s)), wf_train)
joblib.dump({"krr":k7,"scaler_X":sX7,"scaler_y":sY7}, os.path.join(d,"saved_model/model.joblib"))
rec(7,"svd_krr_raw","raw",l,lv,lt,rt,250*80,"SVD(40)+KRR RBF")

# 8. RBF interp raw
print("\n=== 8: RBF raw ===")
d = md(8, "svd_rbf_interp_raw")
from scipy.interpolate import RBFInterpolator
sX8 = StandardScaler().fit(ps["raw"]); X8s, Xv8s = sX8.transform(ps["raw"]), sX8.transform(pv["raw"])
sY8 = StandardScaler().fit(Y); Y8s = sY8.transform(Y)
rb8 = RBFInterpolator(X8s, Y8s, kernel='thin_plate_spline', smoothing=0.1)
t0=time.time(); yp=sY8.inverse_transform(rb8(Xv8s)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY8.inverse_transform(rb8(X8s)), wf_train)
joblib.dump({"rbf":rb8,"scaler_X":sX8,"scaler_y":sY8}, os.path.join(d,"saved_model/model.joblib"))
rec(8,"svd_rbf_interp_raw","raw",l,lv,lt,rt,250*80,"SVD(40)+RBF TPS")

# 9. KNN raw
print("\n=== 9: KNN raw ===")
d = md(9, "svd_knn_raw")
from sklearn.neighbors import KNeighborsRegressor
sX9 = StandardScaler().fit(ps["raw"]); X9s, Xv9s = sX9.transform(ps["raw"]), sX9.transform(pv["raw"])
kn9 = KNeighborsRegressor(5, weights='distance').fit(X9s, Y)
t0=time.time(); yp=kn9.predict(Xv9s); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(kn9.predict(X9s), wf_train)
joblib.dump({"knn":kn9,"scaler_X":sX9}, os.path.join(d,"saved_model/model.joblib"))
rec(9,"svd_knn_raw","raw",l,lv,lt,rt,0,"SVD(40)+5-NN dist-wt")

# 10. MLP eta
print("\n=== 10: MLP eta ===")
d = md(10, "svd_mlp_eta")
sX10 = StandardScaler().fit(ps["eta_chieff"]); X10s, Xv10s = sX10.transform(ps["eta_chieff"]), sX10.transform(pv["eta_chieff"])
sY10 = StandardScaler().fit(Y); Y10s = sY10.transform(Y)
m10 = MLPRegressor(hidden_layer_sizes=(512,256,128), max_iter=3000, early_stopping=True,
                   validation_fraction=0.15, random_state=42, learning_rate_init=0.0005)
m10.fit(X10s, Y10s)
t0=time.time(); yp=sY10.inverse_transform(m10.predict(Xv10s)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY10.inverse_transform(m10.predict(X10s)), wf_train)
joblib.dump({"mlp":m10,"scaler_X":sX10,"scaler_y":sY10}, os.path.join(d,"saved_model/model.joblib"))
rec(10,"svd_mlp_eta","eta_chieff",l,lv,lt,rt,sum(c.size for c in m10.coefs_),"SVD(40)+MLP[512,256,128] eta")

# 11. Poly4 eta
print("\n=== 11: Poly4 eta ===")
d = md(11, "svd_poly4_eta")
sX11 = StandardScaler().fit(ps["eta_chieff"]); X11s, Xv11s = sX11.transform(ps["eta_chieff"]), sX11.transform(pv["eta_chieff"])
p11 = PolynomialFeatures(4); X11p = p11.fit_transform(X11s); Xv11p = p11.transform(Xv11s)
sY11 = StandardScaler().fit(Y); Y11s = sY11.transform(Y)
r11 = Ridge(alpha=10.0).fit(X11p, Y11s)
t0=time.time(); yp=sY11.inverse_transform(r11.predict(Xv11p)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY11.inverse_transform(r11.predict(X11p)), wf_train)
joblib.dump({"ridge":r11,"poly":p11,"scaler_X":sX11,"scaler_y":sY11}, os.path.join(d,"saved_model/model.joblib"))
rec(11,"svd_poly4_eta","eta_chieff",l,lv,lt,rt,X11p.shape[1]*80,"SVD(40)+deg-4 poly eta")

# 13. RF eta
print("\n=== 13: RF eta ===")
d = md(13, "svd_rf_eta")
rf13 = RandomForestRegressor(500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf13.fit(ps["eta_chieff"], Y)
t0=time.time(); yp=rf13.predict(pv["eta_chieff"]); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(rf13.predict(ps["eta_chieff"]), wf_train)
joblib.dump({"rf":rf13}, os.path.join(d,"saved_model/model.joblib"))
rec(13,"svd_rf_eta","eta_chieff",l,lv,lt,rt,rf13.n_estimators*200,"SVD(40)+RF500 eta")

# 14. ExtraTrees sph
print("\n=== 14: ET sph ===")
d = md(14, "svd_et_sph")
from sklearn.ensemble import ExtraTreesRegressor
et14 = ExtraTreesRegressor(500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
et14.fit(ps["spherical"], Y)
t0=time.time(); yp=et14.predict(pv["spherical"]); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(et14.predict(ps["spherical"]), wf_train)
joblib.dump({"et":et14}, os.path.join(d,"saved_model/model.joblib"))
rec(14,"svd_et_sph","spherical",l,lv,lt,rt,et14.n_estimators*200,"SVD(40)+ET500 sph")

# 15. SVR mass_diff
print("\n=== 15: SVR mdiff ===")
d = md(15, "svd_svr_mdiff")
from sklearn.svm import SVR
sX15 = StandardScaler().fit(ps["mass_diff"]); X15s, Xv15s = sX15.transform(ps["mass_diff"]), sX15.transform(pv["mass_diff"])
sY15 = StandardScaler().fit(Y); Y15s = sY15.transform(Y)
sv15 = MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.01), n_jobs=-1)
sv15.fit(X15s, Y15s)
t0=time.time(); yp=sY15.inverse_transform(sv15.predict(Xv15s)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY15.inverse_transform(sv15.predict(X15s)), wf_train)
joblib.dump({"svr":sv15,"scaler_X":sX15,"scaler_y":sY15}, os.path.join(d,"saved_model/model.joblib"))
rec(15,"svd_svr_mdiff","mass_diff",l,lv,lt,rt,250*80,"SVD(40)+SVR mdiff")

# 16. Lasso raw
print("\n=== 16: Lasso raw ===")
d = md(16, "svd_lasso_raw")
from sklearn.linear_model import Lasso
sX16 = StandardScaler().fit(ps["raw"]); X16s, Xv16s = sX16.transform(ps["raw"]), sX16.transform(pv["raw"])
p16 = PolynomialFeatures(3); X16p = p16.fit_transform(X16s); Xv16p = p16.transform(Xv16s)
sY16 = StandardScaler().fit(Y); Y16s = sY16.transform(Y)
la16 = MultiOutputRegressor(Lasso(alpha=0.001, max_iter=5000), n_jobs=-1)
la16.fit(X16p, Y16s)
t0=time.time(); yp=sY16.inverse_transform(la16.predict(Xv16p)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY16.inverse_transform(la16.predict(X16p)), wf_train)
joblib.dump({"lasso":la16,"poly":p16,"scaler_X":sX16,"scaler_y":sY16}, os.path.join(d,"saved_model/model.joblib"))
rec(16,"svd_lasso_raw","raw",l,lv,lt,rt,X16p.shape[1]*80,"SVD(40)+Lasso poly3")

# 17. AdaBoost eta
print("\n=== 17: AdaBoost eta ===")
d = md(17, "svd_adaboost_eta")
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
ab17 = MultiOutputRegressor(AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=6),
       n_estimators=100, learning_rate=0.1, random_state=42), n_jobs=-1)
ab17.fit(ps["eta_chieff"], Y)
t0=time.time(); yp=ab17.predict(pv["eta_chieff"]); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(ab17.predict(ps["eta_chieff"]), wf_train)
joblib.dump({"ada":ab17}, os.path.join(d,"saved_model/model.joblib"))
rec(17,"svd_adaboost_eta","eta_chieff",l,lv,lt,rt,100*80,"SVD(40)+AdaBoost eta")

# 18. MLP large sph
print("\n=== 18: MLP-large sph ===")
d = md(18, "svd_mlp_large_sph")
sX18 = StandardScaler().fit(ps["spherical"]); X18s, Xv18s = sX18.transform(ps["spherical"]), sX18.transform(pv["spherical"])
sY18 = StandardScaler().fit(Y); Y18s = sY18.transform(Y)
m18 = MLPRegressor(hidden_layer_sizes=(512,512,256,128), max_iter=5000, early_stopping=True,
                   validation_fraction=0.15, random_state=42, learning_rate_init=0.0003, batch_size=32)
m18.fit(X18s, Y18s)
t0=time.time(); yp=sY18.inverse_transform(m18.predict(Xv18s)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY18.inverse_transform(m18.predict(X18s)), wf_train)
joblib.dump({"mlp":m18,"scaler_X":sX18,"scaler_y":sY18}, os.path.join(d,"saved_model/model.joblib"))
rec(18,"svd_mlp_large_sph","spherical",l,lv,lt,rt,sum(c.size for c in m18.coefs_),"SVD(40)+MLP[512,512,256,128] sph")

# 19. AmpPhase RF eta
print("\n=== 19: AmpPhase RF ===")
d = md(19, "ampphase_rf_eta")
amp_tr = np.abs(wf_train); phase_tr = np.unwrap(np.angle(wf_train), axis=1)
ca, ba, ma, _ = compute_svd(amp_tr, N_BASIS)
cp, bp, mp, _ = compute_svd(phase_tr, N_BASIS)
y_ap = np.hstack([ca, cp])
rf_ap = RandomForestRegressor(300, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_ap.fit(ps["eta_chieff"], y_ap)
t0=time.time()
yap = rf_ap.predict(pv["eta_chieff"])
rt=(time.time()-t0)/N_VA*1000
a_rec = yap[:,:N_BASIS]@ba+ma; p_rec = yap[:,N_BASIS:]@bp+mp
wf19 = a_rec*np.exp(1j*p_rec)
lv19, l19 = compute_loss_batch(wf19, wf_val, DT)
yap_tr = rf_ap.predict(ps["eta_chieff"])
a_tr = yap_tr[:,:N_BASIS]@ba+ma; p_tr = yap_tr[:,N_BASIS:]@bp+mp
lt19, _ = compute_loss_batch(a_tr*np.exp(1j*p_tr), wf_train, DT)
joblib.dump({"rf":rf_ap,"basis_amp":ba,"mean_amp":ma,"basis_phase":bp,"mean_phase":mp},
            os.path.join(d,"saved_model/model.joblib"))
rec(19,"ampphase_rf_eta","eta_chieff",l19,lv19,lt19,rt,rf_ap.n_estimators*200,"AmpPhase SVD(40)+RF300 eta")

# 20. RBF interp eta
print("\n=== 20: RBF eta ===")
d = md(20, "svd_rbf_interp_eta")
sX20 = StandardScaler().fit(ps["eta_chieff"]); X20s, Xv20s = sX20.transform(ps["eta_chieff"]), sX20.transform(pv["eta_chieff"])
sY20 = StandardScaler().fit(Y); Y20s = sY20.transform(Y)
rb20 = RBFInterpolator(X20s, Y20s, kernel='thin_plate_spline', smoothing=0.1)
t0=time.time(); yp=sY20.inverse_transform(rb20(Xv20s)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY20.inverse_transform(rb20(X20s)), wf_train)
joblib.dump({"rbf":rb20,"scaler_X":sX20,"scaler_y":sY20}, os.path.join(d,"saved_model/model.joblib"))
rec(20,"svd_rbf_interp_eta","eta_chieff",l,lv,lt,rt,250*80,"SVD(40)+RBF TPS eta")

# 21. BayRidge mass_diff
print("\n=== 21: BayRidge mdiff ===")
d = md(21, "svd_bayridge_mdiff")
from sklearn.linear_model import BayesianRidge
sX21 = StandardScaler().fit(ps["mass_diff"]); X21s, Xv21s = sX21.transform(ps["mass_diff"]), sX21.transform(pv["mass_diff"])
p21 = PolynomialFeatures(3); X21p = p21.fit_transform(X21s); Xv21p = p21.transform(Xv21s)
sY21 = StandardScaler().fit(Y); Y21s = sY21.transform(Y)
br21 = MultiOutputRegressor(BayesianRidge(max_iter=500), n_jobs=-1)
br21.fit(X21p, Y21s)
t0=time.time(); yp=sY21.inverse_transform(br21.predict(Xv21p)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY21.inverse_transform(br21.predict(X21p)), wf_train)
joblib.dump({"br":br21,"poly":p21,"scaler_X":sX21,"scaler_y":sY21}, os.path.join(d,"saved_model/model.joblib"))
rec(21,"svd_bayridge_mdiff","mass_diff",l,lv,lt,rt,X21p.shape[1]*80,"SVD(40)+BayRidge poly3 mdiff")

# 22. ElasticNet eta
print("\n=== 22: ElasticNet eta ===")
d = md(22, "svd_elasticnet_eta")
from sklearn.linear_model import ElasticNet
sX22 = StandardScaler().fit(ps["eta_chieff"]); X22s, Xv22s = sX22.transform(ps["eta_chieff"]), sX22.transform(pv["eta_chieff"])
p22 = PolynomialFeatures(3); X22p = p22.fit_transform(X22s); Xv22p = p22.transform(Xv22s)
sY22 = StandardScaler().fit(Y); Y22s = sY22.transform(Y)
en22 = MultiOutputRegressor(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000), n_jobs=-1)
en22.fit(X22p, Y22s)
t0=time.time(); yp=sY22.inverse_transform(en22.predict(Xv22p)); rt=(time.time()-t0)/N_VA*1000
l,lv = ev(yp, wf_val); _,lt = ev(sY22.inverse_transform(en22.predict(X22p)), wf_train)
joblib.dump({"enet":en22,"poly":p22,"scaler_X":sX22,"scaler_y":sY22}, os.path.join(d,"saved_model/model.joblib"))
rec(22,"svd_elasticnet_eta","eta_chieff",l,lv,lt,rt,X22p.shape[1]*80,"SVD(40)+ElasticNet poly3 eta")

# ═══════ GPR APPROACHES (slower) ═══════

# 1. GPR RBF raw (no kernel optimization)
print("\n=== 1: GPR RBF raw ===")
d = md(1, "svd_gpr_rbf_raw")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
n_gpr = 5
sX1 = StandardScaler().fit(ps["raw"]); X1s, Xv1s = sX1.transform(ps["raw"]), sX1.transform(pv["raw"])
yt1 = Y[:, :n_gpr]; sY1 = StandardScaler().fit(yt1); Y1s = sY1.transform(yt1)
k1 = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)
gpr1 = MultiOutputRegressor(GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=0, alpha=1e-4, normalize_y=True), n_jobs=-1)
gpr1.fit(X1s, Y1s)
t0=time.time()
yp1 = sY1.inverse_transform(gpr1.predict(Xv1s))
rt=(time.time()-t0)/N_VA*1000
yf = np.zeros((N_VA,2*N_BASIS)); yf[:,:n_gpr]=yp1
l,lv = ev(yf, wf_val)
yp1t = sY1.inverse_transform(gpr1.predict(X1s))
yft = np.zeros((N_TR,2*N_BASIS)); yft[:,:n_gpr]=yp1t
_,lt = ev(yft, wf_train)
joblib.dump({"gpr":gpr1,"scaler_X":sX1,"scaler_y":sY1,"n_basis":n_gpr}, os.path.join(d,"saved_model/model.joblib"))
rec(1,"svd_gpr_rbf_raw","raw",l,lv,lt,rt,n_gpr*250,"SVD(5)+GPR RBF raw (no opt)")

# 2. GPR Matern eta
print("\n=== 2: GPR Matern eta ===")
d = md(2, "svd_gpr_matern_eta")
sX2 = StandardScaler().fit(ps["eta_chieff"]); X2s, Xv2s = sX2.transform(ps["eta_chieff"]), sX2.transform(pv["eta_chieff"])
yt2 = Y[:, :n_gpr]; sY2 = StandardScaler().fit(yt2); Y2s = sY2.transform(yt2)
k2 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
gpr2 = MultiOutputRegressor(GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=0, alpha=1e-4, normalize_y=True), n_jobs=-1)
gpr2.fit(X2s, Y2s)
t0=time.time()
yp2 = sY2.inverse_transform(gpr2.predict(Xv2s))
rt=(time.time()-t0)/N_VA*1000
yf2 = np.zeros((N_VA,2*N_BASIS)); yf2[:,:n_gpr]=yp2
l,lv = ev(yf2, wf_val)
yp2t = sY2.inverse_transform(gpr2.predict(X2s))
yf2t = np.zeros((N_TR,2*N_BASIS)); yf2t[:,:n_gpr]=yp2t
_,lt = ev(yf2t, wf_train)
joblib.dump({"gpr":gpr2,"scaler_X":sX2,"scaler_y":sY2}, os.path.join(d,"saved_model/model.joblib"))
rec(2,"svd_gpr_matern_eta","eta_chieff",l,lv,lt,rt,n_gpr*250,"SVD(5)+GPR Matern eta (no opt)")

# 12. GPR Matern sph
print("\n=== 12: GPR Matern sph ===")
d = md(12, "svd_gpr_matern_sph")
sX12 = StandardScaler().fit(ps["spherical"]); X12s, Xv12s = sX12.transform(ps["spherical"]), sX12.transform(pv["spherical"])
sY12 = StandardScaler().fit(yt1); Y12s = sY12.transform(yt1)
k12 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3)
gpr12 = MultiOutputRegressor(GaussianProcessRegressor(kernel=k12, n_restarts_optimizer=0, alpha=1e-4, normalize_y=True), n_jobs=-1)
gpr12.fit(X12s, Y12s)
t0=time.time()
yp12 = sY12.inverse_transform(gpr12.predict(Xv12s))
rt=(time.time()-t0)/N_VA*1000
yf12 = np.zeros((N_VA,2*N_BASIS)); yf12[:,:n_gpr]=yp12
l,lv = ev(yf12, wf_val)
yp12t = sY12.inverse_transform(gpr12.predict(X12s))
yf12t = np.zeros((N_TR,2*N_BASIS)); yf12t[:,:n_gpr]=yp12t
_,lt = ev(yf12t, wf_train)
joblib.dump({"gpr":gpr12,"scaler_X":sX12,"scaler_y":sY12}, os.path.join(d,"saved_model/model.joblib"))
rec(12,"svd_gpr_matern_sph","spherical",l,lv,lt,rt,n_gpr*250,"SVD(5)+GPR Matern sph (no opt)")

# ═══════ SAVE ═══════
print("\n=== Saving ===")
os.makedirs(os.path.join(WORK_DIR, "comparison"), exist_ok=True)

with open(os.path.join(WORK_DIR, "comparison/error_data.json"), "w") as f:
    json.dump(err_data, f)

results.sort(key=lambda x: x["loss"])
with open(os.path.join(WORK_DIR, "comparison/summary_table.json"), "w") as f:
    json.dump(results, f, indent=2)

best = results[0]
with open(os.path.join(WORK_DIR, "comparison/best_model.json"), "w") as f:
    json.dump(best, f, indent=2)

with open(os.path.join(WORK_DIR, "CHANGELOG.md"), "w") as f:
    f.write("# Waveform Benchmark CHANGELOG — Opus 4.6\n\n")
    for c in cl:
        f.write(c + "\n")

print(f"\n=== DONE ({len(results)} approaches) ===")
for r in results:
    print(f"  {r['approach_number']:02d} {r['approach']:<30s} loss={r['loss']:.6f}")
print(f"Best: {best['approach']} loss={best['loss']:.6f}")
