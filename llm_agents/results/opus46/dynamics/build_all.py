#!/usr/bin/env python3
"""Build all 26 dynamics models: predict x(t) from binary parameters."""

import sys, os, numpy as np, json, time, warnings, joblib
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../.."))
sys.path.insert(0, ROOT)
import h5py
from gwbenchmarks.metrics import rms_relative_error

COMMON_N = 3195

def load_dynamics(path):
    params_list, x_list = [], []
    with h5py.File(path, "r") as f:
        n = f.attrs["n_simulations"]
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            params_list.append([g.attrs["q"], g.attrs["chi1z"], g.attrs["chi2z"],
                                g.attrs["e0"], g.attrs["zeta0"], g.attrs["omega0"]])
            t = g["t"][:]; x = g["x"][:]
            tau = (t - t[0]) / (t[-1] - t[0])
            tau_common = np.linspace(0, 1, COMMON_N)
            x_interp = np.interp(tau_common, tau, x)
            x_list.append(x_interp)
    return np.array(params_list), np.array(x_list)

print("Loading data...")
X_train, x_train = load_dynamics(os.path.join(ROOT, "datasets/dynamics/dynamics_training.h5"))
X_val, x_val = load_dynamics(os.path.join(ROOT, "datasets/dynamics/dynamics_validation.h5"))
N_TR, N_VA = len(X_train), len(X_val)
print(f"Train: {N_TR}, Val: {N_VA}, x shape: {x_train.shape}")

# SVD decomposition
N_BASIS = 30
mean_x = np.mean(x_train, axis=0)
centered = x_train - mean_x
U, s, Vt = np.linalg.svd(centered, full_matrices=False)
basis = Vt[:N_BASIS]
coeffs_train = U[:, :N_BASIS] * s[:N_BASIS]
coeffs_val = (x_val - mean_x) @ basis.T

Y = coeffs_train
Yv = coeffs_val

# Save SVD
os.makedirs(os.path.join(WORK_DIR, "shared_svd"), exist_ok=True)
np.savez(os.path.join(WORK_DIR, "shared_svd/svd_basis.npz"), basis=basis, mean_x=mean_x, sv=s[:N_BASIS])

# Reparameterizations
def reparam(X, scheme="raw"):
    q, chi1z, chi2z, e0, zeta0, omega0 = X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5]
    if scheme == "raw": return X
    eta = q/(1+q)**2; chi_eff = (q*chi1z+chi2z)/(1+q); chi_a = (q*chi1z-chi2z)/(1+q)
    if scheme == "eta_chieff":
        return np.column_stack([eta, chi_eff, chi_a, np.log(e0+1e-10), zeta0, omega0])
    if scheme == "trig_anomaly":
        return np.column_stack([eta, chi_eff, chi_a, e0, np.cos(zeta0), np.sin(zeta0), omega0])
    if scheme == "log_freq":
        return np.column_stack([eta, chi_eff, chi_a, e0, zeta0, np.log(omega0)])
    if scheme == "fully_transformed":
        return np.column_stack([eta, chi_eff, chi_a, np.log(e0+1e-10), np.cos(zeta0), np.sin(zeta0), np.log(omega0)])
    return X

ps = {k: reparam(X_train, k) for k in ["raw","eta_chieff","trig_anomaly","log_freq","fully_transformed"]}
pv = {k: reparam(X_val, k) for k in ["raw","eta_chieff","trig_anomaly","log_freq","fully_transformed"]}

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor

results = []; err_data = {}; cl = []

def md(n, name):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    os.makedirs(os.path.join(d, "saved_model"), exist_ok=True)
    return d

def ev(coeffs_pred, x_ref):
    x_pred = coeffs_pred @ basis + mean_x
    losses = [rms_relative_error(x_pred[i], x_ref[i]) for i in range(len(x_ref))]
    return float(np.mean(losses)), np.array(losses)

def rec(n, name, sch, loss, lv, lt, rt, np_, notes):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    sc = {"approach":name,"approach_number":n,"benchmark":"dynamics","agent":"opus46",
          "parameterization":sch,"time_convention":"normalized_tau","loss":float(loss),
          "loss_components":{"rms_relative_error_x":float(loss)},
          "runtime_ms":rt,"n_train":N_TR,"n_val":N_VA,"n_params":np_,"notes":notes}
    with open(os.path.join(d, "scorecard.json"), "w") as f:
        json.dump(sc, f, indent=2)
    results.append(sc)
    err_data[name] = {"val_losses":[float(x) for x in lv], "train_losses":[float(x) for x in lt]}
    cl.append(f"## {n}: {name}\n- Param: {sch}, Loss: {loss:.6f}, RT: {rt:.1f}ms\n- {notes}\n")
    print(f"  [{n:02d}] {name}: loss={loss:.6f}")

# ═══════ BUILD MODELS ═══════
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import RBFInterpolator

# Helper: scale, train, predict, evaluate
def build_model(num, name, scheme, model_fn, n_basis_used=N_BASIS):
    d = md(num, name)
    X, Xv = ps[scheme], pv[scheme]
    sX = StandardScaler().fit(X); Xs = sX.transform(X); Xvs = sX.transform(Xv)
    y = Y[:, :n_basis_used] if n_basis_used < N_BASIS else Y
    sY = StandardScaler().fit(y); ys = sY.transform(y)
    model = model_fn(Xs, ys)
    t0 = time.time()
    yp = sY.inverse_transform(model.predict(Xvs) if not callable(getattr(model, '__call__', None)) else model(Xvs))
    rt = (time.time()-t0)/N_VA*1000
    if n_basis_used < N_BASIS:
        full = np.zeros((N_VA, N_BASIS)); full[:, :n_basis_used] = yp; yp = full
    loss, lv = ev(yp, x_val)
    yp_tr = sY.inverse_transform(model.predict(Xs))
    if n_basis_used < N_BASIS:
        full_tr = np.zeros((N_TR, N_BASIS)); full_tr[:, :n_basis_used] = yp_tr; yp_tr = full_tr
    _, lt = ev(yp_tr, x_train)
    joblib.dump({"model": model, "scaler_X": sX, "scaler_y": sY, "n_basis": n_basis_used},
                os.path.join(d, "saved_model/model.joblib"))
    return loss, lv, lt, rt

def build_raw_model(num, name, scheme, model_fn):
    d = md(num, name)
    X, Xv = ps[scheme], pv[scheme]
    model = model_fn(X, Y)
    t0 = time.time(); yp = model.predict(Xv); rt = (time.time()-t0)/N_VA*1000
    loss, lv = ev(yp, x_val)
    yp_tr = model.predict(X); _, lt = ev(yp_tr, x_train)
    joblib.dump({"model": model}, os.path.join(d, "saved_model/model.joblib"))
    return loss, lv, lt, rt

# 1. SVD+GPR RBF raw
print("\n=== 1: SVD+GPR raw ===")
n_gpr = 5
d = md(1, "svd_gpr_raw")
sX = StandardScaler().fit(ps["raw"]); Xs = sX.transform(ps["raw"]); Xvs = sX.transform(pv["raw"])
yt1 = Y[:, :n_gpr]; sy1 = StandardScaler().fit(yt1); y1s = sy1.transform(yt1)
k1 = ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(1e-4)
gpr1 = MultiOutputRegressor(GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=0, alpha=1e-4, normalize_y=True), n_jobs=-1)
gpr1.fit(Xs, y1s)
t0=time.time(); yp1=sy1.inverse_transform(gpr1.predict(Xvs)); rt1=(time.time()-t0)/N_VA*1000
full1 = np.zeros((N_VA, N_BASIS)); full1[:, :n_gpr] = yp1
loss1, lv1 = ev(full1, x_val)
full1_tr = np.zeros((N_TR, N_BASIS)); full1_tr[:, :n_gpr] = sy1.inverse_transform(gpr1.predict(Xs))
_, lt1 = ev(full1_tr, x_train)
joblib.dump({"gpr":gpr1,"scaler_X":sX,"scaler_y":sy1,"n_basis":n_gpr}, os.path.join(d,"saved_model/model.joblib"))
rec(1,"svd_gpr_raw","raw",loss1,lv1,lt1,rt1,n_gpr*250,"SVD(5)+GPR RBF")

# 2. SVD+GPR Matern eta
print("\n=== 2: SVD+GPR Matern eta ===")
d = md(2, "svd_gpr_matern_eta")
sX2 = StandardScaler().fit(ps["eta_chieff"]); X2s = sX2.transform(ps["eta_chieff"]); Xv2s = sX2.transform(pv["eta_chieff"])
k2 = ConstantKernel(1.0)*Matern(1.0, nu=2.5)+WhiteKernel(1e-4)
gpr2 = MultiOutputRegressor(GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=0, alpha=1e-4, normalize_y=True), n_jobs=-1)
gpr2.fit(X2s, y1s)
t0=time.time(); yp2=sy1.inverse_transform(gpr2.predict(Xv2s)); rt2=(time.time()-t0)/N_VA*1000
full2 = np.zeros((N_VA, N_BASIS)); full2[:, :n_gpr] = yp2
loss2, lv2 = ev(full2, x_val)
full2_tr = np.zeros((N_TR, N_BASIS)); full2_tr[:, :n_gpr] = sy1.inverse_transform(gpr2.predict(X2s))
_, lt2 = ev(full2_tr, x_train)
joblib.dump({"gpr":gpr2,"scaler_X":sX2,"scaler_y":sy1}, os.path.join(d,"saved_model/model.joblib"))
rec(2,"svd_gpr_matern_eta","eta_chieff",loss2,lv2,lt2,rt2,n_gpr*250,"SVD(5)+GPR Matern eta")

# 3-22: Build individually to handle poly features correctly
def build_scaled(num, name, scheme, fit_fn, predict_fn, save_extra=None):
    print(f"\n=== {num}: {name} ===")
    d = md(num, name)
    X, Xv = ps[scheme], pv[scheme]
    sXi = StandardScaler().fit(X); Xi = sXi.transform(X); Xvi = sXi.transform(Xv)
    sYi = StandardScaler().fit(Y); Yi = sYi.transform(Y)
    model = fit_fn(Xi, Yi)
    t0=time.time(); yp = sYi.inverse_transform(predict_fn(model, Xvi)); rt = (time.time()-t0)/N_VA*1000
    loss, lv = ev(yp, x_val)
    yp_tr = sYi.inverse_transform(predict_fn(model, Xi)); _, lt = ev(yp_tr, x_train)
    save_dict = {"model":model,"scaler_X":sXi,"scaler_y":sYi}
    if save_extra: save_dict.update(save_extra)
    joblib.dump(save_dict, os.path.join(d,"saved_model/model.joblib"))
    rec(num, name, scheme, loss, lv, lt, rt, 1000, name)

def build_raw(num, name, scheme, fit_fn):
    print(f"\n=== {num}: {name} ===")
    d = md(num, name)
    X, Xv = ps[scheme], pv[scheme]
    model = fit_fn(X, Y)
    t0=time.time(); yp = model.predict(Xv); rt = (time.time()-t0)/N_VA*1000
    loss, lv = ev(yp, x_val)
    yp_tr = model.predict(X); _, lt = ev(yp_tr, x_train)
    joblib.dump({"model":model}, os.path.join(d,"saved_model/model.joblib"))
    rec(num, name, scheme, loss, lv, lt, rt, 1000, name)

# 3. Poly3 raw
p3 = PolynomialFeatures(3)
build_scaled(3, "svd_poly3_raw", "raw",
    lambda X,y: Ridge(1.0).fit(p3.fit_transform(X), y),
    lambda m,X: m.predict(p3.transform(X)), {"poly": p3})

# 4. MLP raw
build_scaled(4, "svd_mlp_raw", "raw",
    lambda X,y: MLPRegressor(hidden_layer_sizes=(256,128,64), max_iter=2000, early_stopping=True, validation_fraction=0.15, random_state=42, learning_rate_init=0.001).fit(X, y),
    lambda m,X: m.predict(X))

# 5. RF raw
build_raw(5, "svd_rf_raw", "raw",
    lambda X,y: RandomForestRegressor(300, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(X, y))

# 6. GBR eta
build_raw(6, "svd_gbr_eta", "eta_chieff",
    lambda X,y: MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42), n_jobs=-1).fit(X, y))

# 7. KRR raw
build_scaled(7, "svd_krr_raw", "raw",
    lambda X,y: KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1).fit(X, y),
    lambda m,X: m.predict(X))

# 8. KNN raw
build_scaled(8, "svd_knn_raw", "raw",
    lambda X,y: KNeighborsRegressor(5, weights='distance').fit(X, y),
    lambda m,X: m.predict(X))

# 9. MLP eta
build_scaled(9, "svd_mlp_eta", "eta_chieff",
    lambda X,y: MLPRegressor(hidden_layer_sizes=(512,256,128), max_iter=3000, early_stopping=True, validation_fraction=0.15, random_state=42, learning_rate_init=0.0005).fit(X, y),
    lambda m,X: m.predict(X))

# 10. Poly4 eta
p10 = PolynomialFeatures(4)
build_scaled(10, "svd_poly4_eta", "eta_chieff",
    lambda X,y: Ridge(10.0).fit(p10.fit_transform(X), y),
    lambda m,X: m.predict(p10.transform(X)), {"poly": p10})

# 11. RF eta
build_raw(11, "svd_rf_eta", "eta_chieff",
    lambda X,y: RandomForestRegressor(500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(X, y))

# 12. ET raw
build_raw(12, "svd_et_raw", "raw",
    lambda X,y: ExtraTreesRegressor(500, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(X, y))

# 13. SVR trig
build_scaled(13, "svd_svr_trig", "trig_anomaly",
    lambda X,y: MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.01), n_jobs=-1).fit(X, y),
    lambda m,X: m.predict(X))

# 14. Lasso eta
p14 = PolynomialFeatures(3)
build_scaled(14, "svd_lasso_eta", "eta_chieff",
    lambda X,y: MultiOutputRegressor(Lasso(alpha=0.001, max_iter=5000), n_jobs=-1).fit(p14.fit_transform(X), y),
    lambda m,X: m.predict(p14.transform(X)), {"poly": p14})

# 15. AdaBoost eta
build_raw(15, "svd_adaboost_eta", "eta_chieff",
    lambda X,y: MultiOutputRegressor(AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=6), n_estimators=100, learning_rate=0.1, random_state=42), n_jobs=-1).fit(X, y))

# 16. BayRidge log
p16 = PolynomialFeatures(3)
build_scaled(16, "svd_bayridge_log", "log_freq",
    lambda X,y: MultiOutputRegressor(BayesianRidge(max_iter=500), n_jobs=-1).fit(p16.fit_transform(X), y),
    lambda m,X: m.predict(p16.transform(X)), {"poly": p16})

# 17. MLP trig
build_scaled(17, "svd_mlp_trig", "trig_anomaly",
    lambda X,y: MLPRegressor(hidden_layer_sizes=(512,512,256), max_iter=5000, early_stopping=True, validation_fraction=0.15, random_state=42, learning_rate_init=0.0003).fit(X, y),
    lambda m,X: m.predict(X))

# 18. ElasticNet full
p18 = PolynomialFeatures(3)
build_scaled(18, "svd_enet_full", "fully_transformed",
    lambda X,y: MultiOutputRegressor(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000), n_jobs=-1).fit(p18.fit_transform(X), y),
    lambda m,X: m.predict(p18.transform(X)), {"poly": p18})

# 20. GBR raw
build_raw(20, "svd_gbr_raw", "raw",
    lambda X,y: MultiOutputRegressor(GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, random_state=42), n_jobs=-1).fit(X, y))

# 19. GPR Matern trig (separate)
print("\n=== 19: SVD+GPR Matern trig ===")
d = md(19, "svd_gpr_matern_trig")
sX19 = StandardScaler().fit(ps["trig_anomaly"]); X19s = sX19.transform(ps["trig_anomaly"]); Xv19s = sX19.transform(pv["trig_anomaly"])
k19 = ConstantKernel(1.0)*Matern(1.0, nu=1.5)+WhiteKernel(1e-4)
gpr19 = MultiOutputRegressor(GaussianProcessRegressor(kernel=k19, n_restarts_optimizer=0, alpha=1e-4, normalize_y=True), n_jobs=-1)
gpr19.fit(X19s, y1s)
t0=time.time(); yp19=sy1.inverse_transform(gpr19.predict(Xv19s)); rt19=(time.time()-t0)/N_VA*1000
full19 = np.zeros((N_VA, N_BASIS)); full19[:, :n_gpr] = yp19
loss19, lv19 = ev(full19, x_val)
full19_tr = np.zeros((N_TR, N_BASIS)); full19_tr[:, :n_gpr] = sy1.inverse_transform(gpr19.predict(X19s))
_, lt19 = ev(full19_tr, x_train)
joblib.dump({"gpr":gpr19,"scaler_X":sX19,"scaler_y":sy1}, os.path.join(d,"saved_model/model.joblib"))
rec(19,"svd_gpr_matern_trig","trig_anomaly",loss19,lv19,lt19,rt19,n_gpr*250,"SVD(5)+GPR Matern trig")

# 21. RBF interp eta
print("\n=== 21: RBF interp eta ===")
d = md(21, "svd_rbf_interp_eta")
sX21 = StandardScaler().fit(ps["eta_chieff"]); X21s = sX21.transform(ps["eta_chieff"]); Xv21s = sX21.transform(pv["eta_chieff"])
sY21 = StandardScaler().fit(Y); Y21s = sY21.transform(Y)
rb21 = RBFInterpolator(X21s, Y21s, kernel='thin_plate_spline', smoothing=0.1)
t0=time.time(); yp21=sY21.inverse_transform(rb21(Xv21s)); rt21=(time.time()-t0)/N_VA*1000
loss21, lv21 = ev(yp21, x_val)
_, lt21 = ev(sY21.inverse_transform(rb21(X21s)), x_train)
joblib.dump({"rbf":rb21,"scaler_X":sX21,"scaler_y":sY21}, os.path.join(d,"saved_model/model.joblib"))
rec(21,"svd_rbf_interp_eta","eta_chieff",loss21,lv21,lt21,rt21,250*30,"SVD(30)+RBF TPS eta")

# 22. RBF interp raw
print("\n=== 22: RBF interp raw ===")
d = md(22, "svd_rbf_interp_raw")
sX22 = StandardScaler().fit(ps["raw"]); X22s = sX22.transform(ps["raw"]); Xv22s = sX22.transform(pv["raw"])
sY22 = StandardScaler().fit(Y); Y22s = sY22.transform(Y)
rb22 = RBFInterpolator(X22s, Y22s, kernel='thin_plate_spline', smoothing=0.1)
t0=time.time(); yp22=sY22.inverse_transform(rb22(Xv22s)); rt22=(time.time()-t0)/N_VA*1000
loss22, lv22 = ev(yp22, x_val)
_, lt22 = ev(sY22.inverse_transform(rb22(X22s)), x_train)
joblib.dump({"rbf":rb22,"scaler_X":sX22,"scaler_y":sY22}, os.path.join(d,"saved_model/model.joblib"))
rec(22,"svd_rbf_interp_raw","raw",loss22,lv22,lt22,rt22,250*30,"SVD(30)+RBF TPS raw")

# ═══════ SYMBOLIC ═══════
# 23. PySR raw
print("\n=== 23: PySR raw ===")
d = md(23, "pysr_raw")
from pysr import PySRRegressor
sXp = StandardScaler().fit(ps["raw"]); Xps = sXp.transform(ps["raw"]); Xpvs = sXp.transform(pv["raw"])
n_coeff = 5
pysr_pred_v = np.zeros((N_VA, n_coeff)); pysr_pred_t = np.zeros((N_TR, n_coeff))
all_expr = []
for k in range(n_coeff):
    print(f"  PySR coeff {k}...")
    m = PySRRegressor(niterations=40, binary_operators=["+","-","*","/"],
                      unary_operators=["sqrt","exp","sin","cos"], maxsize=20, populations=10,
                      procs=1, verbosity=0, progress=False, random_state=42, temp_equation_file=True)
    m.fit(Xps, Y[:, k])
    pysr_pred_v[:, k] = m.predict(Xpvs)
    pysr_pred_t[:, k] = m.predict(Xps)
    try:
        expr = str(m.sympy())
        eqs = [{"expression":str(r.get("equation","")),"complexity":int(r.get("complexity",0)),
                "loss":float(r.get("loss",0))} for _,r in m.equations_.iterrows()]
        all_expr.append({"coefficient":k,"best":expr,"pareto_front":eqs})
    except: all_expr.append({"coefficient":k,"best":"error","pareto_front":[]})
    joblib.dump(m, os.path.join(d, f"saved_model/pysr_coeff_{k}.joblib"))
with open(os.path.join(d,"saved_model/expressions.json"),"w") as f:
    json.dump(all_expr, f, indent=2, default=str)
full_v = np.zeros((N_VA, N_BASIS)); full_v[:, :n_coeff] = pysr_pred_v
full_t = np.zeros((N_TR, N_BASIS)); full_t[:, :n_coeff] = pysr_pred_t
loss23, lv23 = ev(full_v, x_val); _, lt23 = ev(full_t, x_train)
rec(23,"pysr_raw","raw",loss23,lv23,lt23,0.1,n_coeff*20,"PySR on SVD coefficients raw")

# 24. PySR eta
print("\n=== 24: PySR eta ===")
d = md(24, "pysr_eta")
sXpe = StandardScaler().fit(ps["eta_chieff"]); Xpes = sXpe.transform(ps["eta_chieff"]); Xpves = sXpe.transform(pv["eta_chieff"])
pysr_pred2_v = np.zeros((N_VA, n_coeff)); pysr_pred2_t = np.zeros((N_TR, n_coeff))
all_expr2 = []
for k in range(n_coeff):
    print(f"  PySR(eta) coeff {k}...")
    m2 = PySRRegressor(niterations=40, binary_operators=["+","-","*","/"],
                       unary_operators=["sqrt","exp","sin","cos"], maxsize=20, populations=10,
                       procs=1, verbosity=0, progress=False, random_state=42, temp_equation_file=True)
    m2.fit(Xpes, Y[:, k])
    pysr_pred2_v[:, k] = m2.predict(Xpves)
    pysr_pred2_t[:, k] = m2.predict(Xpes)
    try:
        expr = str(m2.sympy())
        eqs = [{"expression":str(r.get("equation","")),"complexity":int(r.get("complexity",0)),
                "loss":float(r.get("loss",0))} for _,r in m2.equations_.iterrows()]
        all_expr2.append({"coefficient":k,"best":expr,"pareto_front":eqs})
    except: all_expr2.append({"coefficient":k,"best":"error","pareto_front":[]})
    joblib.dump(m2, os.path.join(d, f"saved_model/pysr_coeff_{k}.joblib"))
with open(os.path.join(d,"saved_model/expressions.json"),"w") as f:
    json.dump(all_expr2, f, indent=2, default=str)
full2_v = np.zeros((N_VA, N_BASIS)); full2_v[:, :n_coeff] = pysr_pred2_v
full2_t = np.zeros((N_TR, N_BASIS)); full2_t[:, :n_coeff] = pysr_pred2_t
loss24, lv24 = ev(full2_v, x_val); _, lt24 = ev(full2_t, x_train)
rec(24,"pysr_eta","eta_chieff",loss24,lv24,lt24,0.1,n_coeff*20,"PySR on SVD coefficients eta")

# 25. gplearn raw
print("\n=== 25: gplearn raw ===")
d = md(25, "gplearn_raw")
from gplearn.genetic import SymbolicRegressor
gp_v = np.zeros((N_VA, n_coeff)); gp_t = np.zeros((N_TR, n_coeff))
gp_expr = []
for k in range(n_coeff):
    print(f"  gplearn coeff {k}...")
    est = SymbolicRegressor(population_size=2000, generations=30, tournament_size=20,
                            function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                            metric='mse', parsimony_coefficient=0.001, verbose=0, random_state=42)
    est.fit(Xps, Y[:, k])
    gp_v[:, k] = est.predict(Xpvs); gp_t[:, k] = est.predict(Xps)
    gp_expr.append({"coefficient":k,"expression":str(est._program),"complexity":est._program.length_,"fitness":float(est._program.fitness_)})
    joblib.dump(est, os.path.join(d, f"saved_model/gplearn_coeff_{k}.joblib"))
with open(os.path.join(d,"saved_model/expressions.json"),"w") as f:
    json.dump(gp_expr, f, indent=2, default=str)
full_gv = np.zeros((N_VA, N_BASIS)); full_gv[:, :n_coeff] = gp_v
full_gt = np.zeros((N_TR, N_BASIS)); full_gt[:, :n_coeff] = gp_t
loss25, lv25 = ev(full_gv, x_val); _, lt25 = ev(full_gt, x_train)
rec(25,"gplearn_raw","raw",loss25,lv25,lt25,0.1,n_coeff*30,"gplearn on SVD coefficients raw")

# 26. gplearn eta
print("\n=== 26: gplearn eta ===")
d = md(26, "gplearn_eta")
gp2_v = np.zeros((N_VA, n_coeff)); gp2_t = np.zeros((N_TR, n_coeff))
gp_expr2 = []
for k in range(n_coeff):
    print(f"  gplearn(eta) coeff {k}...")
    est2 = SymbolicRegressor(population_size=2000, generations=30, tournament_size=20,
                             function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                             metric='mse', parsimony_coefficient=0.001, verbose=0, random_state=42)
    est2.fit(Xpes, Y[:, k])
    gp2_v[:, k] = est2.predict(Xpves); gp2_t[:, k] = est2.predict(Xpes)
    gp_expr2.append({"coefficient":k,"expression":str(est2._program),"complexity":est2._program.length_,"fitness":float(est2._program.fitness_)})
    joblib.dump(est2, os.path.join(d, f"saved_model/gplearn_coeff_{k}.joblib"))
with open(os.path.join(d,"saved_model/expressions.json"),"w") as f:
    json.dump(gp_expr2, f, indent=2, default=str)
full_g2v = np.zeros((N_VA, N_BASIS)); full_g2v[:, :n_coeff] = gp2_v
full_g2t = np.zeros((N_TR, N_BASIS)); full_g2t[:, :n_coeff] = gp2_t
loss26, lv26 = ev(full_g2v, x_val); _, lt26 = ev(full_g2t, x_train)
rec(26,"gplearn_eta","eta_chieff",loss26,lv26,lt26,0.1,n_coeff*30,"gplearn on SVD coefficients eta")

# ═══════ SAVE & PLOTS ═══════
print("\n=== Saving ===")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    f.write("# Dynamics Benchmark CHANGELOG — Opus 4.6\n\n")
    for c in cl: f.write(c + "\n")

# train.py / predict.py
for r in results:
    n = r["approach_number"]; name = r["approach"]; sch = r["parameterization"]
    dp = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    with open(os.path.join(dp, "train.py"), "w") as f:
        f.write(f'#!/usr/bin/env python3\n"""Training script for {name}."""\n# See build_all.py approach {n}\n')
    with open(os.path.join(dp, "predict.py"), "w") as f:
        f.write(f'#!/usr/bin/env python3\n"""Prediction function for {name}."""\nimport os, numpy as np, joblib\nWORK_DIR = os.path.dirname(os.path.abspath(__file__))\ndef predict(params_raw):\n    saved = joblib.load(os.path.join(WORK_DIR, "saved_model/model.joblib"))\n    return saved["model"].predict(np.atleast_2d(params_raw))\n')

names = [r["approach"] for r in results]; losses = [r["loss"] for r in results]; rts = [r["runtime_ms"] for r in results]
CAT = {"kernel":pset.COLORS["blue"],"symbolic":pset.COLORS["green"],"interp":pset.COLORS["orange"],"ml":pset.COLORS["red"]}
def cat(n):
    nl=n.lower()
    if any(x in nl for x in ["pysr","gplearn"]): return "symbolic"
    if any(x in nl for x in ["rbf_interp","knn"]): return "interp"
    if any(x in nl for x in ["gpr","krr","svr"]): return "kernel"
    return "ml"
cats=[cat(n) for n in names]; cols=[CAT[c] for c in cats]

fig,ax=plt.subplots(figsize=pset.figsize(2,0.5)); y_pos=np.arange(len(results))
ax.barh(y_pos, losses, color=cols, height=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels([n.replace("_"," ") for n in names], fontsize=6)
ax.set_xlabel("RMS Relative Error"); ax.set_xscale("log"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(os.path.join(WORK_DIR,"comparison/progress.png")); fig.savefig(os.path.join(WORK_DIR,"comparison/progress.pdf")); plt.close()

fig,ax=plt.subplots(figsize=pset.figsize(1,0.9))
for c in ["kernel","symbolic","interp","ml"]:
    idx=[i for i,x in enumerate(cats) if x==c]
    if idx: ax.scatter([losses[i] for i in idx],[rts[i] for i in idx],c=CAT[c],label=c.upper(),s=30,zorder=5)
    for i in idx: ax.annotate(names[i].replace("_"," "),(losses[i],max(rts[i],0.001)),fontsize=4.5,ha='left',rotation=10)
ax.set_xlabel("Loss"); ax.set_ylabel("RT (ms)"); ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(WORK_DIR,"comparison/pareto_accuracy_speed.png")); fig.savefig(os.path.join(WORK_DIR,"comparison/pareto_accuracy_speed.pdf")); plt.close()

fig,ax=plt.subplots(figsize=pset.figsize(2,0.5))
ax.barh(y_pos, losses, color=cols, height=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels([n.replace("_"," ") for n in names], fontsize=6)
ax.set_xlabel("RMS Relative Error"); ax.set_xscale("log"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(os.path.join(WORK_DIR,"comparison/loss_only_comparison.png")); fig.savefig(os.path.join(WORK_DIR,"comparison/loss_only_comparison.pdf")); plt.close()

fig,ax=plt.subplots(figsize=pset.figsize(2,0.7))
for name, data in sorted(err_data.items(), key=lambda x: np.mean(x[1]["val_losses"]))[:10]:
    ax.hist(data["val_losses"], bins=50, alpha=0.4, label=name.replace("_"," "), histtype='stepfilled')
ax.set_xlabel("Per-sample RMS Relative Error"); ax.set_ylabel("Count"); ax.legend(fontsize=5, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(WORK_DIR,"comparison/error_histograms.png")); fig.savefig(os.path.join(WORK_DIR,"comparison/error_histograms.pdf")); plt.close()

print(f"\n=== DONE ({len(results)} approaches) ===")
for r in results:
    print(f"  {r['approach_number']:02d} {r['approach']:<30s} loss={r['loss']:.6f}")
print(f"Best: {results[0]['approach']} loss={results[0]['loss']:.6f}")
