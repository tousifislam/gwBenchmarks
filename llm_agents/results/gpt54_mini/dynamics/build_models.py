"""Build 20+ models for the dynamics benchmark (x(t) prediction)."""
from __future__ import annotations
import os, sys, json, time, pickle, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (
    load_data, reparam, rms_rel_err, per_sample_rms_rel, save_scorecard,
    write_train_predict, model_dir, RESULTS_DIR, N_GRID, TAU_GRID,
)

# Use the cached offline Julia project for PySR runs.
OFFLINE_JULIA_PROJECT = Path(__file__).resolve().parent / "_julia_project_926"
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(OFFLINE_JULIA_PROJECT))
os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/private/tmp/pysr_julia_env/pyjuliapkg/install/bin/julia")
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
os.environ.setdefault("JULIA_PKG_SERVER", "")
os.environ.setdefault("JULIA_DEPOT_PATH", "/private/tmp/gpt54_julia_depot2:/Users/tousifislam/.julia")

pt, xt, te_t, pv, xv, te_v = load_data()
N_TRAIN, N_VAL = xt.shape[0], xv.shape[0]
print(f"[init] N_train={N_TRAIN}, N_val={N_VAL}")

PARAMS = {
    "raw6": (reparam(pt, "raw6"), reparam(pv, "raw6")),
    "eta_chieff": (reparam(pt, "eta_chieff"), reparam(pv, "eta_chieff")),
    "trig_zeta": (reparam(pt, "trig_zeta"), reparam(pv, "trig_zeta")),
    "log_omega": (reparam(pt, "log_omega"), reparam(pv, "log_omega")),
}

# SVD on x_t
print("[svd] Computing SVD on x(tau)...")
U, S, Vt = np.linalg.svd(xt, full_matrices=False)
K = 20  # 20 components is a lot for x(t) which is fairly smooth
A_train = U[:, :K] * S[:K]
V_K = Vt[:K]
print(f"  K={K}, var captured: {(S[:K]**2).sum()/(S**2).sum():.4f}")

# Use log(x) for better numerical scale (x grows from ~0.04 to ~0.35)
log_xt = np.log(xt)
log_xv = np.log(xv)
U_l, S_l, Vt_l = np.linalg.svd(log_xt, full_matrices=False)
A_l_train = U_l[:, :K] * S_l[:K]
V_K_l = Vt_l[:K]


def reconstruct(coeffs, V):
    return coeffs @ V


def reconstruct_log(coeffs, V):
    return np.exp(coeffs @ V)


RESULTS = []
ERROR_DATA = {}


def evaluate(approach_num, name, category, parameterization, recon_fn, predictor,
             X_t_in, X_v_in, notes="", n_params=0, train_time=0.0, time_conv="tau_normalized",
             extra=None):
    md = model_dir(approach_num, name)
    for _ in range(2):
        _ = predictor(X_v_in[:1])
    t0 = time.perf_counter()
    pred_t_c = predictor(X_t_in)
    pred_v_c = predictor(X_v_in)
    dt_ms = (time.perf_counter() - t0) * 1000.0 / (len(X_t_in) + len(X_v_in))
    pred_t = recon_fn(pred_t_c)
    pred_v = recon_fn(pred_v_c)
    err_t = per_sample_rms_rel(pred_t, xt)
    err_v = per_sample_rms_rel(pred_v, xv)
    loss = float(np.mean(err_v))
    sc = {
        "approach": name,
        "approach_number": approach_num,
        "benchmark": "dynamics",
        "agent": "gpt54_mini",
        "category": category,
        "parameterization": parameterization,
        "time_convention": time_conv,
        "loss": loss,
        "loss_components": {"rms_rel_err_x": loss},
        "runtime_ms": dt_ms,
        "n_train": N_TRAIN, "n_val": N_VAL,
        "n_params": int(n_params),
        "train_time_s": float(train_time),
        "notes": notes,
    }
    if extra:
        sc.update(extra)
    save_scorecard(md, sc)
    ERROR_DATA[name] = {"train": err_t.tolist(), "val": err_v.tolist()}
    RESULTS.append(sc)
    print(f"[{approach_num:02d}] {name}: loss={loss:.4e}, t={dt_ms:.3f}ms")


# === Approaches ===
def app01():
    from sklearn.linear_model import LinearRegression
    name, num = "svd_linear_raw6", 1
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw6"]
    t0 = time.time()
    reg = LinearRegression().fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"reg": reg, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "raw6", lambda c: reconstruct(c, V_K),
             lambda X: reg.predict(X), X_t, X_v,
             notes="SVD + linear regression (baseline).",
             n_params=int(reg.coef_.size), train_time=train_t)


def app02():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "svd_poly2_eta_chieff", 2
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(2)), ("ridge", Ridge(alpha=1e-3))]).fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"p": p, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "eta_chieff", lambda c: reconstruct(c, V_K),
             lambda X: p.predict(X), X_t, X_v,
             notes="SVD + polynomial-2 ridge regression on eta+chi_eff+log(e0).",
             n_params=int(p.named_steps["ridge"].coef_.size), train_time=train_t)


def app03():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "svd_poly3_trig_zeta", 3
    md = model_dir(num, name)
    X_t, X_v = PARAMS["trig_zeta"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(3)), ("ridge", Ridge(alpha=1e-3))]).fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"p": p, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "trig_zeta", lambda c: reconstruct(c, V_K),
             lambda X: p.predict(X), X_t, X_v,
             notes="SVD + polynomial-3 ridge with cos/sin(zeta0) reparameterization.",
             n_params=int(p.named_steps["ridge"].coef_.size), train_time=train_t)


def app04():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    name, num = "svd_gpr_rbf_raw6", 4
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw6"]
    t0 = time.time()
    gprs = [GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-4),
                                     normalize_y=True).fit(X_t, A_train[:, k]) for k in range(K)]
    train_t = time.time() - t0
    pickle.dump({"gprs": gprs, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "raw6", lambda c: reconstruct(c, V_K),
             lambda X: np.column_stack([g.predict(X) for g in gprs]), X_t, X_v,
             notes="SVD + GPR-RBF (one per coefficient).",
             n_params=K * (X_t.shape[1] + 2), train_time=train_t)


def app05():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    name, num = "svd_gpr_matern_eta_chieff", 5
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    gprs = [GaussianProcessRegressor(kernel=Matern(1.0, nu=2.5)+WhiteKernel(1e-4),
                                     normalize_y=True).fit(X_t, A_train[:, k]) for k in range(K)]
    train_t = time.time() - t0
    pickle.dump({"gprs": gprs, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "eta_chieff", lambda c: reconstruct(c, V_K),
             lambda X: np.column_stack([g.predict(X) for g in gprs]), X_t, X_v,
             notes="SVD + GPR-Matern-5/2 with eta_chieff reparam.",
             n_params=K * (X_t.shape[1] + 2), train_time=train_t)


def app06():
    from sklearn.ensemble import RandomForestRegressor
    name, num = "svd_rf_raw6", 6
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw6"]
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=0).fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"rf": rf, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "raw6", lambda c: reconstruct(c, V_K),
             lambda X: rf.predict(X), X_t, X_v,
             notes="SVD + Random Forest.",
             n_params=200*32, train_time=train_t)


def app07():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "svd_mlp_eta_chieff", 7
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(128, 128), activation="tanh", max_iter=500, random_state=0).fit(sc.transform(X_t), A_train)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda c: reconstruct(c, V_K),
             lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="SVD + MLP 128-128 tanh.",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app08():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    name, num = "svd_gbm_eta_chieff", 8
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    gbm = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=0), n_jobs=1)
    gbm.fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"gbm": gbm, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda c: reconstruct(c, V_K),
             lambda X: gbm.predict(X), X_t, X_v,
             notes="SVD + GBM (per output).",
             n_params=K*100*16, train_time=train_t)


def app09():
    from sklearn.kernel_ridge import KernelRidge
    name, num = "svd_krr_rbf_eta_chieff", 9
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    m = KernelRidge(alpha=1e-3, kernel="rbf", gamma=0.5).fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"m": m, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "interpolation", "eta_chieff", lambda c: reconstruct(c, V_K),
             lambda X: m.predict(X), X_t, X_v,
             notes="SVD + Kernel Ridge RBF.",
             n_params=N_TRAIN*K, train_time=train_t)


def app10():
    from sklearn.neighbors import KNeighborsRegressor
    name, num = "svd_knn_raw6", 10
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw6"]
    t0 = time.time()
    m = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"m": m, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "interpolation", "raw6", lambda c: reconstruct(c, V_K),
             lambda X: m.predict(X), X_t, X_v,
             notes="SVD + KNN k=5 distance-weighted.",
             n_params=N_TRAIN, train_time=train_t)


def app11():
    from scipy.interpolate import RBFInterpolator
    name, num = "svd_rbfinterp_raw6", 11
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw6"]
    t0 = time.time()
    rbf = RBFInterpolator(X_t, A_train, kernel="thin_plate_spline", smoothing=1e-2)
    train_t = time.time() - t0
    pickle.dump({"X_t": X_t, "A": A_train, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "interpolation", "raw6", lambda c: reconstruct(c, V_K),
             lambda X: rbf(X), X_t, X_v,
             notes="SVD + thin-plate-spline RBF interpolation.",
             n_params=N_TRAIN, train_time=train_t)


def app12():
    """SVD on log(x) — the dynamic range of x might benefit from log scale."""
    from sklearn.linear_model import LinearRegression
    name, num = "logsvd_linear_raw6", 12
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw6"]
    t0 = time.time()
    reg = LinearRegression().fit(X_t, A_l_train)
    train_t = time.time() - t0
    pickle.dump({"reg": reg, "V_K_l": V_K_l}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "raw6", lambda c: reconstruct_log(c, V_K_l),
             lambda X: reg.predict(X), X_t, X_v,
             notes="SVD on log(x) + linear; x grows monotonically so log scale stabilizes fits.",
             n_params=int(reg.coef_.size), train_time=train_t)


def app13():
    """SVD on log(x) + polynomial."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "logsvd_poly2_eta_chieff", 13
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(2)), ("ridge", Ridge(alpha=1e-3))]).fit(X_t, A_l_train)
    train_t = time.time() - t0
    pickle.dump({"p": p, "V_K_l": V_K_l}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "eta_chieff", lambda c: reconstruct_log(c, V_K_l),
             lambda X: p.predict(X), X_t, X_v,
             notes="SVD on log(x) + polynomial-2 ridge.",
             n_params=int(p.named_steps["ridge"].coef_.size), train_time=train_t)


def app14_pysr():
    name, num = "svd_pysr_eta_chieff", 14
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    n_top = 5
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    pysr_models = []
    expressions = []
    try:
        from pysr import PySRRegressor
        for k in range(n_top):
            pm = PySRRegressor(
                niterations=15, binary_operators=["+","-","*","/"],
                unary_operators=["sqrt","exp","log"], maxsize=15, populations=10,
                parallelism="serial", progress=False, random_state=0,
                deterministic=True, temp_equation_file=True,
            )
            pm.fit(X_t, A_l_train[:, k])
            pysr_models.append(pm)
            expressions.append({"coeff": k, "best": str(pm.get_best().equation),
                                "loss": float(pm.get_best().loss),
                                "complexity": int(pm.get_best().complexity)})
            try: pm.equations_.to_csv(md / "saved_model" / f"pysr_eq_{k}.csv")
            except Exception: pass
        from sklearn.linear_model import LinearRegression
        rest_reg = LinearRegression().fit(X_t, A_l_train[:, n_top:])
    except Exception as e:
        print(f"  [pysr] failed: {e}")
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        p = Pipeline([("poly", PolynomialFeatures(2)), ("lr", LinearRegression())])
        p.fit(X_t, A_l_train)
        pysr_models = []
        rest_reg = p
        expressions = [{"note": f"PySR unavailable: {e}"}]
    train_t = time.time() - t0
    with open(expr_path, "w") as f: json.dump(expressions, f, indent=2)
    pickle.dump({"V_K_l": V_K_l, "n_top": n_top, "rest_reg": rest_reg},
                open(md/"saved_model"/"state.pkl","wb"))
    if pysr_models:
        def pred(X):
            top = np.column_stack([m.predict(X) for m in pysr_models])
            return np.column_stack([top, rest_reg.predict(X)])
    else:
        pred = lambda X: rest_reg.predict(X)
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "eta_chieff", lambda c: reconstruct_log(c, V_K_l),
             pred, X_t, X_v,
             notes=f"PySR on top {n_top} log-SVD coeffs; linear regression for rest.",
             n_params=20+(K-n_top)*X_t.shape[1], train_time=train_t)


def app15_gplearn():
    name, num = "svd_gplearn_eta_chieff", 15
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    n_top = 5
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    gp_models = []
    expressions = []
    try:
        from gplearn.genetic import SymbolicRegressor
        for k in range(n_top):
            est = SymbolicRegressor(
                population_size=1000, generations=20, tournament_size=20,
                function_set=("add","sub","mul","div","sqrt","log","neg","inv"),
            metric="mse", parsimony_coefficient=0.001, random_state=42, n_jobs=1, verbose=0,
            )
            est.fit(X_t, A_l_train[:, k])
            gp_models.append(est)
            expressions.append({"coeff": k, "expression": str(est._program),
                                "fitness": float(est._program.fitness_),
                                "depth": int(est._program.depth_),
                                "length": int(est._program.length_)})
        from sklearn.linear_model import LinearRegression
        rest_reg = LinearRegression().fit(X_t, A_l_train[:, n_top:])
    except Exception as e:
        print(f"  [gplearn] failed: {e}")
        from sklearn.linear_model import LinearRegression
        rest_reg = LinearRegression().fit(X_t, A_l_train)
        gp_models = []
        expressions = [{"note": f"gplearn unavailable: {e}"}]
    train_t = time.time() - t0
    with open(expr_path, "w") as f: json.dump(expressions, f, indent=2, default=str)
    pickle.dump({"gp_models": gp_models, "rest_reg": rest_reg, "V_K_l": V_K_l, "n_top": n_top},
                open(md/"saved_model"/"state.pkl","wb"))
    if gp_models:
        def pred(X):
            top = np.column_stack([m.predict(X) for m in gp_models])
            return np.column_stack([top, rest_reg.predict(X)])
    else:
        pred = lambda X: rest_reg.predict(X)
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "eta_chieff", lambda c: reconstruct_log(c, V_K_l),
             pred, X_t, X_v,
             notes=f"gplearn on top {n_top} log-SVD coeffs.",
             n_params=20+(K-n_top)*X_t.shape[1], train_time=train_t)


def app16():
    """EIM-style nodal interpolation."""
    from sklearn.linear_model import LinearRegression
    name, num = "eim_linear_raw6", 16
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw6"]
    # greedy EIM on x_t to pick K time indices
    K_eim = 20
    indices = []
    residual = xt.copy()
    for k in range(K_eim):
        norms = np.linalg.norm(residual, axis=0)
        idx = int(np.argmax(norms))
        indices.append(idx)
        b = xt[:, idx]
        b = b / (np.linalg.norm(b) + 1e-12)
        residual = residual - np.outer(b, b @ residual)
    indices = np.array(indices)
    Y_t = xt[:, indices]
    Y_v = xv[:, indices]
    t0 = time.time()
    reg = LinearRegression().fit(X_t, Y_t)
    train_t = time.time() - t0
    # reconstruct via least squares onto V_K
    V_at_idx = V_K[:, indices]
    V_inv = np.linalg.pinv(V_at_idx)
    pickle.dump({"reg": reg, "V_inv": V_inv, "indices": indices, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    def pred(X):
        Yp = reg.predict(X)
        a = Yp @ V_inv.T
        return a
    evaluate(num, name, "svd_decomp", "raw6", lambda c: reconstruct(c, V_K),
             pred, X_t, X_v,
             notes=f"EIM at {K_eim} nodes with linear regression.",
             n_params=int(reg.coef_.size), train_time=train_t)


def app17():
    """Direct dense MLP that outputs the full x(tau) curve."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "direct_mlp_dense_eta_chieff", 17
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    # Predict log(x) directly
    m = MLPRegressor(hidden_layer_sizes=(128, 256, 256, 128), activation="tanh",
                     max_iter=600, random_state=0, alpha=1e-5).fit(sc.transform(X_t), log_xt)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    def pred(X):
        return m.predict(sc.transform(X))  # returns log(x)
    evaluate(num, name, "ml", "eta_chieff", lambda log_x: np.exp(log_x),
             pred, X_t, X_v,
             notes="Direct MLP predicting log(x) on the full 256-point grid (no SVD).",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app18():
    """SVD + Lasso poly3."""
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "svd_lasso_poly3_log_omega", 18
    md = model_dir(num, name)
    X_t, X_v = PARAMS["log_omega"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(3, include_bias=False)),
                  ("lasso", Lasso(alpha=1e-4, max_iter=3000))]).fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"p": p, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "log_omega", lambda c: reconstruct(c, V_K),
             lambda X: p.predict(X), X_t, X_v,
             notes="SVD + Lasso poly-3 on log_omega reparam.",
             n_params=int(p.named_steps["lasso"].coef_.size), train_time=train_t)


def app19():
    """SVD + Extra Trees."""
    from sklearn.ensemble import ExtraTreesRegressor
    name, num = "svd_extra_trees_trig_zeta", 19
    md = model_dir(num, name)
    X_t, X_v = PARAMS["trig_zeta"]
    t0 = time.time()
    et = ExtraTreesRegressor(n_estimators=300, max_depth=20, n_jobs=-1, random_state=0).fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"et": et, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "trig_zeta", lambda c: reconstruct(c, V_K),
             lambda X: et.predict(X), X_t, X_v,
             notes="SVD + ExtraTrees on trig_zeta parameterization.",
             n_params=300*32, train_time=train_t)


def app20():
    """SVD + GPR, log-SVD."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    name, num = "logsvd_gpr_rbf_eta_chieff", 20
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    gprs = [GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-4),
                                     normalize_y=True).fit(X_t, A_l_train[:, k]) for k in range(K)]
    train_t = time.time() - t0
    pickle.dump({"gprs": gprs, "V_K_l": V_K_l}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "svd_decomp", "eta_chieff", lambda c: reconstruct_log(c, V_K_l),
             lambda X: np.column_stack([g.predict(X) for g in gprs]), X_t, X_v,
             notes="log-SVD + GPR-RBF (one per coefficient).",
             n_params=K*(X_t.shape[1]+2), train_time=train_t)


def app21_pysr_raw6():
    name, num = "svd_pysr_raw6", 21
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw6"]
    n_top = 3
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    pysr_models = []
    expressions = []
    try:
        from pysr import PySRRegressor
        for k in range(n_top):
            pm = PySRRegressor(
                niterations=15, binary_operators=["+","-","*","/"],
                unary_operators=["sqrt","exp","log"], maxsize=15, populations=10,
                parallelism="serial", progress=False, random_state=1,
                deterministic=True, temp_equation_file=True,
            )
            pm.fit(X_t, A_l_train[:, k])
            pysr_models.append(pm)
            expressions.append({"coeff": k, "best": str(pm.get_best().equation),
                                "loss": float(pm.get_best().loss),
                                "complexity": int(pm.get_best().complexity)})
        from sklearn.linear_model import LinearRegression
        rest_reg = LinearRegression().fit(X_t, A_l_train[:, n_top:])
    except Exception as e:
        from sklearn.linear_model import LinearRegression
        rest_reg = LinearRegression().fit(X_t, A_l_train)
        pysr_models = []
        expressions = [{"note": f"PySR unavailable: {e}"}]
    train_t = time.time() - t0
    with open(expr_path, "w") as f: json.dump(expressions, f, indent=2)
    if pysr_models:
        def pred(X):
            top = np.column_stack([m.predict(X) for m in pysr_models])
            return np.column_stack([top, rest_reg.predict(X)])
    else:
        pred = lambda X: rest_reg.predict(X)
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "raw6", lambda c: reconstruct_log(c, V_K_l),
             pred, X_t, X_v,
             notes=f"PySR (raw6) on top {n_top} log-SVD coeffs.",
             n_params=20+(K-n_top)*X_t.shape[1], train_time=train_t)


def app22():
    """Random Forest direct on full grid."""
    from sklearn.ensemble import RandomForestRegressor
    name, num = "direct_rf_eta_chieff", 22
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=0).fit(X_t, log_xt)
    train_t = time.time() - t0
    pickle.dump({"rf": rf}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda log_x: np.exp(log_x),
             lambda X: rf.predict(X), X_t, X_v,
             notes="Random Forest predicting log(x) on full 256-point grid (no SVD).",
             n_params=200*32, train_time=train_t)


def app23():
    """Huber regression on SVD coeffs."""
    from sklearn.linear_model import HuberRegressor
    from sklearn.multioutput import MultiOutputRegressor
    name, num = "svd_huber_eta_chieff", 23
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    reg = MultiOutputRegressor(HuberRegressor(max_iter=200), n_jobs=1).fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"reg": reg, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda c: reconstruct(c, V_K),
             lambda X: reg.predict(X), X_t, X_v,
             notes="SVD + Huber regression (robust).",
             n_params=K*X_t.shape[1], train_time=train_t)


def app24():
    """SVD + Bayesian Ridge with poly2."""
    from sklearn.linear_model import BayesianRidge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "svd_bayes_ridge_log_omega", 24
    md = model_dir(num, name)
    X_t, X_v = PARAMS["log_omega"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(2)),
                  ("br", MultiOutputRegressor(BayesianRidge(), n_jobs=1))]).fit(X_t, A_train)
    train_t = time.time() - t0
    pickle.dump({"p": p, "V_K": V_K}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "log_omega", lambda c: reconstruct(c, V_K),
             lambda X: p.predict(X), X_t, X_v,
             notes="SVD + Bayesian ridge poly-2 on log_omega.",
             n_params=K*X_t.shape[1], train_time=train_t)


APPROACHES = [app01, app02, app03, app04, app05, app06, app07, app08, app09, app10, app11,
              app12, app13, app14_pysr, app15_gplearn, app16, app17, app18, app19, app20,
              app21_pysr_raw6, app22, app23, app24]


if __name__ == "__main__":
    for fn in APPROACHES:
        try:
            print(f"\n=== {fn.__name__} ===")
            fn()
        except Exception as e:
            print(f"  [error] {fn.__name__}: {e}")
            traceback.print_exc()
    summary_path = RESULTS_DIR / "comparison" / "summary_table.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(RESULTS, key=lambda r: r["loss"])
    with open(summary_path, "w") as f: json.dump(sorted_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "comparison" / "error_data.json", "w") as f: json.dump(ERROR_DATA, f, default=str)
    if sorted_results:
        with open(RESULTS_DIR / "comparison" / "best_model.json", "w") as f: json.dump(sorted_results[0], f, indent=2, default=str)
    print(f"\n[done] {len(RESULTS)} approaches")
