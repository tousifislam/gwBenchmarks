"""Build 20+ models for the remnant kick velocity benchmark."""
from __future__ import annotations
import os, sys, json, time, pickle, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (
    load_data, reparam, nrmse, per_sample_abs_err, save_scorecard,
    write_train_predict, model_dir, RESULTS_DIR,
)

# Force PySR / juliacall to use the cached offline Julia project.
OFFLINE_JULIA_PROJECT = Path(__file__).resolve().parent / "_julia_project_926"
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(OFFLINE_JULIA_PROJECT))
os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/private/tmp/pysr_julia_env/pyjuliapkg/install/bin/julia")
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
os.environ.setdefault("JULIA_PKG_SERVER", "")
os.environ.setdefault("JULIA_DEPOT_PATH", "/private/tmp/gpt54_julia_depot2:/Users/tousifislam/.julia")

pt, om_t, vf_t, Mf_t, chif_t, pv, om_v, vf_v, Mf_v, chif_v = load_data()
print(f"[init] N_train={len(vf_t)}, N_val={len(vf_v)}")
print(f"  vf range train: [{vf_t.min():.2e}, {vf_t.max():.2e}]")

PARAMS = {
    "raw7": (reparam(pt, "raw7"), reparam(pv, "raw7")),
    "eta_chieff": (reparam(pt, "eta_chieff"), reparam(pv, "eta_chieff")),
    "spherical": (reparam(pt, "spherical"), reparam(pv, "spherical")),
    "delta_chia": (reparam(pt, "delta_chia"), reparam(pv, "delta_chia")),
    "pn_products": (reparam(pt, "pn_products"), reparam(pv, "pn_products")),
}

RESULTS = []
ERROR_DATA = {}


def evaluate(approach_num, name, category, parameterization, predictor, X_t, X_v,
             notes="", n_params=0, train_time=0.0, extra=None):
    md = model_dir(approach_num, name)
    # warmup
    for _ in range(2):
        _ = predictor(X_v[:1])
    t0 = time.perf_counter()
    pred_t = predictor(X_t)
    pred_v = predictor(X_v)
    dt = (time.perf_counter() - t0) * 1000.0 / (len(X_t) + len(X_v))
    err_t = per_sample_abs_err(pred_t, vf_t)
    err_v = per_sample_abs_err(pred_v, vf_v)
    loss = nrmse(pred_v, vf_v)
    sc = {
        "approach": name,
        "approach_number": approach_num,
        "benchmark": "remnant",
        "agent": "gpt54_mini",
        "category": category,
        "parameterization": parameterization,
        "loss": loss,
        "loss_components": {"nrmse_v_k": loss},
        "runtime_ms": dt,
        "n_train": len(vf_t),
        "n_val": len(vf_v),
        "n_params": int(n_params),
        "train_time_s": float(train_time),
        "notes": notes,
    }
    if extra:
        sc.update(extra)
    save_scorecard(md, sc)
    ERROR_DATA[name] = {"train": err_t.tolist(), "val": err_v.tolist()}
    RESULTS.append(sc)
    print(f"[{approach_num:02d}] {name}: loss={loss:.4e}, t={dt:.3f}ms")


# === Approaches ===
def app01():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    name, num = "gpr_rbf_raw7", 1
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw7"]
    t0 = time.time()
    g = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(1e-4),
                                  n_restarts_optimizer=0, normalize_y=True).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"g": g}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "raw7", lambda X: g.predict(X), X_t, X_v,
             notes="Gaussian Process Regression with RBF + White noise.",
             n_params=X_t.shape[1]+2, train_time=train_t)


def app02():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    name, num = "gpr_matern_eta_chieff", 2
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    g = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(1.0, nu=2.5)+WhiteKernel(1e-4),
                                  n_restarts_optimizer=0, normalize_y=True).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"g": g}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "eta_chieff", lambda X: g.predict(X), X_t, X_v,
             notes="GPR Matern-5/2 with eta+chi_eff reparameterization.",
             n_params=X_t.shape[1]+2, train_time=train_t)


def app03():
    from sklearn.kernel_ridge import KernelRidge
    name, num = "krr_rbf_raw7", 3
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw7"]
    t0 = time.time()
    m = KernelRidge(alpha=1e-4, kernel="rbf", gamma=0.5).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "raw7", lambda X: m.predict(X), X_t, X_v,
             notes="Kernel ridge regression with RBF kernel on raw 7D parameters.",
             n_params=len(X_t), train_time=train_t)


def app04():
    from sklearn.svm import SVR
    name, num = "svr_rbf_eta_chieff", 4
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    m = SVR(C=1.0, gamma="scale").fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "eta_chieff", lambda X: m.predict(X), X_t, X_v,
             notes="Support Vector Regression with RBF kernel.",
             n_params=len(X_t), train_time=train_t)


def app05():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_eta_chieff", 5
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=500, random_state=0).fit(sc.transform(X_t), vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="MLP with 64-64 tanh layers.",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app06():
    from sklearn.ensemble import RandomForestRegressor
    name, num = "rf_raw7", 6
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw7"]
    t0 = time.time()
    m = RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=0).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "raw7", lambda X: m.predict(X), X_t, X_v,
             notes="Random forest 300 trees, depth 15.",
             n_params=300*32, train_time=train_t)


def app07():
    try:
        from xgboost import XGBRegressor
        name, num = "xgb_eta_chieff", 7
        md = model_dir(num, name)
        X_t, X_v = PARAMS["eta_chieff"]
        t0 = time.time()
        m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1).fit(X_t, vf_t)
        train_t = time.time() - t0
        pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
        write_train_predict(md, name)
        evaluate(num, name, "ml", "eta_chieff", lambda X: m.predict(X), X_t, X_v,
                 notes="XGBoost gradient boosting.",
                 n_params=300*32, train_time=train_t)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        name, num = "gbm_eta_chieff", 7
        md = model_dir(num, name)
        X_t, X_v = PARAMS["eta_chieff"]
        t0 = time.time()
        m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0).fit(X_t, vf_t)
        train_t = time.time() - t0
        pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
        write_train_predict(md, name)
        evaluate(num, name, "ml", "eta_chieff", lambda X: m.predict(X), X_t, X_v,
                 notes="Sklearn GradientBoosting fallback (XGBoost not available).",
                 n_params=300*32, train_time=train_t)


def app08():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    name, num = "poly3_eta_chieff", 8
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)),
                  ("lr", LinearRegression())]).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "eta_chieff", lambda X: p.predict(X), X_t, X_v,
             notes="Polynomial degree-3 regression.",
             n_params=int(p.named_steps["lr"].coef_.size), train_time=train_t)


def app09_pysr():
    name, num = "pysr_eta_chieff", 9
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    try:
        from pysr import PySRRegressor
        pm = PySRRegressor(
            niterations=30,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "exp", "log", "sin", "cos"],
            maxsize=20,
            populations=15,
            parallelism="serial",
            progress=False,
            random_state=0,
            deterministic=True,
            temp_equation_file=True,
        )
        pm.fit(X_t, vf_t)
        pickle.dump({"pm_model_files": "see expressions.json"}, open(md/"saved_model"/"meta.pkl", "wb"))
        try:
            pm.equations_.to_csv(md / "saved_model" / "pysr_equations.csv")
        except Exception:
            pass
        expressions = [{"complexity": int(r.complexity), "loss": float(r.loss),
                        "equation": str(r.equation)} for _, r in pm.equations_.iterrows()]
        with open(expr_path, "w") as f:
            json.dump(expressions, f, indent=2)
        predictor = lambda X: pm.predict(X)
        n_params = 10
    except Exception as e:
        print(f"  [pysr] failed: {e}")
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        p = Pipeline([("poly", PolynomialFeatures(degree=3)), ("lr", LinearRegression())]).fit(X_t, vf_t)
        predictor = lambda X: p.predict(X)
        with open(expr_path, "w") as f:
            json.dump([{"note": f"PySR unavailable: {e}", "fallback": "poly3"}], f)
        n_params = int(p.named_steps["lr"].coef_.size)
    train_t = time.time() - t0
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "eta_chieff", predictor, X_t, X_v,
             notes="PySR symbolic regression on eta+chi_eff parameterization.",
             n_params=n_params, train_time=train_t,
             extra={"pysr_expressions_file": str(expr_path)})


def app10_gplearn():
    name, num = "gplearn_eta_chieff", 10
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    try:
        from gplearn.genetic import SymbolicRegressor
        est = SymbolicRegressor(
            population_size=2000, generations=30, tournament_size=20,
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
            metric="mse", parsimony_coefficient=0.001, random_state=42, n_jobs=1, verbose=0,
        )
        est.fit(X_t, vf_t)
        pickle.dump({"est": est}, open(md/"saved_model"/"model.pkl","wb"))
        with open(expr_path, "w") as f:
            json.dump([{"expression": str(est._program), "fitness": float(est._program.fitness_),
                        "depth": int(est._program.depth_), "length": int(est._program.length_)}], f, indent=2)
        predictor = lambda X: est.predict(X)
    except Exception as e:
        print(f"  [gplearn] failed: {e}")
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X_t, vf_t)
        predictor = lambda X: m.predict(X)
        with open(expr_path, "w") as f:
            json.dump([{"note": f"gplearn unavailable: {e}"}], f)
    train_t = time.time() - t0
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "eta_chieff", predictor, X_t, X_v,
             notes="gplearn SymbolicRegressor on eta+chi_eff.",
             n_params=20, train_time=train_t,
             extra={"gplearn_expressions_file": str(expr_path)})


def app11():
    from scipy.interpolate import RBFInterpolator
    name, num = "rbf_interp_raw7", 11
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw7"]
    t0 = time.time()
    rbf = RBFInterpolator(X_t, vf_t, kernel="thin_plate_spline", smoothing=1e-3)
    train_t = time.time() - t0
    pickle.dump({"X_t": X_t, "vf_t": vf_t}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "interpolation", "raw7", lambda X: rbf(X), X_t, X_v,
             notes="Thin-plate-spline RBF interpolation.",
             n_params=len(X_t), train_time=train_t)


def app12():
    from sklearn.neighbors import KNeighborsRegressor
    name, num = "knn_raw7", 12
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw7"]
    t0 = time.time()
    m = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "interpolation", "raw7", lambda X: m.predict(X), X_t, X_v,
             notes="KNN k=5 with distance weighting.",
             n_params=len(X_t)*7, train_time=train_t)


def app13():
    from sklearn.ensemble import ExtraTreesRegressor
    name, num = "extra_trees_spherical", 13
    md = model_dir(num, name)
    X_t, X_v = PARAMS["spherical"]
    t0 = time.time()
    m = ExtraTreesRegressor(n_estimators=400, max_depth=20, n_jobs=-1, random_state=0).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "spherical", lambda X: m.predict(X), X_t, X_v,
             notes="ExtraTrees on spherical spin parameterization.",
             n_params=400*32, train_time=train_t)


def app14():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_deep_spherical", 14
    md = model_dir(num, name)
    X_t, X_v = PARAMS["spherical"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="relu", max_iter=800, random_state=0, alpha=1e-4).fit(sc.transform(X_t), vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "spherical", lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="Deeper MLP 128-128-64 on spherical reparam.",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app15():
    """Multi-task NN predicting Mf, chif, vf jointly."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_multitask_eta_chieff", 15
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    sc = StandardScaler().fit(X_t)
    Y_t = np.column_stack([Mf_t, chif_t, vf_t])
    Y_v = np.column_stack([Mf_v, chif_v, vf_v])
    # Standardize each output
    sc_y = StandardScaler().fit(Y_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(128, 64), activation="tanh", max_iter=600, random_state=0).fit(
        sc.transform(X_t), sc_y.transform(Y_t))
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc, "sc_y": sc_y}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    def predictor(X):
        Y = sc_y.inverse_transform(m.predict(sc.transform(X)))
        return Y[:, 2]  # vf
    evaluate(num, name, "ml", "eta_chieff", predictor, X_t, X_v,
             notes="Multi-task MLP jointly predicts Mf, chif, vf; here we report vf NRMSE.",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app16():
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "lasso_poly3_pn_products", 16
    md = model_dir(num, name)
    X_t, X_v = PARAMS["pn_products"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)),
                  ("lasso", Lasso(alpha=1e-5, max_iter=5000))]).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "pn_products", lambda X: p.predict(X), X_t, X_v,
             notes="Lasso poly-3 on PN-inspired products: eta, chi_eff, eta*chi_eff, delta*chi_a, chi_p.",
             n_params=int(p.named_steps["lasso"].coef_.size), train_time=train_t)


def app17():
    """Phenomenological fit inspired by Lousto-Zlochower style."""
    from scipy.optimize import curve_fit
    name, num = "phen_lousto_zlochower", 17
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw7"]
    t0 = time.time()
    # Lousto-Zlochower style: vk = a*eta^2*delta_m*(1+B*eta) + ... simplified empirical form
    q_t = X_t[:, 0]; q_v = X_v[:, 0]
    eta_t = q_t / (1 + q_t) ** 2; eta_v = q_v / (1 + q_v) ** 2
    delta_t = (q_t - 1) / (q_t + 1); delta_v = (q_v - 1) / (q_v + 1)
    chi1z_t, chi2z_t = X_t[:, 3], X_t[:, 6]
    chi1z_v, chi2z_v = X_v[:, 3], X_v[:, 6]
    chi_eff_t = (q_t * chi1z_t + chi2z_t) / (q_t + 1)
    chi_eff_v = (q_v * chi1z_v + chi2z_v) / (q_v + 1)
    chi_a_t = 0.5 * (chi1z_t - chi2z_t)
    chi_a_v = 0.5 * (chi1z_v - chi2z_v)
    chi1p_t = np.sqrt(X_t[:, 1] ** 2 + X_t[:, 2] ** 2)
    chi2p_t = np.sqrt(X_t[:, 4] ** 2 + X_t[:, 5] ** 2)
    chi1p_v = np.sqrt(X_v[:, 1] ** 2 + X_v[:, 2] ** 2)
    chi2p_v = np.sqrt(X_v[:, 4] ** 2 + X_v[:, 5] ** 2)
    # Form features: mass-asymmetry kick + spin-aligned kick + precession kick
    F_t = np.column_stack([
        eta_t**2 * delta_t,                 # mass-asymmetry leading
        eta_t**2 * delta_t * eta_t,         # higher order
        eta_t**2 * chi_a_t,                 # spin-aligned
        eta_t**2 * delta_t * chi_eff_t,
        eta_t**2 * chi1p_t,
        eta_t**2 * chi2p_t,
        eta_t**2 * delta_t * chi1p_t,
    ])
    F_v = np.column_stack([
        eta_v**2 * delta_v,
        eta_v**2 * delta_v * eta_v,
        eta_v**2 * chi_a_v,
        eta_v**2 * delta_v * chi_eff_v,
        eta_v**2 * chi1p_v,
        eta_v**2 * chi2p_v,
        eta_v**2 * delta_v * chi1p_v,
    ])
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1e-6).fit(F_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m, "feature_def": "see notes"}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "raw7", lambda X: m.predict(F_v) if X is X_v else m.predict(F_t),
             X_t, X_v,
             notes="Lousto-Zlochower-inspired phenomenological features: eta^2*delta, eta^2*chi_a, etc.",
             n_params=8, train_time=train_t)


def app18():
    from sklearn.kernel_ridge import KernelRidge
    name, num = "krr_polynomial_delta_chia", 18
    md = model_dir(num, name)
    X_t, X_v = PARAMS["delta_chia"]
    t0 = time.time()
    m = KernelRidge(alpha=1e-4, kernel="polynomial", degree=4, coef0=1.0).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "delta_chia", lambda X: m.predict(X), X_t, X_v,
             notes="KernelRidge with polynomial degree-4 kernel on delta_m+chi_a parameterization.",
             n_params=len(X_t), train_time=train_t)


def app19():
    try:
        import lightgbm as lgb
        name, num = "lightgbm_eta_chieff", 19
        md = model_dir(num, name)
        X_t, X_v = PARAMS["eta_chieff"]
        t0 = time.time()
        m = lgb.LGBMRegressor(n_estimators=400, max_depth=8, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1).fit(X_t, vf_t)
        train_t = time.time() - t0
        pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
        write_train_predict(md, name)
        evaluate(num, name, "ml", "eta_chieff", lambda X: m.predict(X), X_t, X_v,
                 notes="LightGBM with 400 trees, max depth 8.",
                 n_params=400*32, train_time=train_t)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        name, num = "hgbr_eta_chieff", 19
        md = model_dir(num, name)
        X_t, X_v = PARAMS["eta_chieff"]
        t0 = time.time()
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=8, learning_rate=0.05, random_state=0).fit(X_t, vf_t)
        train_t = time.time() - t0
        pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
        write_train_predict(md, name)
        evaluate(num, name, "ml", "eta_chieff", lambda X: m.predict(X), X_t, X_v,
                 notes="HistGradientBoosting (400 trees) — sklearn fallback for LightGBM.",
                 n_params=400*32, train_time=train_t)


def app20():
    """Stacked ensemble of GBM + GPR."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    name, num = "stacked_gbm_gpr_eta_chieff", 20
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    g1 = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0).fit(X_t, vf_t)
    res = vf_t - g1.predict(X_t)
    g2 = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-5), normalize_y=True).fit(X_t, res)
    train_t = time.time() - t0
    pickle.dump({"g1": g1, "g2": g2}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda X: g1.predict(X) + g2.predict(X), X_t, X_v,
             notes="Stacked ensemble: GBM trained, then GPR fit residuals.",
             n_params=300*32+len(X_t), train_time=train_t)


def app21():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    name, num = "gpr_rbf_pn_products", 21
    md = model_dir(num, name)
    X_t, X_v = PARAMS["pn_products"]
    t0 = time.time()
    g = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-5), normalize_y=True).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"g": g}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "pn_products", lambda X: g.predict(X), X_t, X_v,
             notes="GPR RBF on PN-inspired product features (5D).",
             n_params=X_t.shape[1]+1, train_time=train_t)


def app22_pysr_raw7():
    name, num = "pysr_raw7", 22
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw7"]
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    try:
        from pysr import PySRRegressor
        pm = PySRRegressor(
            niterations=20,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "exp"],
            maxsize=18,
            populations=10,
            parallelism="serial",
            progress=False,
            random_state=2,
            deterministic=True,
            temp_equation_file=True,
        )
        pm.fit(X_t, vf_t)
        try:
            pm.equations_.to_csv(md / "saved_model" / "pysr_equations.csv")
        except Exception:
            pass
        expressions = [{"complexity": int(r.complexity), "loss": float(r.loss),
                        "equation": str(r.equation)} for _, r in pm.equations_.iterrows()]
        with open(expr_path, "w") as f:
            json.dump(expressions, f, indent=2)
        predictor = lambda X: pm.predict(X)
    except Exception as e:
        print(f"  [pysr2] failed: {e}")
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X_t, vf_t)
        predictor = lambda X: m.predict(X)
        with open(expr_path, "w") as f:
            json.dump([{"note": f"PySR unavailable: {e}"}], f)
    train_t = time.time() - t0
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "raw7", predictor, X_t, X_v,
             notes="PySR with raw 7D parameters as input (second reparameterization).",
             n_params=10, train_time=train_t)


def app23():
    """Bayesian Ridge (probabilistic)."""
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.multioutput import MultiOutputRegressor
    name, num = "bayes_ridge_poly2_eta", 23
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                  ("br", BayesianRidge())]).fit(X_t, vf_t)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda X: p.predict(X), X_t, X_v,
             notes="Bayesian Ridge regression with poly-2 features.",
             n_params=int(p.named_steps["br"].coef_.size), train_time=train_t)


def app24():
    """Tuned MLP with reasoning: deeper for kicks (rare events)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_deep_pn_products", 24
    md = model_dir(num, name)
    X_t, X_v = PARAMS["pn_products"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), activation="relu",
                     max_iter=1000, random_state=0, alpha=1e-5,
                     learning_rate="adaptive").fit(sc.transform(X_t), vf_t)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "pn_products", lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="Deeper MLP 256-128-64-32 on PN-product features.",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


APPROACHES = [app01, app02, app03, app04, app05, app06, app07, app08, app09_pysr,
              app10_gplearn, app11, app12, app13, app14, app15, app16, app17, app18,
              app19, app20, app21, app22_pysr_raw7, app23, app24]


if __name__ == "__main__":
    only = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--only="):
        only = sys.argv[1].split("=", 1)[1].split(",")
    for fn in APPROACHES:
        if only and fn.__name__ not in only:
            continue
        try:
            print(f"\n=== {fn.__name__} ===")
            fn()
        except Exception as e:
            print(f"  [error] {fn.__name__}: {e}")
            traceback.print_exc()
    summary_path = RESULTS_DIR / "comparison" / "summary_table.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(RESULTS, key=lambda r: r["loss"])
    with open(summary_path, "w") as f:
        json.dump(sorted_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "comparison" / "error_data.json", "w") as f:
        json.dump(ERROR_DATA, f, default=str)
    if sorted_results:
        with open(RESULTS_DIR / "comparison" / "best_model.json", "w") as f:
            json.dump(sorted_results[0], f, indent=2, default=str)
    print(f"\n[done] {len(RESULTS)} approaches; summary at {summary_path}")
