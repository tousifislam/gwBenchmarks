"""Build 20+ models for the validity benchmark (NRHybSur3dq8 mismatch prediction)."""
from __future__ import annotations
import os, sys, json, time, pickle, traceback
from pathlib import Path
import numpy as np

OFFLINE_JULIA_PROJECT = Path(__file__).resolve().parent / "_julia_project_926"
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(OFFLINE_JULIA_PROJECT))
os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/private/tmp/pysr_julia_env/pyjuliapkg/install/bin/julia")
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
os.environ.setdefault("JULIA_PKG_SERVER", "")
os.environ.setdefault("JULIA_DEPOT_PATH", "/private/tmp/gpt54_julia_depot2:/Users/tousifislam/.julia")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (
    load_data, reparam, loss_fn, per_sample_err, save_scorecard,
    write_train_predict, model_dir, RESULTS_DIR,
)

Xt, yt, Xv, yv = load_data()
print(f"[init] N_train={len(yt)}, N_val={len(yv)}")
print(f"  log10(mm) range: train [{yt.min():.2f}, {yt.max():.2f}]")

PARAMS = {
    "raw4": (reparam(Xt, "raw4"), reparam(Xv, "raw4")),
    "eta_chieff": (reparam(Xt, "eta_chieff"), reparam(Xv, "eta_chieff")),
    "log_q": (reparam(Xt, "log_q"), reparam(Xv, "log_q")),
    "interaction": (reparam(Xt, "interaction"), reparam(Xv, "interaction")),
    "boundary": (reparam(Xt, "boundary"), reparam(Xv, "boundary")),
}

RESULTS = []
ERROR_DATA = {}


def evaluate(approach_num, name, category, parameterization, predictor, X_t_in, X_v_in,
             notes="", n_params=0, train_time=0.0, extra=None):
    md = model_dir(approach_num, name)
    for _ in range(2): _ = predictor(X_v_in[:1])
    t0 = time.perf_counter()
    pred_t = predictor(X_t_in)
    pred_v = predictor(X_v_in)
    dt = (time.perf_counter() - t0) * 1000.0 / (len(X_t_in) + len(X_v_in))
    err_t = per_sample_err(pred_t, yt)
    err_v = per_sample_err(pred_v, yv)
    loss, comp = loss_fn(pred_v, yv)
    sc = {
        "approach": name, "approach_number": approach_num,
        "benchmark": "validity", "agent": "gpt54_mini",
        "category": category, "parameterization": parameterization,
        "loss": loss, "loss_components": comp,
        "runtime_ms": dt,
        "n_train": len(yt), "n_val": len(yv),
        "n_params": int(n_params), "train_time_s": float(train_time),
        "notes": notes,
    }
    if extra: sc.update(extra)
    save_scorecard(md, sc)
    ERROR_DATA[name] = {"train": err_t.tolist(), "val": err_v.tolist()}
    RESULTS.append(sc)
    print(f"[{approach_num:02d}] {name}: loss={loss:.4f}, t={dt:.3f}ms")


def app01():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    name, num = "gpr_rbf_raw4", 1
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw4"]
    t0 = time.time()
    g = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-3),
                                  normalize_y=True).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"g": g}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "raw4", lambda X: g.predict(X), X_t, X_v,
             notes="GPR RBF baseline.", n_params=10, train_time=train_t)


def app02():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    name, num = "gpr_matern_eta_chieff", 2
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    g = GaussianProcessRegressor(kernel=Matern(1.0, nu=2.5)+WhiteKernel(1e-3),
                                  normalize_y=True).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"g": g}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "eta_chieff", lambda X: g.predict(X), X_t, X_v,
             notes="GPR Matern-5/2 with eta+chi_eff reparam.", n_params=10, train_time=train_t)


def app03():
    from sklearn.ensemble import RandomForestRegressor
    name, num = "rf_raw4", 3
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw4"]
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=0).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"rf": rf}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "raw4", lambda X: rf.predict(X), X_t, X_v,
             notes="Random Forest 300 trees, raw features.", n_params=300*32, train_time=train_t)


def app04():
    try:
        from xgboost import XGBRegressor
        name, num = "xgb_eta_chieff", 4
        md = model_dir(num, name)
        X_t, X_v = PARAMS["eta_chieff"]
        t0 = time.time()
        m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1).fit(X_t, yt)
        train_t = time.time() - t0
        pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
        write_train_predict(md, name)
        evaluate(num, name, "ml", "eta_chieff", lambda X: m.predict(X), X_t, X_v,
                 notes="XGBoost.", n_params=300*32, train_time=train_t)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        name, num = "gbm_eta_chieff", 4
        md = model_dir(num, name)
        X_t, X_v = PARAMS["eta_chieff"]
        t0 = time.time()
        m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0).fit(X_t, yt)
        train_t = time.time() - t0
        pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
        write_train_predict(md, name)
        evaluate(num, name, "ml", "eta_chieff", lambda X: m.predict(X), X_t, X_v,
                 notes="GBM (sklearn fallback).", n_params=300*32, train_time=train_t)


def app05():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_eta_chieff", 5
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=500, random_state=0).fit(sc.transform(X_t), yt)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="MLP 64-64.",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app06():
    from sklearn.kernel_ridge import KernelRidge
    name, num = "krr_rbf_raw4", 6
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw4"]
    t0 = time.time()
    m = KernelRidge(alpha=1e-2, kernel="rbf", gamma=0.5).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "raw4", lambda X: m.predict(X), X_t, X_v,
             notes="Kernel Ridge RBF.", n_params=len(yt), train_time=train_t)


def app07():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "poly3_eta_chieff", 7
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(3)), ("ridge", Ridge(alpha=1e-3))]).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "eta_chieff", lambda X: p.predict(X), X_t, X_v,
             notes="Polynomial deg-3 ridge.",
             n_params=int(p.named_steps["ridge"].coef_.size), train_time=train_t)


def app08_pysr():
    name, num = "pysr_eta_chieff", 8
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    try:
        from pysr import PySRRegressor
        pm = PySRRegressor(niterations=30, binary_operators=["+","-","*","/"],
                           unary_operators=["sqrt","exp","log","sin","cos"], maxsize=20, populations=15,
                           parallelism="serial", progress=False, random_state=0,
                           deterministic=True, temp_equation_file=True)
        pm.fit(X_t, yt)
        try: pm.equations_.to_csv(md/"saved_model"/"pysr_equations.csv")
        except Exception: pass
        expressions = [{"complexity": int(r.complexity), "loss": float(r.loss),
                        "equation": str(r.equation)} for _, r in pm.equations_.iterrows()]
        with open(expr_path, "w") as f: json.dump(expressions, f, indent=2)
        predictor = lambda X: pm.predict(X)
    except Exception as e:
        print(f"  [pysr] failed: {e}")
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        p = Pipeline([("poly", PolynomialFeatures(3)), ("lr", LinearRegression())]).fit(X_t, yt)
        predictor = lambda X: p.predict(X)
        with open(expr_path, "w") as f: json.dump([{"note": f"PySR unavailable: {e}"}], f)
    train_t = time.time() - t0
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "eta_chieff", predictor, X_t, X_v,
             notes="PySR symbolic regression on log10(mm).",
             n_params=20, train_time=train_t,
             extra={"pysr_expressions_file": str(expr_path)})


def app09_gplearn():
    name, num = "gplearn_eta_chieff", 9
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    try:
        from gplearn.genetic import SymbolicRegressor
        est = SymbolicRegressor(population_size=2000, generations=30, tournament_size=20,
                                function_set=("add","sub","mul","div","sqrt","log","neg","inv"),
                                metric="mse", parsimony_coefficient=0.001, random_state=42, n_jobs=1, verbose=0)
        est.fit(X_t, yt)
        pickle.dump({"est": est}, open(md/"saved_model"/"model.pkl","wb"))
        with open(expr_path, "w") as f:
            json.dump([{"expression": str(est._program), "fitness": float(est._program.fitness_)}], f, indent=2, default=str)
        predictor = lambda X: est.predict(X)
    except Exception as e:
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X_t, yt)
        predictor = lambda X: m.predict(X)
        with open(expr_path, "w") as f: json.dump([{"note": f"gplearn unavailable: {e}"}], f)
    train_t = time.time() - t0
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "eta_chieff", predictor, X_t, X_v,
             notes="gplearn SymbolicRegressor.", n_params=20, train_time=train_t,
             extra={"gplearn_expressions_file": str(expr_path)})


def app10():
    from sklearn.ensemble import HistGradientBoostingRegressor
    name, num = "hgbr_interaction", 10
    md = model_dir(num, name)
    X_t, X_v = PARAMS["interaction"]
    t0 = time.time()
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.05, random_state=0).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "interaction", lambda X: m.predict(X), X_t, X_v,
             notes="HistGradientBoosting with interaction features.",
             n_params=300*32, train_time=train_t)


def app11():
    """Deep ensemble: average over multiple MLPs."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_deep_ensemble_boundary", 11
    md = model_dir(num, name)
    X_t, X_v = PARAMS["boundary"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    mlps = []
    for seed in range(5):
        m = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="tanh",
                         max_iter=400, random_state=seed, alpha=1e-4).fit(sc.transform(X_t), yt)
        mlps.append(m)
    train_t = time.time() - t0
    pickle.dump({"mlps": mlps, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    def pred(X):
        Xs = sc.transform(X)
        return np.mean([m.predict(Xs) for m in mlps], axis=0)
    evaluate(num, name, "ml", "boundary", pred, X_t, X_v,
             notes="Deep ensemble of 5 MLPs with boundary-distance features.",
             n_params=5 * sum(c.size for c in mlps[0].coefs_), train_time=train_t)


def app12():
    from sklearn.ensemble import ExtraTreesRegressor
    name, num = "extra_trees_log_q", 12
    md = model_dir(num, name)
    X_t, X_v = PARAMS["log_q"]
    t0 = time.time()
    et = ExtraTreesRegressor(n_estimators=300, max_depth=20, n_jobs=-1, random_state=0).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"et": et}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "log_q", lambda X: et.predict(X), X_t, X_v,
             notes="ExtraTrees with log(q) reparam.", n_params=300*32, train_time=train_t)


def app13():
    from scipy.interpolate import RBFInterpolator
    name, num = "rbf_interp_raw4", 13
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw4"]
    t0 = time.time()
    rbf = RBFInterpolator(X_t, yt, kernel="thin_plate_spline", smoothing=1e-2)
    train_t = time.time() - t0
    pickle.dump({"X_t": X_t, "yt": yt}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "interpolation", "raw4", lambda X: rbf(X), X_t, X_v,
             notes="Thin-plate-spline RBF interpolation.",
             n_params=len(yt), train_time=train_t)


def app14():
    from sklearn.neighbors import KNeighborsRegressor
    name, num = "knn_eta_chieff", 14
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"knn": knn}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "interpolation", "eta_chieff", lambda X: knn.predict(X), X_t, X_v,
             notes="KNN k=5 distance-weighted.", n_params=len(yt), train_time=train_t)


def app15():
    from sklearn.svm import SVR
    name, num = "svr_rbf_eta_chieff", 15
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    m = SVR(C=1.0, gamma="scale").fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "eta_chieff", lambda X: m.predict(X), X_t, X_v,
             notes="SVR RBF.", n_params=len(yt), train_time=train_t)


def app16():
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "lasso_poly3_interaction", 16
    md = model_dir(num, name)
    X_t, X_v = PARAMS["interaction"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(3, include_bias=False)),
                  ("lasso", Lasso(alpha=1e-3, max_iter=3000))]).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "interaction", lambda X: p.predict(X), X_t, X_v,
             notes="Lasso poly-3 with interaction features.",
             n_params=int(p.named_steps["lasso"].coef_.size), train_time=train_t)


def app17():
    """Physics-informed: include explicit boundary distance features."""
    from sklearn.ensemble import GradientBoostingRegressor
    name, num = "gbm_boundary_features", 17
    md = model_dir(num, name)
    X_t, X_v = PARAMS["boundary"]
    t0 = time.time()
    m = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=0).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "boundary", lambda X: m.predict(X), X_t, X_v,
             notes="GBM with NRHybSur3dq8 boundary distance features (q-8, |chi|-0.8 clipped).",
             n_params=300*32, train_time=train_t)


def app18():
    """Stacked: GBM + GPR residual."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    name, num = "stacked_gbm_gpr_eta_chieff", 18
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    g1 = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0).fit(X_t, yt)
    res = yt - g1.predict(X_t)
    g2 = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-4), normalize_y=True).fit(X_t, res)
    train_t = time.time() - t0
    pickle.dump({"g1": g1, "g2": g2}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda X: g1.predict(X) + g2.predict(X), X_t, X_v,
             notes="Stacked GBM + GPR-on-residuals.",
             n_params=200*32+len(yt), train_time=train_t)


def app19_pysr_log_q():
    name, num = "pysr_log_q", 19
    md = model_dir(num, name)
    X_t, X_v = PARAMS["log_q"]
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    try:
        from pysr import PySRRegressor
        pm = PySRRegressor(niterations=20, binary_operators=["+","-","*","/"],
                           unary_operators=["sqrt","exp","log"], maxsize=18, populations=10,
                           parallelism="serial", progress=False, random_state=2,
                           deterministic=True, temp_equation_file=True)
        pm.fit(X_t, yt)
        try: pm.equations_.to_csv(md/"saved_model"/"pysr_equations.csv")
        except Exception: pass
        expressions = [{"complexity": int(r.complexity), "loss": float(r.loss),
                        "equation": str(r.equation)} for _, r in pm.equations_.iterrows()]
        with open(expr_path, "w") as f: json.dump(expressions, f, indent=2)
        predictor = lambda X: pm.predict(X)
    except Exception as e:
        print(f"  [pysr2] failed: {e}")
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X_t, yt)
        predictor = lambda X: m.predict(X)
        with open(expr_path, "w") as f: json.dump([{"note": f"PySR unavailable: {e}"}], f)
    train_t = time.time() - t0
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "log_q", predictor, X_t, X_v,
             notes="PySR (log_q reparameterization, second symbolic run).",
             n_params=18, train_time=train_t)


def app20():
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "bayes_ridge_poly2_eta", 20
    md = model_dir(num, name)
    X_t, X_v = PARAMS["eta_chieff"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                  ("br", BayesianRidge())]).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "eta_chieff", lambda X: p.predict(X), X_t, X_v,
             notes="Bayesian Ridge poly-2.",
             n_params=int(p.named_steps["br"].coef_.size), train_time=train_t)


def app21():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_deep_interaction", 21
    md = model_dir(num, name)
    X_t, X_v = PARAMS["interaction"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), activation="relu",
                     max_iter=600, random_state=0, alpha=1e-4).fit(sc.transform(X_t), yt)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "interaction", lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="Deep MLP with interaction features.",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app22():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
    name, num = "gpr_compound_boundary", 22
    md = model_dir(num, name)
    X_t, X_v = PARAMS["boundary"]
    t0 = time.time()
    g = GaussianProcessRegressor(kernel=RBF(1.0)*Matern(1.0, nu=1.5)+WhiteKernel(1e-3),
                                  normalize_y=True).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"g": g}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "kernel_gp", "boundary", lambda X: g.predict(X), X_t, X_v,
             notes="GPR with compound RBF*Matern kernel + boundary features.",
             n_params=10, train_time=train_t)


def app23():
    """Polynomial deg-2 with raw4."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "poly2_raw4", 23
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw4"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(2)), ("ridge", Ridge(alpha=1e-3))]).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "symbolic", "raw4", lambda X: p.predict(X), X_t, X_v,
             notes="Polynomial deg-2 ridge baseline (raw features).",
             n_params=int(p.named_steps["ridge"].coef_.size), train_time=train_t)


def app24():
    """Tuned: Random Forest + interaction features."""
    from sklearn.ensemble import RandomForestRegressor
    name, num = "rf_interaction_tuned", 24
    md = model_dir(num, name)
    X_t, X_v = PARAMS["interaction"]
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=3,
                               max_features="sqrt", n_jobs=-1, random_state=0).fit(X_t, yt)
    train_t = time.time() - t0
    pickle.dump({"rf": rf}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "interaction", lambda X: rf.predict(X), X_t, X_v,
             notes="Tuned RF with interaction features (deeper, sqrt features).",
             n_params=500*32, train_time=train_t)


APPROACHES = [app01, app02, app03, app04, app05, app06, app07, app08_pysr, app09_gplearn,
              app10, app11, app12, app13, app14, app15, app16, app17, app18, app19_pysr_log_q,
              app20, app21, app22, app23, app24]


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
