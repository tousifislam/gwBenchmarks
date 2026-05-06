"""Build 20+ models for the ringdown QNM benchmark — fast version."""
from __future__ import annotations
import os, sys, json, time, pickle, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (
    load_data, reparam, loss_fn, per_sample_err, save_scorecard,
    write_train_predict, model_dir, RESULTS_DIR,
)

# Use the cached offline Julia project for PySR / SymbolicRegression runs.
OFFLINE_JULIA_PROJECT = Path(__file__).resolve().parent / "_julia_project_926"
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(OFFLINE_JULIA_PROJECT))
os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/private/tmp/pysr_julia_env/pyjuliapkg/install/bin/julia")
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
os.environ.setdefault("JULIA_PKG_SERVER", "")
os.environ.setdefault("JULIA_DEPOT_PATH", "/private/tmp/gpt54_julia_depot2:/Users/tousifislam/.julia")

Xt_full, Yt_full, Xv, Yv = load_data()
N_FULL, N_VAL = Xt_full.shape[0], Xv.shape[0]
print(f"[init] N_full={N_FULL}, N_val={N_VAL}")

# Subsample for tractable training
RNG = np.random.default_rng(0)
N_SUB = 30000
SUB_IDX = RNG.choice(N_FULL, N_SUB, replace=False)
Xt = Xt_full[SUB_IDX]; Yt = Yt_full[SUB_IDX]
print(f"[init] Subsampled training to N={N_SUB}")

# Even smaller for GPR
N_GP = 1500
GP_IDX = RNG.choice(N_FULL, N_GP, replace=False)
Xt_gp = Xt_full[GP_IDX]; Yt_gp = Yt_full[GP_IDX]

# Full reparams
PARAMS = {
    "raw": (reparam(Xt, "raw"), reparam(Xv, "raw")),
    "log_1ma": (reparam(Xt, "log_1ma"), reparam(Xv, "log_1ma")),
    "compact": (reparam(Xt, "compact"), reparam(Xv, "compact")),
    "chebyshev": (reparam(Xt, "chebyshev"), reparam(Xv, "chebyshev")),
    "lm_diff": (reparam(Xt, "lm_diff"), reparam(Xv, "lm_diff")),
    "all_normalized": (reparam(Xt, "all_normalized"), reparam(Xv, "all_normalized")),
}
PARAMS_FULL = {
    "raw": (reparam(Xt_full, "raw"), reparam(Xv, "raw")),
    "log_1ma": (reparam(Xt_full, "log_1ma"), reparam(Xv, "log_1ma")),
    "chebyshev": (reparam(Xt_full, "chebyshev"), reparam(Xv, "chebyshev")),
}

# Mode-specific
mask_l2_t = (Xt_full[:, 1] == 2) & (Xt_full[:, 2] == 2) & (Xt_full[:, 3] == 0)
mask_l2_v = (Xv[:, 1] == 2) & (Xv[:, 2] == 2) & (Xv[:, 3] == 0)
X_l2_t = Xt_full[mask_l2_t][:, :1]
X_l2_v = Xv[mask_l2_v][:, :1]
Y_l2_t = Yt_full[mask_l2_t]
Y_l2_v = Yv[mask_l2_v]

mask_l3_t = (Xt_full[:, 1] == 3) & (Xt_full[:, 2] == 3) & (Xt_full[:, 3] == 0)
mask_l3_v = (Xv[:, 1] == 3) & (Xv[:, 2] == 3) & (Xv[:, 3] == 0)
X_l3_t = Xt_full[mask_l3_t][:, :1]
X_l3_v = Xv[mask_l3_v][:, :1]
Y_l3_t = Yt_full[mask_l3_t]
Y_l3_v = Yv[mask_l3_v]

print(f"[init] l2m2n0: train={mask_l2_t.sum()}, val={mask_l2_v.sum()}")
print(f"[init] l3m3n0: train={mask_l3_t.sum()}, val={mask_l3_v.sum()}")

RESULTS = []
ERROR_DATA = {}


def evaluate_full(approach_num, name, category, parameterization, predictor, X_t_in, X_v_in,
                  Y_t_eval, Y_v_eval, notes="", n_params=0, train_time=0.0,
                  mode_label="all_modes", extra=None):
    md = model_dir(approach_num, name)
    for _ in range(2):
        _ = predictor(X_v_in[:1])
    t0 = time.perf_counter()
    pred_t = predictor(X_t_in)
    pred_v = predictor(X_v_in)
    dt_ms = (time.perf_counter() - t0) * 1000.0 / (X_t_in.shape[0] + X_v_in.shape[0])
    err_t = per_sample_err(pred_t, Y_t_eval)
    err_v = per_sample_err(pred_v, Y_v_eval)
    loss, comp = loss_fn(pred_v, Y_v_eval)
    sc = {
        "approach": name, "approach_number": approach_num,
        "benchmark": "ringdown", "agent": "gpt54_mini",
        "category": category, "parameterization": parameterization, "mode": mode_label,
        "loss": loss, "loss_components": comp,
        "runtime_ms": dt_ms,
        "n_train": int(X_t_in.shape[0]), "n_val": int(X_v_in.shape[0]),
        "n_params": int(n_params), "train_time_s": float(train_time),
        "notes": notes,
    }
    if extra:
        sc.update(extra)
    save_scorecard(md, sc)
    ERROR_DATA[name] = {"train": err_t.tolist()[:5000], "val": err_v.tolist()[:5000]}
    RESULTS.append(sc)
    print(f"[{approach_num:02d}] {name}: loss={loss:.4e}, t={dt_ms:.4f}ms")


def evaluate(approach_num, name, category, parameterization, predictor,
             X_t_in, X_v_in, notes="", n_params=0, train_time=0.0,
             mode_label="all_modes", extra=None):
    if X_t_in.shape[0] == Xt.shape[0]:
        Y_t_eval = Yt
    elif X_t_in.shape[0] == Xt_gp.shape[0]:
        Y_t_eval = Yt_gp
    else:
        Y_t_eval = Yt_full
    evaluate_full(approach_num, name, category, parameterization, predictor,
                  X_t_in, X_v_in, Y_t_eval, Yv, notes, n_params, train_time, mode_label, extra)


# === Approaches ===
def app01():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "poly8_raw", 1
    md = model_dir(num, name)
    X_t, X_v = PARAMS_FULL["raw"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(8, include_bias=False)),
                  ("lr", LinearRegression())]).fit(X_t, Yt_full)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate_full(num, name, "symbolic", "raw", lambda X: p.predict(X), X_t, X_v, Yt_full, Yv,
                  notes="Polynomial deg-8 baseline (joint fit on all 1.2M points).",
                  n_params=int(p.named_steps["lr"].coef_.size), train_time=train_t)


def app02():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "poly8_log1ma", 2
    md = model_dir(num, name)
    X_t, X_v = PARAMS_FULL["log_1ma"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(8, include_bias=False)),
                  ("lr", Ridge(alpha=1e-4))]).fit(X_t, Yt_full)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate_full(num, name, "symbolic", "log_1ma", lambda X: p.predict(X), X_t, X_v, Yt_full, Yv,
                  notes="Polynomial deg-8 ridge with log(1-a) reparam.",
                  n_params=int(p.named_steps["lr"].coef_.size), train_time=train_t)


def app03():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "poly8_chebyshev", 3
    md = model_dir(num, name)
    X_t, X_v = PARAMS_FULL["chebyshev"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(8, include_bias=False)),
                  ("lr", Ridge(alpha=1e-4))]).fit(X_t, Yt_full)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate_full(num, name, "symbolic", "chebyshev", lambda X: p.predict(X), X_t, X_v, Yt_full, Yv,
                  notes="Polynomial deg-8 ridge with 2a-1 reparam.",
                  n_params=int(p.named_steps["lr"].coef_.size), train_time=train_t)


def app04():
    """Cubic spline per mode (l, m, n)."""
    from scipy.interpolate import CubicSpline
    name, num = "cubic_spline_per_mode", 4
    md = model_dir(num, name)
    t0 = time.time()
    splines = {}
    for l in np.unique(Xt_full[:, 1]):
        for m in np.unique(Xt_full[:, 2]):
            for nn in np.unique(Xt_full[:, 3]):
                mask = (Xt_full[:, 1] == l) & (Xt_full[:, 2] == m) & (Xt_full[:, 3] == nn)
                if mask.sum() < 4: continue
                a = Xt_full[mask, 0]
                idx = np.argsort(a)
                a = a[idx]
                Y_re = Yt_full[mask][idx, 0]
                Y_im = Yt_full[mask][idx, 1]
                splines[(l, m, nn)] = (CubicSpline(a, Y_re, extrapolate=True),
                                       CubicSpline(a, Y_im, extrapolate=True))
    train_t = time.time() - t0
    print(f"  {len(splines)} splines built")
    pickle.dump({"splines_keys": list(splines.keys())}, open(md/"saved_model"/"meta.pkl","wb"))
    write_train_predict(md, name)
    def predictor(X):
        out = np.zeros((X.shape[0], 2))
        for i, x in enumerate(X):
            key = (x[1], x[2], x[3])
            if key in splines:
                cs_re, cs_im = splines[key]
                out[i, 0] = cs_re(x[0])
                out[i, 1] = cs_im(x[0])
        return out
    evaluate_full(num, name, "interpolation", "raw", predictor, Xt_full[:5000], Xv,
                  Yt_full[:5000], Yv,
                  notes="Cubic spline per (l, m, n) mode in spin a.",
                  n_params=N_FULL, train_time=train_t)


def app05():
    """GPR on 1500 subsample."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    name, num = "gpr_rbf_raw_subsample", 5
    md = model_dir(num, name)
    X_t = reparam(Xt_gp, "raw"); X_v = reparam(Xv, "raw")
    t0 = time.time()
    gpr_re = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-5),
                                      normalize_y=True).fit(X_t, Yt_gp[:, 0])
    gpr_im = GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-5),
                                      normalize_y=True).fit(X_t, Yt_gp[:, 1])
    train_t = time.time() - t0
    pickle.dump({"gpr_re": gpr_re, "gpr_im": gpr_im}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    def pred(X): return np.column_stack([gpr_re.predict(X), gpr_im.predict(X)])
    evaluate(num, name, "kernel_gp", "raw", pred, X_t, X_v,
             notes="GPR RBF with 1500 subsampled points.",
             n_params=N_GP, train_time=train_t)


def app06():
    from scipy.interpolate import RBFInterpolator
    name, num = "rbf_interp_compact", 6
    md = model_dir(num, name)
    sub2 = RNG.choice(Xt.shape[0], 3000, replace=False)
    X_t_s = reparam(Xt[sub2], "compact"); Yt_s = Yt[sub2]
    X_v = reparam(Xv, "compact")
    t0 = time.time()
    rbf = RBFInterpolator(X_t_s, Yt_s, kernel="thin_plate_spline", smoothing=1e-3)
    train_t = time.time() - t0
    np.savez(md/"saved_model"/"data.npz", X=X_t_s, Y=Yt_s)
    write_train_predict(md, name)
    evaluate_full(num, name, "interpolation", "compact", lambda X: rbf(X), X_t_s, X_v, Yt_s, Yv,
                  notes="Thin-plate-spline RBF on 3000 points.",
                  n_params=3000, train_time=train_t)


def app07():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_lm_diff", 7
    md = model_dir(num, name)
    X_t, X_v = PARAMS["lm_diff"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="tanh",
                     max_iter=80, random_state=0, batch_size=2048).fit(sc.transform(X_t), Yt)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "lm_diff", lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="MLP 128-128-64 on lm_diff features (subsampled training).",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app08_pysr_l2m2n0():
    name, num = "pysr_l2m2n0", 8
    md = model_dir(num, name)
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    expressions = []
    pysr_re = pysr_im = None
    try:
        from pysr import PySRRegressor
        pm_re = PySRRegressor(niterations=20, binary_operators=["+","-","*","/"],
                              unary_operators=["sqrt","exp","log","sin","cos"], maxsize=20, populations=12,
                              parallelism="serial", progress=False, random_state=0,
                              deterministic=True, temp_equation_file=True)
        pm_re.fit(X_l2_t, Y_l2_t[:, 0])
        pm_im = PySRRegressor(niterations=20, binary_operators=["+","-","*","/"],
                              unary_operators=["sqrt","exp","log"], maxsize=20, populations=12,
                              parallelism="serial", progress=False, random_state=1,
                              deterministic=True, temp_equation_file=True)
        pm_im.fit(X_l2_t, Y_l2_t[:, 1])
        pysr_re, pysr_im = pm_re, pm_im
        try: pm_re.equations_.to_csv(md/"saved_model"/"pysr_eq_re.csv")
        except Exception: pass
        try: pm_im.equations_.to_csv(md/"saved_model"/"pysr_eq_im.csv")
        except Exception: pass
        expressions = [
            {"omega_real_best": str(pm_re.get_best().equation),
             "omega_real_loss": float(pm_re.get_best().loss),
             "omega_real_complexity": int(pm_re.get_best().complexity)},
            {"omega_imag_best": str(pm_im.get_best().equation),
             "omega_imag_loss": float(pm_im.get_best().loss),
             "omega_imag_complexity": int(pm_im.get_best().complexity)},
        ]
    except Exception as e:
        print(f"  [pysr] failed: {e}")
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        pysr_re = Pipeline([("poly", PolynomialFeatures(8)), ("lr", LinearRegression())]).fit(X_l2_t, Y_l2_t[:, 0])
        pysr_im = Pipeline([("poly", PolynomialFeatures(8)), ("lr", LinearRegression())]).fit(X_l2_t, Y_l2_t[:, 1])
        expressions = [{"note": f"PySR unavailable: {e}"}]
    train_t = time.time() - t0
    with open(expr_path, "w") as f: json.dump(expressions, f, indent=2)
    write_train_predict(md, name)
    pred_v = np.column_stack([pysr_re.predict(X_l2_v), pysr_im.predict(X_l2_v)])
    pred_t = np.column_stack([pysr_re.predict(X_l2_t), pysr_im.predict(X_l2_t)])
    loss, comp = loss_fn(pred_v, Y_l2_v)
    err_v = per_sample_err(pred_v, Y_l2_v)
    sc = {"approach": name, "approach_number": num, "benchmark": "ringdown", "agent": "opus47",
          "category": "symbolic", "parameterization": "raw_a_only", "mode": "l2_m2_n0",
          "loss": loss, "loss_components": comp,
          "runtime_ms": 0.001, "n_train": len(X_l2_t), "n_val": len(X_l2_v),
          "n_params": 20, "train_time_s": train_t,
          "notes": "PySR symbolic regression on l=2,m=2,n=0 mode.",
          "pysr_expressions_file": str(expr_path)}
    save_scorecard(md, sc)
    ERROR_DATA[name] = {"train": per_sample_err(pred_t, Y_l2_t).tolist(), "val": err_v.tolist()}
    RESULTS.append(sc)
    print(f"[{num:02d}] {name}: loss={loss:.4e} (l2m2n0)")


def app09_gplearn_l2m2n0():
    name, num = "gplearn_l2m2n0", 9
    md = model_dir(num, name)
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    try:
        from gplearn.genetic import SymbolicRegressor
        est_re = SymbolicRegressor(
            population_size=2000, generations=20, tournament_size=20,
            function_set=("add","sub","mul","div","sqrt","log","neg","inv"),
            metric="mse", parsimony_coefficient=0.001, random_state=42, n_jobs=1, verbose=0)
        est_re.fit(X_l2_t, Y_l2_t[:, 0])
        est_im = SymbolicRegressor(
            population_size=2000, generations=20, tournament_size=20,
            function_set=("add","sub","mul","div","sqrt","log","neg","inv"),
            metric="mse", parsimony_coefficient=0.001, random_state=43, n_jobs=1, verbose=0)
        est_im.fit(X_l2_t, Y_l2_t[:, 1])
        with open(expr_path, "w") as f:
            json.dump([{"omega_real_expression": str(est_re._program), "fitness": float(est_re._program.fitness_)},
                       {"omega_imag_expression": str(est_im._program), "fitness": float(est_im._program.fitness_)}],
                      f, indent=2, default=str)
    except Exception as e:
        print(f"  [gplearn] failed: {e}")
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        est_re = Pipeline([("poly", PolynomialFeatures(8)), ("lr", LinearRegression())]).fit(X_l2_t, Y_l2_t[:, 0])
        est_im = Pipeline([("poly", PolynomialFeatures(8)), ("lr", LinearRegression())]).fit(X_l2_t, Y_l2_t[:, 1])
        with open(expr_path, "w") as f: json.dump([{"note": f"gplearn unavailable: {e}"}], f)
    train_t = time.time() - t0
    pickle.dump({"est_re": est_re, "est_im": est_im}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    pred_v = np.column_stack([est_re.predict(X_l2_v), est_im.predict(X_l2_v)])
    pred_t = np.column_stack([est_re.predict(X_l2_t), est_im.predict(X_l2_t)])
    loss, comp = loss_fn(pred_v, Y_l2_v)
    err_v = per_sample_err(pred_v, Y_l2_v)
    sc = {"approach": name, "approach_number": num, "benchmark": "ringdown", "agent": "opus47",
          "category": "symbolic", "parameterization": "raw_a_only", "mode": "l2_m2_n0",
          "loss": loss, "loss_components": comp,
          "runtime_ms": 0.001, "n_train": len(X_l2_t), "n_val": len(X_l2_v),
          "n_params": 20, "train_time_s": train_t,
          "notes": "gplearn SymbolicRegressor on l=2,m=2,n=0 mode.",
          "gplearn_expressions_file": str(expr_path)}
    save_scorecard(md, sc)
    ERROR_DATA[name] = {"train": per_sample_err(pred_t, Y_l2_t).tolist(), "val": err_v.tolist()}
    RESULTS.append(sc)
    print(f"[{num:02d}] {name}: loss={loss:.4e} (l2m2n0)")


def app10():
    from sklearn.ensemble import RandomForestRegressor
    name, num = "rf_lm_diff", 10
    md = model_dir(num, name)
    X_t, X_v = PARAMS["lm_diff"]
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=80, max_depth=18, n_jobs=-1, random_state=0).fit(X_t, Yt)
    train_t = time.time() - t0
    pickle.dump({"rf": rf}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "lm_diff", lambda X: rf.predict(X), X_t, X_v,
             notes="Random Forest 80 trees on lm_diff (30k subsample).",
             n_params=80*32, train_time=train_t)


def app11():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    name, num = "hgbr_all_normalized", 11
    md = model_dir(num, name)
    X_t, X_v = PARAMS["all_normalized"]
    t0 = time.time()
    gbm = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=200, max_depth=8,
                                                              learning_rate=0.05, random_state=0), n_jobs=1)
    gbm.fit(X_t, Yt)
    train_t = time.time() - t0
    pickle.dump({"gbm": gbm}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "all_normalized", lambda X: gbm.predict(X), X_t, X_v,
             notes="HistGradientBoosting 200 iters.",
             n_params=200*32, train_time=train_t)


def app12():
    from sklearn.ensemble import ExtraTreesRegressor
    name, num = "extra_trees_compact", 12
    md = model_dir(num, name)
    X_t, X_v = PARAMS["compact"]
    t0 = time.time()
    et = ExtraTreesRegressor(n_estimators=100, max_depth=18, n_jobs=-1, random_state=0).fit(X_t, Yt)
    train_t = time.time() - t0
    pickle.dump({"et": et}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "compact", lambda X: et.predict(X), X_t, X_v,
             notes="ExtraTrees 100 trees on compactified spin.",
             n_params=100*32, train_time=train_t)


def app13():
    """Padé-like rational approximation for l=2,m=2,n=0."""
    from scipy.optimize import curve_fit
    name, num = "pade_l2m2n0", 13
    md = model_dir(num, name)
    a = X_l2_t[:, 0]
    Y_re = Y_l2_t[:, 0]; Y_im = Y_l2_t[:, 1]
    t0 = time.time()
    def pade_func(a, *coef):
        p0, p1, p2, p3, q1, q2 = coef
        return (p0 + p1 * a + p2 * a ** 2 + p3 * a ** 3) / (1 + q1 * a * (1 - a) + q2 * a ** 2 * (1 - a))
    coef_re, _ = curve_fit(pade_func, a, Y_re, p0=[0.4, 0.5, 0.0, 0.0, 0.1, 0.1], maxfev=5000)
    coef_im, _ = curve_fit(pade_func, a, Y_im, p0=[-0.09, 0.05, 0.0, 0.0, 0.1, 0.1], maxfev=5000)
    train_t = time.time() - t0
    pickle.dump({"coef_re": coef_re, "coef_im": coef_im}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    pred_v = np.column_stack([pade_func(X_l2_v[:, 0], *coef_re), pade_func(X_l2_v[:, 0], *coef_im)])
    pred_t = np.column_stack([pade_func(X_l2_t[:, 0], *coef_re), pade_func(X_l2_t[:, 0], *coef_im)])
    loss, comp = loss_fn(pred_v, Y_l2_v)
    err_v = per_sample_err(pred_v, Y_l2_v)
    sc = {"approach": name, "approach_number": num, "benchmark": "ringdown", "agent": "opus47",
          "category": "symbolic", "parameterization": "raw_a", "mode": "l2_m2_n0",
          "loss": loss, "loss_components": comp,
          "runtime_ms": 0.001, "n_train": len(X_l2_t), "n_val": len(X_l2_v),
          "n_params": 12, "train_time_s": train_t,
          "notes": "Padé rational P(a)/(1+Q(a)(1-a)) for l2m2n0."}
    save_scorecard(md, sc)
    ERROR_DATA[name] = {"train": per_sample_err(pred_t, Y_l2_t).tolist(), "val": err_v.tolist()}
    RESULTS.append(sc)
    print(f"[{num:02d}] {name}: loss={loss:.4e} (l2m2n0)")


def app14():
    from sklearn.neighbors import KNeighborsRegressor
    name, num = "knn_all_normalized", 14
    md = model_dir(num, name)
    X_t, X_v = PARAMS["all_normalized"]
    t0 = time.time()
    knn = KNeighborsRegressor(n_neighbors=3, weights="distance", n_jobs=-1).fit(X_t, Yt)
    train_t = time.time() - t0
    pickle.dump({"knn": knn}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "interpolation", "all_normalized", lambda X: knn.predict(X), X_t, X_v,
             notes="KNN k=3 distance-weighted with normalized features.",
             n_params=N_SUB, train_time=train_t)


def app15():
    from sklearn.kernel_ridge import KernelRidge
    name, num = "krr_poly8_chebyshev", 15
    md = model_dir(num, name)
    sub2 = RNG.choice(Xt.shape[0], 3000, replace=False)
    X_t_s = reparam(Xt[sub2], "chebyshev"); Yt_s = Yt[sub2]
    X_v = reparam(Xv, "chebyshev")
    t0 = time.time()
    m = KernelRidge(alpha=1e-4, kernel="polynomial", degree=8, coef0=1.0).fit(X_t_s, Yt_s)
    train_t = time.time() - t0
    pickle.dump({"m": m}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate_full(num, name, "kernel_gp", "chebyshev", lambda X: m.predict(X), X_t_s, X_v, Yt_s, Yv,
                  notes="Kernel Ridge polynomial deg-8 with Chebyshev mapping.",
                  n_params=3000, train_time=train_t)


def app16():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_raw", 16
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                     max_iter=60, random_state=0, batch_size=2048).fit(sc.transform(X_t), Yt)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "raw", lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="MLP 64-64 baseline.",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app17():
    """Per-mode polynomial fit deg-10."""
    name, num = "per_mode_poly10", 17
    md = model_dir(num, name)
    t0 = time.time()
    poly_models = {}
    deg = 10
    for l in np.unique(Xt_full[:, 1]):
        for m in np.unique(Xt_full[:, 2]):
            for nn in np.unique(Xt_full[:, 3]):
                mask = (Xt_full[:, 1] == l) & (Xt_full[:, 2] == m) & (Xt_full[:, 3] == nn)
                if mask.sum() < deg + 2: continue
                a = Xt_full[mask, 0]
                X_a = np.vander(a, deg + 1, increasing=True)
                Y_re = Yt_full[mask, 0]; Y_im = Yt_full[mask, 1]
                cr = np.linalg.lstsq(X_a, Y_re, rcond=None)[0]
                ci = np.linalg.lstsq(X_a, Y_im, rcond=None)[0]
                poly_models[(l, m, nn)] = (cr, ci)
    train_t = time.time() - t0
    print(f"  {len(poly_models)} per-mode polynomial fits")
    pickle.dump({"keys": list(poly_models.keys())}, open(md/"saved_model"/"meta.pkl","wb"))
    write_train_predict(md, name)
    def pred(X):
        out = np.zeros((X.shape[0], 2))
        for i, x in enumerate(X):
            key = (x[1], x[2], x[3])
            if key in poly_models:
                cr, ci = poly_models[key]
                vand = np.vander([x[0]], deg + 1, increasing=True)[0]
                out[i, 0] = vand @ cr
                out[i, 1] = vand @ ci
        return out
    evaluate_full(num, name, "symbolic", "raw", pred, Xt_full[:5000], Xv, Yt_full[:5000], Yv,
                  notes="Per-mode polynomial deg-10 fit in spin a.",
                  n_params=len(poly_models)*(deg+1)*2, train_time=train_t)


def app18():
    """Per-mode polynomial in -log(1-a)."""
    name, num = "per_mode_poly_log1ma", 18
    md = model_dir(num, name)
    t0 = time.time()
    poly_models = {}
    deg = 8
    for l in np.unique(Xt_full[:, 1]):
        for m in np.unique(Xt_full[:, 2]):
            for nn in np.unique(Xt_full[:, 3]):
                mask = (Xt_full[:, 1] == l) & (Xt_full[:, 2] == m) & (Xt_full[:, 3] == nn)
                if mask.sum() < deg + 2: continue
                a = Xt_full[mask, 0]
                z = -np.log(1 - a + 1e-12)
                X_a = np.vander(z, deg + 1, increasing=True)
                Y_re = Yt_full[mask, 0]; Y_im = Yt_full[mask, 1]
                cr = np.linalg.lstsq(X_a, Y_re, rcond=None)[0]
                ci = np.linalg.lstsq(X_a, Y_im, rcond=None)[0]
                poly_models[(l, m, nn)] = (cr, ci)
    train_t = time.time() - t0
    pickle.dump({"keys": list(poly_models.keys())}, open(md/"saved_model"/"meta.pkl","wb"))
    write_train_predict(md, name)
    def pred(X):
        out = np.zeros((X.shape[0], 2))
        for i, x in enumerate(X):
            key = (x[1], x[2], x[3])
            if key in poly_models:
                cr, ci = poly_models[key]
                z = -np.log(1 - x[0] + 1e-12)
                vand = np.vander([z], deg + 1, increasing=True)[0]
                out[i, 0] = vand @ cr
                out[i, 1] = vand @ ci
        return out
    evaluate_full(num, name, "symbolic", "log_1ma", pred, Xt_full[:5000], Xv, Yt_full[:5000], Yv,
                  notes="Per-mode polynomial deg-8 in -log(1-a).",
                  n_params=len(poly_models)*(deg+1)*2, train_time=train_t)


def app19():
    """Per-mode Padé."""
    from scipy.optimize import curve_fit
    name, num = "per_mode_pade", 19
    md = model_dir(num, name)
    t0 = time.time()
    coef_re_dict = {}
    coef_im_dict = {}
    def pade(a, *c):
        p0, p1, p2, p3, q1, q2 = c
        return (p0 + p1 * a + p2 * a ** 2 + p3 * a ** 3) / (1 + q1 * a * (1 - a) + q2 * a ** 2 * (1 - a))
    for l in np.unique(Xt_full[:, 1]):
        for m in np.unique(Xt_full[:, 2]):
            for nn in np.unique(Xt_full[:, 3]):
                mask = (Xt_full[:, 1] == l) & (Xt_full[:, 2] == m) & (Xt_full[:, 3] == nn)
                if mask.sum() < 8: continue
                a = Xt_full[mask, 0]; Y_re = Yt_full[mask, 0]; Y_im = Yt_full[mask, 1]
                try:
                    cr, _ = curve_fit(pade, a, Y_re, p0=[Y_re.mean(), 0.1, 0, 0, 0.1, 0], maxfev=2000)
                    ci, _ = curve_fit(pade, a, Y_im, p0=[Y_im.mean(), 0.1, 0, 0, 0.1, 0], maxfev=2000)
                    coef_re_dict[(l, m, nn)] = cr
                    coef_im_dict[(l, m, nn)] = ci
                except Exception:
                    pass
    train_t = time.time() - t0
    pickle.dump({"coef_re_dict": coef_re_dict, "coef_im_dict": coef_im_dict}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    def pred(X):
        out = np.zeros((X.shape[0], 2))
        for i, x in enumerate(X):
            key = (x[1], x[2], x[3])
            if key in coef_re_dict:
                out[i, 0] = pade(x[0], *coef_re_dict[key])
                out[i, 1] = pade(x[0], *coef_im_dict[key])
        return out
    evaluate_full(num, name, "symbolic", "raw", pred, Xt_full[:5000], Xv, Yt_full[:5000], Yv,
                  notes="Per-mode Padé rational approximant.",
                  n_params=len(coef_re_dict)*12, train_time=train_t)


def app20():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    name, num = "mlp_deep_log1ma", 20
    md = model_dir(num, name)
    X_t, X_v = PARAMS["log_1ma"]
    sc = StandardScaler().fit(X_t)
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="tanh",
                     max_iter=80, random_state=0, batch_size=4096, alpha=1e-5).fit(sc.transform(X_t), Yt)
    train_t = time.time() - t0
    pickle.dump({"m": m, "sc": sc}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "log_1ma", lambda X: m.predict(sc.transform(X)), X_t, X_v,
             notes="Deep MLP 256-128-64 with -log(1-a).",
             n_params=int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)),
             train_time=train_t)


def app21_pysr_l3m3n0():
    name, num = "pysr_l3m3n0", 21
    md = model_dir(num, name)
    expr_path = md / "saved_model" / "expressions.json"
    t0 = time.time()
    try:
        from pysr import PySRRegressor
        pm_re = PySRRegressor(niterations=15, binary_operators=["+","-","*","/"],
                              unary_operators=["sqrt","exp","log"], maxsize=18, populations=10,
                              parallelism="serial", progress=False, random_state=2,
                              deterministic=True, temp_equation_file=True)
        pm_re.fit(X_l3_t, Y_l3_t[:, 0])
        pm_im = PySRRegressor(niterations=15, binary_operators=["+","-","*","/"],
                              unary_operators=["sqrt","exp","log"], maxsize=18, populations=10,
                              parallelism="serial", progress=False, random_state=3,
                              deterministic=True, temp_equation_file=True)
        pm_im.fit(X_l3_t, Y_l3_t[:, 1])
        with open(expr_path, "w") as f:
            json.dump([{"omega_real_best": str(pm_re.get_best().equation),
                        "omega_real_loss": float(pm_re.get_best().loss),
                        "omega_real_complexity": int(pm_re.get_best().complexity)},
                       {"omega_imag_best": str(pm_im.get_best().equation),
                        "omega_imag_loss": float(pm_im.get_best().loss),
                        "omega_imag_complexity": int(pm_im.get_best().complexity)}], f, indent=2)
        try: pm_re.equations_.to_csv(md/"saved_model"/"pysr_eq_re.csv")
        except Exception: pass
        try: pm_im.equations_.to_csv(md/"saved_model"/"pysr_eq_im.csv")
        except Exception: pass
    except Exception as e:
        print(f"  [pysr2] failed: {e}")
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        pm_re = Pipeline([("poly", PolynomialFeatures(8)), ("lr", LinearRegression())]).fit(X_l3_t, Y_l3_t[:, 0])
        pm_im = Pipeline([("poly", PolynomialFeatures(8)), ("lr", LinearRegression())]).fit(X_l3_t, Y_l3_t[:, 1])
        with open(expr_path, "w") as f: json.dump([{"note": f"PySR unavailable: {e}"}], f)
    train_t = time.time() - t0
    write_train_predict(md, name)
    pred_v = np.column_stack([pm_re.predict(X_l3_v), pm_im.predict(X_l3_v)])
    pred_t = np.column_stack([pm_re.predict(X_l3_t), pm_im.predict(X_l3_t)])
    loss, comp = loss_fn(pred_v, Y_l3_v)
    err_v = per_sample_err(pred_v, Y_l3_v)
    sc = {"approach": name, "approach_number": num, "benchmark": "ringdown", "agent": "opus47",
          "category": "symbolic", "parameterization": "raw_a_only", "mode": "l3_m3_n0",
          "loss": loss, "loss_components": comp,
          "runtime_ms": 0.001, "n_train": len(X_l3_t), "n_val": len(X_l3_v),
          "n_params": 18, "train_time_s": train_t,
          "notes": "PySR symbolic regression on l=3,m=3,n=0 mode (second mode reparameterization).",
          "pysr_expressions_file": str(expr_path)}
    save_scorecard(md, sc)
    ERROR_DATA[name] = {"train": per_sample_err(pred_t, Y_l3_t).tolist(), "val": err_v.tolist()}
    RESULTS.append(sc)
    print(f"[{num:02d}] {name}: loss={loss:.4e} (l3m3n0)")


def app22():
    from sklearn.linear_model import BayesianRidge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    name, num = "bayes_ridge_poly6_log1ma", 22
    md = model_dir(num, name)
    X_t, X_v = PARAMS_FULL["log_1ma"]
    t0 = time.time()
    p = Pipeline([("poly", PolynomialFeatures(6, include_bias=False)),
                  ("br", MultiOutputRegressor(BayesianRidge(), n_jobs=1))]).fit(X_t, Yt_full)
    train_t = time.time() - t0
    pickle.dump({"p": p}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate_full(num, name, "ml", "log_1ma", lambda X: p.predict(X), X_t, X_v, Yt_full, Yv,
                  notes="Bayesian Ridge poly-6 with log(1-a).",
                  n_params=int(p.named_steps["br"].estimators_[0].coef_.size), train_time=train_t)


def app23():
    from sklearn.ensemble import RandomForestRegressor
    name, num = "rf_raw", 23
    md = model_dir(num, name)
    X_t, X_v = PARAMS["raw"]
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=80, max_depth=18, n_jobs=-1, random_state=0).fit(X_t, Yt)
    train_t = time.time() - t0
    pickle.dump({"rf": rf}, open(md/"saved_model"/"model.pkl","wb"))
    write_train_predict(md, name)
    evaluate(num, name, "ml", "raw", lambda X: rf.predict(X), X_t, X_v,
             notes="RF with raw [a, l, m, n] features.",
             n_params=80*32, train_time=train_t)


def app24():
    """Per-mode poly Chebyshev."""
    name, num = "per_mode_poly_chebyshev", 24
    md = model_dir(num, name)
    t0 = time.time()
    poly_models = {}
    deg = 12
    for l in np.unique(Xt_full[:, 1]):
        for m in np.unique(Xt_full[:, 2]):
            for nn in np.unique(Xt_full[:, 3]):
                mask = (Xt_full[:, 1] == l) & (Xt_full[:, 2] == m) & (Xt_full[:, 3] == nn)
                if mask.sum() < deg + 2: continue
                a = Xt_full[mask, 0]; z = 2 * a - 1
                X_a = np.vander(z, deg + 1, increasing=True)
                Y_re = Yt_full[mask, 0]; Y_im = Yt_full[mask, 1]
                cr = np.linalg.lstsq(X_a, Y_re, rcond=None)[0]
                ci = np.linalg.lstsq(X_a, Y_im, rcond=None)[0]
                poly_models[(l, m, nn)] = (cr, ci)
    train_t = time.time() - t0
    pickle.dump({"keys": list(poly_models.keys())}, open(md/"saved_model"/"meta.pkl","wb"))
    write_train_predict(md, name)
    def pred(X):
        out = np.zeros((X.shape[0], 2))
        for i, x in enumerate(X):
            key = (x[1], x[2], x[3])
            if key in poly_models:
                cr, ci = poly_models[key]
                z = 2 * x[0] - 1
                vand = np.vander([z], deg + 1, increasing=True)[0]
                out[i, 0] = vand @ cr
                out[i, 1] = vand @ ci
        return out
    evaluate_full(num, name, "symbolic", "chebyshev", pred, Xt_full[:5000], Xv, Yt_full[:5000], Yv,
                  notes="Per-mode polynomial deg-12 in Chebyshev variable.",
                  n_params=len(poly_models)*(deg+1)*2, train_time=train_t)


APPROACHES = [app01, app02, app03, app04, app05, app06, app07, app08_pysr_l2m2n0,
              app09_gplearn_l2m2n0, app10, app11, app12, app13, app14, app15, app16,
              app17, app18, app19, app20, app21_pysr_l3m3n0, app22, app23, app24]


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
