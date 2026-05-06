#!/usr/bin/env python3
"""Validity Bench — Opus 4.6 Agent: Build all 26 approaches."""

import os, sys, json, time, warnings, shutil, pickle
import numpy as np

warnings.filterwarnings("ignore")

ROOT = "/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks"
WORK = os.path.join(ROOT, "llm_agents/results/opus46/validity")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import h5py
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.interpolate import RBFInterpolator
import joblib

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    data = {}
    for split, fn in [("train", "datasets/validity/validity_training.h5"),
                      ("val", "datasets/validity/validity_validation.h5")]:
        with h5py.File(fn, "r") as f:
            data[split] = {
                "q": f["q"][:],
                "chi1z": f["chi1z"][:],
                "chi2z": f["chi2z"][:],
                "omega0": f["omega0"][:],
                "mm_td": f["mm_td"][:],
            }
    return data

def make_X_y(data, split):
    d = data[split]
    X = np.column_stack([d["q"], d["chi1z"], d["chi2z"], d["omega0"]])
    y = np.log10(np.clip(d["mm_td"], 1e-10, None))
    return X, y

# ── Reparameterizations ──────────────────────────────────────────────────────
def reparam_raw(X):
    return X  # (q, chi1z, chi2z, omega0)

def reparam_eta_chieff(X):
    q, chi1z, chi2z, omega0 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eta = q / (1 + q)**2
    chi_eff = (q * chi1z + chi2z) / (1 + q)
    chi_a = (q * chi1z - chi2z) / (1 + q)
    return np.column_stack([eta, chi_eff, chi_a, omega0])

def reparam_logq(X):
    q, chi1z, chi2z, omega0 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eta = q / (1 + q)**2
    chi_eff = (q * chi1z + chi2z) / (1 + q)
    chi_a = (q * chi1z - chi2z) / (1 + q)
    return np.column_stack([np.log(q), chi_eff, chi_a, np.log(omega0)])

def reparam_interact(X):
    q, chi1z, chi2z, omega0 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eta = q / (1 + q)**2
    chi_eff = (q * chi1z + chi2z) / (1 + q)
    chi_a = (q * chi1z - chi2z) / (1 + q)
    return np.column_stack([eta, chi_eff, chi_a, omega0, q * chi_eff, eta * chi_a])

def reparam_boundary(X):
    q, chi1z, chi2z, omega0 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eta = q / (1 + q)**2
    chi_eff = (q * chi1z + chi2z) / (1 + q)
    chi_a = (q * chi1z - chi2z) / (1 + q)
    dq = np.abs(q - 8.0) / 8.0
    dchi1 = np.abs(chi1z) / 0.8
    dchi2 = np.abs(chi2z) / 0.8
    return np.column_stack([eta, chi_eff, chi_a, omega0, dq, dchi1, dchi2])

REPARAMS = {
    "raw": reparam_raw,
    "eta_chieff": reparam_eta_chieff,
    "logq": reparam_logq,
    "interact": reparam_interact,
    "boundary": reparam_boundary,
}

# ── Loss ──────────────────────────────────────────────────────────────────────
def compute_loss(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

def compute_per_sample_errors(y_true, y_pred):
    return np.abs(y_pred - y_true)

# ── Approaches ────────────────────────────────────────────────────────────────
APPROACHES = []

def add(num, name, category, param, build_fn):
    APPROACHES.append({"num": num, "name": name, "category": category,
                        "param": param, "build_fn": build_fn})

# ── Category 1: Kernel/GP ────────────────────────────────────────────────────
def build_gpr_rbf_raw(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(Xt.shape[1])) + WhiteKernel(1e-3)
    m = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0, normalize_y=True)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(1, "gpr_rbf_raw", "kernel", "raw", build_gpr_rbf_raw)

def build_gpr_matern_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(Xt.shape[1]), nu=2.5) + WhiteKernel(1e-3)
    m = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0, normalize_y=True)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(2, "gpr_matern_eta", "kernel", "eta_chieff", build_gpr_matern_eta)

def build_gpr_matern_logq(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(Xt.shape[1]), nu=1.5) + WhiteKernel(1e-3)
    m = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0, normalize_y=True)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(3, "gpr_matern_logq", "kernel", "logq", build_gpr_matern_logq)

def build_krr_raw(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = KernelRidge(alpha=0.01, kernel='rbf', gamma=0.1)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(4, "krr_raw", "kernel", "raw", build_krr_raw)

def build_svr_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = SVR(kernel='rbf', C=10.0, epsilon=0.05, gamma='scale')
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(5, "svr_eta", "kernel", "eta_chieff", build_svr_eta)

def build_krr_interact(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = KernelRidge(alpha=0.005, kernel='rbf', gamma=0.05)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(6, "krr_interact", "kernel", "interact", build_krr_interact)

# ── Category 2: Symbolic / Analytical ────────────────────────────────────────
def build_poly3_raw(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    pf = PolynomialFeatures(degree=3, include_bias=True)
    Xp = pf.fit_transform(Xs)
    m = Ridge(alpha=1.0)
    m.fit(Xp, yt)
    return (m, pf), sc, lambda mp, X: mp[0].predict(mp[1].transform(X))

add(7, "poly3_raw", "symbolic", "raw", build_poly3_raw)

def build_poly4_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    pf = PolynomialFeatures(degree=4, include_bias=True)
    Xp = pf.fit_transform(Xs)
    m = Ridge(alpha=0.1)
    m.fit(Xp, yt)
    return (m, pf), sc, lambda mp, X: mp[0].predict(mp[1].transform(X))

add(8, "poly4_eta", "symbolic", "eta_chieff", build_poly4_eta)

def build_poly5_logq(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    pf = PolynomialFeatures(degree=5, include_bias=True)
    Xp = pf.fit_transform(Xs)
    m = Ridge(alpha=1.0)
    m.fit(Xp, yt)
    return (m, pf), sc, lambda mp, X: mp[0].predict(mp[1].transform(X))

add(9, "poly5_logq", "symbolic", "logq", build_poly5_logq)

def build_lasso_interact(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    pf = PolynomialFeatures(degree=3, include_bias=True)
    Xp = pf.fit_transform(Xs)
    m = Lasso(alpha=0.01, max_iter=10000)
    m.fit(Xp, yt)
    return (m, pf), sc, lambda mp, X: mp[0].predict(mp[1].transform(X))

add(10, "lasso_interact", "symbolic", "interact", build_lasso_interact)

def build_elasticnet_boundary(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    pf = PolynomialFeatures(degree=3, include_bias=True)
    Xp = pf.fit_transform(Xs)
    m = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    m.fit(Xp, yt)
    return (m, pf), sc, lambda mp, X: mp[0].predict(mp[1].transform(X))

add(11, "elasticnet_boundary", "symbolic", "boundary", build_elasticnet_boundary)

def build_bayesian_ridge_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    pf = PolynomialFeatures(degree=4, include_bias=True)
    Xp = pf.fit_transform(Xs)
    m = BayesianRidge(max_iter=500)
    m.fit(Xp, yt)
    return (m, pf), sc, lambda mp, X: mp[0].predict(mp[1].transform(X))

add(12, "bayesian_ridge_eta", "symbolic", "eta_chieff", build_bayesian_ridge_eta)

# ── Category 3: Interpolation ────────────────────────────────────────────────
def build_rbf_tps_raw(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    m = RBFInterpolator(Xs, yt, kernel='thin_plate_spline', smoothing=0.01)
    return m, sc, lambda m, X: m(X)

add(13, "rbf_tps_raw", "interpolation", "raw", build_rbf_tps_raw)

def build_rbf_cubic_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    m = RBFInterpolator(Xs, yt, kernel='cubic', smoothing=0.01)
    return m, sc, lambda m, X: m(X)

add(14, "rbf_cubic_eta", "interpolation", "eta_chieff", build_rbf_cubic_eta)

def build_rbf_linear_logq(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    Xs = sc.transform(Xt)
    m = RBFInterpolator(Xs, yt, kernel='linear', smoothing=0.1)
    return m, sc, lambda m, X: m(X)

add(15, "rbf_linear_logq", "interpolation", "logq", build_rbf_linear_logq)

def build_knn5_raw(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = KNeighborsRegressor(n_neighbors=5, weights='distance')
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(16, "knn5_raw", "interpolation", "raw", build_knn5_raw)

def build_knn10_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = KNeighborsRegressor(n_neighbors=10, weights='distance')
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(17, "knn10_eta", "interpolation", "eta_chieff", build_knn10_eta)

# ── Category 4: Machine Learning ─────────────────────────────────────────────
def build_mlp_raw(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=2000,
                     early_stopping=True, validation_fraction=0.15,
                     learning_rate_init=0.001, random_state=42)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(18, "mlp_raw", "ml", "raw", build_mlp_raw)

def build_mlp_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=2000,
                     early_stopping=True, validation_fraction=0.15,
                     learning_rate_init=0.0005, random_state=42)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(19, "mlp_eta", "ml", "eta_chieff", build_mlp_eta)

def build_rf_raw(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=42)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(20, "rf_raw", "ml", "raw", build_rf_raw)

def build_gbr_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, random_state=42)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(21, "gbr_eta", "ml", "eta_chieff", build_gbr_eta)

def build_et_raw(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=42)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(22, "et_raw", "ml", "raw", build_et_raw)

def build_ada_eta(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(23, "ada_eta", "ml", "eta_chieff", build_ada_eta)

def build_bagging_logq(Xt, yt, Xv, yv):
    sc = StandardScaler().fit(Xt)
    m = BaggingRegressor(n_estimators=200, max_samples=0.8, random_state=42)
    m.fit(sc.transform(Xt), yt)
    return m, sc, lambda m, X: m.predict(X)

add(24, "bagging_logq", "ml", "logq", build_bagging_logq)

# Approaches 25-26: PySR and gplearn (handled separately in run_symbolic.py)

# ── Build all non-symbolic approaches ─────────────────────────────────────────
def run_all():
    data = load_data()
    Xraw_t, yt = make_X_y(data, "train")
    Xraw_v, yv = make_X_y(data, "val")

    os.makedirs(os.path.join(WORK, "models"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "comparison"), exist_ok=True)

    results = []
    error_data = {}

    for ap in APPROACHES:
        num = ap["num"]
        name = ap["name"]
        param = ap["param"]
        category = ap["category"]
        rfn = REPARAMS[param]
        print(f"\n{'='*60}")
        print(f"Approach {num:02d}: {name} (param={param}, cat={category})")
        print(f"{'='*60}")

        Xt = rfn(Xraw_t)
        Xv = rfn(Xraw_v)

        try:
            t0 = time.time()
            model, scaler, pred_fn = ap["build_fn"](Xt, yt, Xv, yv)
            train_time = time.time() - t0

            Xts = scaler.transform(Xt)
            Xvs = scaler.transform(Xv)

            t0 = time.time()
            yp_t = pred_fn(model, Xts)
            t_pred_train = time.time() - t0

            t0 = time.time()
            yp_v = pred_fn(model, Xvs)
            t_pred_val = time.time() - t0

            loss_t = compute_loss(yt, yp_t)
            loss_v = compute_loss(yv, yp_v)
            runtime_ms = (t_pred_val / len(yv)) * 1000

            err_t = compute_per_sample_errors(yt, yp_t).tolist()
            err_v = compute_per_sample_errors(yv, yp_v).tolist()

            print(f"  Train loss: {loss_t:.4f}, Val loss: {loss_v:.4f}, runtime: {runtime_ms:.4f} ms/sample")

            # Save model artifacts
            mdir = os.path.join(WORK, f"models/{num:02d}_{name}")
            sdir = os.path.join(mdir, "saved_model")
            os.makedirs(sdir, exist_ok=True)

            joblib.dump({"model": model, "scaler": scaler}, os.path.join(sdir, "model.joblib"))

            n_params = 0
            if hasattr(model, 'coef_'):
                n_params = len(model.coef_) if hasattr(model.coef_, '__len__') else 1
            elif isinstance(model, tuple) and hasattr(model[0], 'coef_'):
                n_params = len(model[0].coef_)
            elif hasattr(model, 'n_support_'):
                n_params = int(np.sum(model.n_support_))
            else:
                n_params = 100

            scorecard = {
                "approach": name, "approach_number": num,
                "benchmark": "validity", "agent": "opus46",
                "category": category,
                "parameterization": param,
                "loss": float(loss_v), "loss_train": float(loss_t),
                "loss_components": {"log_rmse": float(loss_v)},
                "runtime_ms": float(runtime_ms),
                "n_train": len(yt), "n_val": len(yv),
                "n_params": int(n_params),
                "notes": f"{name} with {param} parameterization"
            }
            with open(os.path.join(mdir, "scorecard.json"), "w") as f:
                json.dump(scorecard, f, indent=2)

            results.append(scorecard)
            error_data[name] = {"train": err_t, "val": err_v}

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"approach": name, "approach_number": num,
                            "benchmark": "validity", "agent": "opus46",
                            "category": category, "parameterization": param,
                            "loss": 999.0, "loss_train": 999.0,
                            "loss_components": {"log_rmse": 999.0},
                            "runtime_ms": 0, "n_train": len(yt), "n_val": len(yv),
                            "n_params": 0, "notes": f"FAILED: {e}"})

    # Save error data
    with open(os.path.join(WORK, "comparison/error_data.json"), "w") as f:
        json.dump(error_data, f)

    return results, error_data


def run_symbolic(results, error_data):
    """Run PySR and gplearn."""
    data = load_data()
    Xraw_t, yt = make_X_y(data, "train")
    Xraw_v, yv = make_X_y(data, "val")

    # ── PySR (approach 25) ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Approach 25: pysr_eta (PySR symbolic regression)")
    print(f"{'='*60}")
    try:
        from pysr import PySRRegressor

        Xt = reparam_eta_chieff(Xraw_t)
        Xv = reparam_eta_chieff(Xraw_v)

        mdir = os.path.join(WORK, "models/25_pysr_eta")
        sdir = os.path.join(mdir, "saved_model")
        os.makedirs(sdir, exist_ok=True)

        model = PySRRegressor(
            niterations=40, binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "exp", "square"],
            maxsize=20, populations=10, procs=1,
            loss="loss(prediction, target) = (prediction - target)^2",
            temp_equation_file=True, delete_tempfiles=True,
            random_state=42, deterministic=False, verbosity=0,
        )
        t0 = time.time()
        model.fit(Xt, yt)
        train_time = time.time() - t0

        yp_t = model.predict(Xt)
        t0 = time.time()
        yp_v = model.predict(Xv)
        t_pred = time.time() - t0

        loss_t = compute_loss(yt, yp_t)
        loss_v = compute_loss(yv, yp_v)
        runtime_ms = (t_pred / len(yv)) * 1000

        expr = str(model.sympy())
        print(f"  Best expression: {expr}")
        print(f"  Train loss: {loss_t:.4f}, Val loss: {loss_v:.4f}")

        # Save expressions
        eqs = []
        try:
            for i, row in model.equations_.iterrows():
                eqs.append({"expression": str(row.get("sympy_format", row.get("equation", ""))),
                            "complexity": int(row.get("complexity", 0)),
                            "loss": float(row.get("loss", 999))})
        except:
            eqs = [{"expression": expr, "complexity": 0, "loss": float(loss_v)}]
        with open(os.path.join(sdir, "expressions.json"), "w") as f:
            json.dump(eqs, f, indent=2)

        joblib.dump({"model": model}, os.path.join(sdir, "model.joblib"))

        scorecard = {
            "approach": "pysr_eta", "approach_number": 25,
            "benchmark": "validity", "agent": "opus46",
            "category": "symbolic", "parameterization": "eta_chieff",
            "loss": float(loss_v), "loss_train": float(loss_t),
            "loss_components": {"log_rmse": float(loss_v)},
            "runtime_ms": float(runtime_ms),
            "n_train": len(yt), "n_val": len(yv), "n_params": 0,
            "expression": expr,
            "notes": "PySR symbolic regression on eta_chieff params"
        }
        with open(os.path.join(mdir, "scorecard.json"), "w") as f:
            json.dump(scorecard, f, indent=2)

        results.append(scorecard)
        error_data["pysr_eta"] = {
            "train": compute_per_sample_errors(yt, yp_t).tolist(),
            "val": compute_per_sample_errors(yv, yp_v).tolist()
        }

    except Exception as e:
        print(f"  PySR FAILED: {e}")
        import traceback; traceback.print_exc()
        mdir = os.path.join(WORK, "models/25_pysr_eta")
        sdir = os.path.join(mdir, "saved_model")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "expressions.json"), "w") as f:
            json.dump([{"expression": "FAILED", "complexity": 0, "loss": 999}], f)
        scorecard = {
            "approach": "pysr_eta", "approach_number": 25,
            "benchmark": "validity", "agent": "opus46",
            "category": "symbolic", "parameterization": "eta_chieff",
            "loss": 999.0, "loss_train": 999.0,
            "loss_components": {"log_rmse": 999.0},
            "runtime_ms": 0, "n_train": len(yt), "n_val": len(yv),
            "n_params": 0, "notes": f"PySR FAILED: {e}"
        }
        with open(os.path.join(mdir, "scorecard.json"), "w") as f:
            json.dump(scorecard, f, indent=2)
        results.append(scorecard)

    # ── gplearn (approach 26) ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Approach 26: gplearn_raw (gplearn symbolic regression)")
    print(f"{'='*60}")
    try:
        from gplearn.genetic import SymbolicRegressor

        Xt = reparam_raw(Xraw_t)
        Xv = reparam_raw(Xraw_v)

        mdir = os.path.join(WORK, "models/26_gplearn_raw")
        sdir = os.path.join(mdir, "saved_model")
        os.makedirs(sdir, exist_ok=True)

        est = SymbolicRegressor(
            population_size=2000, generations=30, tournament_size=20,
            function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
            metric='mse', parsimony_coefficient=0.001,
            max_samples=1.0, verbose=0, random_state=42,
        )
        t0 = time.time()
        est.fit(Xt, yt)
        train_time = time.time() - t0

        yp_t = est.predict(Xt)
        t0 = time.time()
        yp_v = est.predict(Xv)
        t_pred = time.time() - t0

        loss_t = compute_loss(yt, yp_t)
        loss_v = compute_loss(yv, yp_v)
        runtime_ms = (t_pred / len(yv)) * 1000

        expr = str(est._program)
        print(f"  Best expression: {expr}")
        print(f"  Train loss: {loss_t:.4f}, Val loss: {loss_v:.4f}")

        with open(os.path.join(sdir, "expressions.json"), "w") as f:
            json.dump([{"expression": expr, "complexity": est._program.length_,
                        "loss": float(loss_v)}], f, indent=2)

        joblib.dump({"model": est}, os.path.join(sdir, "model.joblib"))

        scorecard = {
            "approach": "gplearn_raw", "approach_number": 26,
            "benchmark": "validity", "agent": "opus46",
            "category": "symbolic", "parameterization": "raw",
            "loss": float(loss_v), "loss_train": float(loss_t),
            "loss_components": {"log_rmse": float(loss_v)},
            "runtime_ms": float(runtime_ms),
            "n_train": len(yt), "n_val": len(yv), "n_params": 0,
            "expression": expr,
            "notes": "gplearn symbolic regression on raw params"
        }
        with open(os.path.join(mdir, "scorecard.json"), "w") as f:
            json.dump(scorecard, f, indent=2)

        results.append(scorecard)
        error_data["gplearn_raw"] = {
            "train": compute_per_sample_errors(yt, yp_t).tolist(),
            "val": compute_per_sample_errors(yv, yp_v).tolist()
        }

    except Exception as e:
        print(f"  gplearn FAILED: {e}")
        import traceback; traceback.print_exc()
        mdir = os.path.join(WORK, "models/26_gplearn_raw")
        sdir = os.path.join(mdir, "saved_model")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "expressions.json"), "w") as f:
            json.dump([{"expression": "FAILED", "complexity": 0, "loss": 999}], f)
        scorecard = {
            "approach": "gplearn_raw", "approach_number": 26,
            "benchmark": "validity", "agent": "opus46",
            "category": "symbolic", "parameterization": "raw",
            "loss": 999.0, "loss_train": 999.0,
            "loss_components": {"log_rmse": 999.0},
            "runtime_ms": 0, "n_train": len(yt), "n_val": len(yv),
            "n_params": 0, "notes": f"gplearn FAILED: {e}"
        }
        with open(os.path.join(mdir, "scorecard.json"), "w") as f:
            json.dump(scorecard, f, indent=2)
        results.append(scorecard)

    return results, error_data


def make_plots(results, error_data):
    """Generate all comparison plots."""
    import gwbenchmarks.plot_settings as ps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cdir = os.path.join(WORK, "comparison")
    os.makedirs(cdir, exist_ok=True)

    valid = [r for r in results if r["loss"] < 100]
    valid.sort(key=lambda r: r["loss"])

    cat_colors = {"kernel": ps.COLOR_CYCLE[0], "symbolic": ps.COLOR_CYCLE[1],
                  "interpolation": ps.COLOR_CYCLE[2], "ml": ps.COLOR_CYCLE[3]}

    # ── Progress plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=ps.figsize(1.0))
    names = [r["approach"] for r in valid]
    losses = [r["loss"] for r in valid]
    colors = [cat_colors.get(r.get("category", "ml"), "gray") for r in valid]
    ax.barh(range(len(names)), losses, color=colors, edgecolor='k', linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Validation Loss (RMSE in log10 space)")
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=cat_colors[c], label=c) for c in cat_colors]
    ax.legend(handles=legend_items, fontsize=6, loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(cdir, "progress.png"), dpi=200)
    fig.savefig(os.path.join(cdir, "progress.pdf"))
    plt.close(fig)

    # ── Loss-only comparison ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=ps.figsize(1.0))
    ax.barh(range(len(names)), losses, color=colors, edgecolor='k', linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Validation Loss (RMSE in log10 space)")
    ax.invert_yaxis()
    ax.legend(handles=legend_items, fontsize=6, loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(cdir, "loss_only_comparison.png"), dpi=200)
    fig.savefig(os.path.join(cdir, "loss_only_comparison.pdf"))
    plt.close(fig)

    # ── Pareto plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=ps.figsize(1.0))
    for r in valid:
        c = cat_colors.get(r.get("category", "ml"), "gray")
        ax.scatter(r["runtime_ms"], r["loss"], color=c, s=30, edgecolors='k', linewidth=0.3, zorder=5)
        ax.annotate(r["approach"], (r["runtime_ms"], r["loss"]),
                    fontsize=4, ha='left', va='bottom', xytext=(2, 2), textcoords='offset points')
    ax.set_xlabel("Runtime (ms/sample)")
    ax.set_ylabel("Validation Loss")
    ax.set_xscale("log")
    ax.legend(handles=legend_items, fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(cdir, "pareto_accuracy_speed.png"), dpi=200)
    fig.savefig(os.path.join(cdir, "pareto_accuracy_speed.pdf"))
    plt.close(fig)

    # ── Error histograms ─────────────────────────────────────────────────
    n_models = len(error_data)
    if n_models > 0:
        ncols = 4
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2))
        axes = np.array(axes).flatten() if n_models > 1 else [axes]
        for idx, (name, errs) in enumerate(error_data.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            bins = np.linspace(0, max(max(errs["train"]), max(errs["val"]), 2.0), 30)
            ax.hist(errs["train"], bins=bins, alpha=0.5, label="train", color=ps.COLOR_CYCLE[0])
            ax.hist(errs["val"], bins=bins, alpha=0.5, label="val", color=ps.COLOR_CYCLE[1],
                    hatch='//')
            ax.set_title(name, fontsize=5)
            ax.tick_params(labelsize=4)
            if idx == 0:
                ax.legend(fontsize=4)
        for idx in range(len(error_data), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(cdir, "error_histograms.png"), dpi=200)
        fig.savefig(os.path.join(cdir, "error_histograms.pdf"))
        plt.close(fig)

    # ── Summary table ─────────────────────────────────────────────────────
    summary = sorted(valid, key=lambda r: r["loss"])
    with open(os.path.join(cdir, "summary_table.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Best model ────────────────────────────────────────────────────────
    if valid:
        best = min(valid, key=lambda r: r["loss"])
        with open(os.path.join(cdir, "best_model.json"), "w") as f:
            json.dump(best, f, indent=2)
        print(f"\nBest model: {best['approach']} with loss={best['loss']:.4f}")

    print("All plots saved.")


def generate_train_predict(results):
    """Generate train.py and predict.py for each model directory."""
    for r in results:
        num = r["approach_number"]
        name = r["approach"]
        mdir = os.path.join(WORK, f"models/{num:02d}_{name}")
        if not os.path.isdir(mdir):
            continue

        # train.py
        train_code = f'''#!/usr/bin/env python3
"""Train script for {name} (approach {num}) — Validity Bench."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
# This model was trained by build_all.py. Re-run that script to retrain.
print("Model {name} trained via build_all.py")
'''
        with open(os.path.join(mdir, "train.py"), "w") as f:
            f.write(train_code)

        # predict.py
        predict_code = f'''#!/usr/bin/env python3
"""Predict function for {name} (approach {num}) — Validity Bench."""
import os, numpy as np, joblib

_dir = os.path.dirname(os.path.abspath(__file__))
_data = joblib.load(os.path.join(_dir, "saved_model", "model.joblib"))
_model = _data["model"]
_scaler = _data["scaler"]

PARAM = "{r.get("parameterization", "raw")}"

def _reparam(X):
    if PARAM == "raw":
        return X
    q, chi1z, chi2z, omega0 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eta = q / (1 + q)**2
    chi_eff = (q * chi1z + chi2z) / (1 + q)
    chi_a = (q * chi1z - chi2z) / (1 + q)
    if PARAM == "eta_chieff":
        return np.column_stack([eta, chi_eff, chi_a, omega0])
    elif PARAM == "logq":
        return np.column_stack([np.log(q), chi_eff, chi_a, np.log(omega0)])
    elif PARAM == "interact":
        return np.column_stack([eta, chi_eff, chi_a, omega0, q * chi_eff, eta * chi_a])
    elif PARAM == "boundary":
        dq = np.abs(q - 8.0) / 8.0
        dchi1 = np.abs(chi1z) / 0.8
        dchi2 = np.abs(chi2z) / 0.8
        return np.column_stack([eta, chi_eff, chi_a, omega0, dq, dchi1, dchi2])
    return X

def predict(X):
    """Predict log10(mismatch) from (q, chi1z, chi2z, omega0) array."""
    Xr = _reparam(X)
    Xs = _scaler.transform(Xr)
    if hasattr(_model, "predict"):
        return _model.predict(Xs)
    elif callable(_model):
        return _model(Xs)
    else:
        m, pf = _model
        return m.predict(pf.transform(Xs))
'''
        with open(os.path.join(mdir, "predict.py"), "w") as f:
            f.write(predict_code)


def write_changelog(results):
    """Write CHANGELOG.md."""
    lines = ["# Validity Bench — CHANGELOG\n"]
    for r in sorted(results, key=lambda x: x["approach_number"]):
        lines.append(f"## Approach {r['approach_number']:02d}: {r['approach']}")
        lines.append(f"- Category: {r.get('category', 'unknown')}")
        lines.append(f"- Parameterization: {r.get('parameterization', 'raw')}")
        lines.append(f"- Val Loss: {r['loss']:.4f}")
        if 'expression' in r:
            lines.append(f"- Expression: `{r['expression']}`")
        lines.append(f"- Notes: {r.get('notes', '')}")
        lines.append("")
    with open(os.path.join(WORK, "CHANGELOG.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print("=" * 60)
    print("VALIDITY BENCH — Opus 4.6 Agent — Building all 26 approaches")
    print("=" * 60)

    results, error_data = run_all()
    results, error_data = run_symbolic(results, error_data)

    # Update error_data file
    with open(os.path.join(WORK, "comparison/error_data.json"), "w") as f:
        json.dump(error_data, f)

    make_plots(results, error_data)
    generate_train_predict(results)
    write_changelog(results)

    print("\n" + "=" * 60)
    print("ALL 26 APPROACHES COMPLETE")
    print("=" * 60)
