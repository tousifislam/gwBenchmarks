"""Model registry + runner for the Remnant Bench (opus48, original work).

Scalar regression of the kick magnitude vf_mag. Each approach = (reparam,
estimator, optional log-target). Categories: kernel/GP, symbolic (PySR+gplearn,
mandatory), interpolation, machine-learning.
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import joblib

import rdata as RD

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# wrappers
# --------------------------------------------------------------------------
class LogTarget:
    """Fit estimator on log1p(y) when y>=0; predict expm1. Helps kicks (skewed)."""
    def __init__(self, est):
        self.est = est
    def fit(self, X, y):
        self.shift_ = max(0.0, -np.min(y)) + 1e-9
        self.est.fit(X, np.log(y + self.shift_))
        return self
    def predict(self, X):
        return np.exp(self.est.predict(X)) - self.shift_


class SymbolicScalar:
    """PySR or gplearn for a single scalar target. Saves the Pareto front."""
    def __init__(self, backend="pysr", **kw):
        self.backend = backend; self.kw = kw
    def fit(self, X, y):
        if self.backend == "pysr":
            from pysr import PySRRegressor
            m = PySRRegressor(
                niterations=self.kw.get("niterations", 60),
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "cube", "sqrt", "exp", "log"],
                maxsize=self.kw.get("maxsize", 28),
                populations=self.kw.get("populations", 20),
                progress=False, verbosity=0, temp_equation_file=True,
                random_state=0, deterministic=True, parallelism="serial",
            )
            m.fit(X, y)
            exprs = []
            try:
                for _, r in m.equations_.iterrows():
                    exprs.append({"equation": str(r["equation"]),
                                  "complexity": int(r["complexity"]),
                                  "loss": float(r["loss"])})
            except Exception:
                pass
        else:
            from gplearn.genetic import SymbolicRegressor
            m = SymbolicRegressor(
                population_size=self.kw.get("population_size", 3000),
                generations=self.kw.get("generations", 30),
                tournament_size=20,
                function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
                metric="mse", parsimony_coefficient=0.001,
                random_state=42, verbose=0,
            )
            m.fit(X, y)
            exprs = [{"equation": str(m._program),
                      "complexity": int(m._program.length_),
                      "loss": float(m.run_details_["best_fitness"][-1])}]
        self.m_ = m; self.expressions_ = exprs
        return self
    def predict(self, X):
        return np.asarray(self.m_.predict(X)).ravel()


# --------------------------------------------------------------------------
# estimator factories
# --------------------------------------------------------------------------
def _gpr(kernel="rbf", length=2.0, alpha=1e-4, opt=True):
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel as W
    base = RBF(length) if kernel == "rbf" else Matern(length, nu=2.5)
    k = C(1.0) * base + W(1e-3)
    return GPR(kernel=k, alpha=alpha, normalize_y=True,
               n_restarts_optimizer=(2 if opt else 0),
               optimizer=("fmin_l_bfgs_b" if opt else None))


def _krr(alpha=1e-2, gamma=None):
    from sklearn.kernel_ridge import KernelRidge
    return KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)


def _svr(C=10.0, gamma="scale"):
    from sklearn.svm import SVR
    return SVR(C=C, gamma=gamma, epsilon=1e-3)


def _mlp(hidden=(128, 128), alpha=1e-3, max_iter=4000):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    base = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                        activation="relu", random_state=0, learning_rate_init=2e-3)
    return TransformedTargetRegressor(base, transformer=StandardScaler())


def _rf(n=500):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n, random_state=0, n_jobs=-1)


def _extratrees(n=600):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=n, random_state=0, n_jobs=-1)


def _xgb():
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, random_state=0)


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=800, max_depth=-1, num_leaves=31,
                             learning_rate=0.03, subsample=0.8, random_state=0,
                             verbose=-1)


def _poly(deg=3, alpha=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    return make_pipeline(PolynomialFeatures(deg), Ridge(alpha=alpha))


class RBFInterp:
    """Picklable multi/single-output scipy RBFInterpolator wrapper."""
    def __init__(self, kernel="thin_plate_spline", smoothing=1e-3, kw=None):
        self.kernel = kernel; self.smoothing = smoothing; self.kw = kw or {}
    def fit(self, X, y):
        from scipy.interpolate import RBFInterpolator
        self.r_ = RBFInterpolator(X, y, kernel=self.kernel,
                                  smoothing=self.smoothing, **self.kw)
        self.n_features_in_ = X.shape[1]; return self
    def predict(self, X):
        return self.r_(np.atleast_2d(X))


def _rbf_interp(kernel="thin_plate_spline", smoothing=1e-3, **kw):
    return RBFInterp(kernel, smoothing, kw)


def _knn(k=8):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=k, weights="distance")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_approach(spec, write=True, verbose=True):
    t0 = time.time()
    Ptr, ytr, _ = RD.load("training")
    Pva, yva, nr = RD.load("validation")
    Xtr = RD.reparam(Ptr, spec["reparam"]); Xva = RD.reparam(Pva, spec["reparam"])
    Xtr_s, mean, std = RD.standardize(Xtr)
    Xva_s, _, _ = RD.standardize(Xva, mean, std)

    target = spec.get("target", "vf_mag")
    model = spec["make"]()
    model.fit(Xtr_s, ytr[target])

    t_pred = time.time()
    pred_va = np.asarray(model.predict(Xva_s)).ravel()
    runtime_ms = 1e3 * (time.time() - t_pred) / len(Xva_s)
    pred_tr = np.asarray(model.predict(Xtr_s)).ravel()

    loss = RD.nrmse_score(pred_va, yva[target])
    train_loss = RD.nrmse_score(pred_tr, ytr[target])
    err_va = RD.per_sample_err(pred_va, yva[target])
    err_tr = RD.per_sample_err(pred_tr, ytr[target])

    expr = getattr(model, "expressions_", None)
    sc = {
        "approach": spec["name"], "approach_number": spec["number"],
        "benchmark": "remnant", "agent": "opus48",
        "category": spec["category"], "parameterization": spec["reparam"],
        "target": target,
        "loss": loss, "loss_components": {"nrmse_v_k": loss},
        "train_loss": train_loss,
        "val_median_err": float(np.median(err_va)),
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
        "n_params": int(_count(model)),
        "notes": spec.get("notes", ""),
        "wall_time_s": round(time.time() - t0, 1),
    }
    if write:
        _persist(spec, model, mean, std, sc, err_tr, err_va, expr)
    if verbose:
        print(f"[{sc['approach_number']:2d}] {sc['approach']:26s} "
              f"NRMSE={loss:.4f} train={train_loss:.4f} rt={runtime_ms:.3f}ms")
    return dict(scorecard=sc, err_tr=err_tr, err_va=err_va)


def _count(model):
    for attr in ("coef_", "dual_coef_"):
        v = getattr(model, attr, None)
        if v is not None:
            return int(np.size(v))
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    return 0


def _persist(spec, model, mean, std, sc, err_tr, err_va, expr):
    mdir = MODELS / f"NN_{spec['name']}"; sdir = mdir / "saved_model"
    sdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "mean": mean, "std": std,
                 "reparam": spec["reparam"], "target": sc["target"]},
                sdir / "model.joblib")
    np.savez(sdir / "errors.npz", err_train=err_tr, err_val=err_va)
    (mdir / "scorecard.json").write_text(json.dumps(sc, indent=2))
    if expr:
        (sdir / "expressions.json").write_text(json.dumps(expr, indent=2))
    _write_scripts(mdir, spec)


def _write_scripts(mdir, spec):
    (mdir / "train.py").write_text(f'''"""Reproducible training for remnant approach: {spec['name']}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import rmodels as M, approaches_r as A
res = M.run_approach(A.get_spec({spec['number']!r}), write=True)
print("NRMSE:", res["scorecard"]["loss"])
''')
    (mdir / "predict.py").write_text('''"""Importable prediction: predict(params7) -> kick magnitude v_k."""
import sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import rdata as RD
_M = joblib.load(Path(__file__).resolve().parent / "saved_model" / "model.joblib")

def predict(params7):
    P = np.atleast_2d(np.asarray(params7, float))
    X = RD.reparam(P, _M["reparam"])
    X = (X - _M["mean"]) / _M["std"]
    return np.asarray(_M["model"].predict(X)).ravel()
''')
