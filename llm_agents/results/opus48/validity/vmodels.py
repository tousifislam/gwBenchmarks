"""Models + runner for the Validity Bench (opus48, original work).

Scalar regression of y = ln(mm). predict.py returns mm_hat = exp(y_hat).
Categories: kernel/GP, symbolic (PySR+gplearn), interpolation, ML.
"""
import json, time
from pathlib import Path
import numpy as np
import joblib
import vdata as VD

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)


class RBFInterp:
    def __init__(self, kernel="thin_plate_spline", smoothing=1e-2, kw=None):
        self.kernel = kernel; self.smoothing = smoothing; self.kw = kw or {}
    def fit(self, X, y):
        from scipy.interpolate import RBFInterpolator
        self.r_ = RBFInterpolator(X, y, kernel=self.kernel, smoothing=self.smoothing, **self.kw)
        self.n_features_in_ = X.shape[1]; return self
    def predict(self, X):
        return self.r_(np.atleast_2d(X))


class SymbolicScalar:
    def __init__(self, backend="pysr", **kw):
        self.backend = backend; self.kw = kw
    def fit(self, X, y):
        if self.backend == "pysr":
            from pysr import PySRRegressor
            m = PySRRegressor(niterations=self.kw.get("niterations", 60),
                              binary_operators=["+", "-", "*", "/"],
                              unary_operators=["square", "cube", "sqrt", "exp", "log"],
                              maxsize=self.kw.get("maxsize", 28), populations=20,
                              progress=False, verbosity=0, temp_equation_file=True,
                              random_state=0, deterministic=True, parallelism="serial")
            m.fit(X, y); e = []
            try:
                for _, r in m.equations_.iterrows():
                    e.append({"equation": str(r["equation"]), "complexity": int(r["complexity"]),
                              "loss": float(r["loss"])})
            except Exception:
                pass
        else:
            from gplearn.genetic import SymbolicRegressor
            m = SymbolicRegressor(population_size=self.kw.get("population_size", 3000),
                                  generations=self.kw.get("generations", 30), tournament_size=20,
                                  function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
                                  metric="mse", parsimony_coefficient=0.001, random_state=42, verbose=0)
            m.fit(X, y)
            e = [{"equation": str(m._program), "complexity": int(m._program.length_),
                  "loss": float(m.run_details_["best_fitness"][-1])}]
        self.m_ = m; self.expressions_ = e; return self
    def predict(self, X):
        return np.asarray(self.m_.predict(X)).ravel()


def _gpr(kernel="rbf", length=2.0, alpha=1e-2, opt=True):
    from sklearn.gaussian_process import GaussianProcessRegressor as G
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel as W
    base = RBF(length) if kernel == "rbf" else Matern(length, nu=2.5)
    return G(kernel=C(1.0) * base + W(1e-1), alpha=alpha, normalize_y=True,
             n_restarts_optimizer=(2 if opt else 0), optimizer=("fmin_l_bfgs_b" if opt else None))


def _krr(alpha=1e-1):
    from sklearn.kernel_ridge import KernelRidge
    return KernelRidge(alpha=alpha, kernel="rbf")


def _svr(C=5.0):
    from sklearn.svm import SVR
    return SVR(C=C, gamma="scale", epsilon=0.05)


def _mlp(hidden=(128, 128), alpha=1e-2):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    b = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=5000,
                     activation="relu", random_state=0, learning_rate_init=2e-3)
    return TransformedTargetRegressor(b, transformer=StandardScaler())


def _rf(n=500):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n, random_state=0, n_jobs=-1)


def _extratrees(n=600):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n, random_state=0, n_jobs=-1)


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, random_state=0)


def _lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=700, num_leaves=31, learning_rate=0.03,
                         subsample=0.8, random_state=0, verbose=-1)


def _poly(deg=3, alpha=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    return make_pipeline(PolynomialFeatures(deg), Ridge(alpha=alpha))


def _rbf(kernel="thin_plate_spline", smoothing=1e-2, **kw):
    return RBFInterp(kernel, smoothing, kw)


def _knn(k=8):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(k, weights="distance")


def _ensemble_mlp():
    """Deep ensemble: average of MLPs with different seeds/architectures."""
    return DeepEnsemble()


class DeepEnsemble:
    def __init__(self):
        self.members = []
    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        archs = [(128, 128), (256, 128), (64, 64, 64)]
        self.members = []
        for i, h in enumerate(archs):
            m = MLPRegressor(hidden_layer_sizes=h, alpha=1e-2, max_iter=4000,
                             activation="relu", random_state=i, learning_rate_init=2e-3)
            m.fit(X, y); self.members.append(m)
        self.n_features_in_ = X.shape[1]; return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


def run_approach(spec, write=True, verbose=True):
    t0 = time.time()
    Ptr, ytr = VD.load("training"); Pva, yva = VD.load("validation")
    Xtr = VD.reparam(Ptr, spec["reparam"]); Xva = VD.reparam(Pva, spec["reparam"])
    Xtr_s, mean, std = VD.standardize(Xtr); Xva_s, _, _ = VD.standardize(Xva, mean, std)
    model = spec["make"](); model.fit(Xtr_s, ytr)
    tp = time.time()
    pred_va = np.asarray(model.predict(Xva_s)).ravel()
    runtime_ms = 1e3 * (time.time() - tp) / len(Xva_s)
    pred_tr = np.asarray(model.predict(Xtr_s)).ravel()
    loss = VD.log_rmse(pred_va, yva); train_loss = VD.log_rmse(pred_tr, ytr)
    err_va = VD.per_sample_err(pred_va, yva); err_tr = VD.per_sample_err(pred_tr, ytr)
    expr = getattr(model, "expressions_", None)
    sc = {"approach": spec["name"], "approach_number": spec["number"], "benchmark": "validity",
          "agent": "opus48", "category": spec["category"], "parameterization": spec["reparam"],
          "loss": loss, "loss_components": {"log_rmse": loss}, "train_loss": train_loss,
          "val_median_err": float(np.median(err_va)), "runtime_ms": float(runtime_ms),
          "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_params": int(_count(model)),
          "notes": spec.get("notes", ""), "wall_time_s": round(time.time() - t0, 1)}
    if write:
        _persist(spec, model, mean, std, sc, err_tr, err_va, expr)
    if verbose:
        print(f"[{sc['approach_number']:2d}] {sc['approach']:26s} log_rmse={loss:.4f} train={train_loss:.4f}")
    return dict(scorecard=sc, err_tr=err_tr, err_va=err_va)


def _count(model):
    for a in ("coef_", "dual_coef_"):
        v = getattr(model, a, None)
        if v is not None:
            return int(np.size(v))
    return int(getattr(model, "n_features_in_", 0))


def _persist(spec, model, mean, std, sc, err_tr, err_va, expr):
    mdir = MODELS / f"NN_{spec['name']}"; sdir = mdir / "saved_model"
    sdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "mean": mean, "std": std, "reparam": spec["reparam"]},
                sdir / "model.joblib")
    np.savez(sdir / "errors.npz", err_train=err_tr, err_val=err_va)
    (mdir / "scorecard.json").write_text(json.dumps(sc, indent=2))
    if expr:
        (sdir / "expressions.json").write_text(json.dumps(expr, indent=2))
    (mdir / "train.py").write_text(f'''"""Reproducible training for validity approach: {spec['name']}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vmodels as M, vapproaches as A
print("log_rmse:", M.run_approach(A.get_spec({spec['number']!r}), write=True)["scorecard"]["loss"])
''')
    (mdir / "predict.py").write_text('''"""predict(params4) -> predicted mismatch (q,chi1z,chi2z,omega0)."""
import sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vdata as VD
_M = joblib.load(Path(__file__).resolve().parent / "saved_model" / "model.joblib")

def predict(params4):
    P = np.atleast_2d(np.asarray(params4, float))
    X = (VD.reparam(P, _M["reparam"]) - _M["mean"]) / _M["std"]
    return np.exp(np.asarray(_M["model"].predict(X)).ravel())   # mismatch
''')
