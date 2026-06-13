"""Models + runner for the Ringdown Bench (opus48, original work).

1D regression a -> (omega_R, omega_I). Includes custom analytical models
(Chebyshev, rational/Pade) plus sklearn estimators and symbolic regression.
All models implement fit(a_1d_or_X, Y[n,2]) / predict(...) -> (n,2).
"""
import json, time
from pathlib import Path
import numpy as np
import joblib
import rddata as RD

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)


# ---------------- custom analytical models (operate on raw spin a) ----------
class PolyFit:
    def __init__(self, deg=10, reparam="raw_a"):
        self.deg = deg; self.reparam = reparam
    def fit(self, a, Y):
        x = RD.reparam(a, self.reparam).ravel()
        self.c_ = [np.polyfit(x, Y[:, k], self.deg) for k in range(Y.shape[1])]
        self.n_params = sum(len(c) for c in self.c_); return self
    def predict(self, a):
        x = RD.reparam(a, self.reparam).ravel()
        return np.column_stack([np.polyval(c, x) for c in self.c_])


class ChebyFit:
    def __init__(self, deg=12, reparam="raw_a"):
        self.deg = deg; self.reparam = reparam
    def fit(self, a, Y):
        x = RD.reparam(a, self.reparam).ravel()
        self.lo, self.hi = x.min(), x.max()
        xs = 2 * (x - self.lo) / (self.hi - self.lo) - 1
        self.c_ = [np.polynomial.chebyshev.Chebyshev.fit(xs, Y[:, k], self.deg)
                   for k in range(Y.shape[1])]
        self.n_params = (self.deg + 1) * Y.shape[1]; return self
    def predict(self, a):
        x = RD.reparam(a, self.reparam).ravel()
        xs = 2 * (x - self.lo) / (self.hi - self.lo) - 1
        return np.column_stack([c(xs) for c in self.c_])


class RationalFit:
    """Rational P_m(x)/Q_n(x) with Q(0)=1, fit by linearised least squares
    then one Levenberg-Marquardt refinement (scipy)."""
    def __init__(self, m=6, n=6, reparam="raw_a"):
        self.m = m; self.n = n; self.reparam = reparam
    def _fit_one(self, x, y):
        from numpy.polynomial import polynomial as Pp
        A = np.column_stack([x ** k for k in range(self.m + 1)] +
                            [-y * x ** k for k in range(1, self.n + 1)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        p = coef[:self.m + 1]; q = np.concatenate([[1.0], coef[self.m + 1:]])
        try:
            from scipy.optimize import least_squares
            def res(w):
                pp = w[:self.m + 1]; qq = np.concatenate([[1.0], w[self.m + 1:]])
                num = sum(pp[k] * x ** k for k in range(self.m + 1))
                den = sum(qq[k] * x ** k for k in range(self.n + 1))
                return (num / den - y) / np.abs(y)
            w0 = np.concatenate([p, q[1:]])
            sol = least_squares(res, w0, max_nfev=2000)
            p = sol.x[:self.m + 1]; q = np.concatenate([[1.0], sol.x[self.m + 1:]])
        except Exception:
            pass
        return p, q
    def fit(self, a, Y):
        x = RD.reparam(a, self.reparam).ravel()
        self.pq_ = [self._fit_one(x, Y[:, k]) for k in range(Y.shape[1])]
        self.n_params = (self.m + self.n + 1) * Y.shape[1]; return self
    def predict(self, a):
        x = RD.reparam(a, self.reparam).ravel()
        out = []
        for p, q in self.pq_:
            num = sum(p[k] * x ** k for k in range(len(p)))
            den = sum(q[k] * x ** k for k in range(len(q)))
            out.append(num / den)
        return np.column_stack(out)


class SplineFit:
    def __init__(self, k=3, s=0.0, reparam="raw_a"):
        self.k = k; self.s = s; self.reparam = reparam
    def fit(self, a, Y):
        from scipy.interpolate import UnivariateSpline
        x = RD.reparam(a, self.reparam).ravel()
        idx = np.argsort(x); x = x[idx]; Y = Y[idx]
        self.sp_ = [UnivariateSpline(x, Y[:, k], k=self.k, s=self.s)
                    for k in range(Y.shape[1])]
        self.n_params = 0; return self
    def predict(self, a):
        x = RD.reparam(a, self.reparam).ravel()
        return np.column_stack([sp(x) for sp in self.sp_])


class RBFInterp:
    def __init__(self, kernel="thin_plate_spline", smoothing=0.0, reparam="raw_a", kw=None):
        self.kernel = kernel; self.smoothing = smoothing; self.reparam = reparam; self.kw = kw or {}
    def fit(self, a, Y):
        from scipy.interpolate import RBFInterpolator
        X = RD.reparam(a, self.reparam)
        self.r_ = RBFInterpolator(X, Y, kernel=self.kernel, smoothing=self.smoothing, **self.kw)
        self.n_params = 0; return self
    def predict(self, a):
        return self.r_(RD.reparam(a, self.reparam))


class SymbolicRD:
    """PySR or gplearn, fit omega_R and omega_I separately. Saves expressions."""
    def __init__(self, backend="pysr", reparam="raw_a", **kw):
        self.backend = backend; self.reparam = reparam; self.kw = kw
    def fit(self, a, Y):
        X = RD.reparam(a, self.reparam)
        self.models_ = []; self.expressions_ = {}
        for k, name in enumerate(["omega_r", "omega_i"]):
            m, e = self._one(X, Y[:, k]); self.models_.append(m)
            self.expressions_[name] = e
        self.n_params = 0; return self
    def _one(self, X, y):
        if self.backend == "pysr":
            from pysr import PySRRegressor
            m = PySRRegressor(niterations=self.kw.get("niterations", 80),
                              binary_operators=["+", "-", "*", "/"],
                              unary_operators=["sqrt", "log", "exp", "square", "cube"],
                              maxsize=self.kw.get("maxsize", 30), populations=20,
                              progress=False, verbosity=0, temp_equation_file=True,
                              random_state=0, deterministic=True, parallelism="serial")
            m.fit(X, y); e = []
            try:
                for _, r in m.equations_.iterrows():
                    e.append({"equation": str(r["equation"]), "complexity": int(r["complexity"]),
                              "loss": float(r["loss"])})
            except Exception:
                pass
            return m, e
        from gplearn.genetic import SymbolicRegressor
        m = SymbolicRegressor(population_size=self.kw.get("population_size", 4000),
                              generations=self.kw.get("generations", 40), tournament_size=20,
                              function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
                              metric="mse", parsimony_coefficient=0.0005, random_state=42, verbose=0)
        m.fit(X, y)
        return m, [{"equation": str(m._program), "complexity": int(m._program.length_),
                    "loss": float(m.run_details_["best_fitness"][-1])}]
    def predict(self, a):
        X = RD.reparam(a, self.reparam)
        return np.column_stack([np.asarray(m.predict(X)).ravel() for m in self.models_])


# ----------- sklearn wrappers (standardise the 1D feature) ------------------
class SkWrap:
    def __init__(self, make, reparam="raw_a"):
        self.make = make; self.reparam = reparam
    def fit(self, a, Y):
        X = RD.reparam(a, self.reparam); self.Xs_, self.mean_, self.std_ = RD.standardize(X)
        self.est_ = self.make(); self.est_.fit(self.Xs_, Y); self.n_params = 0; return self
    def predict(self, a):
        X = (RD.reparam(a, self.reparam) - self.mean_) / self.std_
        return np.atleast_2d(self.est_.predict(X))


def _gpr():
    from sklearn.gaussian_process import GaussianProcessRegressor as G
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
    return G(kernel=C(1.0) * RBF(1.0) + W(1e-8), alpha=1e-10, normalize_y=True, n_restarts_optimizer=3)


def _mlp():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    b = MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-5, max_iter=8000,
                     activation="tanh", random_state=0)
    return TransformedTargetRegressor(b, transformer=StandardScaler())


def _rf():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(400, random_state=0, n_jobs=-1)


def _gbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=400, random_state=0))


def _krr():
    from sklearn.kernel_ridge import KernelRidge
    return KernelRidge(alpha=1e-6, kernel="rbf", gamma=1.0)


# ------------------------------- runner -------------------------------------
def run_approach(spec, write=True, verbose=True):
    t0 = time.time()
    a_tr, Y_tr = RD.load("training"); a_va, Y_va = RD.load("validation")
    model = spec["make"]()
    model.fit(a_tr, Y_tr)
    tp = time.time()
    pred_va = np.asarray(model.predict(a_va))
    runtime_ms = 1e3 * (time.time() - tp) / len(a_va)
    pred_tr = np.asarray(model.predict(a_tr))
    L, rr, ri = RD.loss(pred_va, Y_va)
    Lt, _, _ = RD.loss(pred_tr, Y_tr)
    err_va = RD.per_sample_err(pred_va, Y_va); err_tr = RD.per_sample_err(pred_tr, Y_tr)
    expr = getattr(model, "expressions_", None)
    sc = {"approach": spec["name"], "approach_number": spec["number"],
          "benchmark": "ringdown", "agent": "opus48", "category": spec["category"],
          "parameterization": spec["reparam"], "mode": "l2_m2_n0",
          "loss": L, "loss_components": {"rel_error_omega_real": rr, "rel_error_omega_imag": ri},
          "train_loss": Lt, "val_median": float(np.median(err_va)),
          "runtime_ms": float(runtime_ms), "n_train": int(len(a_tr)), "n_val": int(len(a_va)),
          "n_params": int(getattr(model, "n_params", 0)), "notes": spec.get("notes", ""),
          "wall_time_s": round(time.time() - t0, 1)}
    if expr:
        sc["expression_omega_r"] = expr["omega_r"][-1]["equation"] if expr.get("omega_r") else ""
        sc["expression_omega_i"] = expr["omega_i"][-1]["equation"] if expr.get("omega_i") else ""
    if write:
        _persist(spec, model, sc, err_tr, err_va, expr)
    if verbose:
        print(f"[{sc['approach_number']:2d}] {sc['approach']:26s} loss={L:.2e} "
              f"(wR={rr:.2e} wI={ri:.2e}) train={Lt:.2e}")
    return dict(scorecard=sc, err_tr=err_tr, err_va=err_va)


def _persist(spec, model, sc, err_tr, err_va, expr):
    mdir = MODELS / f"NN_{spec['name']}"; sdir = mdir / "saved_model"
    sdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model}, sdir / "model.joblib")
    np.savez(sdir / "errors.npz", err_train=err_tr, err_val=err_va)
    (mdir / "scorecard.json").write_text(json.dumps(sc, indent=2))
    if expr:
        (sdir / "expressions.json").write_text(json.dumps(expr, indent=2))
    (mdir / "train.py").write_text(f'''"""Reproducible training for ringdown approach: {spec['name']}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import rdmodels as M, rdapproaches as A
print("loss:", M.run_approach(A.get_spec({spec['number']!r}), write=True)["scorecard"]["loss"])
''')
    (mdir / "predict.py").write_text('''"""predict(spin_array) -> (omega_r, omega_i)."""
import sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import rddata, rdmodels
_M = joblib.load(Path(__file__).resolve().parent / "saved_model" / "model.joblib")["model"]

def predict(spin_array):
    a = np.atleast_1d(np.asarray(spin_array, float))
    Y = np.atleast_2d(_M.predict(a))
    return Y[:, 0], Y[:, 1]
''')
