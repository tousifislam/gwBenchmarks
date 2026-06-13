"""SVD/EIM surrogate machinery + runner for Dynamics Bench (opus48, original).

Single target function log(x)(tau). Approach = (reparam, basis, rank, regressor).
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import joblib
import ddata as D

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)


class SVDBasis:
    def __init__(self, M=None, rank=None):
        if M is None:
            return
        self.mu = M.mean(0)
        U, s, Vt = np.linalg.svd(M - self.mu, full_matrices=False)
        self.rank = rank; self.s = s[:rank]; self.V = Vt[:rank]

    def project(self, M):
        return (M - self.mu) @ self.V.T

    def reconstruct(self, c):
        return self.mu + np.atleast_2d(c) @ self.V

    def to_dict(self):
        return {"type": "svd", "mu": self.mu, "V": self.V, "rank": self.rank}

    @classmethod
    def from_dict(cls, d):
        o = cls(); o.mu = d["mu"]; o.V = d["V"]; o.rank = int(d["rank"]); return o


class EIMBasis:
    def __init__(self, M=None, rank=None):
        if M is None:
            return
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        B = Vt[:rank].copy()
        nodes = [int(np.argmax(np.abs(B[0])))]
        for i in range(1, rank):
            A = B[:i][:, nodes].T
            c = np.linalg.solve(A, B[i][nodes])
            nodes.append(int(np.argmax(np.abs(B[i] - c @ B[:i]))))
        self.rank = rank; self.B = B; self.nodes = np.array(nodes)
        self.Binv = np.linalg.inv(B[:, self.nodes].T)

    def project(self, M):
        return M[:, self.nodes]

    def reconstruct(self, nv):
        return (np.atleast_2d(nv) @ self.Binv.T) @ self.B

    def to_dict(self):
        return {"type": "eim", "B": self.B, "nodes": self.nodes,
                "Binv": self.Binv, "rank": self.rank}

    @classmethod
    def from_dict(cls, d):
        o = cls(); o.B = d["B"]; o.nodes = d["nodes"]; o.Binv = d["Binv"]
        o.rank = int(d["rank"]); return o


def basis_from_dict(d):
    return EIMBasis.from_dict(d) if d.get("type") == "eim" else SVDBasis.from_dict(d)


# ----- estimator factories -----
class RBFInterp:
    def __init__(self, kernel="thin_plate_spline", smoothing=1e-3, kw=None):
        self.kernel = kernel; self.smoothing = smoothing; self.kw = kw or {}
    def fit(self, X, y):
        from scipy.interpolate import RBFInterpolator
        self.r_ = RBFInterpolator(X, y, kernel=self.kernel, smoothing=self.smoothing, **self.kw)
        self.n_features_in_ = X.shape[1]; return self
    def predict(self, X):
        return self.r_(np.atleast_2d(X))


class SymbolicCoeff:
    """PySR/gplearn on leading SVD coeffs, ridge on the tail."""
    def __init__(self, backend="pysr", n_sym=3, **kw):
        self.backend = backend; self.n_sym = n_sym; self.kw = kw
    def fit(self, X, Y):
        from sklearn.linear_model import Ridge
        Y = np.atleast_2d(Y); nout = Y.shape[1]; nsym = min(self.n_sym, nout)
        self.models_ = []; self.expressions_ = []
        for j in range(nsym):
            m, e = self._one(X, Y[:, j]); self.models_.append(m)
            self.expressions_.append({"coeff_index": j, "expressions": e})
        self.tail_ = Ridge(1.0).fit(X, Y[:, nsym:]) if nout > nsym else None
        self.nout_ = nout; self.nsym_ = nsym; self.n_features_in_ = X.shape[1]
        return self
    def _one(self, X, y):
        if self.backend == "pysr":
            from pysr import PySRRegressor
            m = PySRRegressor(niterations=self.kw.get("niterations", 40),
                              binary_operators=["+", "-", "*", "/"],
                              unary_operators=["square", "cube", "sqrt", "exp", "log"],
                              maxsize=24, populations=15, progress=False, verbosity=0,
                              temp_equation_file=True, random_state=0,
                              deterministic=True, parallelism="serial")
            m.fit(X, y); e = []
            try:
                for _, r in m.equations_.iterrows():
                    e.append({"equation": str(r["equation"]), "complexity": int(r["complexity"]),
                              "loss": float(r["loss"])})
            except Exception:
                pass
            return m, e
        from gplearn.genetic import SymbolicRegressor
        m = SymbolicRegressor(population_size=self.kw.get("population_size", 2000),
                              generations=self.kw.get("generations", 20), tournament_size=20,
                              function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
                              metric="mse", parsimony_coefficient=0.001, random_state=42, verbose=0)
        m.fit(X, y)
        return m, [{"equation": str(m._program), "complexity": int(m._program.length_),
                    "loss": float(m.run_details_["best_fitness"][-1])}]
    def predict(self, X):
        X = np.atleast_2d(X); out = np.zeros((X.shape[0], self.nout_))
        for j, m in enumerate(self.models_):
            out[:, j] = np.asarray(m.predict(X)).ravel()
        if self.tail_ is not None:
            out[:, self.nsym_:] = np.atleast_2d(self.tail_.predict(X))
        return out


def _gpr(kernel="rbf", length=4.0, alpha=1e-4):
    from sklearn.gaussian_process import GaussianProcessRegressor as G
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
    k = C(1.0) * (RBF(length) if kernel == "rbf" else Matern(length, nu=2.5))
    return G(kernel=k, alpha=alpha, normalize_y=True, optimizer=None)


def _poly(deg=3, alpha=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    return make_pipeline(PolynomialFeatures(deg), Ridge(alpha=alpha))


def _mlp(hidden=(128, 128), alpha=1e-3):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    b = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=4000,
                     activation="relu", random_state=0, learning_rate_init=2e-3)
    return TransformedTargetRegressor(b, transformer=StandardScaler())


def _rf(n=400):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n, random_state=0, n_jobs=-1)


def _extratrees(n=500):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n, random_state=0, n_jobs=-1)


def _xgb():
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(XGBRegressor(n_estimators=400, max_depth=4,
                                             learning_rate=0.05, random_state=0))


def _krr(alpha=1e-2):
    from sklearn.kernel_ridge import KernelRidge
    return KernelRidge(alpha=alpha, kernel="rbf")


def _knn(k=8):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(k, weights="distance")


def _rbf(kernel="thin_plate_spline", smoothing=1e-3, **kw):
    return RBFInterp(kernel, smoothing, kw)


# ----- runner -----
def run_approach(spec, n_train_eval=80, write=True, verbose=True):
    t0 = time.time()
    Ptr, wtr = D.load("training"); Pva, wva = D.load("validation")
    L, Etr = D.build_matrix("training"); Lv, Eva = D.build_matrix("validation")
    Xtr = D.reparam(Ptr, spec["reparam"]); Xva = D.reparam(Pva, spec["reparam"])
    Xtr_s, mean, std = D.standardize(Xtr); Xva_s, _, _ = D.standardize(Xva, mean, std)

    basis = (EIMBasis if spec.get("basis") == "eim" else SVDBasis)(L, spec["rank"])
    c = basis.project(L)
    model = spec["make"]()
    model.fit(Xtr_s, c)

    def predict_set(Xs, waves, ends, idx):
        pc = np.atleast_2d(model.predict(Xs[idx]))
        out = []
        for k, i in enumerate(idx):
            t, x = waves[i]
            lx = basis.reconstruct(pc[k])[0]
            out.append((t, D.from_tau(lx, t, t[0], t[-1])))
        return out

    vidx = np.arange(len(wva))
    tp = time.time()
    pred_va = predict_set(Xva_s, wva, Eva, vidx)
    runtime_ms = 1e3 * (time.time() - tp) / len(vidx)
    err_va = D.score(pred_va, [wva[i] for i in vidx])

    tidx = np.linspace(0, len(wtr) - 1, min(n_train_eval, len(wtr))).astype(int)
    pred_tr = predict_set(Xtr_s, wtr, Etr, tidx)
    err_tr = D.score(pred_tr, [wtr[i] for i in tidx])

    expr = getattr(model, "expressions_", None)
    sc = {"approach": spec["name"], "approach_number": spec["number"],
          "benchmark": "dynamics", "agent": "opus48", "category": spec["category"],
          "parameterization": spec["reparam"], "time_convention": "tau_normalized",
          "rank": spec["rank"], "loss": float(np.mean(err_va)),
          "loss_components": {"rms_relative_error_x": float(np.mean(err_va))},
          "val_median": float(np.median(err_va)), "train_loss": float(np.mean(err_tr)),
          "runtime_ms": float(runtime_ms), "n_train": int(len(wtr)), "n_val": int(len(wva)),
          "n_params": int(_count(model)), "notes": spec.get("notes", ""),
          "wall_time_s": round(time.time() - t0, 1)}
    if write:
        _persist(spec, model, basis, mean, std, sc, err_tr, err_va, expr)
    if verbose:
        print(f"[{sc['approach_number']:2d}] {sc['approach']:26s} "
              f"rms_rel={sc['loss']:.4f} med={sc['val_median']:.4f} train={sc['train_loss']:.4f}")
    return dict(scorecard=sc, err_tr=err_tr, err_va=err_va)


def _count(model):
    for a in ("coef_", "dual_coef_"):
        v = getattr(model, a, None)
        if v is not None:
            return int(np.size(v))
    return int(getattr(model, "n_features_in_", 0))


def _persist(spec, model, basis, mean, std, sc, err_tr, err_va, expr):
    mdir = MODELS / f"NN_{spec['name']}"; sdir = mdir / "saved_model"
    sdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "basis": basis.to_dict(), "mean": mean, "std": std,
                 "reparam": spec["reparam"]}, sdir / "model.joblib")
    np.savez(sdir / "errors.npz", err_train=err_tr, err_val=err_va)
    (mdir / "scorecard.json").write_text(json.dumps(sc, indent=2))
    if expr:
        (sdir / "expressions.json").write_text(json.dumps(expr, indent=2))
    (mdir / "train.py").write_text(f'''"""Reproducible training for dynamics approach: {spec['name']}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dmodels as M, dapproaches as A
print("rms_rel:", M.run_approach(A.get_spec({spec['number']!r}), write=True)["scorecard"]["loss"])
''')
    (mdir / "predict.py").write_text('''"""predict(params6, t_grid) -> x(t_grid)."""
import sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ddata as D, dmodels as M
_M = joblib.load(Path(__file__).resolve().parent / "saved_model" / "model.joblib")
_b = M.basis_from_dict(_M["basis"])

def predict(params6, t_grid):
    P = np.atleast_2d(np.asarray(params6, float))
    X = (D.reparam(P, _M["reparam"]) - _M["mean"]) / _M["std"]
    c = np.atleast_2d(_M["model"].predict(X))[0]
    lx = _b.reconstruct(c)[0]
    t = np.asarray(t_grid, float)
    return D.from_tau(lx, t, t[0], t[-1])
''')
