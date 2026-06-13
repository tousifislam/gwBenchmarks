"""Registry of waveform surrogate approaches (opus48, original work).

Each approach = (reparameterisation, basis, ranks, coefficient regressor).
Coverage:
  * decomposition : SVD+GPR, SVD+poly, SVD+NN, SVD+RF, EIM+GPR
  * symbolic      : PySR and gplearn on SVD coeffs (mandatory), 2 reparams
  * interp/kernel : RBF interpolation, kernel ridge, kNN+correction, thin-plate
  * ml            : MLP (deep), random forest, gradient boosting, extra-trees

Reparameterisations exercised: raw_7d, raw_omega, eff_spin, eff_spin_omega,
spherical, massdiff  (>= 3 required).
"""
import numpy as np

# ----- sklearn estimators (lazy imports inside factories so module import is cheap)


class RBFRegressor:
    """Multi-output wrapper around scipy.interpolate.RBFInterpolator."""

    def __init__(self, kernel="thin_plate_spline", smoothing=1e-3, **kw):
        self.kernel = kernel
        self.smoothing = smoothing
        self.kw = kw

    def fit(self, X, y):
        from scipy.interpolate import RBFInterpolator
        self.rbf_ = RBFInterpolator(X, y, kernel=self.kernel,
                                    smoothing=self.smoothing, **self.kw)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.rbf_(np.atleast_2d(X))


class SymbolicCoeffModel:
    """Fit a symbolic regressor (PySR or gplearn) on the leading coefficients
    and a cheap ridge regressor on the remaining (sub-dominant) coefficients.

    Symbolic regression on every one of ~30 coeffs is intractable, and the
    leading coeffs carry almost all the variance, so this is the principled
    split: interpretable closed forms where they matter, ridge for the tail.
    """

    def __init__(self, backend="pysr", n_sym=4, **kw):
        self.backend = backend
        self.n_sym = n_sym
        self.kw = kw

    def fit(self, X, Y):
        from sklearn.linear_model import Ridge
        Y = np.atleast_2d(Y)
        nout = Y.shape[1]
        nsym = min(self.n_sym, nout)
        self.sym_models_ = []
        self.expressions_ = []
        for j in range(nsym):
            m, exprs = self._fit_one(X, Y[:, j], j)
            self.sym_models_.append(m)
            self.expressions_.append({"coeff_index": j, "expressions": exprs})
        # ridge for the rest
        self.tail_ = None
        if nout > nsym:
            self.tail_ = Ridge(alpha=1.0).fit(X, Y[:, nsym:])
        self.nout_ = nout
        self.nsym_ = nsym
        self.n_features_in_ = X.shape[1]
        return self

    def _fit_one(self, X, y, j):
        if self.backend == "pysr":
            from pysr import PySRRegressor
            m = PySRRegressor(
                niterations=self.kw.get("niterations", 40),
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "cube", "sqrt", "exp"],
                maxsize=self.kw.get("maxsize", 22),
                populations=self.kw.get("populations", 15),
                progress=False,
                verbosity=0,
                temp_equation_file=True,
                random_state=0,
                deterministic=True,
                parallelism="serial",
            )
            m.fit(X, y)
            exprs = []
            try:
                for _, row in m.equations_.iterrows():
                    exprs.append({"equation": str(row["equation"]),
                                  "complexity": int(row["complexity"]),
                                  "loss": float(row["loss"])})
            except Exception:
                pass
            return m, exprs
        else:  # gplearn
            from gplearn.genetic import SymbolicRegressor
            m = SymbolicRegressor(
                population_size=self.kw.get("population_size", 2000),
                generations=self.kw.get("generations", 20),
                tournament_size=20,
                function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
                metric="mse",
                parsimony_coefficient=0.001,
                random_state=42,
                verbose=0,
            )
            m.fit(X, y)
            exprs = [{"equation": str(m._program),
                      "complexity": int(m._program.length_),
                      "loss": float(m.run_details_["best_fitness"][-1])}]
            return m, exprs

    def predict(self, X):
        X = np.atleast_2d(X)
        out = np.zeros((X.shape[0], self.nout_))
        for j, m in enumerate(self.sym_models_):
            out[:, j] = np.asarray(m.predict(X)).ravel()
        if self.tail_ is not None:
            out[:, self.nsym_:] = np.atleast_2d(self.tail_.predict(X))
        return out


# ------------------------- estimator factories -------------------------------

def _gpr(kernel="rbf", alpha=1e-3, length=6.0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
    # FIXED long length scale (optimizer="fixed"): with only 250 points in
    # 8-11D, marginal-likelihood optimisation drives the length scale too short
    # and the GP interpolates the training set but extrapolates to garbage on
    # validation. A fixed broad kernel + moderate jitter generalises far better.
    if kernel == "rbf":
        k = ConstantKernel(1.0) * RBF(length)
    else:
        k = ConstantKernel(1.0) * Matern(length, nu=2.5)
    return GaussianProcessRegressor(kernel=k, alpha=alpha, normalize_y=True,
                                    optimizer=None)


def _mlp_scaled(hidden=(128, 128), alpha=1e-3, max_iter=3000):
    """MLP with standardised targets (coeffs span many orders of magnitude)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    base = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                        activation="relu", solver="adam", random_state=0,
                        learning_rate_init=2e-3)
    return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())


def _poly(deg=3, alpha=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    return make_pipeline(PolynomialFeatures(deg), Ridge(alpha=alpha))


def _mlp(hidden=(128, 128), alpha=1e-4, max_iter=2000):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                        max_iter=max_iter, early_stopping=False,
                        activation="tanh", random_state=0)


def _rf(n=300):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n, random_state=0, n_jobs=-1)


def _extratrees(n=400):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=n, random_state=0, n_jobs=-1)


def _gbr(n=200):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=n,
                                                          max_depth=3, random_state=0))


def _krr(alpha=1e-3, gamma=None):
    from sklearn.kernel_ridge import KernelRidge
    return KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)


def _knn(k=5):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=k, weights="distance")


# ------------------------------ the registry ---------------------------------
# Each spec is a dict; make_modelA / make_modelP are zero-arg factories.

def _spec(number, name, category, reparam, rankA, rankP, mkA, mkP,
          basis="svd", notes="", parameterization=None):
    return dict(number=number, name=name, category=category, reparam=reparam,
                rankA=rankA, rankP=rankP, make_modelA=mkA, make_modelP=mkP,
                basis=basis, notes=notes,
                parameterization=parameterization or reparam)


SPECS = [
    # ---- decomposition ----
    _spec(1, "svd_gpr_rbf_raw", "decomposition", "raw_newt", 25, 35,
          lambda: _gpr("rbf"), lambda: _gpr("rbf"),
          notes="Baseline SVD + GPR (RBF) on raw params + Newtonian features."),
    _spec(2, "svd_gpr_matern_eff", "decomposition", "eff_newt", 25, 35,
          lambda: _gpr("matern"), lambda: _gpr("matern"),
          notes="SVD + GPR (Matern 5/2) on effective-spin + Newtonian reparam."),
    _spec(3, "svd_poly3_raw", "decomposition", "raw_omega", 25, 30,
          lambda: _poly(3, 1.0), lambda: _poly(3, 1.0),
          notes="SVD + cubic polynomial ridge regression."),
    _spec(4, "svd_mlp_eff", "decomposition", "eff_spin_omega", 25, 30,
          lambda: _mlp_scaled((128, 128)), lambda: _mlp_scaled((128, 128)),
          notes="SVD + MLP (scaled targets, 128x128) on effective spins."),
    _spec(5, "eim_gpr_raw", "decomposition", "raw_omega", 25, 30,
          lambda: _gpr("matern"), lambda: _gpr("matern"), basis="eim",
          notes="Empirical Interpolation Method nodes + GPR on node values."),
    # ---- symbolic (mandatory PySR + gplearn) ----
    _spec(6, "svd_pysr_eff", "symbolic", "eff_spin_omega", 12, 14,
          lambda: SymbolicCoeffModel("pysr", n_sym=4),
          lambda: SymbolicCoeffModel("pysr", n_sym=4),
          notes="PySR on leading SVD coeffs (eff spins), ridge tail."),
    _spec(7, "svd_gplearn_raw", "symbolic", "raw_omega", 12, 14,
          lambda: SymbolicCoeffModel("gplearn", n_sym=4),
          lambda: SymbolicCoeffModel("gplearn", n_sym=4),
          notes="gplearn SymbolicRegressor on leading SVD coeffs (raw)."),
    _spec(8, "svd_pysr_spherical", "symbolic", "spherical", 12, 14,
          lambda: SymbolicCoeffModel("pysr", n_sym=3),
          lambda: SymbolicCoeffModel("pysr", n_sym=3),
          notes="PySR, second reparam (spherical spins)."),
    # ---- interpolation / kernel ----
    _spec(9, "rbf_thinplate_raw", "interp_kernel", "raw_newt", 25, 35,
          lambda: RBFRegressor("thin_plate_spline", 1e-3),
          lambda: RBFRegressor("thin_plate_spline", 1e-3),
          notes="Thin-plate-spline RBF interpolation on SVD coeffs."),
    _spec(10, "rbf_multiquadric_eff", "interp_kernel", "eff_spin_omega", 25, 30,
          lambda: RBFRegressor("multiquadric", 1e-3, epsilon=1.0),
          lambda: RBFRegressor("multiquadric", 1e-3, epsilon=1.0),
          notes="Multiquadric RBF interpolation, eff-spin reparam."),
    _spec(11, "krr_rbf_eff", "interp_kernel", "eff_newt", 25, 35,
          lambda: _krr(1e-3), lambda: _krr(1e-3),
          notes="Kernel ridge (RBF) on SVD coeffs."),
    _spec(12, "knn_correction_raw", "interp_kernel", "raw_omega", 25, 30,
          lambda: _knn(6), lambda: _knn(6),
          notes="Distance-weighted kNN on SVD coeffs."),
    # ---- machine learning ----
    _spec(13, "mlp_deep_eff", "ml", "eff_newt", 25, 35,
          lambda: _mlp_scaled((256, 256, 128)), lambda: _mlp_scaled((256, 256, 128)),
          notes="Deeper MLP (256-256-128)."),
    _spec(14, "rf_spherical", "ml", "spherical", 25, 30,
          lambda: _rf(400), lambda: _rf(400),
          notes="Random forest on spherical-spin reparam."),
    _spec(15, "gbr_raw", "ml", "raw_omega", 20, 24,
          lambda: _gbr(200), lambda: _gbr(200),
          notes="Gradient boosting (per-coeff) on raw params."),
    _spec(16, "extratrees_eff", "ml", "eff_spin_omega", 25, 30,
          lambda: _extratrees(500), lambda: _extratrees(500),
          notes="Extremely randomised trees."),
    # ---- additional / reasoned variations ----
    _spec(17, "svd_gpr_highrank_raw", "decomposition", "raw_newt", 35, 45,
          lambda: _gpr("matern"), lambda: _gpr("matern"),
          notes="Higher-rank GPR: more basis for precessing modulations."),
    _spec(18, "svd_poly5_eff", "decomposition", "eff_spin_omega", 25, 30,
          lambda: _poly(5, 5.0), lambda: _poly(5, 5.0),
          notes="Quintic polynomial ridge."),
    _spec(19, "amp_phase_asymrank_eff", "decomposition", "eff_newt", 15, 45,
          lambda: _gpr("matern"), lambda: _gpr("matern"),
          notes="Asymmetric ranks: amplitude is smoother than phase, so "
                "fewer amp basis, more phase basis."),
    _spec(20, "mlp_massdiff", "ml", "massdiff", 25, 30,
          lambda: _mlp_scaled((128, 128)), lambda: _mlp_scaled((128, 128)),
          notes="MLP (scaled targets) on mass-difference reparam (third+ reparam)."),
    _spec(21, "rbf_linear_spherical", "interp_kernel", "spherical", 25, 30,
          lambda: RBFRegressor("linear", 1e-2),
          lambda: RBFRegressor("linear", 1e-2),
          notes="Linear-kernel RBF on spherical spins."),
    _spec(22, "gpr_rbf_eff_tuned", "decomposition", "eff_newt", 30, 40,
          lambda: _gpr("rbf", alpha=1e-7, length=2.0),
          lambda: _gpr("rbf", alpha=1e-7, length=2.0),
          notes="Reasoned GPR retune: longer length scale + lower noise."),
]

_BY_NUMBER = {s["number"]: s for s in SPECS}


def get_spec(number):
    return _BY_NUMBER[number]
