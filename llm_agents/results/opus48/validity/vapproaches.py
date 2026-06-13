"""Validity mismatch-prediction approach registry (opus48, original work).

Categories: kernel/GP, symbolic (PySR+gplearn), interpolation, ML.
Reparams: raw_4d, eff_spin, log_q, interactions, boundary.
"""
import vmodels as M


def _s(n, name, cat, rep, make, notes=""):
    return dict(number=n, name=name, category=cat, reparam=rep, make=make, notes=notes)


SPECS = [
    # kernel / GP
    _s(1, "gpr_rbf_raw", "kernel_gp", "raw_4d", lambda: M._gpr("rbf"),
       "Baseline GPR (RBF) on raw params."),
    _s(2, "gpr_matern_eff", "kernel_gp", "eff_spin", lambda: M._gpr("matern"),
       "GPR Matern on effective spins."),
    _s(3, "krr_rbf_logq", "kernel_gp", "log_q", lambda: M._krr(1e-1),
       "Kernel ridge on log(q)+log(omega0)."),
    _s(4, "svr_rbf_interactions", "kernel_gp", "interactions", lambda: M._svr(5.0),
       "SVR with interaction features."),
    _s(5, "gpr_rbf_boundary", "kernel_gp", "boundary", lambda: M._gpr("rbf", opt=False, length=3.0),
       "GPR with boundary-distance features (extrapolation awareness)."),
    # symbolic (mandatory)
    _s(6, "pysr_raw", "symbolic", "raw_4d", lambda: M.SymbolicScalar("pysr"),
       "PySR on log(mm), raw params."),
    _s(7, "gplearn_raw", "symbolic", "raw_4d", lambda: M.SymbolicScalar("gplearn"),
       "gplearn on log(mm), raw params."),
    _s(8, "pysr_boundary", "symbolic", "boundary", lambda: M.SymbolicScalar("pysr"),
       "PySR with boundary features (second reparam)."),
    _s(9, "gplearn_eff", "symbolic", "eff_spin", lambda: M.SymbolicScalar("gplearn"),
       "gplearn on effective spins."),
    # interpolation
    _s(10, "rbf_thinplate_eff", "interpolation", "eff_spin", lambda: M._rbf("thin_plate_spline", 1e-2),
       "Thin-plate RBF interpolation."),
    _s(11, "rbf_multiquadric_interactions", "interpolation", "interactions",
       lambda: M._rbf("multiquadric", 1e-2, epsilon=1.0), "Multiquadric RBF."),
    _s(12, "knn_raw", "interpolation", "raw_4d", lambda: M._knn(8),
       "Distance-weighted kNN."),
    _s(13, "rbf_linear_boundary", "interpolation", "boundary", lambda: M._rbf("linear", 1e-1),
       "Linear RBF on boundary features."),
    # ML
    _s(14, "mlp_eff", "ml", "eff_spin", lambda: M._mlp((128, 128)), "MLP (scaled target)."),
    _s(15, "deep_ensemble_boundary", "ml", "boundary", lambda: M._ensemble_mlp(),
       "Deep ensemble of MLPs (boundary features)."),
    _s(16, "rf_raw", "ml", "raw_4d", lambda: M._rf(500), "Random forest."),
    _s(17, "extratrees_eff", "ml", "eff_spin", lambda: M._extratrees(600), "Extra trees."),
    _s(18, "xgboost_boundary", "ml", "boundary", lambda: M._xgb(), "XGBoost with boundary features."),
    _s(19, "lightgbm_eff", "ml", "eff_spin", lambda: M._lgbm(), "LightGBM."),
    _s(20, "poly3_eff", "ml", "eff_spin", lambda: M._poly(3, 1.0), "Cubic polynomial ridge."),
    _s(21, "poly2_interactions", "ml", "interactions", lambda: M._poly(2, 1.0),
       "Quadratic polynomial on interaction features."),
    _s(22, "xgboost_logq", "ml", "log_q", lambda: M._xgb(), "XGBoost on log(q) reparam."),
    _s(23, "mlp_deep_boundary", "ml", "boundary", lambda: M._mlp((256, 256, 128)),
       "Deeper MLP on boundary features (reasoned: capture sharp validity edge)."),
]

_BY = {s["number"]: s for s in SPECS}


def get_spec(n):
    return _BY[n]
