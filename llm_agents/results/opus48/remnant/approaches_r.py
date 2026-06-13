"""Remnant kick-velocity approach registry (opus48, original work).

Categories: kernel/GP, symbolic (PySR+gplearn mandatory), interpolation, ML.
Reparams: raw_7d, eff_spin, massdiff_antisym, pn_products, spherical.
All approaches target vf_mag except the two auxiliary symbolic runs that fit
Mf and chif (required: run PySR/gplearn separately for Mf, chif, vf).
"""
import rmodels as M


def _s(number, name, category, reparam, make, notes="", target="vf_mag"):
    return dict(number=number, name=name, category=category, reparam=reparam,
                make=make, notes=notes, target=target)


SPECS = [
    # ---- kernel / GP ----
    _s(1, "gpr_rbf_raw", "kernel_gp", "raw_7d", lambda: M._gpr("rbf"),
       "Baseline GPR (RBF) on raw params."),
    _s(2, "gpr_matern_eff", "kernel_gp", "eff_spin", lambda: M._gpr("matern"),
       "GPR Matern 5/2 on effective spins."),
    _s(3, "krr_rbf_eff", "kernel_gp", "eff_spin", lambda: M._krr(1e-2),
       "Kernel ridge (RBF)."),
    _s(4, "svr_rbf_pn", "kernel_gp", "pn_products", lambda: M._svr(20.0),
       "Support-vector regression on PN-product features."),
    _s(5, "gpr_rbf_pn_logt", "kernel_gp", "pn_products",
       lambda: M.LogTarget(M._gpr("rbf")),
       "GPR on log-target (kick distribution is heavy-tailed)."),
    # ---- symbolic (mandatory) ----
    _s(6, "pysr_vf_raw", "symbolic", "raw_7d", lambda: M.SymbolicScalar("pysr"),
       "PySR on vf, raw params."),
    _s(7, "gplearn_vf_raw", "symbolic", "raw_7d", lambda: M.SymbolicScalar("gplearn"),
       "gplearn SymbolicRegressor on vf, raw params."),
    _s(8, "pysr_vf_pn", "symbolic", "pn_products", lambda: M.SymbolicScalar("pysr"),
       "PySR on vf with PN-product reparam (second reparam)."),
    _s(9, "gplearn_vf_eff", "symbolic", "eff_spin", lambda: M.SymbolicScalar("gplearn"),
       "gplearn on vf, effective spins."),
    # ---- interpolation ----
    _s(10, "rbf_thinplate_eff", "interpolation", "eff_spin",
       lambda: M._rbf_interp("thin_plate_spline", 1e-3),
       "Thin-plate-spline RBF interpolation."),
    _s(11, "rbf_multiquadric_pn", "interpolation", "pn_products",
       lambda: M._rbf_interp("multiquadric", 1e-3, epsilon=1.0),
       "Multiquadric RBF interpolation."),
    _s(12, "knn_raw", "interpolation", "raw_7d", lambda: M._knn(8),
       "Distance-weighted k-nearest-neighbour."),
    _s(13, "rbf_linear_spherical", "interpolation", "spherical",
       lambda: M._rbf_interp("linear", 1e-2),
       "Linear-kernel RBF on spherical spins."),
    # ---- machine learning ----
    _s(14, "mlp_eff", "ml", "eff_spin", lambda: M._mlp((128, 128)),
       "MLP (128x128, scaled target)."),
    _s(15, "mlp_deep_pn", "ml", "pn_products", lambda: M._mlp((256, 256, 128)),
       "Deeper MLP on PN products."),
    _s(16, "rf_raw", "ml", "raw_7d", lambda: M._rf(500),
       "Random forest, raw params."),
    _s(17, "extratrees_eff", "ml", "eff_spin", lambda: M._extratrees(600),
       "Extremely randomised trees."),
    _s(18, "xgboost_pn", "ml", "pn_products", lambda: M._xgb(),
       "XGBoost gradient boosting."),
    _s(19, "lightgbm_eff", "ml", "eff_spin", lambda: M._lgbm(),
       "LightGBM gradient boosting."),
    _s(20, "poly3_eff", "ml", "eff_spin", lambda: M._poly(3, 1.0),
       "Cubic polynomial ridge regression."),
    _s(21, "poly5_pn", "ml", "pn_products", lambda: M._poly(5, 5.0),
       "Quintic polynomial ridge."),
    _s(22, "xgboost_logt_massdiff", "ml", "massdiff_antisym",
       lambda: M.LogTarget(M._xgb()),
       "XGBoost on log-target, mass-difference+antisymmetric reparam."),
    # ---- auxiliary symbolic for Mf and chif (required: separately) ----
    _s(23, "pysr_Mf_eff", "symbolic", "eff_spin", lambda: M.SymbolicScalar("pysr"),
       "PySR on remnant mass Mf (auxiliary).", target="Mf"),
    _s(24, "pysr_chif_eff", "symbolic", "eff_spin", lambda: M.SymbolicScalar("pysr"),
       "PySR on remnant spin chif (auxiliary).", target="chif_mag"),
]

_BY = {s["number"]: s for s in SPECS}


def get_spec(n):
    return _BY[n]
