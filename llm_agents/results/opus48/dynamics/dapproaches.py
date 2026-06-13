"""Dynamics x(t) approach registry (opus48, original work).

Categories: decomposition (SVD/EIM + GPR/poly/NN/RF), symbolic (PySR+gplearn),
interp/kernel (RBF/KRR/kNN), ML (MLP/RF/GBM/xgboost).
Reparams: raw_6d, eff_loge, trig_anom, log_freq, full_transform.
"""
import dmodels as M


def _s(n, name, cat, rep, rank, make, basis="svd", notes=""):
    return dict(number=n, name=name, category=cat, reparam=rep, rank=rank,
                make=make, basis=basis, notes=notes)


SPECS = [
    # decomposition
    _s(1, "svd_gpr_raw", "decomposition", "raw_6d", 25, lambda: M._gpr("rbf"),
       notes="Baseline SVD + GPR (RBF) on raw params."),
    _s(2, "svd_gpr_matern_full", "decomposition", "full_transform", 25, lambda: M._gpr("matern"),
       notes="SVD + GPR Matern on fully-transformed reparam."),
    _s(3, "svd_poly3_eff", "decomposition", "eff_loge", 20, lambda: M._poly(3, 1.0),
       notes="SVD + cubic polynomial ridge."),
    _s(4, "svd_mlp_full", "decomposition", "full_transform", 25, lambda: M._mlp((128, 128)),
       notes="SVD + MLP (scaled targets)."),
    _s(5, "eim_gpr_full", "decomposition", "full_transform", 25, lambda: M._gpr("matern"),
       basis="eim", notes="EIM nodes + GPR on node values."),
    # symbolic
    _s(6, "svd_pysr_full", "symbolic", "full_transform", 12, lambda: M.SymbolicCoeff("pysr", 3),
       notes="PySR on leading SVD coeffs (full transform)."),
    _s(7, "svd_gplearn_eff", "symbolic", "eff_loge", 12, lambda: M.SymbolicCoeff("gplearn", 3),
       notes="gplearn on leading SVD coeffs (eff+log e0)."),
    _s(8, "svd_pysr_trig", "symbolic", "trig_anom", 12, lambda: M.SymbolicCoeff("pysr", 3),
       notes="PySR, second reparam (trig anomaly)."),
    # interp / kernel
    _s(9, "rbf_thinplate_full", "interp_kernel", "full_transform", 25,
       lambda: M._rbf("thin_plate_spline", 1e-3), notes="Thin-plate RBF interpolation."),
    _s(10, "rbf_multiquadric_eff", "interp_kernel", "eff_loge", 25,
       lambda: M._rbf("multiquadric", 1e-3, epsilon=1.0), notes="Multiquadric RBF."),
    _s(11, "krr_full", "interp_kernel", "full_transform", 25, lambda: M._krr(1e-2),
       notes="Kernel ridge (RBF)."),
    _s(12, "knn_raw", "interp_kernel", "raw_6d", 25, lambda: M._knn(8),
       notes="Distance-weighted kNN."),
    # ML
    _s(13, "mlp_deep_full", "ml", "full_transform", 25, lambda: M._mlp((256, 256, 128)),
       notes="Deeper MLP."),
    _s(14, "rf_eff", "ml", "eff_loge", 25, lambda: M._rf(400), notes="Random forest."),
    _s(15, "extratrees_full", "ml", "full_transform", 25, lambda: M._extratrees(500),
       notes="Extremely randomised trees."),
    _s(16, "xgboost_eff", "ml", "eff_loge", 25, lambda: M._xgb(), notes="XGBoost (per coeff)."),
    _s(17, "mlp_trig", "ml", "trig_anom", 25, lambda: M._mlp((128, 128)),
       notes="MLP on trig-anomaly reparam (third reparam)."),
    # reasoned variations
    _s(18, "svd_gpr_lowrank_full", "decomposition", "full_transform", 15, lambda: M._gpr("rbf"),
       notes="Lower rank: drop noisy eccentric-oscillation modes that do not regress."),
    _s(19, "svd_gpr_highrank_full", "decomposition", "full_transform", 60, lambda: M._gpr("matern"),
       notes="Higher rank: attempt to capture eccentric oscillations."),
    _s(20, "rbf_thinplate_logfreq", "interp_kernel", "log_freq", 22,
       lambda: M._rbf("thin_plate_spline", 1e-3), notes="RBF on log-frequency reparam."),
    _s(21, "svd_poly4_full", "decomposition", "full_transform", 20, lambda: M._poly(4, 5.0),
       notes="Quartic polynomial ridge."),
    _s(22, "gpr_rbf_tuned_full", "decomposition", "full_transform", 22,
       lambda: M._gpr("rbf", length=6.0, alpha=1e-5), notes="Reasoned GPR retune (broad kernel)."),
]

_BY = {s["number"]: s for s in SPECS}


def get_spec(n):
    return _BY[n]
