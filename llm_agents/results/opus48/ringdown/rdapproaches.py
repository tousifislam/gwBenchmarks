"""Ringdown QNM approach registry (opus48, original work).

Categories: analytical (poly/Chebyshev/rational), symbolic (PySR+gplearn),
interpolation (spline/RBF), ML (GPR/MLP/RF/GBM).
Reparams: raw_a, log_compact, sqrt_irr, compact, cheby_map.
"""
import rdmodels as M


def _s(n, name, cat, rep, make, notes=""):
    return dict(number=n, name=name, category=cat, reparam=rep, make=make, notes=notes)


SPECS = [
    # analytical / classical
    _s(1, "poly10_raw", "analytical", "raw_a", lambda: M.PolyFit(10, "raw_a"),
       "Degree-10 polynomial in raw a (baseline)."),
    _s(2, "poly15_logcompact", "analytical", "log_compact", lambda: M.PolyFit(15, "log_compact"),
       "Degree-15 polynomial in -log(1-a): resolves the a->1 region."),
    _s(3, "cheby14_raw", "analytical", "raw_a", lambda: M.ChebyFit(14, "raw_a"),
       "Chebyshev expansion (avoids Runge oscillation)."),
    _s(4, "cheby20_chebymap", "analytical", "cheby_map", lambda: M.ChebyFit(20, "cheby_map"),
       "High-order Chebyshev on [-1,1]-mapped spin."),
    _s(5, "rational66_raw", "analytical", "raw_a", lambda: M.RationalFit(6, 6, "raw_a"),
       "Rational [6,6] Pade with rel-error LM refinement."),
    _s(6, "rational88_logcompact", "analytical", "log_compact", lambda: M.RationalFit(8, 8, "log_compact"),
       "Rational [8,8] on log-compactified spin."),
    _s(7, "poly12_sqrtirr", "analytical", "sqrt_irr", lambda: M.PolyFit(12, "sqrt_irr"),
       "Degree-12 polynomial in sqrt(1-a^2) (third reparam)."),
    # symbolic (mandatory)
    _s(8, "pysr_raw", "symbolic", "raw_a", lambda: M.SymbolicRD("pysr", "raw_a"),
       "PySR on omega_R and omega_I (raw a)."),
    _s(9, "gplearn_raw", "symbolic", "raw_a", lambda: M.SymbolicRD("gplearn", "raw_a"),
       "gplearn SymbolicRegressor on omega_R, omega_I."),
    _s(10, "pysr_logcompact", "symbolic", "log_compact", lambda: M.SymbolicRD("pysr", "log_compact"),
       "PySR, second reparam (log-compactified)."),
    _s(11, "gplearn_sqrtirr", "symbolic", "sqrt_irr", lambda: M.SymbolicRD("gplearn", "sqrt_irr"),
       "gplearn on sqrt(1-a^2)."),
    # interpolation
    _s(12, "cubic_spline_raw", "interpolation", "raw_a", lambda: M.SplineFit(3, 0.0, "raw_a"),
       "Cubic interpolating spline."),
    _s(13, "spline_smooth_logcompact", "interpolation", "log_compact",
       lambda: M.SplineFit(3, 1e-10, "log_compact"), "Smoothing spline on log-compact."),
    _s(14, "rbf_thinplate_raw", "interpolation", "raw_a",
       lambda: M.RBFInterp("thin_plate_spline", 0.0, "raw_a"), "Thin-plate-spline RBF."),
    _s(15, "rbf_multiquadric_raw", "interpolation", "raw_a",
       lambda: M.RBFInterp("multiquadric", 0.0, "raw_a", {"epsilon": 0.3}), "Multiquadric RBF."),
    _s(16, "rbf_cubic_logcompact", "interpolation", "log_compact",
       lambda: M.RBFInterp("cubic", 0.0, "log_compact"), "Cubic RBF on log-compact."),
    # ML
    _s(17, "gpr_rbf_raw", "ml", "raw_a", lambda: M.SkWrap(M._gpr, "raw_a"),
       "Gaussian process (RBF)."),
    _s(18, "gpr_rbf_logcompact", "ml", "log_compact", lambda: M.SkWrap(M._gpr, "log_compact"),
       "GPR on log-compactified spin."),
    _s(19, "mlp_raw", "ml", "raw_a", lambda: M.SkWrap(M._mlp, "raw_a"),
       "MLP (tanh, 128x128)."),
    _s(20, "rf_raw", "ml", "raw_a", lambda: M.SkWrap(M._rf, "raw_a"),
       "Random forest."),
    _s(21, "histgbm_raw", "ml", "raw_a", lambda: M.SkWrap(M._gbr, "raw_a"),
       "Hist gradient boosting."),
    _s(22, "krr_rbf_logcompact", "ml", "log_compact", lambda: M.SkWrap(M._krr, "log_compact"),
       "Kernel ridge (RBF) on log-compact."),
    # reasoned variation
    _s(23, "rational1010_logcompact", "analytical", "log_compact",
       lambda: M.RationalFit(10, 10, "log_compact"),
       "Higher-order rational [10,10]: reasoned push on a->1 accuracy."),
]

_BY = {s["number"]: s for s in SPECS}


def get_spec(n):
    return _BY[n]
