"""Test phase-detrending strategies to make the regression tractable.

For each strategy: detrend phase per-waveform, SVD the residual, fit RBF coeff
regression train->val, reconstruct (residual phase only; match optimises the
remaining constant+time gauge), score on a val subset.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, common as C, surrogate as S
from scipy.interpolate import RBFInterpolator

Ptr = C.load_split("training")[0]; Pva = C.load_split("validation")[0]
_, wtr = C.load_split("training"); _, wva = C.load_split("validation")
logA_tr, phi_tr, _ = C.build_tau_matrices("training")
logA_va, phi_va, _ = C.build_tau_matrices("validation")
tau = C.TAU
Xtr = C.reparam(Ptr, "raw_omega"); Xva = C.reparam(Pva, "raw_omega")
m, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1
Xtr_s = (Xtr - m) / s; Xva_s = (Xva - m) / s
vi = np.arange(0, 250, 7)


def detrend(phi_mat, deg):
    """Subtract per-row best-fit polynomial in tau of given degree."""
    if deg < 0:
        return phi_mat.copy(), None
    V = np.vander(tau, deg + 1)            # (N, deg+1)
    coef = np.linalg.lstsq(V, phi_mat.T, rcond=None)[0]   # (deg+1, n)
    trend = (V @ coef).T
    return phi_mat - trend, coef


def score(phi_res_tr, phi_res_va_true_unused, rankP):
    bP = S.SVDBasis(phi_res_tr, rankP)
    bA = S.SVDBasis(logA_tr, 30)
    cP = bP.project(phi_res_tr); cA = bA.project(logA_tr)
    rgP = RBFInterpolator(Xtr_s, cP, kernel="thin_plate_spline", smoothing=1e-3)
    rgA = RBFInterpolator(Xtr_s, cA, kernel="thin_plate_spline", smoothing=1e-3)
    pcP = rgP(Xva_s); pcA = rgA(Xva_s)
    pred, true = [], []
    for k, i in enumerate(vi):
        t, h = wva[i]
        la = bA.reconstruct(pcA[i])[0]
        ph = bP.reconstruct(pcP[i])[0]
        hp = C.tau_to_wave(la, ph, t, t[0], t[-1])
        pred.append((t, hp)); true.append((t, h))
    mm = C.score_waveforms(pred, true)
    return np.median(mm), mm.mean(), mm.max()


for deg in [-1, 1, 2, 3, 4]:
    phr, _ = detrend(phi_tr, deg)
    med, mean, mx = score(phr, None, 30)
    print(f"detrend deg={deg:2d}: val median={med:.3e} mean={mean:.3e} max={mx:.3e}")
