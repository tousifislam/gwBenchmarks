import warnings; warnings.filterwarnings("ignore")
import numpy as np, common as C, surrogate as S
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

Ptr = C.load_split("training")[0]; Pva = C.load_split("validation")[0]
_, wtr = C.load_split("training"); _, wva = C.load_split("validation")
logA_tr, phi_tr, _ = C.build_tau_matrices("training")
tau = C.TAU; dtau = tau[1] - tau[0]
Xtr = C.reparam(Ptr, "raw_omega"); Xva = C.reparam(Pva, "raw_omega")
m, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1
Xtr_s = (Xtr - m) / s; Xva_s = (Xva - m) / s
vi = np.arange(0, 250, 5)

# frequency = dphi/dtau
omega_tr = np.gradient(phi_tr, dtau, axis=1)

def mkgpr():
    k = ConstantKernel(1.0) * Matern(2.0, nu=2.5) + WhiteKernel(1e-5)
    return GaussianProcessRegressor(kernel=k, alpha=1e-7, normalize_y=True, n_restarts_optimizer=0)

def run(target_mat, rank, mode, reg="rbf"):
    bA = S.SVDBasis(logA_tr, 30); cA = bA.project(logA_tr)
    bT = S.SVDBasis(target_mat, rank); cT = bT.project(target_mat)
    if reg == "rbf":
        rgT = RBFInterpolator(Xtr_s, cT, kernel="thin_plate_spline", smoothing=1e-3)
        rgA = RBFInterpolator(Xtr_s, cA, kernel="thin_plate_spline", smoothing=1e-3)
        pcT = rgT(Xva_s); pcA = rgA(Xva_s)
    else:
        rgT = mkgpr().fit(Xtr_s, cT); rgA = mkgpr().fit(Xtr_s, cA)
        pcT = rgT.predict(Xva_s); pcA = rgA.predict(Xva_s)
    pred, true = [], []
    for i in vi:
        t, h = wva[i]
        la = bA.reconstruct(pcA[i])[0]
        tt = bT.reconstruct(pcT[i])[0]
        if mode == "freq":
            ph = np.concatenate([[0], np.cumsum(0.5 * (tt[1:] + tt[:-1]) * dtau)])
            ph = ph - ph[-1]   # phi=0 at peak (tau=1 end)
        else:
            ph = tt
        hp = C.tau_to_wave(la, ph, t, t[0], t[-1])
        pred.append((t, hp)); true.append((t, h))
    mm = C.score_waveforms(pred, true)
    return np.median(mm), mm.mean(), mm.max()

print("phase  SVD+RBF rank30:", [f"{x:.3e}" for x in run(phi_tr, 30, "phase", "rbf")])
print("phase  SVD+GPR rank30:", [f"{x:.3e}" for x in run(phi_tr, 30, "phase", "gpr")])
print("freq   SVD+RBF rank30:", [f"{x:.3e}" for x in run(omega_tr, 30, "freq", "rbf")])
print("freq   SVD+GPR rank30:", [f"{x:.3e}" for x in run(omega_tr, 30, "freq", "gpr")])
print("freq   SVD+GPR rank20:", [f"{x:.3e}" for x in run(omega_tr, 20, "freq", "gpr")])
