import warnings; warnings.filterwarnings("ignore")
import numpy as np, common as C, surrogate as S
from scipy.interpolate import RBFInterpolator

Ptr = C.load_split("training")[0]; Pva = C.load_split("validation")[0]
_, wtr = C.load_split("training"); _, wva = C.load_split("validation")
logA_tr, phi_tr, ends_tr = C.build_tau_matrices("training")
logA_va, phi_va, ends_va = C.build_tau_matrices("validation")
tau = C.TAU
Xtr = C.reparam(Ptr, "raw_omega"); Xva = C.reparam(Pva, "raw_omega")
mn, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
Xtr_s = (Xtr - mn) / sd; Xva_s = (Xva - mn) / sd


def eta_of(P):
    q = P[:, 0]; return q / (1 + q) ** 2


def pn_phase(eta, t0, tend, tref_pad=10.0):
    """0PN GW(2,2) phase on the tau grid for one waveform, =0 at peak (tau=1)."""
    t = t0 + tau * (tend - t0)
    tc = tend + tref_pad           # coalescence just past the data end
    Theta = np.clip(eta * (tc - t) / 5.0, 1e-12, None)
    phi_orb = -(1.0 / eta) * Theta ** (0.625)
    phi22 = 2.0 * phi_orb
    return phi22 - phi22[-1]        # anchor =0 at tau=1 (peak)


def baseline_matrix(P, ends):
    et = eta_of(P)
    return np.array([pn_phase(et[i], ends[i, 0], ends[i, 1]) for i in range(len(P))])


B_tr = baseline_matrix(Ptr, ends_tr)
B_va = baseline_matrix(Pva, ends_va)

# how well does 0PN baseline alone match the true phase?
res_tr = phi_tr - B_tr
print("phase std (raw)      :", phi_tr.std(), " range coeff0~", np.abs(phi_tr[:,0]).max())
print("residual std (NR-0PN):", res_tr.std(), " max|res|", np.abs(res_tr).max())

vi = np.arange(0, 250, 5)
bA = S.SVDBasis(logA_tr, 30); cA = bA.project(logA_tr)
for rank in [20, 30, 40]:
    bR = S.SVDBasis(res_tr, rank); cR = bR.project(res_tr)
    rgR = RBFInterpolator(Xtr_s, cR, kernel="thin_plate_spline", smoothing=1e-3)
    rgA = RBFInterpolator(Xtr_s, cA, kernel="thin_plate_spline", smoothing=1e-3)
    pcR = rgR(Xva_s); pcA = rgA(Xva_s)
    pred, true = [], []
    for i in vi:
        t, h = wva[i]
        la = bA.reconstruct(pcA[i])[0]
        ph = B_va[i] + bR.reconstruct(pcR[i])[0]
        hp = C.tau_to_wave(la, ph, t, t[0], t[-1])
        pred.append((t, hp)); true.append((t, h))
    mm = C.score_waveforms(pred, true)
    print(f"PN+residual rank={rank}: val median={np.median(mm):.3e} mean={mm.mean():.3e} max={mm.max():.3e}")
