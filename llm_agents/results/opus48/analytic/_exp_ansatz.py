"""Probe a closed-form IMR ansatz: fit per-waveform, measure FD mismatch."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import least_squares
import adata as A
from gwbenchmarks.metrics import mean_fd_mismatch

data = A.load("training")


def model_h(t, p, eta):
    """Closed-form h22: TaylorT3-like inspiral phase + tanh-blended QNM,
    phenomenological amplitude (inspiral power-law rise * ringdown decay)."""
    tc, om_qnm, tau_rd, Apk, wA, pscale, aw = p
    eps = 1e-6
    # retarded inspiral variable
    Th = np.clip(eta * (tc - t) / 5.0, eps, None)        # TaylorT3 theta
    phi_insp = -(2.0 / eta) * Th ** (0.625) * pscale     # leading PN phase (x2 for GW)
    # blend inspiral phase into a linear ringdown phase at om_qnm
    bl = 0.5 * (1 + np.tanh(t / aw))                     # 0 (early) -> 1 (late)
    phi = (1 - bl) * phi_insp + bl * (om_qnm * t)
    # amplitude: smooth peak at t~tc with asymmetric widths (sech-like)
    x = (t - 0.0)
    amp_rise = (1.0 + (np.clip(-x, 0, None) / wA) ** 2) ** (-0.5)   # inspiral side
    amp_rd = np.exp(-np.clip(x, 0, None) / tau_rd)                  # ringdown side
    Amod = Apk * np.where(x <= 0, amp_rise, amp_rd)
    return Amod * np.exp(-1j * phi)


def fit_one(d):
    t = d["t"]; eta = d["eta"]
    # crude initial guesses
    p0 = np.array([5.0, 0.55, 12.0, d["A"][d["ipk"]], 60.0, 1.0, 40.0])
    # fit amplitude and phase to the data directly (not mismatch) for speed
    A_t = d["A"]; phi_t = d["phi"]
    def res(p):
        h = model_h(t, p, eta)
        # match amplitude (log) and phase shape; subsample
        s = slice(0, len(t), 25)
        am = np.abs(h)[s]; ph = np.unwrap(np.angle(h))[s]
        ra = (np.log(am + 1e-12) - np.log(A_t[s] + 1e-12))
        rp = 0.01 * (ph - (-phi_t[s]))   # model uses exp(-i phi) ~ -unwrap
        return np.concatenate([ra, rp])
    sol = least_squares(res, p0, max_nfev=400)
    return sol.x


for d in data[:4]:
    p = fit_one(d)
    h = model_h(d["t"], p, d["eta"])
    mm = mean_fd_mismatch(h, d["h"], A.DT)
    print(f"q={d['q']:.2f} mismatch={mm:.3e}  params={np.round(p,2)}")
