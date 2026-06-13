"""Data + representation for the Dynamics Bench (opus48, original work).

Target: PN frequency parameter x(t). Loss = pointwise RMS relative error.
x(t) = smooth inspiral growth modulated by eccentric oscillations.
Representation: normalise time to tau in [0,1] (eval grid is given, so t0/tend
are known), resample log(x)(tau) onto a common grid, SVD, regress coeffs.
"""
import sys
from pathlib import Path
import numpy as np
import h5py
from h5py import h5s, h5t

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"; CACHE.mkdir(exist_ok=True)
DATA = REPO / "datasets" / "dynamics"
N_TAU = 1500
TAU = np.linspace(0.0, 1.0, N_TAU)
PARAM_KEYS = ["q", "chi1z", "chi2z", "e0", "zeta0", "omega0"]


def _rd(d):
    sp = d.id.get_space(); sh = sp.get_simple_extent_dims()
    o = np.empty(sh, np.float64); d.id.read(h5s.ALL, h5s.ALL, o, h5t.NATIVE_DOUBLE)
    return o


def _attr(g, k):
    return float(np.array(g.attrs[k]))


def load(split):
    pc = CACHE / f"{split}_P.npy"
    if pc.exists():
        P = np.load(pc)
        waves = []
        i = 0
        while (CACHE / f"{split}_t_{i:04d}.npy").exists():
            waves.append((np.load(CACHE / f"{split}_t_{i:04d}.npy"),
                          np.load(CACHE / f"{split}_x_{i:04d}.npy")))
            i += 1
        if len(waves) == len(P):
            return P, waves
    with h5py.File(DATA / f"dynamics_{split}.h5", "r") as f:
        sims = sorted(k for k in f.keys() if k.startswith("sim"))
        P = np.array([[_attr(f[s], k) for k in PARAM_KEYS] for s in sims])
        waves = [(_rd(f[s]["t"]), _rd(f[s]["x"])) for s in sims]
    np.save(pc, P)
    for i, (t, x) in enumerate(waves):
        np.save(CACHE / f"{split}_t_{i:04d}.npy", t)
        np.save(CACHE / f"{split}_x_{i:04d}.npy", x)
    return P, waves


def to_tau(t, x, tau=TAU):
    t0, tend = t[0], t[-1]
    tau_w = (t - t0) / (tend - t0)
    logx = np.log(np.clip(x, 1e-30, None))
    return np.interp(tau, tau_w, logx), float(t0), float(tend)


def from_tau(logx_tau, t_eval, t0, tend, tau=TAU):
    tau_e = (t_eval - t0) / (tend - t0)
    return np.exp(np.interp(tau_e, tau, logx_tau))


def build_matrix(split):
    fL = CACHE / f"{split}_logx.npy"; fE = CACHE / f"{split}_ends.npy"
    if fL.exists() and fE.exists():
        return np.load(fL), np.load(fE)
    P, waves = load(split)
    L = np.zeros((len(waves), N_TAU)); E = np.zeros((len(waves), 2))
    for i, (t, x) in enumerate(waves):
        lx, t0, tend = to_tau(t, x)
        L[i] = lx; E[i] = [t0, tend]
    np.save(fL, L); np.save(fE, E)
    return L, E


def reparam(P, kind):
    q = P[:, 0]; c1z = P[:, 1]; c2z = P[:, 2]; e0 = P[:, 3]; z0 = P[:, 4]; om = P[:, 5]
    eta = q / (1 + q) ** 2
    m1 = q / (1 + q); m2 = 1.0 / (1 + q)
    chi_eff = m1 * c1z + m2 * c2z
    chi_a = 0.5 * (c1z - c2z)
    if kind == "raw_6d":
        return np.column_stack([q, c1z, c2z, e0, z0, om])
    if kind == "eff_loge":
        return np.column_stack([eta, chi_eff, chi_a, np.log(e0), z0, om])
    if kind == "trig_anom":
        return np.column_stack([eta, chi_eff, chi_a, e0, np.cos(z0), np.sin(z0), om])
    if kind == "log_freq":
        return np.column_stack([eta, chi_eff, chi_a, e0, z0, np.log(om)])
    if kind == "full_transform":
        return np.column_stack([eta, chi_eff, chi_a, np.log(e0),
                                np.cos(z0), np.sin(z0), np.log(om)])
    raise ValueError(kind)


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0); std = X.std(0); std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def score(pred_waves, true_waves):
    from gwbenchmarks.metrics import rms_relative_error
    return np.array([rms_relative_error(p[1], t[1])
                     for p, t in zip(pred_waves, true_waves)])
