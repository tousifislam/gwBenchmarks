"""Shared infrastructure for the Waveform Bench (opus48 agent).

Original implementation. Builds a duration-normalised amplitude/phase
representation of the co-precessing h22 mode so that variable-length
NR waveforms live on a single common grid, suitable for SVD/decomposition
and parameter regression.

Key design decisions (all my own):
  * Decompose h = A * exp(i*phi), with A=|h|, phi=unwrap(angle(h)).
  * Reparameterise time as tau = (t - t0)/(t_end - t0) in [0,1]; this folds
    the (large) duration variation into a fixed [0,1] grid. The evaluation
    time grid is *given* at predict time, so t0/t_end are known and never
    need to be modelled.
  * Model log(A) and phi(tau) on a common tau grid via SVD/other bases.
  * Reconstruct: predict coeffs -> logA(tau), phi(tau) on common grid ->
    interpolate to the requested tau grid -> h = exp(logA) * exp(i*phi).

Environment: run with envs/gwbench/bin/python (numpy 2.x, working pycbc).
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Repo root so we can import gwbenchmarks (read-only).
REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"
CACHE.mkdir(exist_ok=True)

DATA = REPO / "datasets" / "waveform"
DT_GEOMETRIC = 0.1                 # uniform, in units of M
N_TAU = 2000                       # common normalised-time grid resolution
TAU = np.linspace(0.0, 1.0, N_TAU)

# --------------------------------------------------------------------------
# Long-double safe HDF5 reading (datasets stored as float80/96).
# --------------------------------------------------------------------------
def _read_ld(dset):
    """Read an HDF5 dataset stored as long double into float64."""
    from h5py import h5s, h5t
    sp = dset.id.get_space()
    shape = sp.get_simple_extent_dims()
    out = np.empty(shape, dtype=np.float64)
    dset.id.read(h5s.ALL, h5s.ALL, out, h5t.NATIVE_DOUBLE)
    return out


# --------------------------------------------------------------------------
# Raw dataset loading (cached to .npy in our own _cache dir).
# --------------------------------------------------------------------------
PARAM_KEYS = ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z",
              "chi_eff", "chi_p", "omega0"]


def _load_raw(split):
    """Return dict with params array (n, 10) and list of (t, h_complex)."""
    import h5py
    fn = DATA / f"waveform_{split}.h5"
    with h5py.File(fn, "r") as f:
        md = f["metadata"]
        params = {k: _read_ld(md[k]) for k in PARAM_KEYS}
        sims = sorted(k for k in f.keys() if k.startswith("sim"))
        waves = []
        for k in sims:
            g = f[k]
            t = _read_ld(g["t"])
            h = _read_ld(g["h22_real"]) + 1j * _read_ld(g["h22_imag"])
            waves.append((t, h))
    P = np.column_stack([params[k] for k in PARAM_KEYS])
    return P, waves


def load_split(split):
    """Cached raw load. Returns (P[n,10], list of (t,h))."""
    pcache = CACHE / f"{split}_params.npy"
    if pcache.exists():
        P = np.load(pcache)
        waves = []
        i = 0
        while (CACHE / f"{split}_t_{i:04d}.npy").exists():
            t = np.load(CACHE / f"{split}_t_{i:04d}.npy")
            h = np.load(CACHE / f"{split}_h_{i:04d}.npy")
            waves.append((t, h))
            i += 1
        if len(waves) == len(P):
            return P, waves
    P, waves = _load_raw(split)
    np.save(pcache, P)
    for i, (t, h) in enumerate(waves):
        np.save(CACHE / f"{split}_t_{i:04d}.npy", t)
        np.save(CACHE / f"{split}_h_{i:04d}.npy", h)
    return P, waves


# --------------------------------------------------------------------------
# Amplitude/phase representation on the common tau grid.
# --------------------------------------------------------------------------
def wave_to_tau(t, h, tau=TAU):
    """Map a single (t,h) onto the common tau grid.

    Returns (logA_tau, phi_tau, t0, tend). phi is unwrapped and shifted so
    phi=0 at the amplitude peak (matches dataset convention phase=0 at t=0).
    """
    A = np.abs(h)
    phi = np.unwrap(np.angle(h))
    # enforce phase=0 at peak (t closest to 0)
    ipk = int(np.argmin(np.abs(t)))
    phi = phi - phi[ipk]
    # Canonicalise the global phase sign. The co-precessing-frame h22 has an
    # arbitrary rotation direction (frequency sign) that flips between
    # simulations and scrambles regression of the total accumulated phase.
    # The benchmark scores only Re(h)=A*cos(phi) and maximises over phase, so
    # phi -> -phi (equivalently h -> conj(h)) is loss-preserving. Make the net
    # accumulated phase always negative (frequency-positive convention).
    if phi[0] > 0:
        phi = -phi
    t0, tend = t[0], t[-1]
    tau_w = (t - t0) / (tend - t0)
    logA = np.log(np.clip(A, 1e-30, None))
    logA_tau = np.interp(tau, tau_w, logA)
    phi_tau = np.interp(tau, tau_w, phi)
    return logA_tau, phi_tau, float(t0), float(tend)


def tau_to_wave(logA_tau, phi_tau, t_eval, t0, tend, tau=TAU):
    """Inverse map: common-grid logA/phi -> complex h on requested t grid."""
    tau_e = (t_eval - t0) / (tend - t0)
    logA = np.interp(tau_e, tau, logA_tau)
    phi = np.interp(tau_e, tau, phi_tau)
    return np.exp(logA) * np.exp(1j * phi)


def build_tau_matrices(split):
    """Cached: stack logA(tau) and phi(tau) for all waveforms in a split."""
    fA = CACHE / f"{split}_logA_tau.npy"
    fP = CACHE / f"{split}_phi_tau.npy"
    fE = CACHE / f"{split}_endpoints.npy"
    if fA.exists() and fP.exists() and fE.exists():
        return np.load(fA), np.load(fP), np.load(fE)
    P, waves = load_split(split)
    logA = np.zeros((len(waves), N_TAU))
    phi = np.zeros((len(waves), N_TAU))
    ends = np.zeros((len(waves), 2))
    for i, (t, h) in enumerate(waves):
        la, ph, t0, tend = wave_to_tau(t, h)
        logA[i] = la
        phi[i] = ph
        ends[i] = [t0, tend]
    np.save(fA, logA)
    np.save(fP, phi)
    np.save(fE, ends)
    return logA, phi, ends


# --------------------------------------------------------------------------
# Reparameterisations (input feature engineering).
# --------------------------------------------------------------------------
def reparam(P, kind):
    """Map raw params (n,10) -> feature matrix for a named reparameterisation.

    Columns of P: q,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z,chi_eff,chi_p,omega0
    """
    q = P[:, 0]
    c1 = P[:, 1:4]
    c2 = P[:, 4:7]
    chi_eff = P[:, 7]
    chi_p = P[:, 8]
    omega0 = P[:, 9]
    eta = q / (1.0 + q) ** 2
    a1 = np.linalg.norm(c1, axis=1)
    a2 = np.linalg.norm(c2, axis=1)
    th1 = np.arctan2(np.linalg.norm(c1[:, :2], axis=1), c1[:, 2])
    th2 = np.arctan2(np.linalg.norm(c2[:, :2], axis=1), c2[:, 2])
    ph1 = np.arctan2(c1[:, 1], c1[:, 0])
    ph2 = np.arctan2(c2[:, 1], c2[:, 0])
    delta = (q - 1.0) / (q + 1.0)

    if kind == "raw_7d":
        return np.column_stack([q, c1[:, 0], c1[:, 1], c1[:, 2],
                                c2[:, 0], c2[:, 1], c2[:, 2]])
    if kind == "raw_omega":
        return np.column_stack([q, c1[:, 0], c1[:, 1], c1[:, 2],
                                c2[:, 0], c2[:, 1], c2[:, 2], omega0])
    if kind == "eff_spin":
        return np.column_stack([eta, chi_eff, chi_p, a1, a2, th1, th2])
    if kind == "eff_spin_omega":
        return np.column_stack([eta, chi_eff, chi_p, a1, a2, th1, th2, omega0])
    if kind == "spherical":
        return np.column_stack([eta, a1, th1, ph1, a2, th2, ph2])
    if kind == "massdiff":
        return np.column_stack([delta, chi_eff, chi_p, a1, a2, ph1, ph2])
    # Newtonian-augmented variants: append physically-motivated transforms of
    # omega0 that encode the inspiral cycle count (total phase ~ omega0^-5/3/eta).
    # Regressors cannot easily build these nonlinear features themselves.
    newt = np.column_stack([omega0 ** (-5.0 / 3.0) / eta,
                            omega0 ** (-8.0 / 3.0) / eta,
                            np.log(omega0)])
    if kind == "raw_newt":
        base = np.column_stack([q, c1[:, 0], c1[:, 1], c1[:, 2],
                                c2[:, 0], c2[:, 1], c2[:, 2], omega0])
        return np.column_stack([base, newt])
    if kind == "eff_newt":
        base = np.column_stack([eta, chi_eff, chi_p, a1, a2, th1, th2, omega0])
        return np.column_stack([base, newt])
    raise ValueError(f"unknown reparam {kind}")


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0)
        std = X.std(0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


# --------------------------------------------------------------------------
# Scoring (uses the official benchmark metric, read-only import).
# --------------------------------------------------------------------------
def score_waveforms(pred_waves, true_waves, masses=None):
    """Mean FD mismatch per waveform. pred/true are lists of (t, h).

    Returns array of per-waveform mean-FD-mismatch (the benchmark loss).
    """
    from gwbenchmarks.metrics import mean_fd_mismatch, FD_MASSES_MSUN
    if masses is None:
        masses = FD_MASSES_MSUN
    out = np.zeros(len(true_waves))
    for i, ((tt, ht), (tp, hp)) in enumerate(zip(true_waves, pred_waves)):
        out[i] = mean_fd_mismatch(hp, ht, DT_GEOMETRIC, masses=masses)
    return out


NR_FLOOR = 1.4e-3


if __name__ == "__main__":
    # quick smoke test
    P, waves = load_split("training")
    print("train:", P.shape, len(waves))
    logA, phi, ends = build_tau_matrices("training")
    print("logA", logA.shape, "phi", phi.shape)
