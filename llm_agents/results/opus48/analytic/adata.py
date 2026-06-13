"""Data + closed-form scoring for the Analytic Bench (opus48, original work).

Closed-form h22(t;q) = A(tau;q) * exp(-i*phi(tau;q)), tau = -t (retarded time;
peak at t=0 -> tau=0, inspiral tau>0, ringdown tau<0). All models here are
explicit analytic formulas (no SVD/PCA, no stored bases, no ODE solves).
Loss = mean aLIGO FD mismatch (same metric as the Waveform Bench).
"""
import sys
from pathlib import Path
import numpy as np
import h5py
from h5py import h5s, h5t

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA = REPO / "datasets" / "analytic"
DT = 0.1


def _rd(d):
    sp = d.id.get_space(); sh = sp.get_simple_extent_dims()
    o = np.empty(sh, np.float64); d.id.read(h5s.ALL, h5s.ALL, o, h5t.NATIVE_DOUBLE)
    return o


def load(split):
    """Return list of dicts: q, eta, delta, t, h (complex), A, phi(unwrapped)."""
    out = []
    with h5py.File(DATA / f"analytic_{split}.h5", "r") as f:
        G = f["sims"]
        for k in G.keys():
            w = G[k]
            t = _rd(w["t"]); h = _rd(w["h22_real"]) + 1j * _rd(w["h22_imag"])
            q = float(np.array(w.attrs["q"]))
            A = np.abs(h); phi = np.unwrap(np.angle(h))
            ipk = int(np.argmax(A))
            phi = phi - phi[ipk]
            if phi[0] > 0:        # canonicalise sign (metric uses Re only)
                phi = -phi
            eta = q / (1 + q) ** 2; delta = (q - 1) / (q + 1)
            out.append(dict(q=q, eta=eta, delta=delta, t=t, h=h, A=A, phi=phi, ipk=ipk))
    out.sort(key=lambda d: d["q"])
    return out


def mass_var(d, kind):
    if kind == "q":
        return d["q"]
    if kind == "eta":
        return d["eta"]
    if kind == "delta_m":
        return d["delta"]
    if kind == "sqrt_eta":
        return np.sqrt(d["eta"])
    if kind == "eta_pow15":
        return d["eta"] ** 0.2
    raise ValueError(kind)


def score(pred_h_list, data):
    """Mean FD mismatch per waveform. pred_h_list aligned with data order."""
    from gwbenchmarks.metrics import mean_fd_mismatch
    return np.array([mean_fd_mismatch(ph, d["h"], DT) for ph, d in zip(pred_h_list, data)])
