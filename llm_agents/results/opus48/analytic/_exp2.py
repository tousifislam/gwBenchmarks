"""Integrable closed-form frequency -> exact closed-form phase. Fit to data."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import least_squares, nnls
import adata as A
from gwbenchmarks.metrics import mean_fd_mismatch

data = A.load("training")


def phase_model(t, tc, tm, wr, b):
    """phi(t) = b0*t - b1*(8/5)(tc-t)^(5/8) + b2*wr*log(cosh((t-tm)/wr)) + b3.
    Exact integral of omega(t)=b0 + b1*(tc-t)^(-3/8) + b2*tanh((t-tm)/wr)."""
    s = np.clip(tc - t, 1e-6, None)
    return (b[0] * t - b[1] * (8.0 / 5.0) * s ** 0.625
            + b[2] * wr * np.log(np.cosh((t - tm) / wr)) + b[3])


def fit_phase(d):
    t = d["t"]; target = d["phi"]               # canonicalised, =0 at peak
    best = None
    for tc in [2.0, 5.0, 10.0]:
        for tm in [-30.0, 0.0, 30.0]:
            for wr in [15.0, 40.0]:
                s = np.clip(tc - t, 1e-6, None)
                B = np.column_stack([t, -(8/5) * s ** 0.625,
                                     wr * np.log(np.cosh((t - tm) / wr)), np.ones_like(t)])
                coef, *_ = np.linalg.lstsq(B, target, rcond=None)
                resid = np.sqrt(np.mean((B @ coef - target) ** 2))
                if best is None or resid < best[0]:
                    best = (resid, tc, tm, wr, coef)
    return best


def amp_model(t, ap):
    Apk, wrise, td, prise = ap
    x = t
    rise = (1.0 + (np.clip(-x, 0, None) / wrise) ** 2) ** (-prise)
    rd = 1.0 / np.cosh(np.clip(x, 0, None) / td)
    return Apk * np.where(x <= 0, rise, rd)


def fit_amp(d):
    t = d["t"]; At = d["A"]
    def res(ap):
        return (np.log(amp_model(t, ap)[::20] + 1e-12) - np.log(At[::20] + 1e-12))
    from scipy.optimize import least_squares
    sol = least_squares(res, [At[d["ipk"]], 80.0, 11.0, 0.5], max_nfev=300)
    return sol.x


for d in data[:5]:
    resid, tc, tm, wr, b = fit_phase(d)
    ap = fit_amp(d)
    phi = phase_model(d["t"], tc, tm, wr, b)
    h = amp_model(d["t"], ap) * np.exp(-1j * phi)
    mm = mean_fd_mismatch(h, d["h"], A.DT)
    print(f"q={d['q']:.2f} phase_resid={resid:.2f}rad  mismatch={mm:.3e}")
