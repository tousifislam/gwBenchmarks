"""Sanity checks for candidate_waveform.h_of_f (uses pycbc match, standard pkg)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from candidate_waveform import h_of_f

f_low, f_high, df = 15.0, 990.0, 0.125
f = np.arange(0.0, 2048.0 + df, df)

# 1. structural checks
h = h_of_f(f, Mc=30.0, eta=0.24, dL=400.0, lambda_RG=1.0)
print("shape ok:", h.shape == f.shape, "| complex:", np.iscomplexobj(h))
print("finite:", np.all(np.isfinite(h)))
print("zero below f_low:", np.all(h[f < 20.0] == 0))
band = h != 0
print("band f-range:", round(f[band].min(), 1), "to", round(f[band].max(), 1), "Hz")


def match(h1, h2):
    from pycbc.types import FrequencySeries
    from pycbc.filter import match as pmatch
    from pycbc.psd import aLIGOZeroDetHighPower
    a = FrequencySeries(h1.astype(complex), delta_f=df)
    b = FrequencySeries(h2.astype(complex), delta_f=df)
    n = len(a); psd = aLIGOZeroDetHighPower(n, df, f_low)
    m, _ = pmatch(a, b, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
    return float(m)


# 2. lambda_RG sensitivity (benchmark varies it; must change the waveform)
for cases in [(30.0, 0.24), (15.0, 0.16), (60.0, 0.10)]:
    Mc, eta = cases
    h1 = h_of_f(f, Mc, eta, 400.0, lambda_RG=1.0)
    h2 = h_of_f(f, Mc, eta, 400.0, lambda_RG=2.0)
    h0 = h_of_f(f, Mc, eta, 400.0, lambda_RG=0.0)
    print(f"Mc={Mc} eta={eta}: mismatch(lam1,lam2)={1-match(h1,h2):.2e} "
          f"mismatch(lam1,lam0)={1-match(h1,h0):.2e}")

# 3. phase reasonableness vs a standard TaylorF2 (should be a decent inspiral match)
try:
    from pycbc.waveform import get_fd_waveform
    Mc, eta = 30.0, 0.24
    M = Mc / eta ** 0.6
    m1 = m2 = M / 2.0  # equalish; just for a PN phase reference
    hp, _ = get_fd_waveform(approximant="TaylorF2", mass1=m1, mass2=m2,
                            delta_f=df, f_lower=f_low, f_final=f_high)
    hp = np.asarray(hp.data)
    L = min(len(hp), len(f))
    hc = h_of_f(f[:L], Mc, eta, 400.0, lambda_RG=1.0)
    print(f"match vs TaylorF2 (sanity): {match(hc, hp[:L]):.4f}")
except Exception as e:
    print("TaylorF2 check skipped:", type(e).__name__, str(e)[:60])
