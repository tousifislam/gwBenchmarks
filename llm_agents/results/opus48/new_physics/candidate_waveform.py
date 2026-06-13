"""Dominant nonspinning (2,2) RG-tail frequency-domain waveform.

Implements hhat_22 = H_eff * T_22 * rho_22^2 * exp(i*delta_22) from the
Section-IV factorized ingredients (arXiv:2602.08833 formula sheet) and maps it
to a stationary-phase-approximation (SPA) frequency-domain strain h(f).

Conventions (documented in notes.md):
  * Orbital angular frequency Omega, GW(2,2) frequency f = Omega/pi, so the
    dimensionless M*Omega = x^(3/2) with x = (pi M_sec f)^(2/3).
  * Time-domain mode h_22(t) ~ A(t) * exp(-i*[2*phi_orb - arg(hhat_22)]); the
    SPA gives Psi(f) = 2*pi*f*t(f) - 2*phi_orb(f) + arg(hhat_22(f)) - pi/4,
    with t(f), phi_orb(f) from cumulative integration of dt/df and dphi/df.
  * t(f) from dt/dx = -M dE/dx / F_22 (radiation reaction); dE/dx by finite
    differences of E_real/M (which carries an explicit log(16x) term).
  * SPA amplitude |h(f)| ∝ (nu*x*|hhat_22|) * sqrt(dt/df). Overall constant /
    distance / constant phase are irrelevant to the normalised mismatch.
"""
import numpy as np
from scipy.special import loggamma
from scipy.integrate import cumulative_trapezoid

MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 1.0292712503e14
GAMMA_E = 0.5772156649015329
PI = np.pi


def _conservative(x, nu):
    """E_real/M and p_phi,circ/(mu M) on array x."""
    log16x = np.log(16.0 * x)
    c0 = 1.0
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu ** 2 / 24.0
    c3 = (-675.0 / 64.0
          + (34445.0 / 576.0 - 205.0 * PI ** 2 / 96.0) * nu
          - 155.0 * nu ** 2 / 96.0
          - 35.0 * nu ** 3 / 5184.0)
    c4 = (-3969.0 / 128.0
          + (-123671.0 / 5760.0 + 9037.0 * PI ** 2 / 1536.0
             + 896.0 * GAMMA_E / 15.0 + 448.0 * log16x / 15.0) * nu
          + (498449.0 / 3456.0 - 3157.0 * PI ** 2 / 576.0) * nu ** 2
          + 301.0 * nu ** 3 / 1728.0
          + 77.0 * nu ** 4 / 31104.0)
    EperM = 1.0 - (nu * x / 2.0) * (c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4)

    d0 = 1.0
    d1 = 3.0 / 2.0 + nu / 6.0
    d2 = 27.0 / 8.0 - 19.0 * nu / 8.0 + nu ** 2 / 24.0
    d3 = (135.0 / 16.0
          + (-6889.0 / 144.0 + 41.0 * PI ** 2 / 24.0) * nu
          + 31.0 * nu ** 2 / 24.0
          + 7.0 * nu ** 3 / 1296.0)
    d4 = (2835.0 / 128.0
          + (98869.0 / 5760.0 - 128.0 * GAMMA_E / 3.0 - 6455.0 * PI ** 2 / 1536.0
             - 64.0 * log16x / 3.0) * nu
          + (356035.0 / 3456.0 - 2255.0 * PI ** 2 / 576.0) * nu ** 2
          - 215.0 * nu ** 3 / 1728.0
          - 55.0 * nu ** 4 / 31104.0)
    pphi = x ** (-0.5) * (d0 + d1 * x + d2 * x ** 2 + d3 * x ** 3 + d4 * x ** 4)
    return EperM, pphi


def _hhat22(x, nu, lambda_RG):
    """Factorized (2,2) correction hhat_22 (complex) and its |.|."""
    EperM, pphi = _conservative(x, nu)
    H_eff = ((EperM ** 2 - 1.0) / (2.0 * nu)) + 1.0
    m = 2.0
    MOmega = x ** 1.5
    khat = EperM * m * MOmega
    J = pphi / EperM ** 2

    gamma22 = (-214.0 * khat ** 2 / 105.0
               + 2.0 * m * J * khat ** 3 / 3.0
               - 3390466.0 * khat ** 4 / 1157625.0
               + 381863.0 * m * J * khat ** 5 / 99225.0)
    ellhat = 2.0 + lambda_RG * gamma22

    logT = (np.log(120.0)
            + (ellhat - 2.0) * np.log(4.0 * np.sqrt(x))
            + 2j * khat * (17.0 / 12.0 - GAMMA_E)
            + loggamma(ellhat - 1.0 - 2j * khat)
            - loggamma(2.0 * ellhat + 2.0)
            + PI * khat
            - 1j * PI * (ellhat - 2.0) / 2.0)
    T22 = np.exp(logT)

    eulerlog2 = GAMMA_E + np.log(4.0 * np.sqrt(x))
    r1 = -43.0 / 42.0 + 55.0 * nu / 84.0
    r2 = -20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu ** 2 / 42336.0
    r3 = (-4296031.0 / 4889808.0
          + (41.0 * PI ** 2 / 192.0 - 48993925.0 / 9779616.0) * nu
          - 6292061.0 * nu ** 2 / 3259872.0
          + 10620745.0 * nu ** 3 / 39118464.0)
    r4 = (9228174993589.0 / 800950550400.0
          + (-2487107795131.0 / 145627372800.0 + 464.0 * eulerlog2 / 35.0
             - 9953.0 * PI ** 2 / 21504.0) * nu
          + (10815863492353.0 / 640760440320.0 - 3485.0 * PI ** 2 / 5376.0) * nu ** 2
          - 2088847783.0 * nu ** 3 / 11650189824.0
          + 70134663541.0 * nu ** 4 / 512608352256.0)
    rho22 = 1.0 + r1 * x + r2 * x ** 2 + r3 * x ** 3 + r4 * x ** 4

    y = (EperM * x ** 1.5) ** (2.0 / 3.0)
    delta22 = (-17.0 * y ** 1.5 / 3.0
               - 24.0 * nu * y ** 2.5
               + (30995.0 * nu / 1134.0 + 962.0 * nu ** 2 / 135.0) * y ** 3.5
               - 4976.0 * PI * nu * y ** 4 / 105.0)

    hhat = H_eff * T22 * rho22 ** 2 * np.exp(1j * delta22)
    F22 = (32.0 / 5.0) * nu ** 2 * x ** 5 * np.abs(hhat) ** 2
    return hhat, F22, EperM


def h_of_f(f, Mc, eta, dL, tc=0.0, phic=0.0, lambda_RG=1.0,
           f_low=20.0, fmax_over_fisco=1.3, sigma_taper_over_fisco=0.01,
           phase_only=False):
    f = np.asarray(f, dtype=float)
    out = np.zeros(f.shape, dtype=complex)

    nu = float(eta)
    M_sec = Mc * MSUN_SEC / nu ** 0.6
    dL_sec = dL * MPC_SEC

    f_isco = 1.0 / (PI * 6.0 ** 1.5 * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    band = (f >= f_low) & (f < f_cut) & (f > 0)
    if not np.any(band):
        return out

    fb = f[band]
    order = np.argsort(fb)
    fs = fb[order]                      # sorted, ascending
    if len(fs) < 2:                     # too few in-band points for the SPA
        return out

    x = (PI * M_sec * fs) ** (2.0 / 3.0)
    hhat, F22, EperM = _hhat22(x, nu, lambda_RG)

    # radiation reaction: dt/dx = -M dE/dx / F ; dE/dx by finite differences
    dEdx = np.gradient(EperM, x)
    dt_dx = -M_sec * dEdx / F22
    dx_df = (2.0 / 3.0) * (PI * M_sec) ** (2.0 / 3.0) * fs ** (-1.0 / 3.0)
    dt_df = dt_dx * dx_df               # seconds per Hz, > 0

    # SPA phase pieces via cumulative integration over the band
    t_f = cumulative_trapezoid(dt_df, fs, initial=0.0)            # t(f) (sec)
    dphiorb_df = PI * fs * dt_df                                  # dphi_orb/df
    phi_orb = cumulative_trapezoid(dphiorb_df, fs, initial=0.0)   # phi_orb(f)

    arg_hhat = np.unwrap(np.angle(hhat))
    Psi = (2.0 * PI * fs * t_f - 2.0 * phi_orb + arg_hhat - PI / 4.0
           + 2.0 * PI * fs * tc + phic)

    W = 1.0 / (1.0 + np.exp((fs - f_isco) / sigma))              # Fermi taper

    if phase_only:
        amp = W
    else:
        amp = (nu * x * np.abs(hhat) * np.sqrt(np.clip(dt_df, 0, None))
               * (M_sec / dL_sec) * W)

    # Engineering FT convention h~(f) = int h(t) exp(-2 pi i f t) dt (matches
    # PyCBC / standard GW data analysis): the SPA strain carries exp(-i*Psi).
    hb_sorted = amp * np.exp(-1j * Psi)
    hb = np.empty_like(hb_sorted)
    hb[order] = hb_sorted               # undo the sort
    out[band] = hb
    return out
