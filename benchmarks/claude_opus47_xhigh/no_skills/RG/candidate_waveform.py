"""Frequency-domain (2,2)-mode RG-tail inspiral waveform per arXiv:2602.08833.

Implements the factorized (2,2) correction
    hhat_22 = H_eff * T_22 * rho_22^2 * exp(i delta_22)
combined with energy balance
    F_22 = (32/5) nu^2 x^5 |hhat_22|^2,  dt/dx = -M dE/dx / F_22
to construct the SPA frequency-domain strain.
"""
import numpy as np
from scipy.special import loggamma
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458
GAMMA_E = 0.5772156649015329


def _eM_and_prime(x, nu):
    """Return (E_real/M, d(E_real/M)/dx) from the conservative-sector polynomial."""
    pi2 = np.pi ** 2
    log16x = np.log(16.0 * x)

    c0 = 1.0
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu ** 2 / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * pi2 / 96.0) * nu
        - 155.0 * nu ** 2 / 96.0
        - 35.0 * nu ** 3 / 5184.0
    )
    c4_nolog = (
        -3969.0 / 128.0
        + (-123671.0 / 5760.0 + 9037.0 * pi2 / 1536.0 + 896.0 * GAMMA_E / 15.0) * nu
        + (498449.0 / 3456.0 - 3157.0 * pi2 / 576.0) * nu ** 2
        + 301.0 * nu ** 3 / 1728.0
        + 77.0 * nu ** 4 / 31104.0
    )
    c4_log_coef = 448.0 * nu / 15.0
    c4 = c4_nolog + c4_log_coef * log16x

    P = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4
    P_prime = (
        c1
        + 2.0 * c2 * x
        + 3.0 * c3 * x ** 2
        + (4.0 * c4 + c4_log_coef) * x ** 3
    )

    eM = 1.0 - (nu * x / 2.0) * P
    eM_prime = -(nu / 2.0) * (P + x * P_prime)
    return eM, eM_prime


def _pphi(x, nu):
    """p_phi,circ / (mu M) from the conservative-sector polynomial."""
    pi2 = np.pi ** 2
    log16x = np.log(16.0 * x)

    d0 = 1.0
    d1 = 3.0 / 2.0 + nu / 6.0
    d2 = 27.0 / 8.0 - 19.0 * nu / 8.0 + nu ** 2 / 24.0
    d3 = (
        135.0 / 16.0
        + (-6889.0 / 144.0 + 41.0 * pi2 / 24.0) * nu
        + 31.0 * nu ** 2 / 24.0
        + 7.0 * nu ** 3 / 1296.0
    )
    d4_nolog = (
        2835.0 / 128.0
        + (98869.0 / 5760.0 - 128.0 * GAMMA_E / 3.0 - 6455.0 * pi2 / 1536.0) * nu
        + (356035.0 / 3456.0 - 2255.0 * pi2 / 576.0) * nu ** 2
        - 215.0 * nu ** 3 / 1728.0
        - 55.0 * nu ** 4 / 31104.0
    )
    d4_log_coef = -64.0 * nu / 3.0
    d4 = d4_nolog + d4_log_coef * log16x

    return x ** (-0.5) * (d0 + d1 * x + d2 * x ** 2 + d3 * x ** 3 + d4 * x ** 4)


def _hhat22(x, nu, lambda_RG):
    """Factorized (2,2) correction: hhat_22 = H_eff T_22 rho_22^2 e^{i delta_22}."""
    eM, _ = _eM_and_prime(x, nu)
    pphi = _pphi(x, nu)

    H_eff = ((eM ** 2 - 1.0) / (2.0 * nu)) + 1.0

    m_mode = 2
    Omega_M = x ** 1.5  # dimensionless M*Omega
    khat = eM * m_mode * Omega_M
    J = pphi / eM ** 2

    gamma_22 = (
        -214.0 * khat ** 2 / 105.0
        + 2.0 * m_mode * J * khat ** 3 / 3.0
        - 3390466.0 * khat ** 4 / 1157625.0
        + 381863.0 * m_mode * J * khat ** 5 / 99225.0
    )
    ellhat = 2.0 + lambda_RG * gamma_22

    log_2kr_omega = np.log(4.0 * np.sqrt(x))
    phi0 = np.exp(17.0 / 12.0 - GAMMA_E) / 4.0
    log_4phi0 = np.log(4.0 * phi0)

    logT22 = (
        np.log(120.0)
        + (ellhat - 2.0) * log_2kr_omega
        + 2j * khat * log_4phi0
        + loggamma(ellhat - 1.0 - 2j * khat)
        - loggamma(2.0 * ellhat + 2.0)
        + np.pi * khat
        - 1j * np.pi * (ellhat - 2.0) / 2.0
    )
    T22 = np.exp(logT22)

    eulerlog2 = GAMMA_E + np.log(4.0 * np.sqrt(x))
    pi2 = np.pi ** 2
    r1 = -43.0 / 42.0 + 55.0 * nu / 84.0
    r2 = -20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu ** 2 / 42336.0
    r3 = (
        -4296031.0 / 4889808.0
        + (41.0 * pi2 / 192.0 - 48993925.0 / 9779616.0) * nu
        - 6292061.0 * nu ** 2 / 3259872.0
        + 10620745.0 * nu ** 3 / 39118464.0
    )
    r4 = (
        9228174993589.0 / 800950550400.0
        + (
            -2487107795131.0 / 145627372800.0
            + 464.0 * eulerlog2 / 35.0
            - 9953.0 * pi2 / 21504.0
        )
        * nu
        + (10815863492353.0 / 640760440320.0 - 3485.0 * pi2 / 5376.0) * nu ** 2
        - 2088847783.0 * nu ** 3 / 11650189824.0
        + 70134663541.0 * nu ** 4 / 512608352256.0
    )
    rho22 = 1.0 + r1 * x + r2 * x ** 2 + r3 * x ** 3 + r4 * x ** 4

    y = (eM * x ** 1.5) ** (2.0 / 3.0)
    delta22 = (
        -17.0 * y ** 1.5 / 3.0
        - 24.0 * nu * y ** 2.5
        + (30995.0 * nu / 1134.0 + 962.0 * nu ** 2 / 135.0) * y ** 3.5
        - 4976.0 * np.pi * nu * y ** 4 / 105.0
    )

    return H_eff * T22 * rho22 ** 2 * np.exp(1j * delta22)


def h_of_f(
    f,
    Mc,
    eta,
    dL,
    tc=0.0,
    phic=0.0,
    lambda_RG=1.0,
    f_low=20.0,
    fmax_over_fisco=1.3,
    sigma_taper_over_fisco=0.01,
    phase_only=False,
):
    """Frequency-domain (2,2) RG-tail inspiral waveform.

    Parameters
    ----------
    f : array_like
        Frequencies in Hz.
    Mc : float
        Detector-frame chirp mass in M_sun.
    eta : float
        Symmetric mass ratio (0, 0.25].
    dL : float
        Luminosity distance in Mpc.
    tc, phic : float
        Coalescence time (s) and phase (rad).
    lambda_RG : float
        RG-running deformation (1.0 = GR).
    f_low : float
        Low-frequency cutoff (Hz). Strict zero below.
    fmax_over_fisco : float
        Upper edge in units of f_isco. Strict zero at or above.
    sigma_taper_over_fisco : float
        Width of the smooth Fermi-Dirac high-frequency taper in units of f_isco.
    phase_only : bool
        If True, return unit-modulus exp(i Phi(f)) (amplitude and taper dropped).

    Returns
    -------
    h : ndarray, complex
        Same shape as ``f``.
    """
    f = np.asarray(f, dtype=float)
    nu = float(eta)

    M_sec = Mc * MSUN_SEC / nu ** (3.0 / 5.0)
    dL_sec = dL * MPC_SEC

    f_isco = 1.0 / (np.pi * 6.0 ** 1.5 * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    h = np.zeros_like(f, dtype=complex)

    if f_cut <= f_low:
        return h

    # Integration grid in x
    x_low = (np.pi * M_sec * f_low) ** (2.0 / 3.0)
    x_high = (np.pi * M_sec * f_cut) ** (2.0 / 3.0)

    if x_high <= x_low:
        return h

    N_grid = 8192
    x_grid = np.logspace(np.log10(x_low), np.log10(x_high), N_grid)

    eM_grid, eM_p_grid = _eM_and_prime(x_grid, nu)
    hhat22_grid = _hhat22(x_grid, nu, lambda_RG)
    F22_grid = (32.0 / 5.0) * nu ** 2 * x_grid ** 5 * np.abs(hhat22_grid) ** 2

    dphi_dx_grid = -x_grid ** 1.5 * eM_p_grid / F22_grid
    dt_dx_grid = -M_sec * eM_p_grid / F22_grid

    phi_cum = cumulative_trapezoid(dphi_dx_grid, x_grid, initial=0.0)
    t_cum = cumulative_trapezoid(dt_dx_grid, x_grid, initial=0.0)

    # Set integration constants so phi_orb_rel = 0 and t_rel = 0 at x_high (coalescence reference)
    phi_orb_rel_grid = phi_cum - phi_cum[-1]
    t_rel_grid = t_cum - t_cum[-1]

    phi_interp = CubicSpline(x_grid, phi_orb_rel_grid)
    t_interp = CubicSpline(x_grid, t_rel_grid)

    in_band = (f >= f_low) & (f < f_cut)
    f_in = f[in_band]

    if f_in.size == 0:
        return h

    x_f = (np.pi * M_sec * f_in) ** (2.0 / 3.0)
    x_f_clipped = np.clip(x_f, x_grid[0], x_grid[-1])

    eM_at_xf, eM_p_at_xf = _eM_and_prime(x_f, nu)
    hhat22_at_xf = _hhat22(x_f, nu, lambda_RG)
    F22_at_xf = (32.0 / 5.0) * nu ** 2 * x_f ** 5 * np.abs(hhat22_at_xf) ** 2

    df_dt_at_xf = -3.0 * np.sqrt(x_f) * F22_at_xf / (2.0 * np.pi * M_sec ** 2 * eM_p_at_xf)

    phi_rel_at_xf = phi_interp(x_f_clipped)
    t_rel_at_xf = t_interp(x_f_clipped)

    # SPA face-on circular polarization (h_+ + i h_x), carrier convention exp(-2 i phi_orb)
    phase_natural = (
        2.0 * phi_rel_at_xf
        - 2.0 * np.pi * f_in * t_rel_at_xf
        + np.pi / 4.0
        + phic
        - 2.0 * np.pi * f_in * tc
    )

    common_factor = np.conj(hhat22_at_xf) * np.exp(1j * phase_natural)
    W = 1.0 / (1.0 + np.exp((f_in - f_cut) / sigma))

    if phase_only:
        h[in_band] = common_factor / np.abs(common_factor)
    else:
        amp_real = -2.0 * M_sec * nu * x_f / dL_sec / np.sqrt(df_dt_at_xf)
        h[in_band] = amp_real * common_factor * W

    return h
