"""Reference RG-tail frequency-domain inspiral waveform (arXiv:2602.08833).

Ground-truth implementation for the New Physics Bench. Computes the dominant
nonspinning (2,2) mode with a factorized RG-tail correction:

    hhat_22 = H_eff * T_22 * rho_22^2 * exp(i * delta_22)

The beyond-GR parameter lambda_RG scales the running anomalous dimension.
lambda_RG = 1 recovers GR.
"""
from __future__ import annotations

import numpy as np
from scipy.special import loggamma

MSUN_SEC = 4.925491025543576e-06
MPC_SEC = 1.0292712503e14
GAMMA_E = 0.5772156649015329
PI = np.pi

_MAX_LOG_REAL = 500.0


def _asarray(x):
    return np.asarray(x, dtype=float)


def _eulerlog2(x):
    return GAMMA_E + np.log(4.0 * np.sqrt(x))


def _isco_frequency(M_sec):
    return 1.0 / (6.0**1.5 * PI * M_sec)


def energy_real_over_M(x, nu):
    x = _asarray(x)
    nu2, nu3, nu4 = nu**2, nu**3, nu**4

    c0 = 1.0
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu2 / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * PI**2 / 96.0) * nu
        - 155.0 * nu2 / 96.0
        - 35.0 * nu3 / 5184.0
    )
    c4 = (
        -3969.0 / 128.0
        + (
            -123671.0 / 5760.0
            + 9037.0 * PI**2 / 1536.0
            + 896.0 * GAMMA_E / 15.0
            + 448.0 * np.log(16.0 * x) / 15.0
        )
        * nu
        + (498449.0 / 3456.0 - 3157.0 * PI**2 / 576.0) * nu2
        + 301.0 * nu3 / 1728.0
        + 77.0 * nu4 / 31104.0
    )

    bracket = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4
    return 1.0 - 0.5 * nu * x * bracket


def d_energy_real_over_M_dx(x, nu):
    x = _asarray(x)
    nu2, nu3, nu4 = nu**2, nu**3, nu**4

    c0 = 1.0
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu2 / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * PI**2 / 96.0) * nu
        - 155.0 * nu2 / 96.0
        - 35.0 * nu3 / 5184.0
    )
    c4_const = (
        -3969.0 / 128.0
        + (
            -123671.0 / 5760.0
            + 9037.0 * PI**2 / 1536.0
            + 896.0 * GAMMA_E / 15.0
        )
        * nu
        + (498449.0 / 3456.0 - 3157.0 * PI**2 / 576.0) * nu2
        + 301.0 * nu3 / 1728.0
        + 77.0 * nu4 / 31104.0
    )
    c4_log = 448.0 * nu / 15.0
    c4 = c4_const + c4_log * np.log(16.0 * x)

    bracket = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4
    d_bracket_dx = (
        c1 + 2.0 * c2 * x + 3.0 * c3 * x**2 + (4.0 * c4 + c4_log) * x**3
    )
    return -0.5 * nu * (bracket + x * d_bracket_dx)


def p_phi_circ_over_muM(x, nu):
    x = _asarray(x)
    nu2, nu3, nu4 = nu**2, nu**3, nu**4

    c0 = 1.0
    c1 = 3.0 / 2.0 + nu / 6.0
    c2 = 27.0 / 8.0 - 19.0 * nu / 8.0 + nu2 / 24.0
    c3 = (
        135.0 / 16.0
        + (-6889.0 / 144.0 + 41.0 * PI**2 / 24.0) * nu
        + 31.0 * nu2 / 24.0
        + 7.0 * nu3 / 1296.0
    )
    c4 = (
        2835.0 / 128.0
        + (
            98869.0 / 5760.0
            - 128.0 * GAMMA_E / 3.0
            - 6455.0 * PI**2 / 1536.0
            - 64.0 * np.log(16.0 * x) / 3.0
        )
        * nu
        + (356035.0 / 3456.0 - 2255.0 * PI**2 / 576.0) * nu2
        - 215.0 * nu3 / 1728.0
        - 55.0 * nu4 / 31104.0
    )

    bracket = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4
    return bracket / np.sqrt(x)


def Hhat_eff(x, nu):
    E_over_M = energy_real_over_M(x, nu)
    return (E_over_M**2 - 1.0) / (2.0 * nu) + 1.0


def gamma_univ_22(khat, J, m=2):
    khat = _asarray(khat)
    J = _asarray(J)
    return (
        -214.0 / 105.0 * khat**2
        + 2.0 * m * J * khat**3 / 3.0
        - 3390466.0 / 1157625.0 * khat**4
        + 381863.0 * m * J * khat**5 / 99225.0
    )


def ellhathat_22(x, nu, lambda_RG=1.0):
    x = _asarray(x)
    E_over_M = energy_real_over_M(x, nu)
    Omega = x**1.5
    m = 2
    khat = E_over_M * m * Omega
    p_phi = p_phi_circ_over_muM(x, nu)
    J = p_phi / E_over_M**2
    return 2.0 + lambda_RG * gamma_univ_22(khat, J, m=m)


def tail_factor_T22(x, nu, lambda_RG=1.0):
    x = _asarray(x)
    ell = 2.0
    m = 2.0
    phi0 = np.exp(17.0 / 12.0 - GAMMA_E) / 4.0

    E_over_M = energy_real_over_M(x, nu)
    Omega = x**1.5
    k = m * Omega
    khat = E_over_M * k
    lhat = ellhathat_22(x, nu, lambda_RG=lambda_RG)
    r_omega = 1.0 / x

    log_tail = (
        np.log(120.0)
        + (lhat - ell) * np.log(2.0 * k * r_omega)
        + 2j * khat * np.log(2.0 * m * phi0)
        + loggamma(lhat - 1.0 - 2j * khat)
        - loggamma(2.0 * lhat + 2.0)
        + PI * khat
        - 0.5j * PI * (lhat - ell)
    )

    log_tail = np.where(
        np.real(log_tail) > _MAX_LOG_REAL,
        _MAX_LOG_REAL + 1j * np.imag(log_tail),
        log_tail,
    )
    return np.exp(log_tail)


def rho_22(x, nu):
    x = _asarray(x)
    nu2, nu3, nu4 = nu**2, nu**3, nu**4
    eulerlog2 = _eulerlog2(x)

    c1 = -43.0 / 42.0 + 55.0 * nu / 84.0
    c2 = -20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu2 / 42336.0
    c3 = (
        -4296031.0 / 4889808.0
        + (41.0 * PI**2 / 192.0 - 48993925.0 / 9779616.0) * nu
        - 6292061.0 * nu2 / 3259872.0
        + 10620745.0 * nu3 / 39118464.0
    )
    c4 = (
        9228174993589.0 / 800950550400.0
        + (
            -2487107795131.0 / 145627372800.0
            + 464.0 * eulerlog2 / 35.0
            - 9953.0 * PI**2 / 21504.0
        )
        * nu
        + (10815863492353.0 / 640760440320.0 - 3485.0 * PI**2 / 5376.0) * nu2
        - 2088847783.0 * nu3 / 11650189824.0
        + 70134663541.0 * nu4 / 512608352256.0
    )

    return 1.0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4


def delta_22(x, nu):
    x = _asarray(x)
    E_over_M = energy_real_over_M(x, nu)
    y = (E_over_M * x**1.5) ** (2.0 / 3.0)
    nu2 = nu**2

    return (
        -17.0 / 3.0 * y**1.5
        - 24.0 * nu * y**2.5
        + (30995.0 * nu / 1134.0 + 962.0 * nu2 / 135.0) * y**3.5
        - 4976.0 * PI * nu * y**4 / 105.0
    )


def hhat_22(x, nu, lambda_RG=1.0):
    x = _asarray(x)
    source = Hhat_eff(x, nu)
    tail = tail_factor_T22(x, nu, lambda_RG=lambda_RG)
    residual_amp = rho_22(x, nu)
    residual_phase = delta_22(x, nu)
    return source * tail * residual_amp**2 * np.exp(1j * residual_phase)


def flux_22(x, nu, lambda_RG=1.0):
    x = _asarray(x)
    return (32.0 / 5.0) * nu**2 * x**5 * np.abs(hhat_22(x, nu, lambda_RG=lambda_RG)) ** 2


def _cumulative_integral_to_upper(x, y):
    if len(x) == 0:
        return np.array([], dtype=float)
    if len(x) == 1:
        return np.zeros(1, dtype=float)
    dx = np.diff(x)
    trap = 0.5 * (y[:-1] + y[1:]) * dx
    cumulative_from_low = np.concatenate(([0.0], np.cumsum(trap)))
    return cumulative_from_low[-1] - cumulative_from_low


def _spa_phase_from_22_flux(f, M_sec, nu, lambda_RG=1.0, x_ref=None):
    f = _asarray(f)
    if len(f) == 0:
        return np.array([], dtype=float)

    x = (PI * M_sec * f) ** (2.0 / 3.0)
    if x_ref is None:
        x_ref = x[-1]
    x_ref = float(max(x_ref, x[-1]))
    append_ref = x_ref > x[-1] * (1.0 + 1e-12)
    if append_ref:
        x_nodes = np.concatenate((x, [x_ref]))
    else:
        x_nodes = x

    dE_dx = d_energy_real_over_M_dx(x_nodes, nu)
    F22 = flux_22(x_nodes, nu, lambda_RG=lambda_RG)
    F22 = np.maximum(F22, 1e-300)
    dt_dx = -M_sec * dE_dx / F22
    dphi_gw_dx = 2.0 * (x_nodes**1.5 / M_sec) * dt_dx

    time_to_ref = _cumulative_integral_to_upper(x_nodes, dt_dx)
    gw_phase_to_ref = _cumulative_integral_to_upper(x_nodes, dphi_gw_dx)

    if append_ref:
        time_to_ref = time_to_ref[:-1]
        gw_phase_to_ref = gw_phase_to_ref[:-1]

    return gw_phase_to_ref - 2.0 * PI * f * time_to_ref


def _balance_law_spa_amplitude_correction(x, nu):
    dE_dx = d_energy_real_over_M_dx(x, nu)
    correction_sq = -2.0 * dE_dx / nu
    return np.sqrt(np.maximum(correction_sq, 0.0))


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
        Frequency array in Hz.
    Mc : float
        Detector-frame chirp mass in solar masses.
    eta : float
        Symmetric mass ratio.
    dL : float
        Luminosity distance in Mpc.
    tc : float
        Coalescence time offset in seconds.
    phic : float
        Coalescence phase.
    lambda_RG : float
        RG deformation parameter (1.0 = GR).
    f_low : float
        Low-frequency cutoff in Hz.
    fmax_over_fisco : float
        Upper cutoff as fraction of ISCO frequency.
    sigma_taper_over_fisco : float
        Fermi taper width as fraction of ISCO frequency.
    phase_only : bool
        If True, use Newtonian amplitude (ignore RG amplitude corrections).

    Returns
    -------
    h : ndarray
        Complex frequency-domain strain with same shape as f.
    """
    f = _asarray(f)
    Mc_sec = Mc * MSUN_SEC
    M_sec = Mc_sec / eta ** (3.0 / 5.0)
    dL_sec = dL * MPC_SEC
    nu = eta

    f_isco = _isco_frequency(M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    valid = (f >= f_low) & (f < f_cut)
    h = np.zeros_like(f, dtype=complex)
    if not np.any(valid):
        return h

    f_eval = f[valid]
    x = (PI * M_sec * f_eval) ** (2.0 / 3.0)

    amp_newt = (
        np.sqrt(5.0 / 24.0)
        * Mc_sec ** (5.0 / 6.0)
        / (dL_sec * PI ** (2.0 / 3.0))
        * f_eval ** (-7.0 / 6.0)
    )

    hhat_running = hhat_22(x, nu, lambda_RG=lambda_RG)
    x_ref = (PI * M_sec * f_cut) ** (2.0 / 3.0)
    psi_orb = _spa_phase_from_22_flux(
        f_eval, M_sec=M_sec, nu=nu, lambda_RG=lambda_RG, x_ref=x_ref
    )
    phase = (
        2.0 * PI * f_eval * tc - phic - PI / 4.0 + psi_orb + np.angle(hhat_running)
    )

    if phase_only:
        amp = amp_newt
    else:
        amp = amp_newt * _balance_law_spa_amplitude_correction(x, nu)

    exponent = np.clip((f_eval - f_isco) / sigma, -500.0, 500.0)
    taper = 1.0 / (1.0 + np.exp(exponent))

    h[valid] = amp * np.exp(1j * phase) * taper
    return h
