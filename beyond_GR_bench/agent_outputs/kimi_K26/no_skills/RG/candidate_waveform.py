"""Candidate RG-tail waveform for the dominant (2,2) mode.

Uses the factorized waveform construction from arXiv:2602.08833:
    hhat_22 = H_eff * T_22 * rho_22^2 * exp(i * delta_22)

with SPA (Stationary Phase Approximation) for the frequency-domain strain.
"""

import numpy as np
from scipy.special import loggamma


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
    """Return complex frequency-domain strain for the dominant (2,2) mode.

    Parameters
    ----------
    f : ndarray
        Frequency array in Hz.
    Mc : float
        Detector-frame chirp mass in solar masses.
    eta : float
        Symmetric mass ratio (nu = m1*m2/M^2).
    dL : float
        Luminosity distance in Mpc.
    tc : float
        Coalescence time in seconds.
    phic : float
        Coalescence phase in radians.
    lambda_RG : float
        RG tail deformation parameter (1.0 = GR).
    f_low : float
        Low-frequency cutoff in Hz.
    fmax_over_fisco : float
        High-frequency cutoff as multiple of f_isco.
    sigma_taper_over_fisco : float
        Taper width as fraction of f_isco.
    phase_only : bool
        If True, return phase-only waveform (unit amplitude).

    Returns
    -------
    ndarray
        Complex frequency-domain strain array with same shape as f.
    """
    # Constants
    MSUN_SEC = 4.925491025543576e-6
    MPC_SEC = 3.0856775814913673e22 / 299792458
    gamma_E = 0.5772156649015328606

    f = np.asarray(f, dtype=np.float64)
    nu = eta
    M_sec = Mc * MSUN_SEC / nu ** (3.0 / 5.0)
    dL_sec = dL * MPC_SEC

    # Cutoffs
    f_isco = 1.0 / (np.pi * 6.0 ** 1.5 * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    # Valid frequency mask
    valid = (f >= f_low) & (f < f_cut)
    h = np.zeros_like(f, dtype=np.complex128)
    f_v = f[valid]

    if not np.any(valid):
        return h

    x_f = (np.pi * M_sec * f_v) ** (2.0 / 3.0)

    # =========================================================================
    # Conservative sector
    # =========================================================================
    E_real_M = 1.0 - (nu * x_f / 2.0) * (
        1.0
        + (-3.0 / 4.0 - nu / 12.0) * x_f
        + (-27.0 / 8.0 + 19.0 * nu / 8.0 - nu ** 2 / 24.0) * x_f ** 2
        + (
            -675.0 / 64.0
            + (34445.0 / 576.0 - 205.0 * np.pi ** 2 / 96.0) * nu
            - 155.0 * nu ** 2 / 96.0
            - 35.0 * nu ** 3 / 5184.0
        )
        * x_f ** 3
        + (
            -3969.0 / 128.0
            + (
                -123671.0 / 5760.0
                + 9037.0 * np.pi ** 2 / 1536.0
                + 896.0 * gamma_E / 15.0
                + 448.0 * np.log(16.0 * x_f) / 15.0
            )
            * nu
            + (498449.0 / 3456.0 - 3157.0 * np.pi ** 2 / 576.0) * nu ** 2
            + 301.0 * nu ** 3 / 1728.0
            + 77.0 * nu ** 4 / 31104.0
        )
        * x_f ** 4
    )

    p_phi_M = x_f ** (-0.5) * (
        1.0
        + (3.0 / 2.0 + nu / 6.0) * x_f
        + (27.0 / 8.0 - 19.0 * nu / 8.0 + nu ** 2 / 24.0) * x_f ** 2
        + (
            135.0 / 16.0
            + (-6889.0 / 144.0 + 41.0 * np.pi ** 2 / 24.0) * nu
            + 31.0 * nu ** 2 / 24.0
            + 7.0 * nu ** 3 / 1296.0
        )
        * x_f ** 3
        + (
            2835.0 / 128.0
            + (
                98869.0 / 5760.0
                - 128.0 * gamma_E / 3.0
                - 6455.0 * np.pi ** 2 / 1536.0
                - 64.0 * np.log(16.0 * x_f) / 3.0
            )
            * nu
            + (356035.0 / 3456.0 - 2255.0 * np.pi ** 2 / 576.0) * nu ** 2
            - 215.0 * nu ** 3 / 1728.0
            - 55.0 * nu ** 4 / 31104.0
        )
        * x_f ** 4
    )

    H_eff = ((E_real_M) ** 2 - 1.0) / (2.0 * nu) + 1.0

    # =========================================================================
    # Running tail
    # =========================================================================
    m_mode = 2
    khat = E_real_M * m_mode * np.pi * M_sec * f_v
    J = p_phi_M / E_real_M ** 2

    gamma_22_univ = (
        -214.0 * khat ** 2 / 105.0
        + 2.0 * m_mode * J * khat ** 3 / 3.0
        - 3390466.0 * khat ** 4 / 1157625.0
        + 381863.0 * m_mode * J * khat ** 5 / 99225.0
    )

    ellhat_22 = 2.0 + lambda_RG * gamma_22_univ

    phi0 = np.exp(17.0 / 12.0 - gamma_E) / 4.0
    log_2kr = np.log(4.0 * np.sqrt(x_f))

    logT_22 = (
        np.log(120.0)
        + (ellhat_22 - 2.0) * log_2kr
        + 2.0j * khat * np.log(4.0 * phi0)
        + loggamma(ellhat_22 - 1.0 - 2.0j * khat)
        - loggamma(2.0 * ellhat_22 + 2.0)
        + np.pi * khat
        - 1j * np.pi * (ellhat_22 - 2.0) / 2.0
    )

    T_22 = np.exp(logT_22)

    # =========================================================================
    # Residual amplitude rho_22
    # =========================================================================
    eulerlog_2 = gamma_E + np.log(4.0 * np.sqrt(x_f))

    rho_22 = 1.0 + (
        -43.0 / 42.0 + 55.0 * nu / 84.0
    ) * x_f + (
        -20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu ** 2 / 42336.0
    ) * x_f ** 2 + (
        -4296031.0 / 4889808.0
        + (41.0 * np.pi ** 2 / 192.0 - 48993925.0 / 9779616.0) * nu
        - 6292061.0 * nu ** 2 / 3259872.0
        + 10620745.0 * nu ** 3 / 39118464.0
    ) * x_f ** 3 + (
        9228174993589.0 / 800950550400.0
        + (
            -2487107795131.0 / 145627372800.0
            + 464.0 * eulerlog_2 / 35.0
            - 9953.0 * np.pi ** 2 / 21504.0
        )
        * nu
        + (10815863492353.0 / 640760440320.0 - 3485.0 * np.pi ** 2 / 5376.0)
        * nu ** 2
        - 2088847783.0 * nu ** 3 / 11650189824.0
        + 70134663541.0 * nu ** 4 / 512608352256.0
    ) * x_f ** 4

    # =========================================================================
    # Residual phase delta_22
    # =========================================================================
    y = (E_real_M * x_f ** 1.5) ** (2.0 / 3.0)

    delta_22 = (
        -17.0 * y ** 1.5 / 3.0
        - 24.0 * nu * y ** 2.5
        + (30995.0 * nu / 1134.0 + 962.0 * nu ** 2 / 135.0) * y ** 3.5
        - 4976.0 * np.pi * nu * y ** 4.0 / 105.0
    )

    # =========================================================================
    # Factorized correction hhat_22
    # =========================================================================
    hhat_22 = H_eff * T_22 * rho_22 ** 2 * np.exp(1j * delta_22)

    # =========================================================================
    # SPA amplitude
    # =========================================================================
    # F_22 = (32/5) * nu^2 * x^5 * |hhat_22|^2
    F_22 = 32.0 / 5.0 * nu ** 2 * x_f ** 5 * np.abs(hhat_22) ** 2

    # d(E_real/M)/dx
    # E_real/M = 1 - (nu/2) * x * [c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4]
    # where c4 = c4_const + (448/15)*nu*log(16*x)
    # d/dx = -(nu/2) * [1 + 2*c1*x + 3*c2*x^2 + 4*c3*x^3 + 5*c4_const*x^4
    #                   + (448/15)*nu * d(x^5*log(16*x))/dx]
    # d(x^5*log(16*x))/dx = x^4 * (5*log(16*x) + 1)
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu ** 2 / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * np.pi ** 2 / 96.0) * nu
        - 155.0 * nu ** 2 / 96.0
        - 35.0 * nu ** 3 / 5184.0
    )
    c4_const = (
        -3969.0 / 128.0
        + (-123671.0 / 5760.0 + 9037.0 * np.pi ** 2 / 1536.0 + 896.0 * gamma_E / 15.0)
        * nu
        + (498449.0 / 3456.0 - 3157.0 * np.pi ** 2 / 576.0) * nu ** 2
        + 301.0 * nu ** 3 / 1728.0
        + 77.0 * nu ** 4 / 31104.0
    )
    dE_dx = -(nu / 2.0) * (
        1.0
        + 2.0 * c1 * x_f
        + 3.0 * c2 * x_f ** 2
        + 4.0 * c3 * x_f ** 3
        + 5.0 * c4_const * x_f ** 4
        + (448.0 / 15.0) * nu * x_f ** 4 * (5.0 * np.log(16.0 * x_f) + 1.0)
    )

    # dt/dx = -M * d(E_real/M)/dx / F_22
    dt_dx = -M_sec * dE_dx / F_22

    # SPA amplitude: A(f) = sqrt(5/24) * M^(5/6) / (pi^(1/6) * dL) * x^(7/6) / sqrt(F_22 * dt/dx)
    A = (
        np.sqrt(5.0 / 24.0)
        * M_sec ** (5.0 / 6.0)
        / (np.pi ** (1.0 / 6.0) * dL_sec)
        * x_f ** (7.0 / 6.0)
        / np.sqrt(F_22 * dt_dx)
    )

    if phase_only:
        A = np.ones_like(A)

    # =========================================================================
    # SPA phase
    # =========================================================================
    # Newtonian SPA phase
    psi_N = 3.0 / (128.0 * nu) * (np.pi * Mc * MSUN_SEC * f_v) ** (-5.0 / 3.0)

    # Phase from hhat_22: arg(T_22) + delta_22
    arg_T22 = np.imag(logT_22)
    psi_hhat = arg_T22 + delta_22

    # Total phase: 2*pi*f*tc - phic + psi_N + psi_hhat
    psi = 2.0 * np.pi * f_v * tc - phic + psi_N + psi_hhat

    # =========================================================================
    # Assemble waveform
    # =========================================================================
    h[valid] = A * np.exp(1j * psi)

    # Taper near f_cut
    taper = 1.0 / (1.0 + np.exp((f_v - f_cut) / sigma))
    h[valid] *= taper

    return h
