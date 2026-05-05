"""Standalone RG-tail waveform prototype for the no-skill benchmark.

The implementation follows the source packet's dominant (2,2) factorized
ingredients, but uses a restricted stationary-phase amplitude convention:

* the flux-corrected inspiral time and phase are obtained by integrating the
  supplied ``dt/dx`` relation;
* the complex phase of ``hhat_22`` is added to the Fourier phase at the
  stationary point;
* the Fourier amplitude uses the standard Newtonian TaylorF2 scaling
  ``f^(-7/6)`` with the requested smooth taper.

The waveform is zero below ``f_low`` and at/above ``f_cut``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import loggamma


MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0
GAMMA_E = 0.5772156649015328606


def _pn_energy_and_derivative(x: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Return E_real/M and d(E_real/M)/dx for circular orbits."""

    x = np.asarray(x, dtype=float)
    log16x = np.log(16.0 * x)

    c0 = 1.0
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu**2 / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * np.pi**2 / 96.0) * nu
        - 155.0 * nu**2 / 96.0
        - 35.0 * nu**3 / 5184.0
    )
    c4 = (
        -3969.0 / 128.0
        + (-123671.0 / 5760.0 + 9037.0 * np.pi**2 / 1536.0 + 896.0 * GAMMA_E / 15.0 + 448.0 * log16x / 15.0)
        * nu
        + (498449.0 / 3456.0 - 3157.0 * np.pi**2 / 576.0) * nu**2
        + 301.0 * nu**3 / 1728.0
        + 77.0 * nu**4 / 31104.0
    )

    poly = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4
    e_over_m = 1.0 - 0.5 * nu * x * poly

    dc4_dx = 448.0 * nu / (15.0 * x)
    dpoly_dx = c1 + 2.0 * c2 * x + 3.0 * c3 * x**2 + 4.0 * c4 * x**3 + x**4 * dc4_dx
    de_over_m_dx = -0.5 * nu * (poly + x * dpoly_dx)

    return e_over_m, de_over_m_dx


def _pn_angular_momentum(x: np.ndarray, nu: float) -> np.ndarray:
    """Return p_phi,circ/(mu M)."""

    x = np.asarray(x, dtype=float)
    log16x = np.log(16.0 * x)

    d0 = 1.0
    d1 = 3.0 / 2.0 + nu / 6.0
    d2 = 27.0 / 8.0 - 19.0 * nu / 8.0 + nu**2 / 24.0
    d3 = (
        135.0 / 16.0
        + (-6889.0 / 144.0 + 41.0 * np.pi**2 / 24.0) * nu
        + 31.0 * nu**2 / 24.0
        + 7.0 * nu**3 / 1296.0
    )
    d4 = (
        2835.0 / 128.0
        + (98869.0 / 5760.0 - 128.0 * GAMMA_E / 3.0 - 6455.0 * np.pi**2 / 1536.0 - 64.0 * log16x / 3.0) * nu
        + (356035.0 / 3456.0 - 2255.0 * np.pi**2 / 576.0) * nu**2
        - 215.0 * nu**3 / 1728.0
        - 55.0 * nu**4 / 31104.0
    )

    return x ** (-0.5) * (d0 + d1 * x + d2 * x**2 + d3 * x**3 + d4 * x**4)


def _hhat_22(x: np.ndarray, nu: float, lambda_RG: float) -> np.ndarray:
    """Return the complex dominant-mode factor hhat_22."""

    x = np.asarray(x, dtype=float)

    e_over_m, _ = _pn_energy_and_derivative(x, nu)
    p_phi = _pn_angular_momentum(x, nu)

    h_eff = ((e_over_m**2) - 1.0) / (2.0 * nu) + 1.0
    omega_hat = x ** 1.5
    m = 2.0
    khat = e_over_m * m * omega_hat
    j_comb = p_phi / (e_over_m**2)

    gamma_univ = (
        -214.0 * khat**2 / 105.0
        + 2.0 * m * j_comb * khat**3 / 3.0
        - 3390466.0 * khat**4 / 1157625.0
        + 381863.0 * m * j_comb * khat**5 / 99225.0
    )
    ellhat = 2.0 + lambda_RG * gamma_univ

    phi0 = np.exp(17.0 / 12.0 - GAMMA_E) / 4.0
    logT = (
        np.log(120.0)
        + (ellhat - 2.0) * np.log(4.0 * np.sqrt(x))
        + 2.0j * khat * np.log(4.0 * phi0)
        + loggamma(ellhat - 1.0 - 2.0j * khat)
        - loggamma(2.0 * ellhat + 2.0)
        + np.pi * khat
        - 0.5j * np.pi * (ellhat - 2.0)
    )
    t22 = np.exp(logT)

    eulerlog2 = GAMMA_E + np.log(4.0 * np.sqrt(x))
    rho22 = (
        1.0
        + (-43.0 / 42.0 + 55.0 * nu / 84.0) * x
        + (-20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu**2 / 42336.0) * x**2
        + (
            -4296031.0 / 4889808.0
            + (41.0 * np.pi**2 / 192.0 - 48993925.0 / 9779616.0) * nu
            - 6292061.0 * nu**2 / 3259872.0
            + 10620745.0 * nu**3 / 39118464.0
        )
        * x**3
        + (
            9228174993589.0 / 800950550400.0
            + (-2487107795131.0 / 145627372800.0 + 464.0 * eulerlog2 / 35.0 - 9953.0 * np.pi**2 / 21504.0) * nu
            + (10815863492353.0 / 640760440320.0 - 3485.0 * np.pi**2 / 5376.0) * nu**2
            - 2088847783.0 * nu**3 / 11650189824.0
            + 70134663541.0 * nu**4 / 512608352256.0
        )
        * x**4
    )

    y = np.power(e_over_m * x ** 1.5, 2.0 / 3.0)
    delta22 = (
        -17.0 * y ** 1.5 / 3.0
        - 24.0 * nu * y ** 2.5
        + (30995.0 * nu / 1134.0 + 962.0 * nu**2 / 135.0) * y ** 3.5
        - 4976.0 * np.pi * nu * y**4 / 105.0
    )

    return h_eff * t22 * rho22**2 * np.exp(1.0j * delta22)


def _build_phase_and_time_grid(
    x_low: float,
    x_cut: float,
    M_sec: float,
    nu: float,
    tc: float,
    phic: float,
    lambda_RG: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute t(x), phi_orb(x), and arg(hhat_22) on a monotone grid."""

    x_grid = np.geomspace(x_low, x_cut, n_grid)
    _, de_over_m_dx = _pn_energy_and_derivative(x_grid, nu)
    hhat = _hhat_22(x_grid, nu, lambda_RG)

    flux = (32.0 / 5.0) * nu**2 * x_grid**5 * np.abs(hhat) ** 2
    dt_dx = -M_sec * de_over_m_dx / flux
    dphi_dx = x_grid ** 1.5 * (-de_over_m_dx / flux)

    int_t = cumulative_trapezoid(dt_dx, x_grid, initial=0.0)
    int_phi = cumulative_trapezoid(dphi_dx, x_grid, initial=0.0)

    t_grid = tc - (int_t[-1] - int_t)
    phi_orb_grid = 0.5 * phic - (int_phi[-1] - int_phi)
    hhat_phase_grid = np.unwrap(np.angle(hhat))

    return x_grid, t_grid, phi_orb_grid, hhat_phase_grid


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
    """Return the complex frequency-domain strain for the benchmark waveform."""

    f_arr = np.asarray(f, dtype=float)
    f_flat = np.ravel(f_arr)
    out_flat = np.zeros_like(f_flat, dtype=np.complex128)

    nu = float(eta)
    Mc_sec = float(Mc) * MSUN_SEC
    M_sec = Mc_sec / nu ** (3.0 / 5.0)
    dL_sec = float(dL) * MPC_SEC

    f_isco = 1.0 / (np.pi * 6.0 ** 1.5 * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = max(sigma_taper_over_fisco * f_isco, np.finfo(float).tiny)

    valid = (f_flat >= f_low) & (f_flat < f_cut)
    if not np.any(valid):
        return out_flat.reshape(f_arr.shape)

    x_low = np.power(np.pi * M_sec * f_low, 2.0 / 3.0)
    x_cut = np.power(np.pi * M_sec * f_cut, 2.0 / 3.0)

    if x_low <= 0.0:
        x_low = 1.0e-12

    if not np.isfinite(x_low) or not np.isfinite(x_cut) or x_cut <= x_low:
        return out_flat.reshape(f_arr.shape)

    # Dense log-spaced grid gives good phase accuracy across the full inspiral.
    n_grid = max(4096, int(1024 * np.log(x_cut / x_low + 1.0)) + 1)
    x_grid, t_grid, phi_orb_grid, hhat_phase_grid = _build_phase_and_time_grid(
        x_low=x_low,
        x_cut=x_cut,
        M_sec=M_sec,
        nu=nu,
        tc=float(tc),
        phic=float(phic),
        lambda_RG=float(lambda_RG),
        n_grid=n_grid,
    )

    x_valid = np.power(np.pi * M_sec * f_flat[valid], 2.0 / 3.0)
    t_valid = np.interp(x_valid, x_grid, t_grid)
    phi_valid = np.interp(x_valid, x_grid, phi_orb_grid)
    hhat_phase_valid = np.interp(x_valid, x_grid, hhat_phase_grid)

    phase = 2.0 * np.pi * f_flat[valid] * t_valid - 2.0 * phi_valid - np.pi / 4.0 + hhat_phase_valid

    if phase_only:
        out_flat[valid] = np.exp(1.0j * phase)
        return out_flat.reshape(f_arr.shape)

    amp0 = np.sqrt(5.0 / 24.0) * Mc_sec ** (5.0 / 6.0) / (np.pi ** (2.0 / 3.0) * dL_sec)
    taper = 1.0 / (1.0 + np.exp((f_flat[valid] - f_cut) / sigma))
    amp = amp0 * f_flat[valid] ** (-7.0 / 6.0) * taper

    out_flat[valid] = amp * np.exp(1.0j * phase)
    return out_flat.reshape(f_arr.shape)
