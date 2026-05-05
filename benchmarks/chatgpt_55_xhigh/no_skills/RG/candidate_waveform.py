"""Standalone RG-tail dominant-mode frequency-domain waveform prototype.

The public entry point is ``h_of_f``.  It implements the source-packet formulas
for the nonspinning (2,2) mode and wraps them in a simple SPA inspiral model.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import loggamma

MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0
_GAMMA_E = 0.577215664901532860606512090082402431
_PI = np.pi


def _energy_and_derivative(x, eta):
    """Return E_real/M and d(E_real/M)/dx for circular nonspinning orbits."""
    x = np.asarray(x, dtype=float)
    nu = float(eta)
    log16x = np.log(16.0 * x)

    c0 = 1.0
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu**2 / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * _PI**2 / 96.0) * nu
        - 155.0 * nu**2 / 96.0
        - 35.0 * nu**3 / 5184.0
    )
    c4 = (
        -3969.0 / 128.0
        + (
            -123671.0 / 5760.0
            + 9037.0 * _PI**2 / 1536.0
            + 896.0 * _GAMMA_E / 15.0
            + 448.0 * log16x / 15.0
        )
        * nu
        + (498449.0 / 3456.0 - 3157.0 * _PI**2 / 576.0) * nu**2
        + 301.0 * nu**3 / 1728.0
        + 77.0 * nu**4 / 31104.0
    )

    series = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4
    energy = 1.0 - 0.5 * nu * x * series

    # c4 contains (448/15) nu log(16 x), so d(c4)/dx contributes this/x.
    dc4_log_coeff = 448.0 * nu / 15.0
    dseries = (
        c1
        + 2.0 * c2 * x
        + 3.0 * c3 * x**2
        + 4.0 * c4 * x**3
        + dc4_log_coeff * x**3
    )
    denergy_dx = -0.5 * nu * (series + x * dseries)
    return energy, denergy_dx


def _circular_angular_momentum(x, eta):
    """Return p_phi,circ/(mu M)."""
    x = np.asarray(x, dtype=float)
    nu = float(eta)
    log16x = np.log(16.0 * x)

    d0 = 1.0
    d1 = 3.0 / 2.0 + nu / 6.0
    d2 = 27.0 / 8.0 - 19.0 * nu / 8.0 + nu**2 / 24.0
    d3 = (
        135.0 / 16.0
        + (-6889.0 / 144.0 + 41.0 * _PI**2 / 24.0) * nu
        + 31.0 * nu**2 / 24.0
        + 7.0 * nu**3 / 1296.0
    )
    d4 = (
        2835.0 / 128.0
        + (
            98869.0 / 5760.0
            - 128.0 * _GAMMA_E / 3.0
            - 6455.0 * _PI**2 / 1536.0
            - 64.0 * log16x / 3.0
        )
        * nu
        + (356035.0 / 3456.0 - 2255.0 * _PI**2 / 576.0) * nu**2
        - 215.0 * nu**3 / 1728.0
        - 55.0 * nu**4 / 31104.0
    )

    series = d0 + d1 * x + d2 * x**2 + d3 * x**3 + d4 * x**4
    return x ** (-0.5) * series


def _hhat_22(x, eta, lambda_RG):
    """Dimensionless factorized (2,2) correction hhat_22."""
    x = np.asarray(x, dtype=float)
    nu = float(eta)
    lam = float(lambda_RG)

    energy, _ = _energy_and_derivative(x, nu)
    p_phi = _circular_angular_momentum(x, nu)
    h_eff = ((energy**2 - 1.0) / (2.0 * nu)) + 1.0

    m = 2.0
    omega_hat = x ** 1.5
    khat = energy * m * omega_hat
    J = p_phi / energy**2
    gamma_univ = (
        -214.0 * khat**2 / 105.0
        + 2.0 * m * J * khat**3 / 3.0
        - 3390466.0 * khat**4 / 1157625.0
        + 381863.0 * m * J * khat**5 / 99225.0
    )
    ellhat = 2.0 + lam * gamma_univ

    log_2kr = np.log(4.0 * np.sqrt(x))
    log_4phi0 = 17.0 / 12.0 - _GAMMA_E
    logT = (
        np.log(120.0)
        + (ellhat - 2.0) * log_2kr
        + 2.0j * khat * log_4phi0
        + loggamma(ellhat - 1.0 - 2.0j * khat)
        - loggamma(2.0 * ellhat + 2.0)
        + _PI * khat
        - 0.5j * _PI * (ellhat - 2.0)
    )
    T22 = np.exp(logT)

    eulerlog_2 = _GAMMA_E + np.log(4.0 * np.sqrt(x))
    r1 = -43.0 / 42.0 + 55.0 * nu / 84.0
    r2 = -20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu**2 / 42336.0
    r3 = (
        -4296031.0 / 4889808.0
        + (41.0 * _PI**2 / 192.0 - 48993925.0 / 9779616.0) * nu
        - 6292061.0 * nu**2 / 3259872.0
        + 10620745.0 * nu**3 / 39118464.0
    )
    r4 = (
        9228174993589.0 / 800950550400.0
        + (
            -2487107795131.0 / 145627372800.0
            + 464.0 * eulerlog_2 / 35.0
            - 9953.0 * _PI**2 / 21504.0
        )
        * nu
        + (10815863492353.0 / 640760440320.0 - 3485.0 * _PI**2 / 5376.0) * nu**2
        - 2088847783.0 * nu**3 / 11650189824.0
        + 70134663541.0 * nu**4 / 512608352256.0
    )
    rho22 = 1.0 + r1 * x + r2 * x**2 + r3 * x**3 + r4 * x**4

    y = (energy * x**1.5) ** (2.0 / 3.0)
    delta22 = (
        -17.0 * y**1.5 / 3.0
        - 24.0 * nu * y**2.5
        + (30995.0 * nu / 1134.0 + 962.0 * nu**2 / 135.0) * y**3.5
        - 4976.0 * _PI * nu * y**4 / 105.0
    )

    return h_eff * T22 * rho22**2 * np.exp(1.0j * delta22)


def _flux_22(x, eta, lambda_RG):
    hhat = _hhat_22(x, eta, lambda_RG)
    return (32.0 / 5.0) * eta**2 * x**5 * np.abs(hhat) ** 2


def _dt_dx(x, M_sec, eta, lambda_RG):
    _, denergy_dx = _energy_and_derivative(x, eta)
    return -M_sec * denergy_dx / _flux_22(x, eta, lambda_RG)


def _phase_integrals_to_cut(x_eval, x_cut, M_sec, eta, lambda_RG):
    """Return time and orbital-phase intervals from x_eval to x_cut."""
    x_eval = np.asarray(x_eval, dtype=float)
    if x_eval.size == 0:
        return x_eval.copy(), x_eval.copy()

    x_min = float(np.min(x_eval))
    if not (x_cut > x_min):
        return np.zeros_like(x_eval), np.zeros_like(x_eval)

    # Subtract the Newtonian singular pieces analytically; the remaining
    # integrands are much smoother on a logarithmic grid.
    t_newt = lambda x: 5.0 * M_sec / (64.0 * eta) * x ** (-5.0)
    phi_newt = lambda x: 5.0 / (64.0 * eta) * x ** (-3.5)

    log_span = np.log(x_cut / x_min)
    n_grid = int(max(12000, 9000 * log_span + 4 * x_eval.size))
    n_grid = min(n_grid, 180000)
    base = np.geomspace(x_min, x_cut, n_grid)
    grid = np.unique(np.concatenate((base, x_eval, np.array([x_cut]))))
    grid.sort()

    dt = _dt_dx(grid, M_sec, eta, lambda_RG)
    dphi = (grid**1.5 / M_sec) * dt
    t_corr_integrand = dt - t_newt(grid)
    phi_corr_integrand = dphi - phi_newt(grid)

    t_corr_cum = cumulative_trapezoid(t_corr_integrand, grid, initial=0.0)
    phi_corr_cum = cumulative_trapezoid(phi_corr_integrand, grid, initial=0.0)
    t_corr_to_cut = t_corr_cum[-1] - np.interp(x_eval, grid, t_corr_cum)
    phi_corr_to_cut = phi_corr_cum[-1] - np.interp(x_eval, grid, phi_corr_cum)

    t_to_cut = (
        5.0 * M_sec / (256.0 * eta) * (x_eval ** (-4.0) - x_cut ** (-4.0))
        + t_corr_to_cut
    )
    phi_to_cut = (
        1.0 / (32.0 * eta) * (x_eval ** (-2.5) - x_cut ** (-2.5))
        + phi_corr_to_cut
    )
    return t_to_cut, phi_to_cut


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
    """Return a complex frequency-domain strain array for frequencies ``f``.

    Parameters are detector-frame chirp mass ``Mc`` in solar masses, symmetric
    mass ratio ``eta``, and luminosity distance ``dL`` in Mpc.  Frequencies are
    in Hz.  Values below ``f_low`` and at/above the high-frequency cutoff are
    returned as exact zeros.
    """
    f_arr = np.asarray(f, dtype=float)
    scalar_input = f_arr.ndim == 0
    f_flat = np.atleast_1d(f_arr).astype(float, copy=False).ravel()
    out = np.zeros(f_flat.shape, dtype=complex)

    eta = float(eta)
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if Mc <= 0.0 or dL <= 0.0:
        raise ValueError("Mc and dL must be positive")

    M_sec = float(Mc) * MSUN_SEC / eta ** (3.0 / 5.0)
    Mc_sec = float(Mc) * MSUN_SEC
    dL_sec = float(dL) * MPC_SEC

    f_isco = 1.0 / (_PI * 6.0 ** 1.5 * M_sec)
    f_cut = float(fmax_over_fisco) * f_isco
    sigma = float(sigma_taper_over_fisco) * f_isco

    active = np.isfinite(f_flat) & (f_flat >= float(f_low)) & (f_flat < f_cut) & (f_flat > 0.0)
    if not np.any(active):
        return out[0] if scalar_input else out.reshape(f_arr.shape)

    f_act = f_flat[active]
    x = (_PI * M_sec * f_act) ** (2.0 / 3.0)
    x_cut = (_PI * M_sec * f_cut) ** (2.0 / 3.0)

    hhat = _hhat_22(x, eta, lambda_RG)
    hhat_abs = np.abs(hhat)
    mode_phase = hhat / np.maximum(hhat_abs, np.finfo(float).tiny)

    t_to_cut, phi_to_cut = _phase_integrals_to_cut(x, x_cut, M_sec, eta, lambda_RG)
    spa_phase = (
        2.0 * _PI * f_act * float(tc)
        - float(phic)
        - _PI / 4.0
        + 2.0 * phi_to_cut
        - 2.0 * _PI * f_act * t_to_cut
    )

    amp0 = (
        np.sqrt(5.0 / 24.0)
        * _PI ** (-2.0 / 3.0)
        * Mc_sec ** (5.0 / 6.0)
        / dL_sec
        * f_act ** (-7.0 / 6.0)
    )
    if phase_only:
        amp_corr = 1.0
    else:
        _, denergy_dx = _energy_and_derivative(x, eta)
        amp_corr = np.sqrt(np.maximum(-2.0 * denergy_dx / eta, 0.0))

    if sigma > 0.0:
        taper = 1.0 / (1.0 + np.exp((f_act - f_cut) / sigma))
    else:
        taper = np.ones_like(f_act)

    out[active] = taper * amp0 * amp_corr * mode_phase * np.exp(1.0j * spa_phase)
    return out[0] if scalar_input else out.reshape(f_arr.shape)
