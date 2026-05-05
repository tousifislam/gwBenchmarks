"""Standalone finite-size balance-law SPA waveform prototype.

The implementation follows the compact source packet for arXiv:2410.00294:
finite-size effects enter through the binding energy, flux to infinity, and
absorbed flux, and the frequency-domain waveform is built from balance law SPA.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0
_PI = np.pi


def _spin_variables(eta, chi1, chi2):
    eta = float(eta)
    if eta <= 0.0 or eta > 0.25 + 1e-14:
        raise ValueError("eta must satisfy 0 < eta <= 0.25")
    delta = np.sqrt(max(0.0, 1.0 - 4.0 * eta))
    chi_s = 0.5 * (float(chi1) + float(chi2))
    chi_a = 0.5 * (float(chi1) - float(chi2))
    return delta, chi_s, chi_a


def _energy_coefficients(eta, chi1, chi2, kappa_s, kappa_a):
    """Return E_SS_SIM(v)=ess0+ess2 v^2 and constant E_SSS_SIM."""
    eta = float(eta)
    ks = float(kappa_s)
    ka = float(kappa_a)
    delta, chi_s, chi_a = _spin_variables(eta, chi1, chi2)

    ess0 = (
        chi_a * chi_s * (-delta**2 * ka / 2.0 - delta * ks - ka / 2.0)
        + chi_s**2 * (-delta**2 * ks / 4.0 - delta * ka / 2.0 - ks / 4.0)
        + chi_a**2 * (-delta * ka / 2.0 + eta * ks - ks / 2.0)
    )
    ess2 = (
        chi_a**2
        * (
            25.0 * delta * eta * ka / 12.0
            - 35.0 * delta * ka / 12.0
            - 5.0 * eta**2 * ks / 6.0
            + 95.0 * eta * ks / 12.0
            - 35.0 * ks / 12.0
        )
        + chi_a
        * chi_s
        * (
            25.0 * delta * eta * ks / 6.0
            - 35.0 * delta * ks / 6.0
            - 5.0 * eta**2 * ka / 3.0
            + 95.0 * eta * ka / 6.0
            - 35.0 * ka / 6.0
        )
        + chi_s**2
        * (
            25.0 * delta * eta * ka / 12.0
            - 35.0 * delta * ka / 12.0
            - 5.0 * eta**2 * ks / 6.0
            + 95.0 * eta * ks / 12.0
            - 35.0 * ks / 12.0
        )
    )
    esss = (
        chi_a
        * chi_s**2
        * (
            -2.0 * delta**2 * eta * ka
            - 5.0 * delta**2 * ka
            + 6.0 * delta * eta * ks
            - 6.0 * delta * ks
            - ka
        )
        + chi_s**3
        * (
            -delta**3 * ka
            - delta**2 * eta * ks
            - 9.0 * delta**2 * ks / 4.0
            - delta * ka
            + ks / 4.0
        )
        + chi_a**2
        * chi_s
        * (-6.0 * delta * ka + 4.0 * eta**2 * ks + 12.0 * eta * ks - 6.0 * ks)
        + chi_a**3
        * (-2.0 * delta * eta * ks - 2.0 * delta * ks + 2.0 * eta * ka - 2.0 * ka)
    )
    return ess0, ess2, esss


def _flux_coefficients(eta, chi1, chi2, kappa_s, kappa_a):
    """Return F_SS_SIM(v)=fss0+fss2 v^2+fss3 v^3 and F_SSS_SIM."""
    eta = float(eta)
    ks = float(kappa_s)
    ka = float(kappa_a)
    delta, chi_s, chi_a = _spin_variables(eta, chi1, chi2)

    fss0 = (
        chi_a * chi_s * (delta**2 * ka + 2.0 * delta * ks + ka)
        + chi_a**2 * (delta * ka - 2.0 * eta * ks + ks)
        + chi_s**2 * (delta * ka - 2.0 * eta * ks + ks)
    )
    fss2 = (
        chi_a**2
        * (
            -127.0 * delta * eta * ka / 16.0
            + delta * ka / 14.0
            + 43.0 * eta**2 * ks / 4.0
            - 905.0 * eta * ks / 112.0
            + ks / 14.0
        )
        + chi_a
        * chi_s
        * (
            -127.0 * delta * eta * ks / 8.0
            + delta * ks / 7.0
            + 43.0 * eta**2 * ka / 2.0
            - 905.0 * eta * ka / 56.0
            + ka / 7.0
        )
        + chi_s**2
        * (
            -127.0 * delta * eta * ka / 16.0
            + delta * ka / 14.0
            + 43.0 * eta**2 * ks / 4.0
            - 905.0 * eta * ks / 112.0
            + ks / 14.0
        )
    )
    fss3 = (
        chi_a**2 * (4.0 * _PI * delta * ka - 8.0 * _PI * eta * ks + 4.0 * _PI * ks)
        + chi_a
        * chi_s
        * (8.0 * _PI * delta * ks - 16.0 * _PI * eta * ka + 8.0 * _PI * ka)
        + chi_s**2 * (4.0 * _PI * delta * ka - 8.0 * _PI * eta * ks + 4.0 * _PI * ks)
    )
    fsss = (
        chi_a
        * chi_s**2
        * (
            13.0 * delta**2 * eta * ka / 3.0
            + 27.0 * delta**2 * ka / 16.0
            + 4.0 * delta * eta * ks / 3.0
            + 15.0 * delta * ks / 8.0
            + 3.0 * ka / 16.0
        )
        + chi_a**2
        * chi_s
        * (
            95.0 * delta * eta * ka / 12.0
            + 15.0 * delta * ka / 8.0
            - 26.0 * eta**2 * ks / 3.0
            + 25.0 * eta * ks / 6.0
            + 15.0 * ks / 8.0
        )
        + chi_s**3
        * (
            -7.0 * delta * eta * ka / 4.0
            + 5.0 * delta * ka / 8.0
            - 26.0 * eta**2 * ks / 3.0
            - 3.0 * eta * ks
            + 5.0 * ks / 8.0
        )
        + chi_a**3
        * (
            29.0 * delta * eta * ks / 6.0
            + 5.0 * delta * ks / 8.0
            + 43.0 * eta * ka / 12.0
            + 5.0 * ka / 8.0
        )
    )
    return fss0, fss2, fss3, fsss


def _de_dv(v, eta, chi1, chi2, kappa_s, kappa_a):
    v = np.asarray(v, dtype=float)
    ess0, ess2, esss = _energy_coefficients(eta, chi1, chi2, kappa_s, kappa_a)
    return -0.5 * eta * (2.0 * v + 6.0 * ess0 * v**5 + 8.0 * ess2 * v**7 + 9.0 * esss * v**8)


def _f_infty(v, eta, chi1, chi2, kappa_s, kappa_a, Lambda_tilde):
    v = np.asarray(v, dtype=float)
    fss0, fss2, fss3, fsss = _flux_coefficients(eta, chi1, chi2, kappa_s, kappa_a)
    fss = fss0 + fss2 * v**2 + fss3 * v**3
    bracket = 1.0 + fss * v**4 + fsss * v**7 + (39.0 / 8.0) * float(Lambda_tilde) * v**10
    return (32.0 / 5.0) * eta**2 * v**10 * bracket


def _dotM(v, eta, chi1, chi2, H0, H1E, H1B, H1E_bar, H1B_bar):
    v = np.asarray(v, dtype=float)
    eta = float(eta)
    delta, chi_s, chi_a = _spin_variables(eta, chi1, chi2)
    H0 = float(H0)
    H1E = float(H1E)
    H1B = float(H1B)
    H1E_bar = float(H1E_bar)
    H1B_bar = float(H1B_bar)

    leading_spin = 0.5 * (9.0 * H1E_bar * eta**2 * chi_a + 9.0 * H1E * eta**2 * chi_s) * v**15
    next_spin = 0.5 * (
        (
            9.0 * H1B * eta**2
            + 45.0 * H1E * eta**2 / 2.0
            + 9.0 * H1E_bar * delta * eta**2 / 2.0
            - 27.0 * H1E * eta**3
        )
        * chi_s
        + (
            9.0 * H1B_bar * eta**2
            + 45.0 * H1E_bar * eta**2 / 2.0
            + 9.0 * H1E * delta * eta**2 / 2.0
            - 27.0 * H1E_bar * eta**3
        )
        * chi_a
    ) * v**17
    spin_independent = 18.0 * H0 * eta**2 * v**18
    return leading_spin + next_spin + spin_independent


def _total_flux(v, eta, chi1, chi2, kappa_s, kappa_a, H0, Lambda_tilde, H1E, H1B, H1E_bar, H1B_bar):
    return _f_infty(v, eta, chi1, chi2, kappa_s, kappa_a, Lambda_tilde) + _dotM(
        v, eta, chi1, chi2, H0, H1E, H1B, H1E_bar, H1B_bar
    )


def _dt_dv(v, M_sec, eta, chi1, chi2, kappa_s, kappa_a, H0, Lambda_tilde, H1E, H1B, H1E_bar, H1B_bar):
    flux = _total_flux(v, eta, chi1, chi2, kappa_s, kappa_a, H0, Lambda_tilde, H1E, H1B, H1E_bar, H1B_bar)
    return -M_sec * _de_dv(v, eta, chi1, chi2, kappa_s, kappa_a) / flux


def _phase_integrals_to_cut(v_eval, v_cut, M_sec, eta, chi1, chi2, kappa_s, kappa_a, H0, Lambda_tilde, H1E, H1B, H1E_bar, H1B_bar):
    """Return int_v^vcut dt/dv dv and int_v^vcut 2 Omega dt/dv dv."""
    v_eval = np.asarray(v_eval, dtype=float)
    if v_eval.size == 0:
        return v_eval.copy(), v_eval.copy()

    v_min = float(np.min(v_eval))
    if not (v_cut > v_min):
        return np.zeros_like(v_eval), np.zeros_like(v_eval)

    dt_newt = lambda v: 5.0 * M_sec / (32.0 * eta) * v ** (-9.0)
    phase_newt = lambda v: 5.0 / (16.0 * eta) * v ** (-6.0)

    log_span = np.log(v_cut / v_min)
    n_grid = int(max(10000, 8000 * log_span + 4 * v_eval.size))
    n_grid = min(n_grid, 160000)
    base = np.geomspace(v_min, v_cut, n_grid)
    grid = np.unique(np.concatenate((base, v_eval, np.array([v_cut]))))
    grid.sort()

    dt = _dt_dv(grid, M_sec, eta, chi1, chi2, kappa_s, kappa_a, H0, Lambda_tilde, H1E, H1B, H1E_bar, H1B_bar)
    gw_phase_integrand = (2.0 * grid**3 / M_sec) * dt

    t_corr_cum = cumulative_trapezoid(dt - dt_newt(grid), grid, initial=0.0)
    p_corr_cum = cumulative_trapezoid(gw_phase_integrand - phase_newt(grid), grid, initial=0.0)
    t_corr = t_corr_cum[-1] - np.interp(v_eval, grid, t_corr_cum)
    p_corr = p_corr_cum[-1] - np.interp(v_eval, grid, p_corr_cum)

    t_newt_to_cut = 5.0 * M_sec / (256.0 * eta) * (v_eval ** (-8.0) - v_cut ** (-8.0))
    p_newt_to_cut = 1.0 / (16.0 * eta) * (v_eval ** (-5.0) - v_cut ** (-5.0))
    return t_newt_to_cut + t_corr, p_newt_to_cut + p_corr


def h_of_f(
    f,
    Mc,
    eta,
    dL,
    chi1=0.0,
    chi2=0.0,
    tc=0.0,
    phic=0.0,
    kappa_s=1.0,
    kappa_a=0.0,
    H0=0.0,
    Lambda_tilde=0.0,
    H1E=0.0,
    H1B=0.0,
    H1E_bar=0.0,
    H1B_bar=0.0,
    f_low=20.0,
    fmax_over_fisco=1.0,
    sigma_taper_over_fisco=0.01,
    phase_only=False,
):
    """Return the complex finite-size SPA waveform on the input frequency grid.

    ``Mc`` is the detector-frame chirp mass in solar masses, ``eta`` is the
    symmetric mass ratio, ``dL`` is in Mpc, and frequencies are in Hz.
    """
    f_arr = np.asarray(f, dtype=float)
    scalar_input = f_arr.ndim == 0
    f_flat = np.atleast_1d(f_arr).astype(float, copy=False).ravel()
    out = np.zeros(f_flat.shape, dtype=complex)

    eta = float(eta)
    _spin_variables(eta, chi1, chi2)
    if float(Mc) <= 0.0 or float(dL) <= 0.0:
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
    v = (_PI * M_sec * f_act) ** (1.0 / 3.0)
    v_cut = (_PI * M_sec * f_cut) ** (1.0 / 3.0)

    t_to_cut, phase_to_cut = _phase_integrals_to_cut(
        v,
        v_cut,
        M_sec,
        eta,
        chi1,
        chi2,
        kappa_s,
        kappa_a,
        H0,
        Lambda_tilde,
        H1E,
        H1B,
        H1E_bar,
        H1B_bar,
    )
    psi = 2.0 * _PI * f_act * float(tc) - float(phic) - _PI / 4.0 + phase_to_cut - 2.0 * _PI * f_act * t_to_cut

    amp_newtonian = (
        np.sqrt(5.0 / 24.0)
        * Mc_sec ** (5.0 / 6.0)
        * _PI ** (-2.0 / 3.0)
        * f_act ** (-7.0 / 6.0)
        / dL_sec
    )
    if phase_only:
        amp = amp_newtonian
    else:
        dt_dv = _dt_dv(
            v,
            M_sec,
            eta,
            chi1,
            chi2,
            kappa_s,
            kappa_a,
            H0,
            Lambda_tilde,
            H1E,
            H1B,
            H1E_bar,
            H1B_bar,
        )
        dfdt = (3.0 * v**2 / (_PI * M_sec)) / dt_dv
        dfdt_newtonian = 96.0 * eta * v**11 / (5.0 * _PI * M_sec**2)
        amp = amp_newtonian * np.sqrt(np.maximum(dfdt_newtonian / dfdt, 0.0))

    if sigma > 0.0:
        taper = 1.0 / (1.0 + np.exp((f_act - f_cut) / sigma))
    else:
        taper = np.ones_like(f_act)

    out[active] = taper * amp * np.exp(-1.0j * psi)
    return out[0] if scalar_input else out.reshape(f_arr.shape)
