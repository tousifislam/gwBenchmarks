"""Frequency-domain inspiral waveform with finite-size corrections from
arXiv:2410.00294, Appendix A.

The SPA chirp rate is built from the balance law

    -F_infty(v) - dotM(v) = dE/dt,

with a Newtonian point-particle baseline and the spin-induced quadrupole
(``E_SS_SIM``, ``F_SS_SIM``) and cubic (``E_SSS_SIM``, ``F_SSS_SIM``) pieces,
the effective leading Love-number flux correction ``(39/8) Lambda_tilde v^10``,
and the absorbed-flux ``H0/H1`` terms (with the doubled BBH convention
``18 H0 eta^2 v^18``). The waveform amplitude is the standard restricted
Newtonian SPA, rescaled by ``sqrt(dfdt_N / dfdt)``.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458


def _binding_energy(v, eta, delta, chi_s, chi_a, kappa_s, kappa_a):
    """Return (e(v), de/dv) per the compact-benchmark binding energy.

    e(v) = -eta v^2/2 [1 + E_SS_SIM(v) v^4 + E_SSS_SIM v^7]
    """
    # E_SS_SIM(v) = E_SS_v0 + E_SS_v2 * v^2
    E_SS_v0 = (
        chi_a * chi_s * (-(delta ** 2) * kappa_a / 2.0 - delta * kappa_s - kappa_a / 2.0)
        + chi_s ** 2 * (-(delta ** 2) * kappa_s / 4.0 - delta * kappa_a / 2.0 - kappa_s / 4.0)
        + chi_a ** 2 * (-delta * kappa_a / 2.0 + eta * kappa_s - kappa_s / 2.0)
    )
    E_SS_v2 = (
        chi_a ** 2
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta ** 2 * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
        + chi_a
        * chi_s
        * (
            25.0 * delta * eta * kappa_s / 6.0
            - 35.0 * delta * kappa_s / 6.0
            - 5.0 * eta ** 2 * kappa_a / 3.0
            + 95.0 * eta * kappa_a / 6.0
            - 35.0 * kappa_a / 6.0
        )
        + chi_s ** 2
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta ** 2 * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
    )

    E_SSS = (
        chi_a
        * chi_s ** 2
        * (
            -2.0 * delta ** 2 * eta * kappa_a
            - 5.0 * delta ** 2 * kappa_a
            + 6.0 * delta * eta * kappa_s
            - 6.0 * delta * kappa_s
            - kappa_a
        )
        + chi_s ** 3
        * (
            -(delta ** 3) * kappa_a
            - delta ** 2 * eta * kappa_s
            - 9.0 * delta ** 2 * kappa_s / 4.0
            - delta * kappa_a
            + kappa_s / 4.0
        )
        + chi_a ** 2
        * chi_s
        * (
            -6.0 * delta * kappa_a
            + 4.0 * eta ** 2 * kappa_s
            + 12.0 * eta * kappa_s
            - 6.0 * kappa_s
        )
        + chi_a ** 3
        * (
            -2.0 * delta * eta * kappa_s
            - 2.0 * delta * kappa_s
            + 2.0 * eta * kappa_a
            - 2.0 * kappa_a
        )
    )

    # bracket(v) = 1 + E_SS_v0 v^4 + E_SS_v2 v^6 + E_SSS v^7
    bracket = 1.0 + E_SS_v0 * v ** 4 + E_SS_v2 * v ** 6 + E_SSS * v ** 7
    d_bracket_dv = 4.0 * E_SS_v0 * v ** 3 + 6.0 * E_SS_v2 * v ** 5 + 7.0 * E_SSS * v ** 6

    e = -eta * v ** 2 / 2.0 * bracket
    de_dv = -eta * v * bracket - eta * v ** 2 / 2.0 * d_bracket_dv
    return e, de_dv


def _flux_infinity(v, eta, delta, chi_s, chi_a, kappa_s, kappa_a, Lambda_tilde):
    """F_infty(v) = (32/5) eta^2 v^10 [1 + F_SS_SIM v^4 + F_SSS_SIM v^7
    + (39/8) Lambda_tilde v^10]."""
    F_SS_v0 = (
        chi_a * chi_s * (delta ** 2 * kappa_a + 2.0 * delta * kappa_s + kappa_a)
        + chi_a ** 2 * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
        + chi_s ** 2 * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
    )
    F_SS_v2 = (
        chi_a ** 2
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta ** 2 * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
        + chi_a
        * chi_s
        * (
            -127.0 * delta * eta * kappa_s / 8.0
            + delta * kappa_s / 7.0
            + 43.0 * eta ** 2 * kappa_a / 2.0
            - 905.0 * eta * kappa_a / 56.0
            + kappa_a / 7.0
        )
        + chi_s ** 2
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta ** 2 * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
    )
    F_SS_v3 = (
        chi_a ** 2
        * (4.0 * np.pi * delta * kappa_a - 8.0 * np.pi * eta * kappa_s + 4.0 * np.pi * kappa_s)
        + chi_a
        * chi_s
        * (8.0 * np.pi * delta * kappa_s - 16.0 * np.pi * eta * kappa_a + 8.0 * np.pi * kappa_a)
        + chi_s ** 2
        * (4.0 * np.pi * delta * kappa_a - 8.0 * np.pi * eta * kappa_s + 4.0 * np.pi * kappa_s)
    )

    F_SSS = (
        chi_a
        * chi_s ** 2
        * (
            13.0 * delta ** 2 * eta * kappa_a / 3.0
            + 27.0 * delta ** 2 * kappa_a / 16.0
            + 4.0 * delta * eta * kappa_s / 3.0
            + 15.0 * delta * kappa_s / 8.0
            + 3.0 * kappa_a / 16.0
        )
        + chi_a ** 2
        * chi_s
        * (
            95.0 * delta * eta * kappa_a / 12.0
            + 15.0 * delta * kappa_a / 8.0
            - 26.0 * eta ** 2 * kappa_s / 3.0
            + 25.0 * eta * kappa_s / 6.0
            + 15.0 * kappa_s / 8.0
        )
        + chi_s ** 3
        * (
            -7.0 * delta * eta * kappa_a / 4.0
            + 5.0 * delta * kappa_a / 8.0
            - 26.0 * eta ** 2 * kappa_s / 3.0
            - 3.0 * eta * kappa_s
            + 5.0 * kappa_s / 8.0
        )
        + chi_a ** 3
        * (
            29.0 * delta * eta * kappa_s / 6.0
            + 5.0 * delta * kappa_s / 8.0
            + 43.0 * eta * kappa_a / 12.0
            + 5.0 * kappa_a / 8.0
        )
    )

    # F_SS_SIM(v) = F_SS_v0 + F_SS_v2 v^2 + F_SS_v3 v^3
    bracket = (
        1.0
        + (F_SS_v0 + F_SS_v2 * v ** 2 + F_SS_v3 * v ** 3) * v ** 4
        + F_SSS * v ** 7
        + (39.0 / 8.0) * Lambda_tilde * v ** 10
    )
    return (32.0 / 5.0) * eta ** 2 * v ** 10 * bracket


def _absorbed_flux(v, eta, delta, chi_s, chi_a, H0, H1E, H1B, H1E_bar, H1B_bar):
    """dotM(v) per the doubled BBH convention (18 H0 eta^2 v^18)."""
    term_v15 = 0.5 * (9.0 * H1E_bar * eta ** 2 * chi_a + 9.0 * H1E * eta ** 2 * chi_s)
    term_v17 = 0.5 * (
        (
            9.0 * H1B * eta ** 2
            + 45.0 * H1E * eta ** 2 / 2.0
            + 9.0 * H1E_bar * delta * eta ** 2 / 2.0
            - 27.0 * H1E * eta ** 3
        )
        * chi_s
        + (
            9.0 * H1B_bar * eta ** 2
            + 45.0 * H1E_bar * eta ** 2 / 2.0
            + 9.0 * H1E * delta * eta ** 2 / 2.0
            - 27.0 * H1E_bar * eta ** 3
        )
        * chi_a
    )
    term_v18 = 18.0 * H0 * eta ** 2
    return term_v15 * v ** 15 + term_v17 * v ** 17 + term_v18 * v ** 18


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
    """Restricted-SPA inspiral waveform with finite-size corrections.

    LIGO-style sign convention: ``h(f) = A(f) exp[-i psi(f)]``.
    The SPA phase is built from the balance-law integrals,

        psi(f) = 2 pi f t(v_f) - phi_GW(v_f) - pi/4 + 2 pi f tc - phic,

    with ``phi_GW = 2 phi_orb`` and integration constants set so that
    ``t(v_high) = phi_GW(v_high) = 0`` at the upper grid edge ``v_high =
    (pi M_sec f_cut)^{1/3}``. Constant/linear-in-f offsets from this choice
    are absorbed into ``tc`` and ``phic``.
    """
    f = np.asarray(f, dtype=float)
    eta = float(eta)

    M_sec = Mc * MSUN_SEC / eta ** (3.0 / 5.0)
    Mc_sec = Mc * MSUN_SEC
    dL_sec = dL * MPC_SEC

    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0

    f_isco = 1.0 / (np.pi * 6.0 ** 1.5 * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    h = np.zeros_like(f, dtype=complex)
    if f_cut <= f_low:
        return h

    v_low = (np.pi * M_sec * f_low) ** (1.0 / 3.0)
    v_high = (np.pi * M_sec * f_cut) ** (1.0 / 3.0)
    if v_high <= v_low:
        return h

    N_grid = 8192
    v_grid = np.logspace(np.log10(v_low), np.log10(v_high), N_grid)

    _, de_dv_grid = _binding_energy(v_grid, eta, delta, chi_s, chi_a, kappa_s, kappa_a)
    F_inf_grid = _flux_infinity(
        v_grid, eta, delta, chi_s, chi_a, kappa_s, kappa_a, Lambda_tilde
    )
    dotM_grid = _absorbed_flux(
        v_grid, eta, delta, chi_s, chi_a, H0, H1E, H1B, H1E_bar, H1B_bar
    )
    F_tot_grid = F_inf_grid + dotM_grid

    # Energy-balance chirp rate: dt/dv = -M_sec de/dv / F_tot
    dt_dv_grid = -M_sec * de_dv_grid / F_tot_grid

    # phi_GW = 2 phi_orb, dphi_GW/dv = 2 Omega dt/dv = 2 (v^3/M_sec) dt/dv
    dphiGW_dv_grid = 2.0 * (v_grid ** 3 / M_sec) * dt_dv_grid

    phiGW_cum = cumulative_trapezoid(dphiGW_dv_grid, v_grid, initial=0.0)
    t_cum = cumulative_trapezoid(dt_dv_grid, v_grid, initial=0.0)

    # Set integration constants so t = phi_GW = 0 at the high-frequency edge
    phiGW_rel_grid = phiGW_cum - phiGW_cum[-1]
    t_rel_grid = t_cum - t_cum[-1]

    phiGW_interp = CubicSpline(v_grid, phiGW_rel_grid)
    t_interp = CubicSpline(v_grid, t_rel_grid)

    in_band = (f >= f_low) & (f < f_cut)
    f_in = f[in_band]
    if f_in.size == 0:
        return h

    v_f = (np.pi * M_sec * f_in) ** (1.0 / 3.0)
    v_f_clipped = np.clip(v_f, v_grid[0], v_grid[-1])

    _, de_dv_at = _binding_energy(v_f, eta, delta, chi_s, chi_a, kappa_s, kappa_a)
    F_inf_at = _flux_infinity(
        v_f, eta, delta, chi_s, chi_a, kappa_s, kappa_a, Lambda_tilde
    )
    dotM_at = _absorbed_flux(
        v_f, eta, delta, chi_s, chi_a, H0, H1E, H1B, H1E_bar, H1B_bar
    )
    F_tot_at = F_inf_at + dotM_at
    dt_dv_at = -M_sec * de_dv_at / F_tot_at
    dfdt_at = (3.0 * v_f ** 2 / (np.pi * M_sec)) / dt_dv_at

    A_N = (
        np.sqrt(5.0 / 24.0)
        * Mc_sec ** (5.0 / 6.0)
        * np.pi ** (-2.0 / 3.0)
        * f_in ** (-7.0 / 6.0)
        / dL_sec
    )
    dfdt_N = 96.0 * eta * v_f ** 11 / (5.0 * np.pi * M_sec ** 2)
    A = A_N * np.sqrt(dfdt_N / dfdt_at)

    phiGW_at = phiGW_interp(v_f_clipped)
    t_at = t_interp(v_f_clipped)

    psi = (
        2.0 * np.pi * f_in * t_at
        - phiGW_at
        - np.pi / 4.0
        + 2.0 * np.pi * f_in * tc
        - phic
    )

    W = 1.0 / (1.0 + np.exp((f_in - f_cut) / sigma))

    if phase_only:
        h[in_band] = np.exp(-1j * psi)
    else:
        h[in_band] = A * np.exp(-1j * psi) * W

    return h
