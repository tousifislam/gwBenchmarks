"""Standalone finite-size balance-law SPA waveform candidate.

The implementation follows the compact source packet for the finite-size
benchmark. It uses the Newtonian restricted SPA amplitude corrected by the
balance-law chirp-rate ratio and a numerically integrated SPA phase anchored at
the high-frequency cutoff.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0
PI = np.pi
TINY = np.finfo(float).tiny


def _spin_symmetric_asymmetric(chi1: float, chi2: float) -> tuple[float, float]:
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    return chi_s, chi_a


def _finite_size_energy_and_derivative(
    v: np.ndarray,
    eta: float,
    chi1: float,
    chi2: float,
    kappa_s: float,
    kappa_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return e(v)=E/M and de/dv using the compact packet formulas."""
    delta = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
    chi_s, chi_a = _spin_symmetric_asymmetric(chi1, chi2)

    eta2 = eta * eta
    delta2 = delta * delta
    chi_s2 = chi_s * chi_s
    chi_a2 = chi_a * chi_a
    chi_a_chi_s = chi_a * chi_s

    e_ss0 = (
        chi_a_chi_s * (-0.5 * delta2 * kappa_a - delta * kappa_s - 0.5 * kappa_a)
        + chi_s2 * (-0.25 * delta2 * kappa_s - 0.5 * delta * kappa_a - 0.25 * kappa_s)
        + chi_a2 * (-0.5 * delta * kappa_a + eta * kappa_s - 0.5 * kappa_s)
    )
    e_ss2 = (
        chi_a2
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta2 * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
        + chi_a_chi_s
        * (
            25.0 * delta * eta * kappa_s / 6.0
            - 35.0 * delta * kappa_s / 6.0
            - 5.0 * eta2 * kappa_a / 3.0
            + 95.0 * eta * kappa_a / 6.0
            - 35.0 * kappa_a / 6.0
        )
        + chi_s2
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta2 * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
    )
    e_sss = (
        chi_a * chi_s2
        * (
            -2.0 * delta2 * eta * kappa_a
            - 5.0 * delta2 * kappa_a
            + 6.0 * delta * eta * kappa_s
            - 6.0 * delta * kappa_s
            - kappa_a
        )
        + chi_s * chi_s2
        * (
            -delta * delta2 * kappa_a
            - delta2 * eta * kappa_s
            - 9.0 * delta2 * kappa_s / 4.0
            - delta * kappa_a
            + 0.25 * kappa_s
        )
        + chi_a2 * chi_s
        * (
            -6.0 * delta * kappa_a
            + 4.0 * eta2 * kappa_s
            + 12.0 * eta * kappa_s
            - 6.0 * kappa_s
        )
        + chi_a2 * chi_a
        * (
            -2.0 * delta * eta * kappa_s
            - 2.0 * delta * kappa_s
            + 2.0 * eta * kappa_a
            - 2.0 * kappa_a
        )
    )

    e = -0.5 * eta * (
        v**2 + e_ss0 * v**6 + e_ss2 * v**8 + e_sss * v**9
    )
    de_dv = -0.5 * eta * (
        2.0 * v + 6.0 * e_ss0 * v**5 + 8.0 * e_ss2 * v**7 + 9.0 * e_sss * v**8
    )
    return e, de_dv


def _finite_size_fluxes(
    v: np.ndarray,
    eta: float,
    chi1: float,
    chi2: float,
    kappa_s: float,
    kappa_a: float,
    Lambda_tilde: float,
    H0: float,
    H1E: float,
    H1B: float,
    H1E_bar: float,
    H1B_bar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return F_infty and dotM from the compact packet formulas."""
    delta = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
    chi_s, chi_a = _spin_symmetric_asymmetric(chi1, chi2)

    eta2 = eta * eta
    eta3 = eta2 * eta
    delta2 = delta * delta
    chi_s2 = chi_s * chi_s
    chi_a2 = chi_a * chi_a
    chi_a_chi_s = chi_a * chi_s

    f_ss0 = (
        chi_a_chi_s * (delta2 * kappa_a + 2.0 * delta * kappa_s + kappa_a)
        + chi_a2 * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
        + chi_s2 * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
    )
    f_ss2 = (
        chi_a2
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta2 * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
        + chi_a_chi_s
        * (
            -127.0 * delta * eta * kappa_s / 8.0
            + delta * kappa_s / 7.0
            + 43.0 * eta2 * kappa_a / 2.0
            - 905.0 * eta * kappa_a / 56.0
            + kappa_a / 7.0
        )
        + chi_s2
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta2 * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
    )
    f_ss3 = (
        chi_a2
        * (
            4.0 * PI * delta * kappa_a
            - 8.0 * PI * eta * kappa_s
            + 4.0 * PI * kappa_s
        )
        + chi_a_chi_s
        * (
            8.0 * PI * delta * kappa_s
            - 16.0 * PI * eta * kappa_a
            + 8.0 * PI * kappa_a
        )
        + chi_s2
        * (
            4.0 * PI * delta * kappa_a
            - 8.0 * PI * eta * kappa_s
            + 4.0 * PI * kappa_s
        )
    )
    f_sss = (
        chi_a * chi_s2
        * (
            13.0 * delta2 * eta * kappa_a / 3.0
            + 27.0 * delta2 * kappa_a / 16.0
            + 4.0 * delta * eta * kappa_s / 3.0
            + 15.0 * delta * kappa_s / 8.0
            + 3.0 * kappa_a / 16.0
        )
        + chi_a2 * chi_s
        * (
            95.0 * delta * eta * kappa_a / 12.0
            + 15.0 * delta * kappa_a / 8.0
            - 26.0 * eta2 * kappa_s / 3.0
            + 25.0 * eta * kappa_s / 6.0
            + 15.0 * kappa_s / 8.0
        )
        + chi_s * chi_s2
        * (
            -7.0 * delta * eta * kappa_a / 4.0
            + 5.0 * delta * kappa_a / 8.0
            - 26.0 * eta2 * kappa_s / 3.0
            - 3.0 * eta * kappa_s
            + 5.0 * kappa_s / 8.0
        )
        + chi_a2 * chi_a
        * (
            29.0 * delta * eta * kappa_s / 6.0
            + 5.0 * delta * kappa_s / 8.0
            + 43.0 * eta * kappa_a / 12.0
            + 5.0 * kappa_a / 8.0
        )
    )

    finfty = (32.0 / 5.0) * eta2 * v**10 * (
        1.0
        + f_ss0 * v**4
        + f_ss2 * v**6
        + (f_ss3 + f_sss) * v**7
        + (39.0 / 8.0) * Lambda_tilde * v**10
    )

    dotM15 = 4.5 * eta2 * (H1E_bar * chi_a + H1E * chi_s) * v**15
    dotM17 = 0.5 * (
        (
            9.0 * H1B * eta2
            + 22.5 * H1E * eta2
            + 4.5 * H1E_bar * delta * eta2
            - 27.0 * H1E * eta3
        )
        * chi_s
        + (
            9.0 * H1B_bar * eta2
            + 22.5 * H1E_bar * eta2
            + 4.5 * H1E * delta * eta2
            - 27.0 * H1E_bar * eta3
        )
        * chi_a
    ) * v**17
    dotM18 = 18.0 * H0 * eta2 * v**18
    dotM = dotM15 + dotM17 + dotM18
    return finfty, dotM


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
    """Return the complex finite-size inspiral strain on the input grid."""
    if not np.isfinite(Mc) or Mc <= 0.0:
        raise ValueError("Mc must be a positive finite chirp mass in solar masses")
    if not np.isfinite(eta) or eta <= 0.0 or eta > 0.25:
        raise ValueError("eta must lie in the physical interval (0, 0.25]")
    if not np.isfinite(dL) or dL <= 0.0:
        raise ValueError("dL must be a positive finite luminosity distance in Mpc")

    f_arr = np.asarray(f, dtype=float)
    f_flat = f_arr.reshape(-1)
    out = np.zeros(f_flat.shape, dtype=np.complex128)

    Mc_sec = Mc * MSUN_SEC
    M_sec = Mc_sec / eta ** (3.0 / 5.0)
    dL_sec = dL * MPC_SEC

    f_isco = 1.0 / (PI * (6.0 ** 1.5) * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    valid = np.isfinite(f_flat) & (f_flat >= f_low) & (f_flat < f_cut)
    if not np.any(valid):
        return out.reshape(f_arr.shape)

    f_valid = f_flat[valid]
    f_nodes = np.unique(np.concatenate([f_valid, np.array([f_cut], dtype=float)]))
    f_eval = f_nodes[:-1]

    v_nodes = (PI * M_sec * f_nodes) ** (1.0 / 3.0)
    v_eval = v_nodes[:-1]

    _, de_dv = _finite_size_energy_and_derivative(
        v_nodes,
        eta,
        chi1,
        chi2,
        kappa_s,
        kappa_a,
    )
    finfty, dotM = _finite_size_fluxes(
        v_nodes,
        eta,
        chi1,
        chi2,
        kappa_s,
        kappa_a,
        Lambda_tilde,
        H0,
        H1E,
        H1B,
        H1E_bar,
        H1B_bar,
    )

    F_tot = np.maximum(finfty + dotM, TINY)
    dt_dv = np.maximum(-M_sec * de_dv / F_tot, TINY)
    df_dv = np.maximum(3.0 * v_nodes**2 / (PI * M_sec), TINY)
    dt_df = dt_dv / df_dv
    dfdt = df_dv / dt_dv
    dfdt_N = 96.0 * eta * v_nodes**11 / (5.0 * PI * M_sec**2)

    dphi_df = 2.0 * PI * f_nodes * dt_df
    time_to_ref = cumulative_trapezoid(dt_df, f_nodes, initial=0.0)
    phase_to_ref = cumulative_trapezoid(dphi_df, f_nodes, initial=0.0)
    time_to_ref = time_to_ref[-1] - time_to_ref[:-1]
    phase_to_ref = phase_to_ref[-1] - phase_to_ref[:-1]

    time_valid = np.interp(f_eval, f_eval, time_to_ref)
    phase_valid = np.interp(f_eval, f_eval, phase_to_ref)

    psi = 2.0 * PI * f_eval * tc - phic - PI / 4.0 + phase_valid - 2.0 * PI * f_eval * time_valid
    amp_newt = (
        np.sqrt(5.0 / 24.0)
        * Mc_sec ** (5.0 / 6.0)
        * PI ** (-2.0 / 3.0)
        * f_eval ** (-7.0 / 6.0)
        / dL_sec
    )
    amp = amp_newt * np.sqrt(np.maximum(dfdt_N[:-1] / dfdt[:-1], 0.0))
    if sigma > 0.0:
        taper = 1.0 / (1.0 + np.exp((f_eval - f_cut) / sigma))
    else:
        taper = np.ones_like(f_eval)
    amp *= taper

    if phase_only:
        h_eval = np.exp(-1.0j * psi)
    else:
        h_eval = amp * np.exp(-1.0j * psi)

    out[valid] = h_eval[np.searchsorted(f_eval, f_valid)]
    return out.reshape(f_arr.shape)

