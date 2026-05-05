"""Balance-law finite-size inspiral waveform benchmark.

This module implements an inspiral-only frequency-domain waveform based on the
energy-balance construction in Appendix A of arXiv:2410.00294:

    -F_infty(v) - dot{M}(v) = dE/dt.

The Fourier phase and SPA amplitude are both built from the same chirp rate
dv/dt.  This mirrors the self-consistent balance-law strategy used by the local
RG-tail benchmark.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0
PI = np.pi

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PSD_DATA_DIR = _PROJECT_ROOT / "data" / "psds"


def _asarray(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _isco_frequency(M_sec: float) -> float:
    return 1.0 / (6.0 ** 1.5 * PI * M_sec)


def _fermi_taper(f: np.ndarray, f_cut: float, sigma: float) -> np.ndarray:
    exponent = np.clip((f - f_cut) / sigma, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(exponent))


def _cumulative_integral_to_upper(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=float)
    if len(x) == 1:
        return np.zeros(1, dtype=float)
    dx = np.diff(x)
    trap = 0.5 * (y[:-1] + y[1:]) * dx
    cumulative_from_low = np.concatenate(([0.0], np.cumsum(trap)))
    return cumulative_from_low[-1] - cumulative_from_low


def _mass_spin_combinations(
    eta: float,
    chi1: float,
    chi2: float,
) -> tuple[float, float, float]:
    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    return delta, chi_s, chi_a


def newtonian_spa_amplitude(f: np.ndarray, Mc_sec: float, dL_sec: float) -> np.ndarray:
    """Restricted Newtonian TaylorF2 amplitude normalization."""
    return (
        np.sqrt(5.0 / 24.0)
        * Mc_sec ** (5.0 / 6.0)
        / (dL_sec * PI ** (2.0 / 3.0))
        * f ** (-7.0 / 6.0)
    )


def _energy_ss_sim(
    v: np.ndarray,
    eta: float,
    chi1: float,
    chi2: float,
    kappa_s: float,
    kappa_a: float,
) -> np.ndarray:
    """Spin-induced quadrupole piece E_SS^SIM(v) from Appendix A."""
    delta, chi_s, chi_a = _mass_spin_combinations(eta, chi1, chi2)
    eta2 = eta * eta
    delta2 = delta * delta
    leading = (
        chi_a
        * chi_s
        * (-0.5 * delta2 * kappa_a - delta * kappa_s - 0.5 * kappa_a)
        + chi_s ** 2 * (-0.25 * delta2 * kappa_s - 0.5 * delta * kappa_a - 0.25 * kappa_s)
        + chi_a ** 2 * (-0.5 * delta * kappa_a + eta * kappa_s - 0.5 * kappa_s)
    )
    subleading = (
        chi_a ** 2
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta2 * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
        + chi_a
        * chi_s
        * (
            25.0 * delta * eta * kappa_s / 6.0
            - 35.0 * delta * kappa_s / 6.0
            - 5.0 * eta2 * kappa_a / 3.0
            + 95.0 * eta * kappa_a / 6.0
            - 35.0 * kappa_a / 6.0
        )
        + chi_s ** 2
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta2 * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
    )
    return leading + v ** 2 * subleading


def _energy_sss_sim(
    eta: float,
    chi1: float,
    chi2: float,
    kappa_s: float,
    kappa_a: float,
) -> float:
    """Spin-induced cubic-in-spin piece E_SSS^SIM from Appendix A."""
    delta, chi_s, chi_a = _mass_spin_combinations(eta, chi1, chi2)
    delta2 = delta * delta
    delta3 = delta2 * delta
    eta2 = eta * eta
    return (
        chi_a
        * chi_s ** 2
        * (-2.0 * delta2 * eta * kappa_a - 5.0 * delta2 * kappa_a + 6.0 * delta * eta * kappa_s - 6.0 * delta * kappa_s - kappa_a)
        + chi_s ** 3
        * (-delta3 * kappa_a - delta2 * eta * kappa_s - 9.0 * delta2 * kappa_s / 4.0 - delta * kappa_a + kappa_s / 4.0)
        + chi_a ** 2
        * chi_s
        * (-6.0 * delta * kappa_a + 4.0 * eta2 * kappa_s + 12.0 * eta * kappa_s - 6.0 * kappa_s)
        + chi_a ** 3 * (-2.0 * delta * eta * kappa_s - 2.0 * delta * kappa_s + 2.0 * eta * kappa_a - 2.0 * kappa_a)
    )


def binding_energy_over_M(
    v: np.ndarray,
    eta: float,
    chi1: float = 0.0,
    chi2: float = 0.0,
    kappa_s: float = 1.0,
    kappa_a: float = 0.0,
) -> np.ndarray:
    """Binding energy per total mass from the paper's finite-size pieces.

    The point-particle baseline is kept at Newtonian order.  The finite-size
    spin-induced pieces follow the Appendix A structure

        E = -M eta v^2/2 [1 + E_SS^SIM v^4 + E_SSS^SIM v^7].
    """
    v = _asarray(v)
    bracket = (
        1.0
        + _energy_ss_sim(v, eta, chi1, chi2, kappa_s, kappa_a) * v ** 4
        + _energy_sss_sim(eta, chi1, chi2, kappa_s, kappa_a) * v ** 7
    )
    return -0.5 * eta * v ** 2 * bracket


def d_binding_energy_over_M_dv(
    v: np.ndarray,
    eta: float,
    chi1: float = 0.0,
    chi2: float = 0.0,
    kappa_s: float = 1.0,
    kappa_a: float = 0.0,
) -> np.ndarray:
    """Numerical derivative d(E/M)/dv using a five-point stencil."""
    v = _asarray(v)
    step = np.maximum(1e-5 * v, 1e-8)

    def energy(v_eval: np.ndarray) -> np.ndarray:
        return binding_energy_over_M(
            v_eval,
            eta=eta,
            chi1=chi1,
            chi2=chi2,
            kappa_s=kappa_s,
            kappa_a=kappa_a,
        )

    return (
        -energy(v + 2.0 * step)
        + 8.0 * energy(v + step)
        - 8.0 * energy(v - step)
        + energy(v - 2.0 * step)
    ) / (12.0 * step)


def _flux_ss_sim(
    v: np.ndarray,
    eta: float,
    chi1: float,
    chi2: float,
    kappa_s: float,
    kappa_a: float,
) -> np.ndarray:
    """Spin-induced quadrupole piece F_SS(v) from Appendix A."""
    delta, chi_s, chi_a = _mass_spin_combinations(eta, chi1, chi2)
    eta2 = eta * eta
    leading = (
        chi_a * chi_s * (delta ** 2 * kappa_a + 2.0 * delta * kappa_s + kappa_a)
        + chi_a ** 2 * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
        + chi_s ** 2 * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
    )
    v2_piece = (
        chi_a ** 2
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta2 * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
        + chi_a
        * chi_s
        * (
            -127.0 * delta * eta * kappa_s / 8.0
            + delta * kappa_s / 7.0
            + 43.0 * eta2 * kappa_a / 2.0
            - 905.0 * eta * kappa_a / 56.0
            + kappa_a / 7.0
        )
        + chi_s ** 2
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta2 * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
    )
    v3_piece = (
        chi_a ** 2 * (4.0 * PI * delta * kappa_a - 8.0 * PI * eta * kappa_s + 4.0 * PI * kappa_s)
        + chi_a * chi_s * (8.0 * PI * delta * kappa_s - 16.0 * PI * eta * kappa_a + 8.0 * PI * kappa_a)
        + chi_s ** 2 * (4.0 * PI * delta * kappa_a - 8.0 * PI * eta * kappa_s + 4.0 * PI * kappa_s)
    )
    return leading + v ** 2 * v2_piece + v ** 3 * v3_piece


def _flux_sss_sim(
    eta: float,
    chi1: float,
    chi2: float,
    kappa_s: float,
    kappa_a: float,
) -> float:
    """Spin-induced cubic-in-spin piece F_SSS from Appendix A."""
    delta, chi_s, chi_a = _mass_spin_combinations(eta, chi1, chi2)
    eta2 = eta * eta
    return (
        chi_a
        * chi_s ** 2
        * (13.0 * delta ** 2 * eta * kappa_a / 3.0 + 27.0 * delta ** 2 * kappa_a / 16.0 + 4.0 * delta * eta * kappa_s / 3.0 + 15.0 * delta * kappa_s / 8.0 + 3.0 * kappa_a / 16.0)
        + chi_a ** 2
        * chi_s
        * (95.0 * delta * eta * kappa_a / 12.0 + 15.0 * delta * kappa_a / 8.0 - 26.0 * eta2 * kappa_s / 3.0 + 25.0 * eta * kappa_s / 6.0 + 15.0 * kappa_s / 8.0)
        + chi_s ** 3
        * (-7.0 * delta * eta * kappa_a / 4.0 + 5.0 * delta * kappa_a / 8.0 - 26.0 * eta2 * kappa_s / 3.0 - 3.0 * eta * kappa_s + 5.0 * kappa_s / 8.0)
        + chi_a ** 3
        * (29.0 * delta * eta * kappa_s / 6.0 + 5.0 * delta * kappa_s / 8.0 + 43.0 * eta * kappa_a / 12.0 + 5.0 * kappa_a / 8.0)
    )


def flux_infinity(
    v: np.ndarray,
    eta: float,
    chi1: float = 0.0,
    chi2: float = 0.0,
    kappa_s: float = 1.0,
    kappa_a: float = 0.0,
    Lambda_tilde: float = 0.0,
) -> np.ndarray:
    """Flux at infinity from the paper's finite-size structure.

    The paper points to Flanagan-Iyer-Will for the Love-number energy and flux.
    For this compact benchmark we include the leading effective Love flux
    correction that reproduces Eq. (2.16)'s leading \tilde Lambda phasing when
    combined with the Newtonian point-particle energy.
    """
    v = _asarray(v)
    finite_size_factor = (
        1.0
        + _flux_ss_sim(v, eta, chi1, chi2, kappa_s, kappa_a) * v ** 4
        + _flux_sss_sim(eta, chi1, chi2, kappa_s, kappa_a) * v ** 7
        + (39.0 / 8.0) * Lambda_tilde * v ** 10
    )
    return (32.0 / 5.0) * eta ** 2 * v ** 10 * finite_size_factor


def horizon_flux(
    v: np.ndarray,
    eta: float,
    chi1: float = 0.0,
    chi2: float = 0.0,
    H0: float = 0.0,
    H1E: float = 0.0,
    H1B: float = 0.0,
    H1E_bar: float = 0.0,
    H1B_bar: float = 0.0,
    h0_dual: bool = True,
) -> np.ndarray:
    """Absorbed power dot{M}(v) from Appendix A.

    With ``h0_dual=True`` the leading spin-independent H0 term is doubled to
    implement the BBH electric-magnetic-duality convention used in the paper's
    BBH analysis.
    """
    v = _asarray(v)
    delta, chi_s, chi_a = _mass_spin_combinations(eta, chi1, chi2)
    eta2 = eta * eta
    h0_coeff = 18.0 if h0_dual else 9.0
    return (
        0.5 * (9.0 * H1E_bar * eta2 * chi_a + 9.0 * H1E * eta2 * chi_s) * v ** 15
        + 0.5
        * (
            (
                9.0 * H1B * eta2
                + 45.0 * H1E * eta2 / 2.0
                + 9.0 * H1E_bar * delta * eta2 / 2.0
                - 27.0 * H1E * eta ** 3
            )
            * chi_s
            + (
                9.0 * H1B_bar * eta2
                + 45.0 * H1E_bar * eta2 / 2.0
                + 9.0 * H1E * delta * eta2 / 2.0
                - 27.0 * H1E_bar * eta ** 3
            )
            * chi_a
        )
        * v ** 17
        + h0_coeff * H0 * eta2 * v ** 18
    )


def total_flux(
    v: np.ndarray,
    eta: float,
    chi1: float = 0.0,
    chi2: float = 0.0,
    kappa_s: float = 1.0,
    kappa_a: float = 0.0,
    H0: float = 0.0,
    Lambda_tilde: float = 0.0,
    H1E: float = 0.0,
    H1B: float = 0.0,
    H1E_bar: float = 0.0,
    H1B_bar: float = 0.0,
    h0_dual: bool = True,
) -> np.ndarray:
    return flux_infinity(
        v,
        eta=eta,
        chi1=chi1,
        chi2=chi2,
        kappa_s=kappa_s,
        kappa_a=kappa_a,
        Lambda_tilde=Lambda_tilde,
    ) + horizon_flux(
        v,
        eta=eta,
        chi1=chi1,
        chi2=chi2,
        H0=H0,
        H1E=H1E,
        H1B=H1B,
        H1E_bar=H1E_bar,
        H1B_bar=H1B_bar,
        h0_dual=h0_dual,
    )


def _chirp_rate(
    v: np.ndarray,
    M_sec: float,
    eta: float,
    chi1: float,
    chi2: float,
    kappa_s: float,
    kappa_a: float,
    H0: float,
    Lambda_tilde: float,
    H1E: float,
    H1B: float,
    H1E_bar: float,
    H1B_bar: float,
    h0_dual: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dE_dv = d_binding_energy_over_M_dv(
        v,
        eta=eta,
        chi1=chi1,
        chi2=chi2,
        kappa_s=kappa_s,
        kappa_a=kappa_a,
    )
    flux = total_flux(
        v,
        eta=eta,
        chi1=chi1,
        chi2=chi2,
        kappa_s=kappa_s,
        kappa_a=kappa_a,
        H0=H0,
        Lambda_tilde=Lambda_tilde,
        H1E=H1E,
        H1B=H1B,
        H1E_bar=H1E_bar,
        H1B_bar=H1B_bar,
        h0_dual=h0_dual,
    )
    flux = np.maximum(flux, 1e-300)
    dt_dv = -M_sec * dE_dv / flux
    df_dt = (3.0 * v ** 2 / (PI * M_sec)) / dt_dv
    return dt_dv, df_dt, flux


def _spa_phase_from_balance(
    f: np.ndarray,
    M_sec: float,
    eta: float,
    chi1: float,
    chi2: float,
    kappa_s: float,
    kappa_a: float,
    H0: float,
    Lambda_tilde: float,
    H1E: float,
    H1B: float,
    H1E_bar: float,
    H1B_bar: float,
    h0_dual: bool,
    v_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return intrinsic SPA phase and df/dt on the requested frequencies."""
    order = np.argsort(f)
    f_sorted = f[order]
    v_sorted = (PI * M_sec * f_sorted) ** (1.0 / 3.0)

    append_ref = v_ref > v_sorted[-1] * (1.0 + 1e-12)
    if append_ref:
        v_nodes = np.concatenate((v_sorted, [v_ref]))
    else:
        v_nodes = v_sorted

    dt_dv, df_dt_nodes, _ = _chirp_rate(
        v_nodes,
        M_sec=M_sec,
        eta=eta,
        chi1=chi1,
        chi2=chi2,
        kappa_s=kappa_s,
        kappa_a=kappa_a,
        H0=H0,
        Lambda_tilde=Lambda_tilde,
        H1E=H1E,
        H1B=H1B,
        H1E_bar=H1E_bar,
        H1B_bar=H1B_bar,
        h0_dual=h0_dual,
    )
    dphi_gw_dv = 2.0 * (v_nodes ** 3 / M_sec) * dt_dv

    time_to_ref = _cumulative_integral_to_upper(v_nodes, dt_dv)
    gw_phase_to_ref = _cumulative_integral_to_upper(v_nodes, dphi_gw_dv)

    if append_ref:
        time_to_ref = time_to_ref[:-1]
        gw_phase_to_ref = gw_phase_to_ref[:-1]
        df_dt_nodes = df_dt_nodes[:-1]

    psi_sorted = gw_phase_to_ref - 2.0 * PI * f_sorted * time_to_ref
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return psi_sorted[inv], df_dt_nodes[inv]


def h_of_f(
    f: np.ndarray,
    Mc: float,
    eta: float,
    dL: float,
    chi1: float = 0.0,
    chi2: float = 0.0,
    tc: float = 0.0,
    phic: float = 0.0,
    kappa_s: float = 1.0,
    kappa_a: float = 0.0,
    H0: float = 0.0,
    Lambda_tilde: float = 0.0,
    H1E: float = 0.0,
    H1B: float = 0.0,
    H1E_bar: float = 0.0,
    H1B_bar: float = 0.0,
    f_low: float = 20.0,
    fmax_over_fisco: float = 1.0,
    sigma_taper_over_fisco: float = 0.01,
    phase_only: bool = False,
    h0_dual: bool = True,
) -> np.ndarray:
    """Return the balance-law finite-size SPA waveform."""
    f = _asarray(f)
    h = np.zeros_like(f, dtype=complex)
    if f.size == 0:
        return h

    Mc_sec = Mc * MSUN_SEC
    M_sec = Mc_sec / eta ** (3.0 / 5.0)
    dL_sec = dL * MPC_SEC
    f_isco = _isco_frequency(M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = max(sigma_taper_over_fisco * f_isco, 1e-12 * f_isco)

    valid = (f >= f_low) & (f < f_cut) & np.isfinite(f) & (f > 0.0)
    if not np.any(valid):
        return h

    f_eval = f[valid]
    v_ref = (PI * M_sec * f_cut) ** (1.0 / 3.0)
    psi_intrinsic, df_dt = _spa_phase_from_balance(
        f_eval,
        M_sec=M_sec,
        eta=eta,
        chi1=chi1,
        chi2=chi2,
        kappa_s=kappa_s,
        kappa_a=kappa_a,
        H0=H0,
        Lambda_tilde=Lambda_tilde,
        H1E=H1E,
        H1B=H1B,
        H1E_bar=H1E_bar,
        H1B_bar=H1B_bar,
        h0_dual=h0_dual,
        v_ref=v_ref,
    )

    v = (PI * M_sec * f_eval) ** (1.0 / 3.0)
    df_dt_newtonian = 96.0 * eta * v ** 11 / (5.0 * PI * M_sec ** 2)
    amp_newtonian = newtonian_spa_amplitude(f_eval, Mc_sec=Mc_sec, dL_sec=dL_sec)
    amp = amp_newtonian * np.sqrt(df_dt_newtonian / np.maximum(df_dt, 1e-300))
    psi = 2.0 * PI * f_eval * tc - phic - PI / 4.0 + psi_intrinsic
    taper = _fermi_taper(f_eval, f_cut=f_cut, sigma=sigma)

    if phase_only:
        h[valid] = taper * np.exp(-1j * psi)
    else:
        h[valid] = amp * np.exp(-1j * psi) * taper
    return h


@lru_cache(maxsize=None)
def _load_psd_table(filename: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(_PSD_DATA_DIR / filename)
    return data[:, 0], data[:, 1]


def _interp_tabulated_psd(f: np.ndarray | float, filename: str, f_low: float) -> np.ndarray:
    f = _asarray(f)
    f_tab, psd_tab = _load_psd_table(filename)
    Sn = np.full_like(f, np.inf, dtype=float)
    valid = (f >= max(f_low, f_tab[0])) & (f <= f_tab[-1]) & np.isfinite(f) & (f > 0.0)
    if np.any(valid):
        Sn[valid] = np.exp(np.interp(f[valid], f_tab, np.log(psd_tab)))
    return Sn


def psd_aLIGO(f: np.ndarray | float) -> np.ndarray:
    f = _asarray(f)
    f0, S0 = 215.0, 1e-49
    x = f / f0
    Sn = S0 * (
        x ** (-4.14)
        - 5.0 * x ** (-2.0)
        + 111.0 * (1.0 - x ** 2 + 0.5 * x ** 4) / (1.0 + 0.5 * x ** 2)
    )
    return np.where((f < 10.0) | (Sn <= 0.0), np.inf, Sn)


def psd_ET_D(f: np.ndarray | float) -> np.ndarray:
    f = _asarray(f)
    f0, S0 = 200.0, 1.449e-52
    x = f / f0
    Sn = S0 * (
        x ** (-4.05)
        + 185.62 * x ** (-0.69)
        + 232.56
        * (
            1.0
            + 31.18 * x
            - 64.72 * x ** 2
            + 52.24 * x ** 3
            - 42.16 * x ** 4
            + 10.17 * x ** 5
            + 11.53 * x ** 6
        )
    )
    return np.where((f < 1.0) | (Sn <= 0.0), np.inf, Sn)


def psd_CE_20km(f: np.ndarray | float) -> np.ndarray:
    return _interp_tabulated_psd(f, "CE1_20km_from_gwfast_psd.txt", f_low=5.0)


def psd_CE_40km(f: np.ndarray | float) -> np.ndarray:
    f = _asarray(f)
    f0, S0 = 200.0, 4.0e-54
    x = f / f0
    Sn = S0 * (x ** (-4.1) + 30.0 * x ** (-0.6) + 45.0 * (1.0 + 2.0 * x ** 2))
    return np.where((f < 3.0) | (Sn <= 0.0), np.inf, Sn)


def psd_Voyager(f: np.ndarray | float) -> np.ndarray:
    return _interp_tabulated_psd(f, "Voyager_from_gwfast_psd.txt", f_low=5.0)


def compute_snr(f: np.ndarray, h: np.ndarray, Sn: np.ndarray) -> float:
    f = _asarray(f)
    df = f[1] - f[0]
    valid = np.isfinite(Sn) & (Sn > 0.0)
    return float(np.sqrt(4.0 * np.sum(np.where(valid, np.abs(h) ** 2 / Sn, 0.0)) * df))
