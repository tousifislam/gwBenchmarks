"""
Standalone finite-size balance-law SPA inspiral waveform (source-packet / no-skills).

Implements the benchmark prompt in:
  gw-forecast-agent/benchmarks/finite_size_waveform_source_packet/prompt_source_packet_no_skills.md
using formulas from:
  gw-forecast-agent/benchmarks/finite_size_waveform_source_packet/2410.00294_relevant_formulas.md

The exported API is h_of_f(...), returning a complex frequency-domain strain.
"""

from __future__ import annotations

import numpy as np


# Unit conversions (given explicitly in the prompt).
MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0


def _logistic_window(f: np.ndarray, f_cut: float, sigma: float) -> np.ndarray:
    """W(f) = 1 / (1 + exp((f - f_cut)/sigma)), computed stably."""
    z = (f - f_cut) / sigma
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral of y(x), with initial value 0."""
    if y.size != x.size:
        raise ValueError("cumtrapz requires y and x to have the same shape")
    if y.size == 0:
        return y.copy()
    if y.size == 1:
        return np.zeros_like(y)
    dx = np.diff(x)
    seg = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.empty_like(y)
    out[0] = 0.0
    out[1:] = np.cumsum(seg)
    return out


def _spin_combinations(chi1: float, chi2: float, eta: float):
    delta = np.sqrt(max(0.0, 1.0 - 4.0 * eta))
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    return delta, chi_s, chi_a


def _E_SS_SIM(v: np.ndarray, eta: float, chi_s: float, chi_a: float, delta: float, kappa_s: float, kappa_a: float):
    """
    Spin-induced quadrupole binding-energy piece:
      E_SS_SIM(v) = E0 + v^2 * E2
    """
    # Constant-in-v part.
    E0 = (
        chi_a * chi_s * (-(delta**2) * kappa_a / 2.0 - delta * kappa_s - kappa_a / 2.0)
        + chi_s**2 * (-(delta**2) * kappa_s / 4.0 - delta * kappa_a / 2.0 - kappa_s / 4.0)
        + chi_a**2 * (-delta * kappa_a / 2.0 + eta * kappa_s - kappa_s / 2.0)
    )

    # v^2 coefficient.
    E2 = (
        chi_a**2
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta**2 * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
        + chi_a
        * chi_s
        * (
            25.0 * delta * eta * kappa_s / 6.0
            - 35.0 * delta * kappa_s / 6.0
            - 5.0 * eta**2 * kappa_a / 3.0
            + 95.0 * eta * kappa_a / 6.0
            - 35.0 * kappa_a / 6.0
        )
        + chi_s**2
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta**2 * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
    )
    return E0 + (v**2) * E2, E0, E2


def _E_SSS_SIM(eta: float, chi_s: float, chi_a: float, delta: float, kappa_s: float, kappa_a: float) -> float:
    """Cubic spin-induced binding-energy coefficient (constant in v)."""
    return (
        chi_a
        * chi_s**2
        * (
            -2.0 * delta**2 * eta * kappa_a
            - 5.0 * delta**2 * kappa_a
            + 6.0 * delta * eta * kappa_s
            - 6.0 * delta * kappa_s
            - kappa_a
        )
        + chi_s**3
        * (
            -delta**3 * kappa_a
            - delta**2 * eta * kappa_s
            - 9.0 * delta**2 * kappa_s / 4.0
            - delta * kappa_a
            + kappa_s / 4.0
        )
        + chi_a**2 * chi_s * (-6.0 * delta * kappa_a + 4.0 * eta**2 * kappa_s + 12.0 * eta * kappa_s - 6.0 * kappa_s)
        + chi_a**3 * (-2.0 * delta * eta * kappa_s - 2.0 * delta * kappa_s + 2.0 * eta * kappa_a - 2.0 * kappa_a)
    )


def _F_SS_SIM(v: np.ndarray, eta: float, chi_s: float, chi_a: float, delta: float, kappa_s: float, kappa_a: float):
    """
    Spin-induced quadrupole flux piece:
      F_SS_SIM(v) = F0 + v^2 * F2 + v^3 * F3
    """
    F0 = (
        chi_a * chi_s * (delta**2 * kappa_a + 2.0 * delta * kappa_s + kappa_a)
        + chi_a**2 * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
        + chi_s**2 * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
    )

    F2 = (
        chi_a**2
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta**2 * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
        + chi_a
        * chi_s
        * (
            -127.0 * delta * eta * kappa_s / 8.0
            + delta * kappa_s / 7.0
            + 43.0 * eta**2 * kappa_a / 2.0
            - 905.0 * eta * kappa_a / 56.0
            + kappa_a / 7.0
        )
        + chi_s**2
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta**2 * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
    )

    F3 = (
        chi_a**2 * (4.0 * np.pi * delta * kappa_a - 8.0 * np.pi * eta * kappa_s + 4.0 * np.pi * kappa_s)
        + chi_a * chi_s * (8.0 * np.pi * delta * kappa_s - 16.0 * np.pi * eta * kappa_a + 8.0 * np.pi * kappa_a)
        + chi_s**2 * (4.0 * np.pi * delta * kappa_a - 8.0 * np.pi * eta * kappa_s + 4.0 * np.pi * kappa_s)
    )
    return F0 + (v**2) * F2 + (v**3) * F3, F0, F2, F3


def _F_SSS_SIM(eta: float, chi_s: float, chi_a: float, delta: float, kappa_s: float, kappa_a: float) -> float:
    """Cubic spin-induced flux coefficient (constant in v)."""
    return (
        chi_a
        * chi_s**2
        * (
            13.0 * delta**2 * eta * kappa_a / 3.0
            + 27.0 * delta**2 * kappa_a / 16.0
            + 4.0 * delta * eta * kappa_s / 3.0
            + 15.0 * delta * kappa_s / 8.0
            + 3.0 * kappa_a / 16.0
        )
        + chi_a**2
        * chi_s
        * (
            95.0 * delta * eta * kappa_a / 12.0
            + 15.0 * delta * kappa_a / 8.0
            - 26.0 * eta**2 * kappa_s / 3.0
            + 25.0 * eta * kappa_s / 6.0
            + 15.0 * kappa_s / 8.0
        )
        + chi_s**3
        * (
            -7.0 * delta * eta * kappa_a / 4.0
            + 5.0 * delta * kappa_a / 8.0
            - 26.0 * eta**2 * kappa_s / 3.0
            - 3.0 * eta * kappa_s
            + 5.0 * kappa_s / 8.0
        )
        + chi_a**3 * (29.0 * delta * eta * kappa_s / 6.0 + 5.0 * delta * kappa_s / 8.0 + 43.0 * eta * kappa_a / 12.0 + 5.0 * kappa_a / 8.0)
    )


def _dotM(v: np.ndarray, eta: float, chi_s: float, chi_a: float, delta: float, H0: float, H1E: float, H1B: float, H1E_bar: float, H1B_bar: float):
    """Absorbed flux dotM(v) (polynomial in v)."""
    term15 = 0.5 * (9.0 * H1E_bar * eta**2 * chi_a + 9.0 * H1E * eta**2 * chi_s) * v**15

    term17_coeff = 0.5 * (
        (
            (9.0 * H1B * eta**2 + 45.0 * H1E * eta**2 / 2.0 + 9.0 * H1E_bar * delta * eta**2 / 2.0 - 27.0 * H1E * eta**3)
            * chi_s
        )
        + (
            (9.0 * H1B_bar * eta**2 + 45.0 * H1E_bar * eta**2 / 2.0 + 9.0 * H1E * delta * eta**2 / 2.0 - 27.0 * H1E_bar * eta**3)
            * chi_a
        )
    )
    term17 = term17_coeff * v**17

    # Uses doubled BBH convention (18 instead of 9).
    term18 = 18.0 * H0 * eta**2 * v**18
    return term15 + term17 + term18


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
    """
    Frequency-domain inspiral strain h(f) with finite-size effects derived from the balance law.

    Returns complex frequency-domain strain with the same shape as `f`.
    """
    f_arr = np.asarray(f, dtype=float)
    out = np.zeros(f_arr.shape, dtype=np.complex128)
    if f_arr.size == 0:
        return out

    eta = float(eta)
    if not (eta > 0.0):
        return out

    M_sec = float(Mc) * MSUN_SEC / (eta ** (3.0 / 5.0))
    dL_sec = float(dL) * MPC_SEC
    Mc_sec = float(Mc) * MSUN_SEC

    f_isco = 1.0 / (np.pi * (6.0 ** (3.0 / 2.0)) * M_sec)
    f_cut = float(fmax_over_fisco) * f_isco
    sigma = float(sigma_taper_over_fisco) * f_isco
    if sigma <= 0.0:
        sigma = 1e-12 * f_isco

    mask = (f_arr >= float(f_low)) & (f_arr < f_cut)
    if not np.any(mask):
        return out

    f_work = f_arr[mask]
    W_work = _logistic_window(f_work, f_cut=f_cut, sigma=sigma)

    # Sort for one-pass quadrature in v.
    order = np.argsort(f_work)
    f_s = f_work[order]
    W_s = W_work[order]
    v = np.power(np.pi * M_sec * f_s, 1.0 / 3.0)

    delta, chi_s, chi_a = _spin_combinations(float(chi1), float(chi2), eta)
    kappa_s = float(kappa_s)
    kappa_a = float(kappa_a)

    # Binding energy e(v) = E/M and derivative de/dv.
    _, E0, E2 = _E_SS_SIM(v, eta, chi_s, chi_a, delta, kappa_s, kappa_a)
    E_SSS = _E_SSS_SIM(eta, chi_s, chi_a, delta, kappa_s, kappa_a)

    e = -(eta / 2.0) * (v**2 + E0 * v**6 + E2 * v**8 + E_SSS * v**9)
    de_dv = -(eta / 2.0) * (2.0 * v + 6.0 * E0 * v**5 + 8.0 * E2 * v**7 + 9.0 * E_SSS * v**8)

    # Flux at infinity.
    _, F0, F2, F3 = _F_SS_SIM(v, eta, chi_s, chi_a, delta, kappa_s, kappa_a)
    F_SSS = _F_SSS_SIM(eta, chi_s, chi_a, delta, kappa_s, kappa_a)
    Lambda_tilde = float(Lambda_tilde)

    F_infty = (32.0 / 5.0) * eta**2 * v**10 * (1.0 + F0 * v**4 + F2 * v**6 + F3 * v**7 + F_SSS * v**7 + (39.0 / 8.0) * Lambda_tilde * v**10)

    # Absorbed flux.
    dotM = _dotM(
        v,
        eta=eta,
        chi_s=chi_s,
        chi_a=chi_a,
        delta=delta,
        H0=float(H0),
        H1E=float(H1E),
        H1B=float(H1B),
        H1E_bar=float(H1E_bar),
        H1B_bar=float(H1B_bar),
    )

    F_tot = F_infty + dotM

    # Guard against pathological inputs (negative/zero flux => no chirp).
    restricted = False
    good = np.isfinite(F_tot) & (F_tot > 0.0) & np.isfinite(de_dv)
    if not np.all(good):
        # If some points are pathological, integrate only over the longest
        # contiguous good segment to avoid gaps inside the quadrature.
        idx = np.where(good)[0]
        if idx.size == 0:
            return out
        # Segment boundaries where consecutive indices break.
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks + 1, idx.size]
        seg_lens = ends - starts
        j = int(np.argmax(seg_lens))
        lo = int(idx[starts[j]])
        hi = int(idx[ends[j] - 1] + 1)

        restricted = True
        v = v[lo:hi]
        f_s = f_s[lo:hi]
        W_s = W_s[lo:hi]
        de_dv = de_dv[lo:hi]
        F_tot = F_tot[lo:hi]

    # Balance law: dt/dv = -M d(e)/dv / F_tot
    dt_dv = -M_sec * de_dv / F_tot

    # df/dt from dt/dv and f(v) = v^3/(pi M).
    df_dv = 3.0 * v**2 / (np.pi * M_sec)
    dfdt = df_dv / dt_dv

    # Newtonian reference chirp rate for amplitude renormalization.
    dfdt_N = 96.0 * eta * v**11 / (5.0 * np.pi * M_sec**2)

    # SPA phase via time and GW phase integrals.
    # GW phase satisfies dphi_gw/dt = 2*pi*f = 2*v^3/M.
    dphi_dv = (2.0 * v**3 / M_sec) * dt_dv

    It = _cumtrapz(dt_dv, v)
    Iphi = _cumtrapz(dphi_dv, v)
    total_t = It[-1]
    total_phi = Iphi[-1]

    # Anchor integration constants at the high-frequency end (largest v in-band):
    #   t(v_max) = tc, phi(v_max) = phic.
    t_of_v = float(tc) - (total_t - It)
    phi_gw = float(phic) - (total_phi - Iphi)

    psi = 2.0 * np.pi * f_s * t_of_v - phi_gw - np.pi / 4.0

    # Restricted SPA amplitude and amplitude correction.
    A_N = (np.sqrt(5.0 / 24.0) * (Mc_sec ** (5.0 / 6.0)) * (np.pi ** (-2.0 / 3.0)) * (f_s ** (-7.0 / 6.0))) / dL_sec
    A = A_N * np.sqrt(dfdt_N / dfdt)

    if phase_only:
        h = np.exp(-1.0j * psi)
    else:
        h = A * np.exp(-1.0j * psi)

    h = W_s * h

    # Unsort back into original masked order.
    # If we had to restrict to a subrange due to bad points, map carefully.
    if restricted:
        # We only filled a subset of the sorted array; build a full sorted buffer.
        hs = np.zeros(order.size, dtype=np.complex128)
        hs[lo:hi] = h
        h_unsorted = hs[np.argsort(order)]
    else:
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        h_unsorted = h[inv]

    out[mask] = h_unsorted
    return out


if __name__ == "__main__":
    # Tiny smoke test (not used by the evaluator).
    f = np.linspace(10.0, 512.0, 4096)
    h = h_of_f(f, Mc=30.0, eta=0.25, dL=1000.0)
    print("h(f):", h.dtype, h.shape, "nonzero:", np.count_nonzero(h))
