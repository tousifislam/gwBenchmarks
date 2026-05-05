"""
Standalone RG-tail inspiral waveform prototype (dominant nonspinning 2,2 mode).

Implements the benchmark prompt in:
  gw-forecast-agent/benchmarks/rg_waveform_source_packet/prompt_source_packet_no_skills.md
using the formulas in:
  gw-forecast-agent/benchmarks/rg_waveform_source_packet/2602.08833_relevant_formulas.md

The exported API is h_of_f(...), returning a complex frequency-domain strain.
"""

from __future__ import annotations

import numpy as np
from scipy import special


# Unit conversions (given explicitly in the prompt).
MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0


def _logistic_window(f: np.ndarray, f_cut: float, sigma: float) -> np.ndarray:
    """W(f) = 1 / (1 + exp((f - f_cut)/sigma)), computed stably."""
    z = (f - f_cut) / sigma
    # Avoid overflow in exp for extreme frequencies.
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


def _conservative_sector(x: np.ndarray, nu: float):
    """
    Conservative circular dynamics (GR):
      E_real/M and p_phi,circ/(mu M), plus d(E_real/M)/dx.
    """
    gamma_e = float(np.euler_gamma)
    log16x = np.log(16.0 * x)

    # Energy coefficients c0..c4 (c4 has log(16x)).
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
        + (
            -123671.0 / 5760.0
            + 9037.0 * np.pi**2 / 1536.0
            + 896.0 * gamma_e / 15.0
            + 448.0 * log16x / 15.0
        )
        * nu
        + (498449.0 / 3456.0 - 3157.0 * np.pi**2 / 576.0) * nu**2
        + 301.0 * nu**3 / 1728.0
        + 77.0 * nu**4 / 31104.0
    )

    # E_real/M = 1 - (nu x / 2) [c0 + c1 x + c2 x^2 + c3 x^3 + c4 x^4]
    S = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4
    E = 1.0 - 0.5 * nu * x * S

    # d/dx of c4 comes only from log(16x) term.
    dc4_dx = nu * (448.0 / 15.0) * (1.0 / x)
    dS_dx = (
        c1
        + 2.0 * c2 * x
        + 3.0 * c3 * x**2
        + 4.0 * c4 * x**3
        + dc4_dx * x**4
    )
    dE_dx = -0.5 * nu * (S + x * dS_dx)

    # Angular momentum coefficients d0..d4 (d4 has log(16x)).
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
        + (
            98869.0 / 5760.0
            - 128.0 * gamma_e / 3.0
            - 6455.0 * np.pi**2 / 1536.0
            - 64.0 * log16x / 3.0
        )
        * nu
        + (356035.0 / 3456.0 - 2255.0 * np.pi**2 / 576.0) * nu**2
        - 215.0 * nu**3 / 1728.0
        - 55.0 * nu**4 / 31104.0
    )

    pphi = x ** (-0.5) * (d0 + d1 * x + d2 * x**2 + d3 * x**3 + d4 * x**4)
    return E, pphi, dE_dx


def _hhat_22(x: np.ndarray, nu: float, lambda_rg: float) -> np.ndarray:
    """Compute hhat_22 = H_eff T_22 rho_22^2 exp(i delta_22)."""
    E, pphi, _ = _conservative_sector(x, nu)

    # Even-parity effective source.
    H_eff = ((E**2 - 1.0) / (2.0 * nu)) + 1.0

    # Dimensionless orbital frequency: M*Omega = x^(3/2)
    omega_hat = x ** (3.0 / 2.0)

    # Running-tail ingredients (m=2).
    m = 2.0
    khat = E * m * omega_hat
    J = pphi / (E**2)

    gamma_univ = (
        -214.0 * khat**2 / 105.0
        + 2.0 * m * J * khat**3 / 3.0
        - 3390466.0 * khat**4 / 1157625.0
        + 381863.0 * m * J * khat**5 / 99225.0
    )

    ellhat = 2.0 + lambda_rg * gamma_univ

    # Tail factor T_22 = exp(logT_22)
    # Use the circular-orbit specializations given in the formula sheet.
    gamma_e = float(np.euler_gamma)
    phi0 = np.exp(17.0 / 12.0 - gamma_e) / 4.0
    log_2krw = np.log(4.0 * np.sqrt(x))  # log(2 k r_omega) = log(4 sqrt(x))

    logT = (
        np.log(120.0)
        + (ellhat - 2.0) * log_2krw
        + 2.0j * khat * np.log(4.0 * phi0)
        + special.loggamma(ellhat - 1.0 - 2.0j * khat)
        - special.loggamma(2.0 * ellhat + 2.0)
        + np.pi * khat
        - 0.5j * np.pi * (ellhat - 2.0)
    )
    T22 = np.exp(logT)

    # Residual amplitude rho_22 and residual phase delta_22.
    eulerlog2 = gamma_e + np.log(4.0 * np.sqrt(x))
    r1 = -43.0 / 42.0 + 55.0 * nu / 84.0
    r2 = -20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu**2 / 42336.0
    r3 = (
        -4296031.0 / 4889808.0
        + (41.0 * np.pi**2 / 192.0 - 48993925.0 / 9779616.0) * nu
        - 6292061.0 * nu**2 / 3259872.0
        + 10620745.0 * nu**3 / 39118464.0
    )
    r4 = (
        9228174993589.0 / 800950550400.0
        + (
            -2487107795131.0 / 145627372800.0
            + 464.0 * eulerlog2 / 35.0
            - 9953.0 * np.pi**2 / 21504.0
        )
        * nu
        + (10815863492353.0 / 640760440320.0 - 3485.0 * np.pi**2 / 5376.0) * nu**2
        - 2088847783.0 * nu**3 / 11650189824.0
        + 70134663541.0 * nu**4 / 512608352256.0
    )
    rho22 = 1.0 + r1 * x + r2 * x**2 + r3 * x**3 + r4 * x**4

    # y = [(E_real/M) x^(3/2)]^(2/3) = x * (E_real/M)^(2/3)
    y = x * np.power(E, 2.0 / 3.0)
    delta22 = (
        -17.0 * y ** (3.0 / 2.0) / 3.0
        - 24.0 * nu * y ** (5.0 / 2.0)
        + (30995.0 * nu / 1134.0 + 962.0 * nu**2 / 135.0) * y ** (7.0 / 2.0)
        - 4976.0 * np.pi * nu * y**4 / 105.0
    )

    return H_eff * T22 * (rho22**2) * np.exp(1.0j * delta22)


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
    """
    Frequency-domain inspiral strain h(f) with RG-tail deformation.

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    Mc : float
        Detector-frame chirp mass in Msun.
    eta : float
        Symmetric mass ratio (0 < eta <= 0.25).
    dL : float
        Luminosity distance in Mpc.
    tc, phic : float
        Coalescence time (s) and phase (rad) shifts.
    lambda_RG : float
        Scales the running correction inside the tail factor (1.0 is GR).
    f_low : float
        Low-frequency cutoff (Hz). Set waveform exactly to zero for f < f_low.
    fmax_over_fisco : float
        High-frequency cutoff is f_cut = fmax_over_fisco * f_isco.
    sigma_taper_over_fisco : float
        Logistic taper width: sigma = sigma_taper_over_fisco * f_isco.
    phase_only : bool
        If True, return only a phase factor (still respecting cutoffs/taper).

    Returns
    -------
    h : ndarray (complex128)
        Complex frequency-domain strain with the same shape as `f`.
    """
    f_arr = np.asarray(f, dtype=float)
    out = np.zeros(f_arr.shape, dtype=np.complex128)

    if f_arr.size == 0:
        return out

    nu = float(eta)
    if not (nu > 0.0):
        return out

    M_sec = float(Mc) * MSUN_SEC / (nu ** (3.0 / 5.0))
    dL_sec = float(dL) * MPC_SEC

    f_isco = 1.0 / (np.pi * (6.0 ** (3.0 / 2.0)) * M_sec)
    f_cut = float(fmax_over_fisco) * f_isco
    sigma = float(sigma_taper_over_fisco) * f_isco
    if sigma <= 0.0:
        sigma = 1e-12 * f_isco

    # Apply exact cut rules from the prompt.
    mask = (f_arr >= float(f_low)) & (f_arr < f_cut)
    if not np.any(mask):
        return out

    f_work = f_arr[mask]
    W = _logistic_window(f_work, f_cut=f_cut, sigma=sigma)

    # Sort for stable one-pass quadrature in x.
    order = np.argsort(f_work)
    f_s = f_work[order]
    x = np.power(np.pi * M_sec * f_s, 2.0 / 3.0)

    # Radiative correction factor (complex).
    hhat = _hhat_22(x, nu=nu, lambda_rg=float(lambda_RG))

    # Energy balance phasing ingredients.
    E, _, dE_dx = _conservative_sector(x, nu)
    F22 = (32.0 / 5.0) * (nu**2) * (x**5) * (np.abs(hhat) ** 2)

    # dt/dx = -M d(E_real/M)/dx / F_22
    dt_dx = -M_sec * dE_dx / F22

    # Orbital frequency: Omega = (M*Omega)/M = x^(3/2)/M.
    dphi_gw_dx = 2.0 * (x ** (3.0 / 2.0)) * dt_dx / M_sec

    It = _cumtrapz(dt_dx, x)
    Iphi = _cumtrapz(dphi_gw_dx, x)
    total_t = It[-1]
    total_phi = Iphi[-1]

    # Enforce tc, phic at the high-frequency end (near x_cut / f_cut).
    t_of_x = float(tc) - (total_t - It)
    phi_gw = float(phic) - (total_phi - Iphi)

    # f-dot from dt/dx and the exact f(x) relation.
    df_dx = (3.0 / (2.0 * np.pi * M_sec)) * np.sqrt(x)
    dfdt = df_dx / dt_dx

    # SPA waveform:
    #   h(f) ~ [ (2 nu M / dL) x hhat_22(x) ] / sqrt(df/dt)
    #          * exp(i [2π f t(x) - phi_gw(x) - π/4])
    prefac = 2.0 * nu * M_sec / dL_sec
    phase = 2.0 * np.pi * f_s * t_of_x - phi_gw - np.pi / 4.0
    h = (prefac * x * hhat / np.sqrt(dfdt)) * np.exp(1.0j * phase)

    # Apply logistic taper and (optional) phase-only mode.
    if phase_only:
        h = W * np.exp(1.0j * np.angle(h))
    else:
        h = W * h

    # Unsort back into the original masked order.
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    out[mask] = h[inv]
    return out


if __name__ == "__main__":
    # Tiny smoke test (not used by the evaluator).
    f = np.linspace(10.0, 512.0, 4096)
    h = h_of_f(f, Mc=30.0, eta=0.25, dL=1000.0)
    print("h(f):", h.dtype, h.shape, "nonzero:", np.count_nonzero(h))
