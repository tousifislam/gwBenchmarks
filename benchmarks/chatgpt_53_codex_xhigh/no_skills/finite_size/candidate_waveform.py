import numpy as np


MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0


def _suffix_trapezoid(y, x):
    """I_i = int_{x_i}^{x_max} y(x) dx for ascending x."""
    out = np.zeros_like(x, dtype=float)
    if x.size < 2:
        return out
    seg = 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def _energy_terms(v, eta, delta, chi_s, chi_a, kappa_s, kappa_a):
    # E_SS_SIM(v) = E_SS0 + E_SS2 v^2
    e_ss0 = (
        chi_a * chi_s * (-0.5 * delta * delta * kappa_a - delta * kappa_s - 0.5 * kappa_a)
        + chi_s * chi_s * (-0.25 * delta * delta * kappa_s - 0.5 * delta * kappa_a - 0.25 * kappa_s)
        + chi_a * chi_a * (-0.5 * delta * kappa_a + eta * kappa_s - 0.5 * kappa_s)
    )
    e_ss2 = (
        chi_a * chi_a
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta * eta * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
        + chi_a * chi_s
        * (
            25.0 * delta * eta * kappa_s / 6.0
            - 35.0 * delta * kappa_s / 6.0
            - 5.0 * eta * eta * kappa_a / 3.0
            + 95.0 * eta * kappa_a / 6.0
            - 35.0 * kappa_a / 6.0
        )
        + chi_s * chi_s
        * (
            25.0 * delta * eta * kappa_a / 12.0
            - 35.0 * delta * kappa_a / 12.0
            - 5.0 * eta * eta * kappa_s / 6.0
            + 95.0 * eta * kappa_s / 12.0
            - 35.0 * kappa_s / 12.0
        )
    )
    e_ss = e_ss0 + e_ss2 * v * v

    e_sss = (
        chi_a
        * chi_s
        * chi_s
        * (
            -2.0 * delta * delta * eta * kappa_a
            - 5.0 * delta * delta * kappa_a
            + 6.0 * delta * eta * kappa_s
            - 6.0 * delta * kappa_s
            - kappa_a
        )
        + chi_s
        * chi_s
        * chi_s
        * (
            -delta**3 * kappa_a
            - delta * delta * eta * kappa_s
            - 9.0 * delta * delta * kappa_s / 4.0
            - delta * kappa_a
            + kappa_s / 4.0
        )
        + chi_a
        * chi_a
        * chi_s
        * (-6.0 * delta * kappa_a + 4.0 * eta * eta * kappa_s + 12.0 * eta * kappa_s - 6.0 * kappa_s)
        + chi_a
        * chi_a
        * chi_a
        * (-2.0 * delta * eta * kappa_s - 2.0 * delta * kappa_s + 2.0 * eta * kappa_a - 2.0 * kappa_a)
    )
    return e_ss, e_ss2, e_sss


def _flux_terms(v, eta, delta, chi_s, chi_a, kappa_s, kappa_a, Lambda_tilde):
    # F_SS_SIM(v) = F_SS0 + F_SS2 v^2 + F_SS3 v^3
    f_ss0 = (
        chi_a * chi_s * (delta * delta * kappa_a + 2.0 * delta * kappa_s + kappa_a)
        + chi_a * chi_a * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
        + chi_s * chi_s * (delta * kappa_a - 2.0 * eta * kappa_s + kappa_s)
    )
    f_ss2 = (
        chi_a
        * chi_a
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta * eta * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
        + chi_a
        * chi_s
        * (
            -127.0 * delta * eta * kappa_s / 8.0
            + delta * kappa_s / 7.0
            + 43.0 * eta * eta * kappa_a / 2.0
            - 905.0 * eta * kappa_a / 56.0
            + kappa_a / 7.0
        )
        + chi_s
        * chi_s
        * (
            -127.0 * delta * eta * kappa_a / 16.0
            + delta * kappa_a / 14.0
            + 43.0 * eta * eta * kappa_s / 4.0
            - 905.0 * eta * kappa_s / 112.0
            + kappa_s / 14.0
        )
    )
    f_ss3 = (
        chi_a * chi_a * (4.0 * np.pi * delta * kappa_a - 8.0 * np.pi * eta * kappa_s + 4.0 * np.pi * kappa_s)
        + chi_a * chi_s * (8.0 * np.pi * delta * kappa_s - 16.0 * np.pi * eta * kappa_a + 8.0 * np.pi * kappa_a)
        + chi_s * chi_s * (4.0 * np.pi * delta * kappa_a - 8.0 * np.pi * eta * kappa_s + 4.0 * np.pi * kappa_s)
    )
    f_ss = f_ss0 + f_ss2 * v * v + f_ss3 * v**3

    f_sss = (
        chi_a
        * chi_s
        * chi_s
        * (
            13.0 * delta * delta * eta * kappa_a / 3.0
            + 27.0 * delta * delta * kappa_a / 16.0
            + 4.0 * delta * eta * kappa_s / 3.0
            + 15.0 * delta * kappa_s / 8.0
            + 3.0 * kappa_a / 16.0
        )
        + chi_a
        * chi_a
        * chi_s
        * (
            95.0 * delta * eta * kappa_a / 12.0
            + 15.0 * delta * kappa_a / 8.0
            - 26.0 * eta * eta * kappa_s / 3.0
            + 25.0 * eta * kappa_s / 6.0
            + 15.0 * kappa_s / 8.0
        )
        + chi_s
        * chi_s
        * chi_s
        * (
            -7.0 * delta * eta * kappa_a / 4.0
            + 5.0 * delta * kappa_a / 8.0
            - 26.0 * eta * eta * kappa_s / 3.0
            - 3.0 * eta * kappa_s
            + 5.0 * kappa_s / 8.0
        )
        + chi_a
        * chi_a
        * chi_a
        * (
            29.0 * delta * eta * kappa_s / 6.0
            + 5.0 * delta * kappa_s / 8.0
            + 43.0 * eta * kappa_a / 12.0
            + 5.0 * kappa_a / 8.0
        )
    )

    f_infty = (32.0 / 5.0) * eta * eta * v**10 * (
        1.0 + f_ss * v**4 + f_sss * v**7 + (39.0 / 8.0) * Lambda_tilde * v**10
    )
    return f_infty


def _absorbed_flux(v, eta, delta, chi_s, chi_a, H0, H1E, H1B, H1E_bar, H1B_bar):
    eta2 = eta * eta
    eta3 = eta2 * eta

    term15 = 0.5 * (9.0 * H1E_bar * eta2 * chi_a + 9.0 * H1E * eta2 * chi_s) * v**15
    term17 = 0.5 * (
        (
            9.0 * H1B * eta2
            + 45.0 * H1E * eta2 / 2.0
            + 9.0 * H1E_bar * delta * eta2 / 2.0
            - 27.0 * H1E * eta3
        )
        * chi_s
        + (
            9.0 * H1B_bar * eta2
            + 45.0 * H1E_bar * eta2 / 2.0
            + 9.0 * H1E * delta * eta2 / 2.0
            - 27.0 * H1E_bar * eta3
        )
        * chi_a
    ) * v**17
    # Benchmark convention uses 18 H0 eta^2 v^18.
    term18 = 18.0 * H0 * eta2 * v**18
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
    f = np.asarray(f, dtype=float)
    h = np.zeros(f.shape, dtype=np.complex128)

    if f.size == 0:
        return h
    if eta <= 0.0 or eta > 0.25:
        return h

    eta = float(eta)
    Mc = float(Mc)
    dL = float(dL)
    if Mc <= 0.0 or dL <= 0.0:
        return h

    M_sec = Mc * MSUN_SEC / (eta ** (3.0 / 5.0))
    Mc_sec = Mc * MSUN_SEC
    dL_sec = dL * MPC_SEC

    if M_sec <= 0.0:
        return h

    f_isco = 1.0 / (np.pi * (6.0 ** 1.5) * M_sec)
    f_cut = float(fmax_over_fisco) * f_isco
    sigma = float(sigma_taper_over_fisco) * f_isco
    if f_cut <= 0.0:
        return h

    mask = (f >= float(f_low)) & (f < f_cut) & (f > 0.0)
    if not np.any(mask):
        return h

    fv = f[mask]
    order = np.argsort(fv)
    fs = fv[order]
    v = (np.pi * M_sec * fs) ** (1.0 / 3.0)

    delta = np.sqrt(max(0.0, 1.0 - 4.0 * eta))
    chi_s = 0.5 * (float(chi1) + float(chi2))
    chi_a = 0.5 * (float(chi1) - float(chi2))
    kappa_s = float(kappa_s)
    kappa_a = float(kappa_a)
    Lambda_tilde = float(Lambda_tilde)
    H0 = float(H0)
    H1E = float(H1E)
    H1B = float(H1B)
    H1E_bar = float(H1E_bar)
    H1B_bar = float(H1B_bar)

    e_ss, e_ss2, e_sss = _energy_terms(v, eta, delta, chi_s, chi_a, kappa_s, kappa_a)
    bracket = 1.0 + e_ss * v**4 + e_sss * v**7
    # de/dv with E_SS_SIM(v) = e_ss0 + e_ss2 v^2.
    de_ss_dv = 2.0 * e_ss2 * v
    dbracket_dv = de_ss_dv * v**4 + e_ss * 4.0 * v**3 + e_sss * 7.0 * v**6
    de_dv = -0.5 * eta * (2.0 * v * bracket + v * v * dbracket_dv)

    f_infty = _flux_terms(v, eta, delta, chi_s, chi_a, kappa_s, kappa_a, Lambda_tilde)
    dot_m = _absorbed_flux(v, eta, delta, chi_s, chi_a, H0, H1E, H1B, H1E_bar, H1B_bar)
    f_tot = f_infty + dot_m

    tiny = np.finfo(float).tiny
    f_tot = np.maximum(f_tot, tiny)

    dt_dv = -M_sec * de_dv / f_tot
    dt_dv = np.maximum(dt_dv, tiny)

    dfdt = (3.0 * v * v / (np.pi * M_sec)) / dt_dv
    dfdt = np.maximum(dfdt, tiny)
    dt_df = 1.0 / dfdt

    # Phase convention: integrate from highest available frequency and anchor
    # t(f_max)=tc, phi(f_max)=phic.
    tau_to_high = _suffix_trapezoid(dt_df, fs)
    t_of_f = float(tc) - tau_to_high

    dphi_df = 2.0 * np.pi * fs * dt_df
    phi_to_high = _suffix_trapezoid(dphi_df, fs)
    phi_of_f = float(phic) - phi_to_high

    psi = 2.0 * np.pi * fs * t_of_f - phi_of_f - np.pi / 4.0

    amp_n = np.sqrt(5.0 / 24.0) * (Mc_sec ** (5.0 / 6.0)) * (np.pi ** (-2.0 / 3.0)) * fs ** (-7.0 / 6.0) / dL_sec
    dfdt_n = 96.0 * eta * v**11 / (5.0 * np.pi * M_sec * M_sec)
    dfdt_n = np.maximum(dfdt_n, tiny)
    amp = amp_n * np.sqrt(dfdt_n / dfdt)

    if sigma > 0.0:
        taper = 1.0 / (1.0 + np.exp((fs - f_cut) / sigma))
    else:
        taper = np.ones_like(fs)

    hs = amp * taper * np.exp(-1j * psi)
    if phase_only:
        hs = np.exp(1j * np.angle(hs))

    h_valid = np.empty_like(fv, dtype=np.complex128)
    h_valid[order] = hs
    h[mask] = h_valid
    return h
