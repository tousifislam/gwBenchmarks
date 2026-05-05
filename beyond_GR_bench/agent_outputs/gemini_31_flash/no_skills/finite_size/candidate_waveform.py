import numpy as np
from scipy.integrate import cumtrapz

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
    # Constants
    MSUN_SEC = 4.925491025543576e-6
    MPC_SEC = 3.0856775814913673e22 / 299792458

    # Derived parameters
    M_sec = Mc * MSUN_SEC / eta**0.6
    dL_sec = dL * MPC_SEC
    Mc_sec = Mc * MSUN_SEC
    delta = np.sqrt(1 - 4 * eta)
    chi_s = (chi1 + chi2) / 2
    chi_a = (chi1 - chi2) / 2

    f_isco = 1 / (np.pi * 6**1.5 * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    # Frequency array and masking
    f = np.atleast_1d(f)
    mask = (f >= f_low) & (f < f_cut)
    
    if not np.any(mask):
        return np.zeros_like(f, dtype=complex)

    f_masked = f[mask]
    v = (np.pi * M_sec * f_masked)**(1/3)

    # Binding Energy E_SS_SIM coefficients
    # E_SS_SIM(v) = E_SS_0 + E_SS_2 * v^2
    E_SS_0 = (
        chi_a * chi_s * (-delta**2 * kappa_a / 2 - delta * kappa_s - kappa_a / 2)
        + chi_s**2 * (-delta**2 * kappa_s / 4 - delta * kappa_a / 2 - kappa_s / 4)
        + chi_a**2 * (-delta * kappa_a / 2 + eta * kappa_s - kappa_s / 2)
    )
    E_SS_2 = (
        chi_a**2 * (25 * delta * eta * kappa_a / 12 - 35 * delta * kappa_a / 12
                     - 5 * eta**2 * kappa_s / 6 + 95 * eta * kappa_s / 12
                     - 35 * kappa_s / 12)
        + chi_a * chi_s * (25 * delta * eta * kappa_s / 6 - 35 * delta * kappa_s / 6
                         - 5 * eta**2 * kappa_a / 3 + 95 * eta * kappa_a / 6
                         - 35 * kappa_a / 6)
        + chi_s**2 * (25 * delta * eta * kappa_a / 12 - 35 * delta * kappa_a / 12
                     - 5 * eta**2 * kappa_s / 6 + 95 * eta * kappa_s / 12
                     - 35 * kappa_s / 12)
    )

    # E_SSS_SIM
    E_SSS_SIM = (
        chi_a * chi_s**2 * (-2 * delta**2 * eta * kappa_a - 5 * delta**2 * kappa_a
                           + 6 * delta * eta * kappa_s - 6 * delta * kappa_s - kappa_a)
        + chi_s**3 * (-delta**3 * kappa_a - delta**2 * eta * kappa_s
                       - 9 * delta**2 * kappa_s / 4 - delta * kappa_a + kappa_s / 4)
        + chi_a**2 * chi_s * (-6 * delta * kappa_a + 4 * eta**2 * kappa_s
                             + 12 * eta * kappa_s - 6 * kappa_s)
        + chi_a**3 * (-2 * delta * eta * kappa_s - 2 * delta * kappa_s
                       + 2 * eta * kappa_a - 2 * kappa_a)
    )

    # de/dv calculation
    # e(v) = -eta/2 * (v^2 + E_SS_0 * v^6 + E_SS_2 * v^8 + E_SSS_SIM * v^9)
    dedv = -eta/2 * (2 * v + 6 * E_SS_0 * v**5 + 8 * E_SS_2 * v**7 + 9 * E_SSS_SIM * v**8)

    # Flux at Infinity F_SS_SIM coefficients
    # F_SS_SIM(v) = F_SS_0 + F_SS_2 * v^2 + F_SS_3 * v^3
    F_SS_0 = (
        chi_a * chi_s * (delta**2 * kappa_a + 2 * delta * kappa_s + kappa_a)
        + chi_a**2 * (delta * kappa_a - 2 * eta * kappa_s + kappa_s)
        + chi_s**2 * (delta * kappa_a - 2 * eta * kappa_s + kappa_s)
    )
    F_SS_2 = (
        chi_a**2 * (-127 * delta * eta * kappa_a / 16 + delta * kappa_a / 14
                     + 43 * eta**2 * kappa_s / 4 - 905 * eta * kappa_s / 112
                     + kappa_s / 14)
        + chi_a * chi_s * (-127 * delta * eta * kappa_s / 8 + delta * kappa_s / 7
                         + 43 * eta**2 * kappa_a / 2 - 905 * eta * kappa_a / 56
                         + kappa_a / 7)
        + chi_s**2 * (-127 * delta * eta * kappa_a / 16 + delta * kappa_a / 14
                     + 43 * eta**2 * kappa_s / 4 - 905 * eta * kappa_s / 112
                     + kappa_s / 14)
    )
    F_SS_3 = (
        chi_a**2 * (4 * np.pi * delta * kappa_a - 8 * np.pi * eta * kappa_s + 4 * np.pi * kappa_s)
        + chi_a * chi_s * (8 * np.pi * delta * kappa_s - 16 * np.pi * eta * kappa_a + 8 * np.pi * kappa_a)
        + chi_s**2 * (4 * np.pi * delta * kappa_a - 8 * np.pi * eta * kappa_s + 4 * np.pi * kappa_s)
    )

    # F_SSS_SIM
    F_SSS_SIM = (
        chi_a * chi_s**2 * (13 * delta**2 * eta * kappa_a / 3 + 27 * delta**2 * kappa_a / 16
                           + 4 * delta * eta * kappa_s / 3 + 15 * delta * kappa_s / 8
                           + 3 * kappa_a / 16)
        + chi_a**2 * chi_s * (95 * delta * eta * kappa_a / 12 + 15 * delta * kappa_a / 8
                             - 26 * eta**2 * kappa_s / 3 + 25 * eta * kappa_s / 6
                             + 15 * kappa_s / 8)
        + chi_s**3 * (-7 * delta * eta * kappa_a / 4 + 5 * delta * kappa_a / 8
                       - 26 * eta**2 * kappa_s / 3 - 3 * eta * kappa_s + 5 * kappa_s / 8)
        + chi_a**3 * (29 * delta * eta * kappa_s / 6 + 5 * delta * kappa_s / 8
                       + 43 * eta * kappa_a / 12 + 5 * kappa_a / 8)
    )

    F_infty = (32/5) * eta**2 * v**10 * (
        1 + F_SS_0 * v**4 + F_SS_2 * v**6 + (F_SS_3 + F_SSS_SIM) * v**7
        + (39/8) * Lambda_tilde * v**10
    )

    # Absorbed Flux dotM
    dotM = (
        0.5 * (9 * H1E_bar * eta**2 * chi_a + 9 * H1E * eta**2 * chi_s) * v**15
        + 0.5 * (
            (9 * H1B * eta**2 + 45 * H1E * eta**2 / 2
             + 9 * H1E_bar * delta * eta**2 / 2 - 27 * H1E * eta**3) * chi_s
            + (9 * H1B_bar * eta**2 + 45 * H1E_bar * eta**2 / 2
               + 9 * H1E * delta * eta**2 / 2 - 27 * H1E_bar * eta**3) * chi_a
        ) * v**17
        + 18 * H0 * eta**2 * v**18
    )

    F_tot = F_infty + dotM

    # dt/dv and df/dt
    dtdv = -M_sec * dedv / F_tot
    dfdt = (3 * v**2 / (np.pi * M_sec)) / dtdv

    # Amplitude
    A_N = np.sqrt(5/24) * Mc_sec**(5/6) * np.pi**(-2/3) * f_masked**(-7/6) / dL_sec
    dfdt_N = 96 * eta * v**11 / (5 * np.pi * M_sec**2)
    A = A_N * np.sqrt(dfdt_N / dfdt)

    # Phase
    # dpsi/df = 2*pi*t(f)
    # t(f) = tc - \int_f^{f_cut} (dt/df') df'
    # dt/df = (dt/dv) / (df/dv) = dtdv / (3 * v**2 / (pi * M_sec))
    dtdf = dtdv / (3 * v**2 / (np.pi * M_sec))
    
    # We need to integrate from some reference. 
    # Let's integrate from f_low.
    # psi(f) = 2*pi*f*tc - phic - pi/4 + 2*pi * \int_{f_low}^f (f - f') (dt/df') df'
    # This choice is equivalent to setting tc and phic at f_low.
    
    # Using cumulative trapezoidal rule for \int (f-f') dtdf(f') df'
    # psi_rel(f) = 2*pi * (f * \int_{f_low}^f dtdf(f') df' - \int_{f_low}^f f' * dtdf(f') df')
    int_dtdf = cumtrapz(dtdf, f_masked, initial=0)
    int_f_dtdf = cumtrapz(f_masked * dtdf, f_masked, initial=0)
    psi_rel = 2 * np.pi * (f_masked * int_dtdf - int_f_dtdf)
    
    psi = 2 * np.pi * f_masked * tc - phic - np.pi/4 + psi_rel

    # Taper
    W = 1 / (1 + np.exp((f_masked - f_cut) / sigma))
    
    # Final waveform
    h_masked = A * np.exp(-1j * psi) * W
    if phase_only:
        h_masked = np.exp(-1j * psi)

    h = np.zeros_like(f, dtype=complex)
    h[mask] = h_masked
    
    return h
