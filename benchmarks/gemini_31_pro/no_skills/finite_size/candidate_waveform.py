import numpy as np
from scipy.integrate import solve_ivp

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
    MSUN_SEC = 4.925491025543576e-6
    MPC_SEC  = 3.0856775814913673e22 / 299792458
    M_sec    = Mc * MSUN_SEC / (eta**(3/5))
    dL_sec   = dL * MPC_SEC
    delta    = np.sqrt(1 - 4 * eta)
    chi_s    = (chi1 + chi2) / 2
    chi_a    = (chi1 - chi2) / 2

    # Frequencies
    f_isco = 1 / (np.pi * 6**(1.5) * M_sec)
    f_cut  = fmax_over_fisco * f_isco
    sigma  = sigma_taper_over_fisco * f_isco

    h_out = np.zeros_like(f, dtype=np.complex128)
    
    valid_mask = (f >= f_low) & (f < f_cut)
    if not np.any(valid_mask):
        return h_out
    
    f_valid = f[valid_mask]
    
    f_unique, inv_idx = np.unique(f_valid, return_inverse=True)
    v_unique = (np.pi * M_sec * f_unique)**(1/3)

    E_SS_0 = chi_a * chi_s * (-delta**2 * kappa_a / 2 - delta * kappa_s - kappa_a / 2) \
             + chi_s**2 * (-delta**2 * kappa_s / 4 - delta * kappa_a / 2 - kappa_s / 4) \
             + chi_a**2 * (-delta * kappa_a / 2 + eta * kappa_s - kappa_s / 2)
    E_SS_2 = chi_a**2 * (25 * delta * eta * kappa_a / 12 - 35 * delta * kappa_a / 12 - 5 * eta**2 * kappa_s / 6 + 95 * eta * kappa_s / 12 - 35 * kappa_s / 12) \
             + chi_a * chi_s * (25 * delta * eta * kappa_s / 6 - 35 * delta * kappa_s / 6 - 5 * eta**2 * kappa_a / 3 + 95 * eta * kappa_a / 6 - 35 * kappa_a / 6) \
             + chi_s**2 * (25 * delta * eta * kappa_a / 12 - 35 * delta * kappa_a / 12 - 5 * eta**2 * kappa_s / 6 + 95 * eta * kappa_s / 12 - 35 * kappa_s / 12)
    E_SSS = chi_a * chi_s**2 * (-2 * delta**2 * eta * kappa_a - 5 * delta**2 * kappa_a + 6 * delta * eta * kappa_s - 6 * delta * kappa_s - kappa_a) \
            + chi_s**3 * (-delta**3 * kappa_a - delta**2 * eta * kappa_s - 9 * delta**2 * kappa_s / 4 - delta * kappa_a + kappa_s / 4) \
            + chi_a**2 * chi_s * (-6 * delta * kappa_a + 4 * eta**2 * kappa_s + 12 * eta * kappa_s - 6 * kappa_s) \
            + chi_a**3 * (-2 * delta * eta * kappa_s - 2 * delta * kappa_s + 2 * eta * kappa_a - 2 * kappa_a)

    def dedv_func(v):
        return -eta/2 * (2*v + 6*E_SS_0 * v**5 + 8*E_SS_2 * v**7 + 9*E_SSS * v**8)

    F_SS_0 = chi_a * chi_s * (delta**2 * kappa_a + 2 * delta * kappa_s + kappa_a) \
             + chi_a**2 * (delta * kappa_a - 2 * eta * kappa_s + kappa_s) \
             + chi_s**2 * (delta * kappa_a - 2 * eta * kappa_s + kappa_s)
    F_SS_2 = chi_a**2 * (-127 * delta * eta * kappa_a / 16 + delta * kappa_a / 14 + 43 * eta**2 * kappa_s / 4 - 905 * eta * kappa_s / 112 + kappa_s / 14) \
             + chi_a * chi_s * (-127 * delta * eta * kappa_s / 8 + delta * kappa_s / 7 + 43 * eta**2 * kappa_a / 2 - 905 * eta * kappa_a / 56 + kappa_a / 7) \
             + chi_s**2 * (-127 * delta * eta * kappa_a / 16 + delta * kappa_a / 14 + 43 * eta**2 * kappa_s / 4 - 905 * eta * kappa_s / 112 + kappa_s / 14)
    F_SS_3 = chi_a**2 * (4 * np.pi * delta * kappa_a - 8 * np.pi * eta * kappa_s + 4 * np.pi * kappa_s) \
             + chi_a * chi_s * (8 * np.pi * delta * kappa_s - 16 * np.pi * eta * kappa_a + 8 * np.pi * kappa_a) \
             + chi_s**2 * (4 * np.pi * delta * kappa_a - 8 * np.pi * eta * kappa_s + 4 * np.pi * kappa_s)
    
    F_SSS_SIM = chi_a * chi_s**2 * (13 * delta**2 * eta * kappa_a / 3 + 27 * delta**2 * kappa_a / 16 + 4 * delta * eta * kappa_s / 3 + 15 * delta * kappa_s / 8 + 3 * kappa_a / 16) \
                + chi_a**2 * chi_s * (95 * delta * eta * kappa_a / 12 + 15 * delta * kappa_a / 8 - 26 * eta**2 * kappa_s / 3 + 25 * eta * kappa_s / 6 + 15 * kappa_s / 8) \
                + chi_s**3 * (-7 * delta * eta * kappa_a / 4 + 5 * delta * kappa_a / 8 - 26 * eta**2 * kappa_s / 3 - 3 * eta * kappa_s + 5 * kappa_s / 8) \
                + chi_a**3 * (29 * delta * eta * kappa_s / 6 + 5 * delta * kappa_s / 8 + 43 * eta * kappa_a / 12 + 5 * kappa_a / 8)

    def F_tot_func(v):
        F_SS_SIM_v = F_SS_0 + F_SS_2 * v**2 + F_SS_3 * v**3
        F_infty = (32/5) * eta**2 * v**10 * (1 + F_SS_SIM_v * v**4 + F_SSS_SIM * v**7 + (39/8) * Lambda_tilde * v**10)
        
        dotM = 0.5 * (9 * H1E_bar * eta**2 * chi_a + 9 * H1E * eta**2 * chi_s) * v**15 \
               + 0.5 * ((9 * H1B * eta**2 + 45 * H1E * eta**2 / 2 + 9 * H1E_bar * delta * eta**2 / 2 - 27 * H1E * eta**3) * chi_s \
                        + (9 * H1B_bar * eta**2 + 45 * H1E_bar * eta**2 / 2 + 9 * H1E * delta * eta**2 / 2 - 27 * H1E_bar * eta**3) * chi_a) * v**17 \
               + 18 * H0 * eta**2 * v**18
        return F_infty + dotM

    def odefun(v, y):
        v = np.maximum(v, 1e-12)
        dt_dv = -M_sec * dedv_func(v) / F_tot_func(v)
        f_val = v**3 / (np.pi * M_sec)
        dphi_dv = 2 * np.pi * f_val * dt_dv
        return [dt_dv, dphi_dv]

    if len(v_unique) == 1:
        t_out = np.array([0.0])
        phi_out = np.array([0.0])
    else:
        v_span = (v_unique[0], v_unique[-1])
        res = solve_ivp(odefun, v_span, [0.0, 0.0], t_eval=v_unique, method='RK45', rtol=1e-10, atol=1e-10)
        t_out = res.y[0]
        phi_out = res.y[1]

    t_valid = t_out[inv_idx]
    phi_valid = phi_out[inv_idx]
    v_valid = v_unique[inv_idx]

    dedv_val = dedv_func(v_valid)
    F_tot_val = F_tot_func(v_valid)
    dt_dv_valid = -M_sec * dedv_val / F_tot_val
    df_dv_valid = 3 * v_valid**2 / (np.pi * M_sec)
    df_dt_valid = df_dv_valid / dt_dv_valid

    psi_valid = 2 * np.pi * f_valid * t_valid - phi_valid - np.pi/4 + 2 * np.pi * f_valid * tc - phic

    if phase_only:
        h_out[valid_mask] = np.exp(-1j * psi_valid)
    else:
        Mc_sec = Mc * MSUN_SEC
        A_N = np.sqrt(5/24) * Mc_sec**(5/6) * np.pi**(-2/3) * f_valid**(-7/6) / dL_sec
        dfdt_N = 96 * eta * v_valid**11 / (5 * np.pi * M_sec**2)
        A = A_N * np.sqrt(dfdt_N / df_dt_valid)
        
        W_f = 1 / (1 + np.exp((f_valid - f_cut) / sigma))
        A = A * W_f
        
        h_out[valid_mask] = A * np.exp(-1j * psi_valid)

    return h_out

if __name__ == "__main__":
    f = np.linspace(20, 2000, 1000)
    h = h_of_f(f, Mc=1.2, eta=0.25, dL=100)
    print(f"Generated waveform. Shape: {h.shape}, Max amplitude: {np.max(np.abs(h))}")
