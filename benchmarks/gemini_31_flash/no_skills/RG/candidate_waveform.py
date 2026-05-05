import numpy as np
from scipy.special import loggamma
from scipy.integrate import cumulative_trapezoid

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
    RG Waveform Frequency-Domain Template.
    
    Parameters:
        f (array_like): Frequency array in Hz.
        Mc (float): Chirp mass in Solar masses.
        eta (float): Symmetric mass ratio.
        dL (float): Luminosity distance in Mpc.
        tc (float): Time of coalescence in seconds.
        phic (float): Coalescence phase.
        lambda_RG (float): RG running deformation parameter (1.0 for GR).
        f_low (float): Lower frequency cutoff in Hz.
        fmax_over_fisco (float): Upper cutoff in units of f_isco.
        sigma_taper_over_fisco (float): Tapering width in units of f_isco.
        phase_only (bool): If True, returns only the phase factor exp(i*Psi).
        
    Returns:
        ndarray: Complex frequency-domain strain.
    """
    # Constants
    MSUN_SEC = 4.925491025543576e-6
    MPC_SEC  = 3.0856775814913673e22 / 299792458
    gamma_E  = 0.57721566490153286
    
    nu = eta
    M_sec = Mc * MSUN_SEC / eta**0.6
    dL_sec = dL * MPC_SEC
    
    # ISCO and cutoffs
    f_isco = 1.0 / (np.pi * 6**1.5 * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco
    
    # Frequency handling
    f = np.atleast_1d(f)
    sort_idx = np.argsort(f)
    f_sorted = f[sort_idx]
    
    mask = (f_sorted >= f_low) & (f_sorted < f_cut)
    res_sorted = np.zeros_like(f_sorted, dtype=complex)
    
    if not np.any(mask):
        inv_sort_idx = np.argsort(sort_idx)
        return res_sorted[inv_sort_idx]
    
    f_m = f_sorted[mask]
    x = (np.pi * M_sec * f_m)**(2/3)
    
    # Conservative Sector: PN Coefficients for Energy
    c0 = 1.0
    c1 = -3/4 - nu/12
    c2 = -27/8 + 19*nu/8 - nu**2/24
    c3 = -675/64 + (34445/576 - 205*np.pi**2/96)*nu - 155*nu**2/96 - 35*nu**3/5184
    c4 = -3969/128 + (-123671/5760 + 9037*np.pi**2/1536 + 896*gamma_E/15 + 448*np.log(16*x)/15)*nu \
         + (498449/3456 - 3157*np.pi**2/576)*nu**2 + 301*nu**3/1728 + 77*nu**4/31104
         
    E_over_M = 1 - (nu * x / 2) * (c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4)
    # Derivative d(E/M)/dx
    dE_dx = - (nu / 2) * (c0 + 2*c1*x + 3*c2*x**2 + 4*c3*x**3 + 5*c4*x**4)
    # Account for log(16x) in c4: d(nu/2 * c4 * x^5)/dx includes (nu/2) * x^5 * d c4 / dx
    # d c4 / dx = nu * (448/15) * (1/x) => x^5 d c4 / dx = nu * (448/15) * x^4
    dE_dx -= (nu/2) * (nu * 448/15 * x**4)
    
    # PN Coefficients for Angular Momentum (needed for J and khat)
    d0 = 1.0
    d1 = 3/2 + nu/6
    d2 = 27/8 - 19*nu/8 + nu**2/24
    d3 = 135/16 + (-6889/144 + 41*np.pi**2/24)*nu + 31*nu**2/24 + 7*nu**3/1296
    d4 = 2835/128 + (98869/5760 - 128*gamma_E/3 - 6455*np.pi**2/1536 - 64*np.log(16*x)/3)*nu \
         + (356035/3456 - 2255*np.pi**2/576)*nu**2 - 215*nu**3/1728 - 55*nu**4/31104
         
    p_phi_over_muM = x**(-0.5) * (d0 + d1*x + d2*x**2 + d3*x**3 + d4*x**4)
    
    H_eff = ((E_over_M)**2 - 1) / (2 * nu) + 1
    
    # Running Tail Factor
    m = 2
    Omega_dim = np.pi * M_sec * f_m
    khat = E_over_M * m * Omega_dim
    J = p_phi_over_muM / (E_over_M**2)
    
    gamma_22_univ = -214 * khat**2 / 105 + 2 * m * J * khat**3 / 3 \
                    - 3390466 * khat**4 / 1157625 + 381863 * m * J * khat**5 / 99225
    
    ellhat_22 = 2 + lambda_RG * gamma_22_univ
    
    phi0 = np.exp(17/12 - gamma_E) / 4
    # logT_22 formula from Section IV
    logT_22 = np.log(120.0) \
              + (ellhat_22 - 2) * np.log(4 * np.sqrt(x)) \
              + 2j * khat * np.log(4 * phi0) \
              + loggamma(ellhat_22 - 1 - 2j * khat) \
              - loggamma(2 * ellhat_22 + 2) \
              + np.pi * khat \
              - 0.5j * np.pi * (ellhat_22 - 2)
    
    T_22 = np.exp(logT_22)
    
    # Residual Amplitude and Phase
    eulerlog_2 = gamma_E + np.log(4 * np.sqrt(x))
    r1 = -43/42 + 55*nu/84
    r2 = -20555/10584 - 33025*nu/21168 + 19583*nu**2/42336
    r3 = -4296031/4889808 + (41*np.pi**2/192 - 48993925/9779616)*nu - 6292061*nu**2/3259872 + 10620745*nu**3/39118464
    r4 = 9228174993589/800950550400 \
         + (-2487107795131/145627372800 + 464*eulerlog_2/35 - 9953*np.pi**2/21504)*nu \
         + (10815863492353/640760440320 - 3485*np.pi**2/5376)*nu**2 \
         - 2088847783*nu**3/11650189824 + 70134663541*nu**4/512608352256
         
    rho_22 = 1 + r1*x + r2*x**2 + r3*x**3 + r4*x**4
    
    y = (E_over_M * x**1.5)**(2/3)
    delta_22 = -17 * y**1.5 / 3 - 24 * nu * y**2.5 + (30995 * nu / 1134 + 962 * nu**2 / 135) * y**3.5 - 4976 * np.pi * nu * y**4 / 105
    
    # Complex correction factor
    hhat_22 = H_eff * T_22 * (rho_22**2) * np.exp(1j * delta_22)
    
    # Flux and Orbital Phase Integration
    F_22 = (32/5) * (nu**2) * (x**5) * (np.abs(hhat_22)**2)
    
    # dt/df = -M (dE/dx) (dx/df) / F_22
    dx_df = (2/3) * np.pi * M_sec * (np.pi * M_sec * f_m)**(-1/3)
    dt_df = - M_sec * dE_dx * dx_df / F_22
    
    # Numerically integrate for orbital time and phase
    # t(f) = int_{f_start}^f (dt/df') df'
    t_of_f = cumulative_trapezoid(dt_df, f_m, initial=0.0)
    # psi_orb = 2*pi * int_{f_start}^f t(f') df'
    psi_orb = 2 * np.pi * cumulative_trapezoid(t_of_f, f_m, initial=0.0)
    
    # Total SPA Phase
    # Psi(f) = 2 pi f tc - phic - pi/4 + arg(hhat_22) + psi_orb
    phase = 2 * np.pi * f_m * tc - phic - np.pi/4 + np.angle(hhat_22) + psi_orb
    
    if phase_only:
        res_sorted[mask] = np.exp(1j * phase)
    else:
        # Amplitude A(f) = (M^2/dL) * (2 pi / sqrt(3)) * x^{-7/4} * sqrt(-dE/dx)
        # Note: |hhat_22| cancels out in SPA amplitude when same factor used in flux.
        amp = (M_sec**2 / dL_sec) * (2 * np.pi / np.sqrt(3)) * (x**(-1.75)) * np.sqrt(-dE_dx)
        # Tapering window
        W_f = 1.0 / (1.0 + np.exp((f_m - f_cut) / sigma))
        res_sorted[mask] = amp * W_f * np.exp(1j * phase)
        
    # Unsort to match input order
    inv_sort_idx = np.argsort(sort_idx)
    return res_sorted[inv_sort_idx]
