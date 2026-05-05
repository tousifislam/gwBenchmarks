import numpy as np
from scipy.special import loggamma
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

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
    RG-tail frequency-domain strain h(f).
    """
    # Constants
    MSUN_SEC = 4.925491025543576e-6
    MPC_SEC  = 3.0856775814913673e22 / 299792458
    GAMMA_E  = 0.5772156649015328606

    # Mass and distance in seconds
    M_sec = Mc * MSUN_SEC / (eta**0.6)
    dL_sec = dL * MPC_SEC
    nu = eta

    # Frequencies and Cutoffs
    f_isco = 1.0 / (np.pi * (6.0**1.5) * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    # Frequency mask
    mask = (f >= f_low) & (f < f_cut)
    h = np.zeros_like(f, dtype=complex)
    if not np.any(mask):
        return h

    f_active = f[mask]
    
    # Internal variables for active frequencies
    x_active = (np.pi * M_sec * f_active)**(2/3)

    # We need to integrate to find the phase. 
    # Use a log-spaced grid to handle the steep x^-4 dependence of dt/dx.
    x_min = (np.pi * M_sec * f_low)**(2/3)
    x_max = (np.pi * M_sec * f_cut)**(2/3)
    
    N_grid = 3000
    x_grid = np.geomspace(x_min, x_max, N_grid)
    
    def get_ingredients(x):
        # Conservative Sector
        c0 = 1.0
        c1 = -3/4 - nu/12
        c2 = -27/8 + 19*nu/8 - nu**2/24
        c3 = -675/64 + (34445/576 - 205*np.pi**2/96)*nu - 155*nu**2/96 - 35*nu**3/5184
        
        # c4 and its derivative terms
        log16x = np.log(16*x)
        c4_part = -3969/128 + (-123671/5760 + 9037*np.pi**2/1536 + 896*GAMMA_E/15 + 448*log16x/15)*nu \
                  + (498449/3456 - 3157*np.pi**2/576)*nu**2 + 301*nu**3/1728 + 77*nu**4/31104
        
        # E_real/M = 1 - (nu*x/2) * poly
        poly = (c0 + c1*x + c2*x**2 + c3*x**3 + c4_part*x**4)
        e_real_over_m = 1.0 - (nu * x / 2.0) * poly
        
        # d(E/M)/dx = - (nu/2) * (d/dx(x*poly))
        # d/dx(x*poly) = poly + x*poly'
        # x*poly' = x * (c1 + 2c2x + 3c3x^2 + 4c4x^3 + x^4*dc4/dx)
        # x^5 * dc4/dx = nu * 448/15 * x^4
        d_xpoly_dx = poly + x * (c1 + 2*c2*x + 3*c3*x**2 + 4*c4_part*x**3 + (nu * 448/15) * x**3)
        de_dm_dx = - (nu / 2.0) * d_xpoly_dx

        # Angular Momentum
        d0 = 1.0
        d1 = 3/2 + nu/6
        d2 = 27/8 - 19*nu/8 + nu**2/24
        d3 = 135/16 + (-6889/144 + 41*np.pi**2/24)*nu + 31*nu**2/24 + 7*nu**3/1296
        d4_part = 2835/128 + (98869/5760 - 128*GAMMA_E/3 - 6455*np.pi**2/1536 - 64*log16x/3)*nu \
                  + (356035/3456 - 2255*np.pi**2/576)*nu**2 - 215*nu**3/1728 - 55*nu**4/31104
        
        p_phi_circ_mu_m = x**(-0.5) * (d0 + d1*x + d2*x**2 + d3*x**3 + d4_part*x**4)
        
        # Effective Source: H_eff = ((e^2 - 1)/(2 nu)) + 1
        # e = 1 - nu * A, where A = (x/2)*poly
        # (e^2 - 1)/(2 nu) = (-2 nu A + nu^2 A^2) / (2 nu) = -A + 0.5 * nu * A^2
        A_val = (x / 2.0) * poly
        H_eff = 1.0 - A_val + 0.5 * nu * (A_val**2)
        
        # Running Tail
        m = 2
        omega_m = x**1.5 # Dimensionless Omega*M
        khat = e_real_over_m * m * omega_m
        J = p_phi_circ_mu_m / (e_real_over_m**2)
        
        gamma_22_univ = -214 * khat**2 / 105 \
                        + 2 * m * J * khat**3 / 3 \
                        - 3390466 * khat**4 / 1157625 \
                        + 381863 * m * J * khat**5 / 99225
        
        ellhat_22 = 2.0 + lambda_RG * gamma_22_univ
        
        log4sqrtx = np.log(4.0 * np.sqrt(x))
        log4phi0 = 17/12 - GAMMA_E
        
        logT_22 = np.log(120.0) + (ellhat_22 - 2.0) * log4sqrtx + 2j * khat * log4phi0 \
                  + loggamma(ellhat_22 - 1.0 - 2j * khat) - loggamma(2.0 * ellhat_22 + 2.0) \
                  + np.pi * khat - 0.5j * np.pi * (ellhat_22 - 2.0)
        
        T_22 = np.exp(logT_22)
        
        # Residual Amplitude and Phase
        eulerlog_2 = GAMMA_E + log4sqrtx
        r1 = -43/42 + 55*nu/84
        r2 = -20555/10584 - 33025*nu/21168 + 19583*nu**2/42336
        r3 = -4296031/4889808 + (41*np.pi**2/192 - 48993925/9779616)*nu - 6292061*nu**2/3259872 + 10620745*nu**3/39118464
        r4 = 9228174993589/800950550400 \
             + (-2487107795131/145627372800 + 464*eulerlog_2/35 - 9953*np.pi**2/21504)*nu \
             + (10815863492353/640760440320 - 3485*np.pi**2/5376)*nu**2 \
             - 2088847783*nu**3/11650189824 + 70134663541*nu**4/512608352256
        
        rho_22 = 1.0 + r1*x + r2*x**2 + r3*x**3 + r4*x**4
        
        y = (e_real_over_m * x**1.5)**(2/3)
        delta_22 = -17 * y**1.5 / 3.0 \
                   - 24 * nu * y**2.5 \
                   + (30995*nu/1134 + 962*nu**2/135) * y**3.5 \
                   - 4976 * np.pi * nu * y**4 / 105.0
        
        hhat_22 = H_eff * T_22 * (rho_22**2) * np.exp(1j * delta_22)
        
        return de_dm_dx, hhat_22

    # Vectorized evaluation for the grid
    de_dm_dx_grid, hhat_22_grid = get_ingredients(x_grid)
    
    # Flux and dt/dx
    F_22_grid = (32.0/5.0) * (nu**2) * (x_grid**5) * np.abs(hhat_22_grid)**2
    dtdx_grid = - M_sec * de_dm_dx_grid / F_22_grid
    
    # Time and orbital phase integrals from x up to x_max
    # t(x) = tc - integral_{x}^{x_max} (dt/dx') dx'
    # phi(x) = phic - integral_{x}^{x_max} (Omega*dt/dx') dx'
    # We integrate from x_max down to x. Since cumtrapz integrates in forward order,
    # we flip the grid and flip back.
    omega_grid = x_grid**1.5 / M_sec
    
    # Integral from x_max down to x
    it_grid = np.concatenate([[0], cumtrapz(dtdx_grid[::-1], x_grid[::-1])])[::-1]
    iphi_grid = np.concatenate([[0], cumtrapz((omega_grid * dtdx_grid)[::-1], x_grid[::-1])])[::-1]
    
    # Interpolate time and phase integrals
    it_interp = interp1d(x_grid, it_grid, kind='cubic', bounds_error=False, fill_value="extrapolate")(x_active)
    iphi_interp = interp1d(x_grid, iphi_grid, kind='cubic', bounds_error=False, fill_value="extrapolate")(x_active)
    
    t_f = tc + it_interp
    phi_f = phic + iphi_interp
    
    # Re-evaluate exactly for active frequencies
    de_dm_dx_active, hhat_22_active = get_ingredients(x_active)
    
    # SPA Phase
    omega_active = x_active**1.5 / M_sec
    psi_f = 2.0 * omega_active * t_f - 2.0 * phi_f + np.angle(hhat_22_active) - np.pi/4.0
    
    if phase_only:
        h[mask] = np.exp(1j * psi_f)
    else:
        # SPA Amplitude
        # Note: de_dm_dx_active is negative, so use abs
        amp_f = (M_sec**2 / dL_sec) * np.sqrt( (8.0 * np.pi**2 * np.abs(de_dm_dx_active)) / (3.0 * x_active**3.5) )
        
        # Window
        W_f = 1.0 / (1.0 + np.exp((f_active - f_cut) / sigma))
        
        h[mask] = amp_f * np.exp(1j * psi_f) * W_f
        
    return h
