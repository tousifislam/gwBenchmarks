"""Frequency-domain (2,2) inspiral waveform with RG-tail corrections (arXiv:2602.08833).

Implements the factorized EOB inspiral mode with running-gravitational
anomalous dimension, using the balance-law SPA for FD construction.
"""

import numpy as np
from scipy.special import loggamma
from scipy.integrate import cumulative_trapezoid

MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 1.0292712503e14


def _energy_over_M(x, nu):
    """E_real/M from the conservative PN expansion."""
    nu2, nu3, nu4 = nu**2, nu**3, nu**4
    gE = np.euler_gamma

    c0 = 1.0
    c1 = -3.0/4 - nu/12
    c2 = -27.0/8 + 19*nu/8 - nu2/24
    c3 = -675.0/64 + (34445.0/576 - 205*np.pi**2/96)*nu - 155*nu2/96 - 35*nu3/5184
    c4 = (-3969.0/128
          + (-123671.0/5760 + 9037*np.pi**2/1536 + 896*gE/15 + 448*np.log(16*x)/15)*nu
          + (498449.0/3456 - 3157*np.pi**2/576)*nu2
          + 301*nu3/1728 + 77*nu4/31104)

    bracket = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
    return 1.0 - (nu*x/2) * bracket


def _d_energy_dx(x, nu):
    """Analytic derivative d(E_real/M)/dx.

    Since c4 contains log(16*x), derivative requires chain rule on that term.
    E/M = 1 - (nu*x/2)*S where S = sum c_n * x^n
    d(E/M)/dx = -(nu/2)*(S + x*dS/dx)
    d(c4*x^4)/dx = 4*c4*x^3 + (dc4/dx)*x^4 = 4*c4*x^3 + (448*nu/15)*(1/x)*x^4
                 = (4*c4 + 448*nu/15)*x^3
    """
    nu2, nu3, nu4 = nu**2, nu**3, nu**4
    gE = np.euler_gamma

    c0 = 1.0
    c1 = -3.0/4 - nu/12
    c2 = -27.0/8 + 19*nu/8 - nu2/24
    c3 = -675.0/64 + (34445.0/576 - 205*np.pi**2/96)*nu - 155*nu2/96 - 35*nu3/5184

    c4_log_coeff = 448*nu/15
    c4_rest = (-3969.0/128
               + (-123671.0/5760 + 9037*np.pi**2/1536 + 896*gE/15)*nu
               + (498449.0/3456 - 3157*np.pi**2/576)*nu2
               + 301*nu3/1728 + 77*nu4/31104)
    c4 = c4_rest + c4_log_coeff * np.log(16*x)

    S = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
    dS_dx = c1 + 2*c2*x + 3*c3*x**2 + (4*c4 + c4_log_coeff)*x**3

    return -(nu/2) * (S + x*dS_dx)


def _angular_momentum(x, nu):
    """p_phi/(mu*M) circular orbital angular momentum."""
    nu2, nu3, nu4 = nu**2, nu**3, nu**4
    gE = np.euler_gamma

    d0 = 1.0
    d1 = 3.0/2 + nu/6
    d2 = 27.0/8 - 19*nu/8 + nu2/24
    d3 = 135.0/16 + (-6889.0/144 + 41*np.pi**2/24)*nu + 31*nu2/24 + 7*nu3/1296
    d4 = (2835.0/128
          + (98869.0/5760 - 128*gE/3 - 6455*np.pi**2/1536 - 64*np.log(16*x)/3)*nu
          + (356035.0/3456 - 2255*np.pi**2/576)*nu2
          - 215*nu3/1728 - 55*nu4/31104)

    bracket = d0 + d1*x + d2*x**2 + d3*x**3 + d4*x**4
    return bracket * x**(-0.5)


def _build_hhat22(x, nu, lambda_RG):
    """Factorized hhat_22 = H_eff * T_22 * rho_22^2 * exp(i*delta_22)."""
    gE = np.euler_gamma
    m = 2  # mode number

    E = _energy_over_M(x, nu)
    pphi = _angular_momentum(x, nu)

    # Effective source
    H_eff = (E**2 - 1) / (2*nu) + 1

    # RG running tail
    Omega = x**1.5
    khat = E * m * Omega
    J = pphi / E**2

    gamma_univ = (-214*khat**2/105
                  + 2*m*J*khat**3/3
                  - 3390466*khat**4/1157625
                  + 381863*m*J*khat**5/99225)
    ellhat = 2.0 + lambda_RG * gamma_univ

    # Tail factor
    k = 2*Omega
    r_omega = 1.0/x
    phi0 = np.exp(17.0/12 - gE) / 4

    logT = (np.log(120)
            + (ellhat - 2) * np.log(2*k*r_omega)
            + 2j*khat * np.log(2*m*phi0)
            + loggamma(ellhat - 1 - 2j*khat)
            - loggamma(2*ellhat + 2)
            + np.pi*khat
            - 1j*np.pi*(ellhat - 2)/2)
    T_22 = np.exp(logT)

    # Residual amplitude
    nu2, nu3, nu4 = nu**2, nu**3, nu**4
    eulerlog2 = gE + np.log(4*np.sqrt(x))

    r1 = -43.0/42 + 55*nu/84
    r2 = -20555.0/10584 - 33025*nu/21168 + 19583*nu2/42336
    r3 = (-4296031.0/4889808
          + (41*np.pi**2/192 - 48993925.0/9779616)*nu
          - 6292061*nu2/3259872 + 10620745*nu3/39118464)
    r4 = (9228174993589.0/800950550400
          + (-2487107795131.0/145627372800 + 464*eulerlog2/35 - 9953*np.pi**2/21504)*nu
          + (10815863492353.0/640760440320 - 3485*np.pi**2/5376)*nu2
          - 2088847783*nu3/11650189824 + 70134663541*nu4/512608352256)
    rho22 = 1 + r1*x + r2*x**2 + r3*x**3 + r4*x**4

    # Residual phase
    y = (E * x**1.5)**(2.0/3)
    delta22 = (-17*y**1.5/3
               - 24*nu*y**2.5
               + (30995*nu/1134 + 962*nu2/135)*y**3.5
               - 4976*np.pi*nu*y**4/105)

    return H_eff * T_22 * rho22**2 * np.exp(1j*delta22)


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
    """Return complex frequency-domain strain array with same shape as f."""
    f = np.asarray(f, dtype=np.float64)
    h = np.zeros_like(f, dtype=np.complex128)
    nu = eta

    Mc_sec = Mc * MSUN_SEC
    M_sec = Mc_sec / eta**(3.0/5)
    dL_sec = dL * MPC_SEC

    f_isco = 1.0 / (np.pi * 6**1.5 * M_sec)
    f_cut = fmax_over_fisco * f_isco
    sigma = sigma_taper_over_fisco * f_isco

    # Valid band
    mask = (f >= f_low) & (f < f_cut)
    if not np.any(mask):
        return h

    fv = f[mask]
    x = (np.pi * M_sec * fv)**(2.0/3)

    # hhat_22 at each frequency
    hhat22 = _build_hhat22(x, nu, lambda_RG)

    # --- SPA phase via balance-law integration ---
    # x_ref is at f_cut
    x_ref = (np.pi * M_sec * f_cut)**(2.0/3)

    # Build a fine grid for integration from x[0] to x_ref
    # Include all evaluation points plus x_ref as endpoint
    if x[-1] < x_ref:
        x_grid = np.append(x, x_ref)
    else:
        x_grid = x.copy()

    # Compute dt/dx on the grid
    dE_grid = _d_energy_dx(x_grid, nu)
    hhat_grid = _build_hhat22(x_grid, nu, lambda_RG)
    F22_grid = (32.0/5) * nu**2 * x_grid**5 * np.abs(hhat_grid)**2
    dt_dx_grid = -M_sec * dE_grid / F22_grid

    # Two integrands:
    # I1 = 2*(x^(3/2)/M_sec) * dt/dx  (for orbital phase)
    # I2 = dt/dx                        (for time delay)
    I1 = 2 * (x_grid**1.5 / M_sec) * dt_dx_grid
    I2 = dt_dx_grid

    # Need integral from x[i] to x_ref for each point
    # = total_integral - integral_from_x[0]_to_x[i]
    # Use cumulative_trapezoid from scipy
    cum_I1 = np.concatenate(([0.0], cumulative_trapezoid(I1, x_grid)))
    cum_I2 = np.concatenate(([0.0], cumulative_trapezoid(I2, x_grid)))

    # Integral from x[i] to x_ref = total - cum[i]
    total_I1 = cum_I1[-1]
    total_I2 = cum_I2[-1]

    # Extract values at the original x points (not the appended x_ref)
    n_orig = len(x)
    int_phase = total_I1 - cum_I1[:n_orig]
    int_time = total_I2 - cum_I2[:n_orig]

    Psi_orb = int_phase - 2*np.pi*fv * int_time

    # Total phase
    phase = 2*np.pi*fv*tc - phic - np.pi/4 + Psi_orb + np.angle(hhat22)

    # --- Amplitude ---
    A_N = np.sqrt(5.0/24) * Mc_sec**(5.0/6) / (dL_sec * np.pi**(2.0/3)) * fv**(-7.0/6)

    if phase_only:
        amp = A_N
    else:
        dE_dx = _d_energy_dx(x, nu)
        amp = A_N * np.sqrt(np.maximum(-2*dE_dx/nu, 0.0))

    # Fermi taper
    W = 1.0 / (1.0 + np.exp((fv - f_isco)/sigma))

    h[mask] = amp * np.exp(1j*phase) * W
    return h
