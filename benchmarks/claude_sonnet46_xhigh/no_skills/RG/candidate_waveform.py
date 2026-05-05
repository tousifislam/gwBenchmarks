"""
RG-tail frequency-domain inspiral waveform (dominant nonspinning (2,2) mode).
Based on arXiv:2602.08833 Section IV.
SPA (stationary phase approximation) assembly.
"""
import numpy as np
from scipy.special import gammaln, loggamma

MSUN_SEC = 4.925491025543576e-6
MPC_SEC  = 3.0856775814913673e22 / 299792458.0
GAMMA_E  = 0.5772156649015328606


def _energy_and_pphi(x, nu):
    """Circular-orbit energy E_real/M and p_phi/(mu M) at PN order x^4."""
    log16x = np.log(16.0 * x)

    c0 = 1.0
    c1 = -3.0/4.0 - nu/12.0
    c2 = -27.0/8.0 + 19.0*nu/8.0 - nu**2/24.0
    c3 = (-675.0/64.0
          + (34445.0/576.0 - 205.0*np.pi**2/96.0)*nu
          - 155.0*nu**2/96.0
          - 35.0*nu**3/5184.0)
    c4 = (-3969.0/128.0
          + (-123671.0/5760.0 + 9037.0*np.pi**2/1536.0
             + 896.0*GAMMA_E/15.0 + 448.0*log16x/15.0)*nu
          + (498449.0/3456.0 - 3157.0*np.pi**2/576.0)*nu**2
          + 301.0*nu**3/1728.0
          + 77.0*nu**4/31104.0)

    E = 1.0 - (nu*x/2.0)*(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4)

    d0 = 1.0
    d1 = 3.0/2.0 + nu/6.0
    d2 = 27.0/8.0 - 19.0*nu/8.0 + nu**2/24.0
    d3 = (135.0/16.0
          + (-6889.0/144.0 + 41.0*np.pi**2/24.0)*nu
          + 31.0*nu**2/24.0
          + 7.0*nu**3/1296.0)
    d4 = (2835.0/128.0
          + (98869.0/5760.0 - 128.0*GAMMA_E/3.0 - 6455.0*np.pi**2/1536.0
             - 64.0*log16x/3.0)*nu
          + (356035.0/3456.0 - 2255.0*np.pi**2/576.0)*nu**2
          - 215.0*nu**3/1728.0
          - 55.0*nu**4/31104.0)

    pphi = x**(-0.5)*(d0 + d1*x + d2*x**2 + d3*x**3 + d4*x**4)

    return E, pphi


def _H_eff(E, nu):
    return ((E**2 - 1.0)/(2.0*nu)) + 1.0


def _tail_factor(x, E, pphi, nu, lambda_RG):
    """Complex T_22 via the RG-tail formula."""
    Omega = x**(3.0/2.0)          # M Omega = x^(3/2)
    m = 2
    khat = E * m * Omega           # (E_real/M) * 2 * Omega
    J = pphi / E**2

    gamma_univ = (-214.0*khat**2/105.0
                  + 2.0*m*J*khat**3/3.0
                  - 3390466.0*khat**4/1157625.0
                  + 381863.0*m*J*khat**5/99225.0)

    ellhat = 2.0 + lambda_RG * gamma_univ

    # 2 k r_omega = 4 sqrt(x)
    log_2krw = np.log(4.0 * np.sqrt(x))

    phi0 = np.exp(17.0/12.0 - GAMMA_E) / 4.0

    # logT_22 using loggamma for complex argument
    arg_gamma = ellhat - 1.0 - 2.0j*khat
    log_gamma_num = loggamma(arg_gamma)
    # log Gamma(2 ellhat + 2): ellhat is real-valued array
    log_gamma_den = gammaln(2.0*ellhat + 2.0)

    logT = (np.log(120.0)
            + (ellhat - 2.0)*log_2krw
            + 2.0j*khat*np.log(4.0*phi0)
            + log_gamma_num
            - log_gamma_den
            + np.pi*khat
            - 1.0j*np.pi*(ellhat - 2.0)/2.0)

    return np.exp(logT)


def _rho22(x, nu):
    """Residual amplitude rho_22 (real)."""
    eulerlog2 = GAMMA_E + np.log(4.0*np.sqrt(x))

    r1 = -43.0/42.0 + 55.0*nu/84.0
    r2 = -20555.0/10584.0 - 33025.0*nu/21168.0 + 19583.0*nu**2/42336.0
    r3 = (-4296031.0/4889808.0
          + (41.0*np.pi**2/192.0 - 48993925.0/9779616.0)*nu
          - 6292061.0*nu**2/3259872.0
          + 10620745.0*nu**3/39118464.0)
    r4 = (9228174993589.0/800950550400.0
          + (-2487107795131.0/145627372800.0
             + 464.0*eulerlog2/35.0
             - 9953.0*np.pi**2/21504.0)*nu
          + (10815863492353.0/640760440320.0 - 3485.0*np.pi**2/5376.0)*nu**2
          - 2088847783.0*nu**3/11650189824.0
          + 70134663541.0*nu**4/512608352256.0)

    return 1.0 + r1*x + r2*x**2 + r3*x**3 + r4*x**4


def _delta22(x, E, nu):
    """Residual phase delta_22."""
    y = (E * x**(3.0/2.0))**(2.0/3.0)
    return (-17.0*y**(3.0/2.0)/3.0
            - 24.0*nu*y**(5.0/2.0)
            + (30995.0*nu/1134.0 + 962.0*nu**2/135.0)*y**(7.0/2.0)
            - 4976.0*np.pi*nu*y**4/105.0)


def _dEdx(x, nu):
    """d(E_real/M)/dx, numerical derivative."""
    dx = x * 1e-5
    Ep, _ = _energy_and_pphi(x + dx, nu)
    Em, _ = _energy_and_pphi(x - dx, nu)
    return (Ep - Em) / (2.0*dx)


def _spa_phase(f_arr, M_sec, nu, lambda_RG, tc, phic):
    """
    Stationary phase approximation Psi(f).
    Integrates dt/df = (dt/dx)(dx/df) from f_low to f.
    Uses the energy balance equation:
       dt/dx = -M dE/dx / F_22
    where F_22 = (32/5) nu^2 x^5 |hhat_22|^2.
    Returns real phase array (SPA Psi).
    """
    # Build x grid for SPA integration
    x_arr = (np.pi * M_sec * f_arr)**(2.0/3.0)
    return x_arr   # placeholder: computed per-frequency below


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
    Frequency-domain RG-tail inspiral waveform h(f).

    Parameters
    ----------
    f : array_like, Hz
    Mc : float, chirp mass [Msun]
    eta : float, symmetric mass ratio
    dL : float, luminosity distance [Mpc]
    tc : float, coalescence time [s]
    phic : float, coalescence phase [rad]
    lambda_RG : float, RG running deformation (1 = GR)
    f_low : float, lower frequency cutoff [Hz]
    fmax_over_fisco : float
    sigma_taper_over_fisco : float
    phase_only : bool, if True return exp(i Psi) only

    Returns
    -------
    h : complex ndarray, same shape as f
    """
    f = np.asarray(f, dtype=float)
    out = np.zeros(f.shape, dtype=complex)

    M_sec  = Mc * MSUN_SEC / eta**(3.0/5.0)   # total mass in seconds
    dL_sec = dL * MPC_SEC

    f_isco  = 1.0 / (np.pi * 6.0**(1.5) * M_sec)
    f_cut   = fmax_over_fisco * f_isco
    sigma   = sigma_taper_over_fisco * f_isco

    # Active frequency mask
    mask = (f >= f_low) & (f < f_cut)
    if not np.any(mask):
        return out

    fa = f[mask]
    x_a = (np.pi * M_sec * fa)**(2.0/3.0)

    # Conservative sector
    E_a, pphi_a = _energy_and_pphi(x_a, eta)
    H_a = _H_eff(E_a, eta)

    # Radiative sector
    T_a   = _tail_factor(x_a, E_a, pphi_a, eta, lambda_RG)
    rho_a = _rho22(x_a, eta)
    d22_a = _delta22(x_a, E_a, eta)

    hhat_a = H_a * T_a * rho_a**2 * np.exp(1.0j * d22_a)

    # SPA: Psi(f) = 2*pi*f*tc - phic - pi/4 + phi_orb(f)
    # phi_orb via energy balance: d phi_orb / df = 2*pi * t(f)
    # t(f) via dt/dx = -M * dE/dx / F_22,  dx/df = (2/3)*(pi*M)^(2/3) * f^(-1/3)
    # We integrate numerically using cumulative trapezoid.

    # Build a finer integration grid (use fa itself, already sorted)
    dEdx_a = _dEdx(x_a, eta)
    F22_a  = (32.0/5.0) * eta**2 * x_a**5 * np.abs(hhat_a)**2

    # dt/dx = -M dE/dx / F_22
    dtdx_a = -M_sec * dEdx_a / F22_a

    # dx/df = (2/3)*(pi M)^(2/3) f^(-1/3)
    dxdf_a = (2.0/3.0) * (np.pi * M_sec)**(2.0/3.0) * fa**(-1.0/3.0)

    dtdf_a = dtdx_a * dxdf_a   # dt/df

    # t(f) relative to t(f[0]) via cumulative trapezoid
    from scipy.integrate import cumulative_trapezoid
    t_rel = np.zeros(len(fa))
    t_rel[1:] = cumulative_trapezoid(dtdf_a, fa)

    # phi_orb(f) = 2*pi * integral t df  (SPA: dPsi/df = 2*pi*t)
    # We need the double integral: phi_orb = 2*pi * integral_{f0}^{f} t(f') df'
    # which equals 2*pi * integral_{f0}^{f} [t(f0) + t_rel(f')] df'
    # But t(f0) is unknown (it is the absolute coalescence time reference).
    # Standard SPA: Psi(f) = 2*pi*f*tc - 2*phic - pi/4 + Psi_PN(f)
    # where Psi_PN(f) is constructed by integrating d^2Psi/df^2 = 2*pi * dt/df twice.
    # We use the standard SPA result by directly integrating twice.

    # Second integral: phi_orb(f) = 2*pi * int_{fa[0]}^{f} t_rel(f') df'  + const
    phi_orb = np.zeros(len(fa))
    phi_orb[1:] = 2.0*np.pi * cumulative_trapezoid(t_rel, fa)

    Psi_a = 2.0*np.pi*fa*tc - 2.0*phic - np.pi/4.0 + phi_orb

    # SPA amplitude: A(f) = sqrt(2*pi/|ddPsi/df^2|) * (source factor)
    # ddPsi/df^2 = 2*pi * dt/df
    ddPsi_a = 2.0*np.pi * dtdf_a   # should be negative (dt/df < 0 inspiral)

    # Newtonian prefactor from SPA:
    # h(f) = -(sqrt(5/24)/pi^(2/3)) * (Mc_sec)^(5/6) / dL_sec * f^(-7/6)
    #        * hhat * sqrt(2*pi / |ddPsi|) * exp(i Psi)
    # We fold everything together.

    Mc_sec = Mc * MSUN_SEC
    # Standard SPA Newtonian amplitude (single-detector, face-on, optimal):
    amp_newt = (np.sqrt(5.0/24.0) / np.pi**(2.0/3.0)
                * Mc_sec**(5.0/6.0) / dL_sec
                * fa**(-7.0/6.0))

    # SPA correction from second derivative:
    # sqrt(2*pi / |ddPsi/df^2|)  -- note ddPsi is negative, take abs
    # This factor modifies the Newtonian amplitude by |hhat| * sqrt correction.
    # But we already encode hhat corrections; combine carefully.
    # Full SPA: h(f) = A_N * hhat * sqrt(2*pi/|ddPsi|) * exp(i Psi)
    # The Newtonian ddPsi = -2*pi*(96/5)*pi^(8/3)*Mc_sec^(5/3)*f^(-11/3)/M_sec^? ...
    # It is cleanest to keep the sqrt(1/|ddPsi|) factor from our numerical ddPsi.

    amp_spa = amp_newt * np.abs(hhat_a) * np.sqrt(2.0*np.pi / np.abs(ddPsi_a))

    # Taper
    W_a = 1.0 / (1.0 + np.exp((fa - f_cut) / sigma))

    if phase_only:
        h_a = np.exp(1.0j * Psi_a) * W_a
    else:
        h_a = amp_spa * np.exp(1.0j * Psi_a) * W_a

    out[mask] = h_a
    return out
