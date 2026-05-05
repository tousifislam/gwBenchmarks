"""
Finite-size balance-law SPA waveform.
Based on arXiv:2410.00294 Appendix A.

We use:
  e(v)     = E(v)/M  (binding energy per unit mass, simplified spin-induced form)
  F_infty  = flux radiated to infinity
  dotM     = absorbed flux (BBH 'doubled' convention)
  F_tot    = F_infty + dotM
  dt/dv    = -M_sec * (de/dv) / F_tot
  dphi_orb/dv = (v^3/M_sec) * dt/dv  =  -v^3 * (de/dv) / F_tot
  f        = v^3 / (pi M_sec)
  Psi(f)   = 2*pi*f*tc - 2*phic - pi/4 + 2*pi*f*t(v_f) - 2*phi_orb(v_f)
  h(f)     = A(f) * exp[-i Psi(f)]
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

MSUN_SEC = 4.925491025543576e-6
MPC_SEC  = 3.0856775814913673e22 / 299792458.0


def _binding_energy_terms(v, eta, chi_s, chi_a, delta, kappa_s, kappa_a):
    """e(v) = E/M and de/dv (analytic) from the simplified Appendix-A form.

    e(v) = -eta v^2/2 * [1 + E_SS_SIM(v) v^4 + E_SSS_SIM v^7]
    where E_SS_SIM(v) = a + b v^2  (constant + v^2 piece).
    """
    # E_SS_SIM: constant piece (a)
    a = (chi_a*chi_s*(-delta**2*kappa_a/2 - delta*kappa_s - kappa_a/2)
         + chi_s**2*(-delta**2*kappa_s/4 - delta*kappa_a/2 - kappa_s/4)
         + chi_a**2*(-delta*kappa_a/2 + eta*kappa_s - kappa_s/2))

    # E_SS_SIM: v^2 piece (b)
    b = (chi_a**2*(25*delta*eta*kappa_a/12 - 35*delta*kappa_a/12
                   - 5*eta**2*kappa_s/6 + 95*eta*kappa_s/12 - 35*kappa_s/12)
         + chi_a*chi_s*(25*delta*eta*kappa_s/6 - 35*delta*kappa_s/6
                        - 5*eta**2*kappa_a/3 + 95*eta*kappa_a/6 - 35*kappa_a/6)
         + chi_s**2*(25*delta*eta*kappa_a/12 - 35*delta*kappa_a/12
                     - 5*eta**2*kappa_s/6 + 95*eta*kappa_s/12 - 35*kappa_s/12))

    # E_SSS_SIM (constant, no v dependence)
    c = (chi_a*chi_s**2*(-2*delta**2*eta*kappa_a - 5*delta**2*kappa_a
                         + 6*delta*eta*kappa_s - 6*delta*kappa_s - kappa_a)
         + chi_s**3*(-delta**3*kappa_a - delta**2*eta*kappa_s
                     - 9*delta**2*kappa_s/4 - delta*kappa_a + kappa_s/4)
         + chi_a**2*chi_s*(-6*delta*kappa_a + 4*eta**2*kappa_s
                           + 12*eta*kappa_s - 6*kappa_s)
         + chi_a**3*(-2*delta*eta*kappa_s - 2*delta*kappa_s
                     + 2*eta*kappa_a - 2*kappa_a))

    # Expand e(v) = -eta/2 * [v^2 + a v^6 + b v^8 + c v^9]
    e_val   = -eta/2.0 * (v**2 + a*v**6 + b*v**8 + c*v**9)
    de_dv   = -eta/2.0 * (2.0*v + 6.0*a*v**5 + 8.0*b*v**7 + 9.0*c*v**8)
    return e_val, de_dv


def _flux_infty(v, eta, chi_s, chi_a, delta, kappa_s, kappa_a, Lambda_tilde):
    """F_infty(v) per Appendix-A simplified form."""
    # F_SS_SIM constant piece
    f_ss0 = (chi_a*chi_s*(delta**2*kappa_a + 2*delta*kappa_s + kappa_a)
             + chi_a**2*(delta*kappa_a - 2*eta*kappa_s + kappa_s)
             + chi_s**2*(delta*kappa_a - 2*eta*kappa_s + kappa_s))

    # F_SS_SIM v^2 piece
    f_ss2 = (chi_a**2*(-127*delta*eta*kappa_a/16 + delta*kappa_a/14
                       + 43*eta**2*kappa_s/4 - 905*eta*kappa_s/112
                       + kappa_s/14)
             + chi_a*chi_s*(-127*delta*eta*kappa_s/8 + delta*kappa_s/7
                            + 43*eta**2*kappa_a/2 - 905*eta*kappa_a/56
                            + kappa_a/7)
             + chi_s**2*(-127*delta*eta*kappa_a/16 + delta*kappa_a/14
                         + 43*eta**2*kappa_s/4 - 905*eta*kappa_s/112
                         + kappa_s/14))

    # F_SS_SIM v^3 piece
    f_ss3 = (chi_a**2*(4*np.pi*delta*kappa_a - 8*np.pi*eta*kappa_s + 4*np.pi*kappa_s)
             + chi_a*chi_s*(8*np.pi*delta*kappa_s - 16*np.pi*eta*kappa_a + 8*np.pi*kappa_a)
             + chi_s**2*(4*np.pi*delta*kappa_a - 8*np.pi*eta*kappa_s + 4*np.pi*kappa_s))

    F_SS_SIM = f_ss0 + f_ss2*v**2 + f_ss3*v**3

    # F_SSS_SIM
    F_SSS_SIM = (chi_a*chi_s**2*(13*delta**2*eta*kappa_a/3 + 27*delta**2*kappa_a/16
                                 + 4*delta*eta*kappa_s/3 + 15*delta*kappa_s/8
                                 + 3*kappa_a/16)
                 + chi_a**2*chi_s*(95*delta*eta*kappa_a/12 + 15*delta*kappa_a/8
                                   - 26*eta**2*kappa_s/3 + 25*eta*kappa_s/6
                                   + 15*kappa_s/8)
                 + chi_s**3*(-7*delta*eta*kappa_a/4 + 5*delta*kappa_a/8
                             - 26*eta**2*kappa_s/3 - 3*eta*kappa_s + 5*kappa_s/8)
                 + chi_a**3*(29*delta*eta*kappa_s/6 + 5*delta*kappa_s/8
                             + 43*eta*kappa_a/12 + 5*kappa_a/8))

    bracket = (1.0 + F_SS_SIM*v**4 + F_SSS_SIM*v**7
               + (39.0/8.0)*Lambda_tilde*v**10)
    return (32.0/5.0) * eta**2 * v**10 * bracket


def _dotM(v, eta, chi_s, chi_a, H0, H1E, H1B, H1E_bar, H1B_bar, delta):
    """Absorbed flux dotM(v).  Uses BBH '18 H0 eta^2 v^18' convention."""
    term15 = 0.5 * (9.0*H1E_bar*eta**2*chi_a + 9.0*H1E*eta**2*chi_s) * v**15

    coeff_chis = (9.0*H1B*eta**2 + 45.0*H1E*eta**2/2.0
                  + 9.0*H1E_bar*delta*eta**2/2.0 - 27.0*H1E*eta**3)
    coeff_chia = (9.0*H1B_bar*eta**2 + 45.0*H1E_bar*eta**2/2.0
                  + 9.0*H1E*delta*eta**2/2.0 - 27.0*H1E_bar*eta**3)
    term17 = 0.5 * (coeff_chis*chi_s + coeff_chia*chi_a) * v**17

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
    """Frequency-domain SPA inspiral waveform with finite-size corrections."""
    f = np.asarray(f, dtype=float)
    out = np.zeros(f.shape, dtype=complex)

    M_sec  = Mc * MSUN_SEC / eta**(3.0/5.0)
    Mc_sec = Mc * MSUN_SEC
    dL_sec = dL * MPC_SEC

    f_isco = 1.0 / (np.pi * 6.0**1.5 * M_sec)
    f_cut  = fmax_over_fisco * f_isco
    sigma  = sigma_taper_over_fisco * f_isco

    mask = (f >= f_low) & (f < f_cut)
    if not np.any(mask):
        return out

    fa = f[mask]
    va = (np.pi * M_sec * fa)**(1.0/3.0)

    # Spin combinations
    delta = np.sqrt(max(0.0, 1.0 - 4.0*eta))
    chi_s = 0.5*(chi1 + chi2)
    chi_a = 0.5*(chi1 - chi2)

    # Binding energy and derivative
    e_val, de_dv = _binding_energy_terms(va, eta, chi_s, chi_a, delta,
                                          kappa_s, kappa_a)

    # Fluxes
    F_inf = _flux_infty(va, eta, chi_s, chi_a, delta, kappa_s, kappa_a,
                        Lambda_tilde)
    M_dot = _dotM(va, eta, chi_s, chi_a, H0, H1E, H1B, H1E_bar, H1B_bar, delta)
    F_tot = F_inf + M_dot

    # Balance-law derivatives
    dtdv = -M_sec * de_dv / F_tot                   # [s] per unit v
    dphidv = (va**3 / M_sec) * dtdv                  # dphi_orb/dv (rad)

    # dv/df = pi M_sec / (3 v^2)
    dvdf = np.pi * M_sec / (3.0 * va**2)

    # dt/df and df/dt
    dtdf = dtdv * dvdf
    dfdt = 1.0 / dtdf

    # Newtonian chirp rate
    dfdt_N = 96.0 * eta * va**11 / (5.0 * np.pi * M_sec**2)

    # Integrate t(v) and phi_orb(v) on the v-grid (sorted ascending)
    # Constant of integration is degenerate with tc and phic.
    if va.size >= 2:
        t_v = np.zeros_like(va)
        phi_v = np.zeros_like(va)
        t_v[1:]   = cumulative_trapezoid(dtdv,   va)
        phi_v[1:] = cumulative_trapezoid(dphidv, va)
    else:
        t_v   = np.zeros_like(va)
        phi_v = np.zeros_like(va)

    # SPA phase
    psi = 2.0*np.pi*fa*tc - 2.0*phic - np.pi/4.0 + 2.0*np.pi*fa*t_v - 2.0*phi_v

    # Amplitude
    A_N = (np.sqrt(5.0/24.0) * Mc_sec**(5.0/6.0) * np.pi**(-2.0/3.0)
           * fa**(-7.0/6.0) / dL_sec)
    A   = A_N * np.sqrt(dfdt_N / dfdt)

    # Logistic taper
    W = 1.0 / (1.0 + np.exp((fa - f_cut) / sigma))

    if phase_only:
        h_a = np.exp(-1.0j * psi) * W
    else:
        h_a = A * np.exp(-1.0j * psi) * W

    out[mask] = h_a
    return out
