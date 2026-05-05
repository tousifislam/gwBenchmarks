# Implementation Notes

This document provides a brief overview of the implementation choices and equation checks performed when constructing the finite-size balance-law SPA waveform.

## Waveform Convention

1. **Integration for SPA Phase:** 
   The phase $\psi(f)$ is determined by integrating the coupled differential equations:
   $$ \frac{dt}{dv} = - M_{\text{sec}} \frac{de/dv}{F_{\text{tot}}} $$
   $$ \frac{d\phi_{GW}}{dv} = 2 \pi f(v) \frac{dt}{dv} $$
   We solve this initial value problem numerically using `scipy.integrate.solve_ivp` starting from $v_{\text{min}}$ (the velocity at the lowest valid evaluated frequency).
   
2. **Degeneracy of Integration Constants:**
   The arbitrary constants of integration at $v_{\text{min}}$ implicitly set $t(v_{\text{min}}) = 0$ and $\phi_{GW}(v_{\text{min}}) = 0$. Since time offsets and constant phase offsets represent linear-in-frequency and constant additions to the frequency-domain phase $\psi(f)$, they are perfectly degenerate with the $t_c$ and $\phi_c$ parameters. We include these as:
   $$ \psi(f) = 2 \pi f t(f) - \phi_{GW}(f) - \pi/4 + 2 \pi f t_c - \phi_c $$
   The resulting $h(f)$ is then computed as $A(f) \exp[-i \psi(f)]$.

## Equation Checks Performed

- **Analytic Binding Energy Derivative:** 
  The binding energy $e(v)$ formula features a spin-induced quadrupole $E_{SS}(v)$ scaling with $v^4$ inside the bracket. By analytically grouping the polynomial coefficients as $E_{SS\_0}$ and $E_{SS\_2}$, the scaled energy expands to $-(\eta/2)(v^2 + E_{SS\_0} v^6 + E_{SS\_2} v^8 + E_{SSS} v^9)$. The exact derivative with respect to $v$ was computed analytically as $-(\eta/2)(2v + 6 E_{SS\_0} v^5 + 8 E_{SS\_2} v^7 + 9 E_{SSS} v^8)$ and implemented without numerical differentiation artifacts.
- **Mass Frame Variables:** 
  Differentiated correctly between total mass and chirp mass. $M_{\text{sec}} = M_c \cdot M_{\odot,\text{sec}} / \eta^{3/5}$ correctly represents the total mass $M$, which is used consistently for the $v$ relation and timescale $dt/dv$. The Newtonian amplitude $A_N$ correctly uses the chirp mass $M_{c,\text{sec}} = M_c \cdot M_{\odot,\text{sec}}$.
- **Cutoffs and Taper:**
  Enforced $h(f) = 0$ for $f < f_{low}$ and $f \ge f_{cut}$ via boolean masking. Amplitude correctly incorporates $W(f) = 1/[1 + \exp((f - f_{cut})/\sigma)]$ applied smoothly leading up to $f_{cut}$.