# RG Waveform Implementation Notes

## Waveform Convention

The implementation follows the factorized frequency-domain (FD) prototype for a dominant nonspinning (2,2) mode.

- **Amplitude**: Derived using the Stationary Phase Approximation (SPA).
  $A(f) = \frac{M^2}{d_L} \sqrt{\frac{8\pi^2 |d(E/M)/dx|}{3 x^{7/2}}}$
- **Phase**: Calculated via numerical integration of the phasing equations:
  $\Psi(f) = 2\Omega t(f) - 2\phi(f) + \arg(\hat{h}_{22}) - \pi/4$
  where $t(f) = t_c + \int_{x_c}^x (dt/dx') dx'$ and $\phi(f) = \phi_c + \int_{x_c}^x \Omega (dt/dx') dx'$.
- **Reference**: $t_c$ and $\phi_c$ are defined at the cutoff frequency $f_{cut}$.

## Numerical Stability Improvements

The implementation includes several measures to ensure numerical robustness:

1. **Stable Effective Source**: The $H_{eff}$ factor is computed using an expanded form that avoids $0/\nu$ cancellation errors at small symmetric mass ratios ($\eta \ll 1$):
   $H_{eff} = 1 - A + \frac{1}{2} \nu A^2$, where $E/M = 1 - \nu A$.
2. **Logarithmic Integration Grid**: Given the steep power-law dependence of the time-to-merger ($t \sim x^{-4}$), a log-spaced grid (`np.geomspace`) is used for the numerical integration. This ensures high density at low frequencies where the integration is most sensitive.
3. **Phase Unwrapping**: The phase of the complex correction factor $\hat{h}_{22}$ is unwrapped before interpolation to avoid artificial $2\pi$ jumps in the Fourier phase.
4. **Sign Consistency**: Time $t(f)$ and orbital phase $\phi(f)$ are integrated backwards from $f_{cut}$, ensuring that $t(f) < t_c$ for the inspiral phase.

## Equation Checks

- **Derivatives**: The derivative $dE/dx$ correctly accounts for the $x^4 \log(16x)$ terms in the 4PN energy.
- **SPA Consistency**: The amplitude and phase evolution are consistently derived from the same conservative and radiative ingredients, ensuring the stationary phase condition $d\Psi/df = 2\pi t$ is satisfied.
