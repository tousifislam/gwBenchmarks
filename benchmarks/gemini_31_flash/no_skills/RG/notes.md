# RG Waveform Implementation Notes

## Conventions

- **Waveform Model**: Dominant nonspinning `(2,2)` RG-tail inspiral waveform using Section-IV ingredients from arXiv:2602.08833.
- **Approximation**: Stationary Phase Approximation (SPA).
- **Amplitude**: The frequency-domain amplitude is derived from the flux and energy balance:
  $$A(f) = |h(t)| \sqrt{ \frac{2\pi}{\ddot{\Phi}} } = \frac{|h_{22}(t)|}{\sqrt{df/dt}}$$
  Using $F_{22} = \frac{32}{5} \nu^2 x^5 |hhat_{22}|^2$ and $|h_{22}(t)| = 8\sqrt{\pi/5} (M/d_L) \nu x |hhat_{22}|$, the factor $|hhat_{22}|$ cancels out of the SPA amplitude, leaving:
  $$A(f) = \frac{M^2}{d_L} \frac{2\pi}{\sqrt{3}} x^{-7/4} \sqrt{-\frac{d(E/M)}{dx}}$$
- **Phase**: The total phase $\Psi(f)$ includes:
  1. The linear terms $2\pi f t_c - \phi_c - \pi/4$.
  2. The correction factor phase $\arg(\hat{h}_{22}) = \arg(T_{22}) + \delta_{22}$.
  3. The orbital phase $\Psi_{orb}(f)$ obtained by numerically integrating $dt/df = -M \frac{d(E/M)/dx \cdot dx/df}{F_{22}}$.
     Specifically, $\Psi_{orb}(f) = 2\pi \int (f - f') (dt/df') df'$.

## Implementation Details

- **Conservative Sector**: 4PN circular energy $E(x)$ and angular momentum $p_\phi(x)$. The derivative $dE/dx$ correctly accounts for the logarithmic term in the 4PN coefficient $c_4$.
- **Running Tail**: The anomalous dimension $\gamma_{22}^{univ}$ and the corresponding tail factor $T_{22}$ are implemented with the $\lambda_{RG}$ deformation.
- **Numerical Integration**: Used `scipy.integrate.cumulative_trapezoid` for both $t(f)$ and $\Psi_{orb}(f)$. The input frequency array is sorted internally to ensure correct integration and then unsorted to return the original order.
- **Tapering**: A Fermi-Dirac style window $W(f)$ is applied near $f_{cut}$ as specified in the prompt.

## Checks

- **ISCO**: Verified $f_{isco} = 1/(\pi 6^{1.5} M)$ matches the Schwarzschild ISCO at $x=1/6$.
- **Newtonian Limit**: Verified that for $\hat{h}_{22} = 1$ and $dE/dx = -\nu/2$, the amplitude and phase evolution reduce to the standard Newtonian SPA forms (modulo normalization conventions for the $(2,2)$ mode).
- **Complex Gamma**: Used `scipy.special.loggamma` which supports complex arguments for the tail factor $T_{22}$.
