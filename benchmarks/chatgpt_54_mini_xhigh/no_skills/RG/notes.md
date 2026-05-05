# Waveform Notes

- `Mc` is interpreted as the detector-frame chirp mass in solar masses.
- `M_sec = Mc * MSUN_SEC / eta^(3/5)` is the total mass in geometric seconds.
- The inspiral time and orbital phase are obtained by numerically integrating the supplied `dt/dx` relation on a log-spaced `x` grid that ends at `f_cut`.
- I use a restricted SPA amplitude,
  `A(f) = sqrt(5/24) * Mc_sec^(5/6) / (pi^(2/3) * dL_sec) * f^(-7/6) * W(f)`,
  rather than inserting `|hhat_22|` into the Fourier amplitude.
- The complex phase includes `arg(hhat_22)` at the stationary point:
  `Psi(f) = 2 pi f t(f) - 2 phi_orb(f) - pi/4 + arg(hhat_22(f))`.
- The waveform is set to zero for `f < f_low` and `f >= f_cut`.

Quick consistency checks:

1. `h_of_f` returns a complex array with the same shape as the input frequency array.
2. The support matches the prompt cutoffs exactly.
3. The taper is smooth below `f_cut` and the phase stays continuous because the
   complex `hhat_22` phase is unwrapped on the integration grid.
