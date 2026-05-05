# Finite-Size Waveform Notes

- `Mc` is the detector-frame chirp mass in solar masses, converted with
  `M_sec = Mc * MSUN_SEC / eta^(3/5)`.
- The waveform uses the compact packet's Newtonian restricted SPA amplitude
  corrected by the balance-law chirp-rate ratio `sqrt(dfdt_N / dfdt)`.
- The finite-size phase is obtained by integrating the balance-law chirp rate on
  the frequency grid and anchoring the SPA phase at the upper cutoff
  `f_cut = fmax_over_fisco * f_isco`.
- I treat `phic` as the coalescence GW phase, so the final phase is written as
  `2 pi f tc - phic - pi/4 + phase_to_ref - 2 pi f time_to_ref`.
- The waveform is set to exactly zero for `f < f_low` and `f >= f_cut`.

Quick checks:

1. `h_of_f` returns a complex array with the same shape as the input.
2. The low-frequency and cutoff masks are exact.
3. The phase anchor is self-consistent with the integration convention used in
   the candidate.
