# Finite-size waveform notes

- `candidate_waveform.py` is standalone and only uses the source-packet balance-law formulas.
- Detector-frame masses and distances are converted to seconds with the prompt constants.
- The implemented energy is `e(v) = -eta v^2/2 [1 + E_SS_SIM(v) v^4 + E_SSS_SIM v^7]`, with analytic `de/dv`.
- The total flux is `F_tot = F_infty + dotM`, including the effective `(39/8) Lambda_tilde v^10` flux term and the doubled `18 H0 eta^2 v^18` absorbed-flux convention.
- The SPA phase is cutoff-anchored: `psi = 2 pi f tc - phic - pi/4 + int_v^vcut 2 Omega dt/dv dv - 2 pi f int_v^vcut dt/dv dv`, and the waveform uses `h = A exp(-i psi)`. In the Newtonian limit this differs from the usual coalescence-at-infinity TaylorF2 phase only by constant and linear-in-frequency offsets.
- The amplitude is `A_N sqrt(dfdt_N / dfdt)` when `phase_only=False`; with `phase_only=True`, the amplitude is the Newtonian restricted SPA amplitude while the balance-law phase still includes finite-size effects.
- The waveform is exactly zero for `f < f_low` and `f >= f_cut`; the logistic taper is applied only inside the active band.
