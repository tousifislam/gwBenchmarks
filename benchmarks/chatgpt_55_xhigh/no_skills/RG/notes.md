# RG waveform source-packet notes

- `candidate_waveform.py` is standalone and uses only the source-packet formulas for the nonspinning dominant `(2,2)` mode.
- Masses and distance are converted to geometric seconds with the constants given in the prompt.
- The SPA phase convention is
  `Psi(f) = 2 pi f tc - phic - pi/4 + 2 int_x^xcut Omega dt/dx dx - 2 pi f int_x^xcut dt/dx dx + arg(hhat_22)`.
  Equivalently, `tc` and `phic` are anchored at the high-frequency cutoff used by the benchmark.
- The chirp rate uses `F_22 = (32/5) eta^2 x^5 |hhat_22|^2` and `dt/dx = -M d(E/M)/dx / F_22`.
- The amplitude starts from the standard restricted SPA amplitude.  For `phase_only=False` it includes the consistent single-mode SPA factor `sqrt(-2 d(E/M)/dx / eta)`; the magnitude of `hhat_22` enters through the flux/chirp rate rather than being multiplied a second time.  For `phase_only=True`, this amplitude correction is omitted but the RG-dependent phase evolution and `arg(hhat_22)` are retained.
- The waveform is exactly zero for `f < f_low` and `f >= f_cut`; the logistic taper is applied only inside the active band.
