# Waveform Convention and Checks

## Construction

Restricted-amplitude SPA waveform built directly from the energy-balance law

```
-F_infty - dotM = dE/dt.
```

The pipeline is:

1. **Inputs in seconds**: `M_sec = Mc * MSUN_SEC / eta^(3/5)`, `dL_sec = dL * MPC_SEC`,
   `v = (pi M_sec f)^(1/3)`, `delta = sqrt(1 - 4 eta)`, `chi_{s,a} = (chi1 +/- chi2)/2`.
2. **Binding energy**: e(v) = E/M from the simplified Appendix-A form
   `e(v) = -eta v^2/2 [1 + E_SS_SIM(v) v^4 + E_SSS_SIM v^7]`, expanded analytically.
   `de/dv` is computed analytically to avoid finite-difference noise.
3. **Flux at infinity** F_infty(v) including:
   - Newtonian point-particle baseline,
   - quadratic spin-induced correction `F_SS_SIM(v) v^4` (with v^0, v^2, v^3 pieces),
   - cubic spin-induced correction `F_SSS_SIM v^7`,
   - leading effective tidal flux term `(39/8) Lambda_tilde v^10`.
4. **Absorbed flux** dotM(v) with the BBH "doubled" `18 H0 eta^2 v^18` convention
   and optional H1E/H1B/H1E_bar/H1B_bar spin-linear pieces.
5. **Total** F_tot = F_infty + dotM.
6. **Chirp rates**:
   - dt/dv = -M_sec * (de/dv) / F_tot
   - dphi_orb/dv = (v^3/M_sec) * dt/dv
   - dt/df = dt/dv * dv/df  with  dv/df = pi M_sec / (3 v^2)
   - df/dt = 1 / dt/df
   - dfdt_N = 96 eta v^11 / (5 pi M_sec^2)
7. **SPA phase**: integrate `dt/dv` and `dphi_orb/dv` on the v-grid using
   `cumulative_trapezoid` (scipy). Constant integration offsets are degenerate
   with `tc` (linear in f) and `phic` (constant), so the integrals are zeroed
   at the lowest active frequency.
   ```
   Psi(f) = 2*pi*f*tc - 2*phic - pi/4 + 2*pi*f*t(v_f) - 2*phi_orb(v_f).
   ```
8. **Amplitude**: restricted SPA
   ```
   A(f) = sqrt(5/24) * Mc_sec^(5/6) * pi^(-2/3) * f^(-7/6) / dL_sec
        * sqrt(dfdt_N / dfdt).
   ```
9. **Sign / cutoffs**: `h(f) = A(f) exp(-i Psi(f)) * W(f)`, with logistic taper
   `W(f) = 1/(1 + exp((f-f_cut)/sigma))`. Exact zero for `f < f_low` and
   `f >= f_cut`, enforced by Boolean masking.

## Sanity Checks

- **Frequency coverage**: for Mc=28, eta=0.25, fmax_over_fisco=1.0, the active
  band is 20–68 Hz (123/5000 points), as expected.
- **Amplitude**: max |h| ≈ 1e-22 at 400 Mpc — appropriate magnitude.
- **No NaN/Inf** anywhere in output.
- **tc linearity**: setting `tc = 0.1 s` produces a phase difference
  `arg(h_{tc=0.1}/h_{tc=0}) = -2*pi*f*0.1`, residual std ≈ 1.8e-14 rad
  (round-off level), confirming the SPA phase enters with the prescribed sign.
- **kappa-induced spin sensitivity**: changing chi1, chi2 from 0 to 0.5
  produces a relative amplitude change up to ~66%, confirming the
  spin-induced quadrupole and cubic terms are wired in.
- **Lambda_tilde sensitivity**: Lambda_tilde=500 gives a sub-rad to rad
  phase drift across a BNS band, consistent with the leading-order tidal
  expectation.
- **H0 sensitivity**: enabling H0=1 with chi=0.7 spins yields a few-percent
  change, in line with horizon flux entering at v^18.

## Limitations

- The Newtonian point-particle baseline is exact (no PN spin-orbit,
  spin-orbit-induced phasing beyond what the simplified formulas include).
- Restricted amplitude (no SPA Psi-to-amplitude corrections beyond the
  Newtonian prefactor and the chirp-rate ratio).
- Constant offsets t(v_first) and phi_orb(v_first) are absorbed into tc, phic,
  as noted in the prompt.
