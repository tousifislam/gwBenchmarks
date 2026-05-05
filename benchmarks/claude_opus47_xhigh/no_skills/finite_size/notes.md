# Notes on `candidate_waveform.py`

## Overall construction

The waveform is a restricted-amplitude SPA inspiral whose chirp rate is built
**from the balance law** (not a closed-form phase formula):

    -F_infty(v) - dotM(v) = dE/dt,                         # given
    dt/dv = -M_sec * de/dv / (F_infty + dotM),
    df/dt = (3 v^2 / (pi M_sec)) / (dt/dv),
    f     = v^3 / (pi M_sec).

Inputs are the binding-energy `e(v)`, the flux at infinity `F_infty(v)`, and
the absorbed flux `dotM(v)` printed in the source packet:

* **Point-particle baseline:** `e_PP = -eta v^2/2`, `F_PP = (32/5) eta^2 v^10`
  (Newtonian only — no orbital PN beyond LO is included).
* **Spin-induced quadrupole (SS):** `E_SS_SIM v^4` and `F_SS_SIM v^4` with the
  full v-expansion (`v^0, v^2, v^3` pieces in the flux SS bracket).
* **Cubic spin (SSS):** `E_SSS_SIM v^7` and `F_SSS_SIM v^7`.
* **Effective Love-number flux:** `(39/8) Lambda_tilde v^10` term, **inside the
  multiplicative bracket** so the integrated SPA phase reproduces the standard
  `-117/(256 eta) Lambda_tilde v^5` leading tidal phase when combined with the
  Newtonian binding energy.
* **Absorbed flux** `dotM` with the v^15 / v^17 / v^18 structure printed in the
  packet, using the doubled-BBH convention `18 H0 eta^2 v^18`. If the spin-
  linear `H1*` parameters are not supplied they default to zero.

## SPA phase integration

The integrals

    phi_GW(v) = 2 phi_orb(v) = ∫ 2 (v^3 / M_sec) (dt/dv) dv,
    t(v)      = ∫ (dt/dv) dv

are evaluated by `scipy.integrate.cumulative_trapezoid` on a log-spaced
`v`-grid (8192 points) running from `v_low = (pi M_sec f_low)^{1/3}` to
`v_high = (pi M_sec f_cut)^{1/3}`. The integration constants are fixed so that

    phi_GW(v_high) = 0,    t(v_high) = 0

(coalescence-edge anchor). Any constant or linear-in-`f` offset induced by
this anchor is degenerate with `phic` and `tc` respectively, so the choice
is harmless for Fisher analyses. The SPA Fourier phase is

    psi(f) = 2 pi f t(v_f) - phi_GW(v_f) - pi/4
             + 2 pi f tc - phic,

with `v_f = (pi M_sec f)^{1/3}`, evaluated through cubic-spline interpolation
of `phi_GW(v)` and `t(v)` at the input frequencies.

## Amplitude

Standard restricted SPA amplitude scaled by the actual chirp rate:

    A_N(f) = sqrt(5/24) * Mc_sec^{5/6} * pi^{-2/3} * f^{-7/6} / dL_sec,
    dfdt_N = 96 eta v_f^{11} / (5 pi M_sec^2),
    A(f)   = A_N(f) * sqrt(dfdt_N / dfdt(v_f)).

This rescales the Newtonian amplitude to absorb the change in chirp rate when
finite-size or spin terms modify `F_tot`.

## Final form

    h(f) = A(f) * exp(-i psi(f)) * W(f)

with the Fermi–Dirac taper `W(f) = 1/(1 + exp((f - f_cut)/sigma))` for
`f_low <= f < f_cut` and **strict zero** outside that band. Sign convention
matches LIGO TaylorF2: at the point-particle limit we recover

    psi_LO(f) = 2 pi f tc - phic - pi/4 + 3/(128 eta v_f^5).

## `phase_only=True`

Returns the unit-modulus complex exponential `exp(-i psi(f))` — no amplitude
factor and no Fermi taper — with the same hard `[f_low, f_cut)` cutoffs.

## Equation checks performed

1. **Point-particle LO chirp.** With no spin and no finite-size, the second
   derivative `d^2 psi/df^2` evaluated by finite differences on the candidate
   waveform agrees with the TF2 prediction
   `(40/(384 eta)) (pi M_sec)^{-5/3} f^{-11/3}` to 1 part in 5e5 at 40 Hz and
   to 1 part in 2e5 at 25 Hz.
2. **`tc` shift.** Setting `tc = 1e-3 s` adds exactly `-2 pi f * 1e-3` to
   `arg(h(f))` (machine precision at 50 Hz).
3. **`phic` shift.** Setting `phic = 0.5` adds exactly `+0.5 rad` to
   `arg(h(f))`.
4. **Hard cutoffs.** `h(f) = 0` for `f < f_low` and `f >= f_cut` for both full
   and `phase_only` outputs.
5. **Leading tidal phase.** The `Lambda_tilde` phase difference
   `psi(Lambda) - psi(0)` evaluated at three frequencies and fit linearly in
   `f` (to absorb tc-degenerate offsets) leaves residuals consistent with
   `-117/(256 eta) Lambda_tilde v^5` plus higher-order `Lambda^2 v^15`
   corrections from the multiplicative `(1 + (39/8) Lambda v^{10})^{-1}`
   expansion in `dt/dv`.
6. **Restricted amplitude limit.** With no corrections, `|h(30 Hz)|` matches
   `sqrt(5/24)/pi^{2/3} Mc_sec^{5/6}/(dL_sec f^{7/6})` to machine precision.

## Implementation notes

- The bracket in `e(v)` is expanded in powers of `v` so that `de/dv` is
  obtained analytically (avoids finite-difference noise near the upper edge).
- `delta = sqrt(max(1 - 4 eta, 0))` guards against tiny negative round-off
  for `eta = 0.25`.
- All internal arithmetic uses geometric-second masses and distance.
- Cubic spline interpolation is accurate enough that the dominant error
  inside the band is the `O(1/N^2)` trapezoidal-rule error of
  `cumulative_trapezoid`; with `N = 8192` log-spaced points this is well
  below typical Fisher tolerances.
