---
name: beyond-gr-fti-phase-deformation
description: Recipe for building a beyond-GR frequency-domain waveform by deforming a post-Newtonian ingredient (worked through for RG-deformed tail factors, ell_hat = ell + lambda * gamma) and applying the result as an inspiral-only phase correction to each mode of IMRPhenomXPHM from LALSuite, switched off before merger with the FTI (flexible theory-independent) construction. Use when a task asks for a parameterized or theory-specific deviation on top of a LAL IMR baseline, per-mode phase corrections, an energy-balance phase deviation, or an FTI taper.
---

# Beyond-GR phase deformations on a LAL baseline with FTI

This skill gives the construction step by step for a deformed tail factor,
`ell_hat = ell + lambda * gamma` with `lambda = 1` as GR. The same pipeline
works for other deformed ingredients.

The skill fixes the structure, the formulas that connect the pieces, and
numerical sanity values. You still take the PN coefficients from the papers.

## 0. Pipeline

1. Masses and frequency variables (section 2).
2. On an internal (2,2)-frequency grid, the PN ingredients (section 3):
   - `E`, `dE/dx`, `p_phi`, `H_eff`, `khat`, `J`;
   - `gamma_lm`, `ell_hat_lm`, `T_lm`, `rho_lm`;
   - the Newtonian mode fluxes.
3. The total flux, deformed (`F_lambda`) and GR (`F_1`), summed over all multipoles
   (section 3.4).
4. Second derivatives of the orbital phase deviation (section 4) and of each mode's direct
   phase deviation (section 5).
5. The FTI window, double integral and boundary conditions, per part (section 6).
6. The LAL baseline, one call per co-precessing mode (section 7).
7. Each mode multiplied by its phase factor and summed; the checks of section 8.

## 1. What the deformation changes

`lambda` sits inside each mode's tail factor `T_lm`. That changes two things:

1. **The phase of `T_lm`.** This is a direct shift of that mode's phase only.
   It is small: about 1e-2 rad in the example of section 8.
2. **The modulus `|T_lm|`.** The modulus changes the energy flux of every mode,
   and with it the total radiated power and the inspiral rate. Every mode
   inherits the resulting orbital phase. This part is large: a few radians in
   the same example.

The orbital (flux-driven) part dominates by about two orders of magnitude. An
implementation that applies only `arg T_lm` misses almost the whole effect.

"`lambda` multiplies the anomalous dimension; nothing else changes" means that
no other ingredient is modified. The consequences through the flux still have
to be propagated.

The conservative binding energy does not depend on `lambda`.

## 2. Variables and conventions

- **Masses.**
  - total mass `M = Mc * eta^(-3/5)`;
  - `nu = eta`;
  - `m1 = M (1 + sqrt(1 - 4 eta)) / 2 >= m2`;
  - `M` in seconds is `M * 4.925491025543576e-6`.
- **Frequencies.**
  - `x = (pi M f22)^(2/3)`, where `f22` is the (2,2)-mode frequency, twice the
    orbital frequency;
  - mode (ell, m) at Fourier frequency `f` is evaluated at `f22 = 2 f / m`;
  - a condition stated at orbital frequency `F` is a condition at `f22 = 2 F`.
- **Fourier convention.** Use `h(f) = ∫ h(t) exp(+2 pi i f t) dt`. In this
  convention the SPA phase satisfies `d^2 Psi / df^2 = +2 pi dt/df > 0` during
  the inspiral. LAL uses `exp(-2 pi i f t)`, so conjugate its output
  (section 7).

## 3. PN ingredients (from the RG paper unless noted)

Write each ingredient as a separate function of `(x, nu)` and test it on its own.

### 3.1 Conservative pieces (comparable-mass section)

- **Energy.** `E_hat = E_real / M` on circular orbits at 4PN, and its derivative
  `dE_hat/dx`. Differentiate analytically, or with a fine finite difference,
  including the `log x` term.
- **Angular momentum.** `p_phi = P_phi / (mu M)` on circular orbits.
- **Effective energy.** Use the exact EOB map from the real energy:
  `H_eff = 1 + b + nu b^2 / 2` with `b = (E_hat - 1) / nu`. Do not re-expand it.
- **Source factor.** `S = H_eff` for even `ell + m`; `S = sqrt(x) * p_phi` for
  odd `ell + m`.
- **Tail variables.** `khat = E_hat * m * x^(3/2)` and `J = nu * p_phi / E_hat^2`.
  The paper's notation for `J` is not uniform. With the reduced `p_phi` used
  here, the factor `nu` belongs in `J`, so `J -> 0` in the test-mass limit.
  Leaving it out is a 4PN error that moves the (2,2) orbital phase of the
  section 8 example from -2.81 to -2.46 rad.

### 3.2 Anomalous dimension and tail factor

- **Anomalous dimension.** Take `gamma_lm(khat, J)` from the paper's appendix
  that lists it for `ell = 2..8`. Keep every printed term, which goes up to
  `khat^5`, with the odd powers linear in `m J`. Do not cut at strict 4PN.
  - You need **every `ell` from 2 to 8 and every `m` from 1 to `ell`**, because
    the flux uses all of them, not only the five output modes.
  - The (2,2) mode uses the comparable-mass Section IV expression.
  - Then `ell_hat = ell + lambda * gamma_lm`.
- **Tail factor.** Use the radiative-coordinate form of `T_lm`:
  - two Gamma-function ratios, `Gamma(2 ell + 2) / Gamma(ell - 1)` and
    `Gamma(ell_hat - 1 - 2i khat) / Gamma(2 ell_hat + 2)`;
  - `e^(pi khat)`;
  - `(2 m sqrt(x))^(ell_hat - ell)`, which is `(2 k r_Omega)^(ell_hat - ell)`;
  - `e^(-i pi (ell_hat - ell) / 2)`;
  - a `lambda`-independent pure phase.

  Evaluate `log T_lm` with complex `loggamma`, which avoids overflow and phase
  wrapping.
- **What you need from `T_lm`.**
  - `|T_lm|^2`, for the flux;
  - the direct phase change,

        dArg_lm = Im loggamma(ell_hat_lambda - 1 - 2i khat) - Im loggamma(ell_hat_1 - 1 - 2i khat) - (pi/2) (ell_hat_lambda - ell_hat_1)

    All other phase factors are real or do not depend on `lambda`, so they cancel.

### 3.3 Residual amplitudes

- `rho_22` comes from Section IV, through `x^4`.
- For the other pairs with `ell <= 5`, use the appendix table with every
  printed term, through `x^10`. Do not stop at `x^4`.
- For `ell >= 6`, set `rho_lm = 1`. The printed `ell = 6..8` entries are not
  worth transcribing. Including them moves the (2,2) orbital phase by about
  1e-4 rad. The mismatch between the two deformed waveforms, maximized over
  time and phase, is below 1e-7. Do not compare instead how much the
  deformed-vs-GR mismatch changes; that quantity moves at first order and
  looks much larger.
- `rho_lm` enters the flux as `rho_lm^(2 ell)`.
- **Getting the numbers.** Extract the page text with PyMuPDF (`import pymupdf`;
  the old name `fitz` may be missing) and parse the coefficients
  programmatically into exact fractions. Then compare a few entries with the
  rendered page image. Never type a long coefficient from memory or guess a
  digit; if a number is unreadable, render that page as an image.
- Get the flux structure right first, then fill in the table.

### 3.4 Flux

- **Newtonian mode flux.** `F^N_lm` comes from the Newtonian multipolar
  normalization in `0811.2069v2.pdf`. The `+m` and `-m` modes together give
  `(1 / 8 pi) (m Omega)^2 |R h^N_lm|^2`, in units where the flux is
  dimensionless (`c^5 / G = 1`). Check:
  - `F^N_22 = (32/5) nu^2 x^5`;
  - `F^N_21 / F^N_22 = (1 - 4 nu) x / 36`;
  - `F^N_33 / F^N_22 = (1215/896) (1 - 4 nu) x`.
- **Mode flux.** `F_lm = F^N_lm * S^2 * rho_lm^(2 ell) * |T_lm(lambda)|^2`.
- **Total flux.** `F_lambda = sum over ell = 2..8, m = 1..ell of F_lm`.
  - Compute `F_1` (GR) with the same code.
  - The sum is the same whichever modes the function returns.
- **Check (`nu = 0.2`).**
  - `lambda = 0` lowers the total flux by about 1% at `x = 0.05` and about 6% at
    `x = 0.1`;
  - `lambda = 2` raises it by the same amounts;
  - the non-(2,2) modes carry about 3% of the flux at `x = 0.1`.

## 4. Orbital phase deviation (energy balance)

Energy balance gives `dt/dx = -M (dE_hat/dx) / F`. Combined with the SPA relation
of section 2, this gives the second derivative of the orbital phase deviation,
normalized to (2,2), as a function of `f22`:

    dPsi_orb''(f22) = (4 pi M x / (3 f22)) * |dE_hat/dx| * (1 / F_lambda(x) - 1 / F_1(x))

Here `M` is in seconds and the flux is dimensionless. Only the flux depends on
`lambda`, so this is the whole orbital source term.

## 5. Direct phase deviation

For each output mode, compute `dArg_lm` (section 3.2) at `x = (pi M f22)^(2/3)`
on the same `f22` grid. Then take its second derivative numerically with respect
to `f22`.

## 6. FTI construction (FTI paper; integrate on f22)

**Window.**

- `v = (pi M f22)^(1/3)`;
- `v_tape = (pi M f_tape)^(1/3)` with `f_tape = alpha_tape * f_peak`;
- `Delta v = (128 nu / 3) * pi * v_tape^6 * gamma_tape * n_gw_tape`;
- `W(v) = 1 / (1 + exp((v - v_tape) / Delta v))`. Clip the exponent to avoid
  overflow.

**Input.**

- `I(f22) = W * (second-derivative source)`;
- set it to zero above `fmax_model_over_fisco * f_ISCO22`, where
  `f_ISCO22 = 1 / (6^(3/2) pi M)` is the (2,2) frequency at the Schwarzschild
  ISCO.

**Double integral.** These two steps give zero slope at orbital frequency
`f_peak / 2` and zero value at `f_low / 2`:

    G(f22)   = ∫ from f_peak to f22 of I      (so G(f_peak) = 0)
    dPsi(f22) = ∫ from f_low to f22 of G      (so dPsi(f_low) = 0)

**Apply it to both parts.** Integrate the orbital source of section 4 and each
mode's direct source of section 5 separately, with the same window and the same
boundary conditions.

**Grid.**

- geometric in `f22`, from `min(f_low, 2 f_low / m_max)` up to `f_peak`, with
  `n_grid` points;
- cumulative trapezoid sums for both integrals;
- above `f_peak`, hold each deviation at its value at `f_peak`; it is constant
  there because the slope is zero;
- below the grid start, hold the first value.

The grid depends only on the physical parameters and `n_grid`. Interpolate to the
requested frequencies afterwards. That makes the output independent of frequency
order, batching and repetition.

**Mode phase.**

    dPsi_lm(f) = (m / 2) * dPsi_orb(f22 = 2 f / m) + dArg_lm(f22 = 2 f / m)

**GR limit.** At `lambda == 1`, return a zero deviation directly. The output is
then exactly the LAL baseline.

## 7. LAL baseline, one mode at a time

1. Sort the frequencies `>= f_low`. Put them in a `lal.CreateREAL8Vector`. If
   there is only one distinct value, pad with a second frequency and drop it
   afterwards: LAL rejects one-point sequences.
2. For each output mode (ell, m) in (2,2), (2,1), (3,3), (3,2), (4,4):
   1. Create a `lal.CreateDict()`.
   2. Make a mode array with `lalsimulation.SimInspiralCreateModeArray()`.
      Activate (ell, m) and (ell, -m) with
      `SimInspiralModeArrayActivateMode`. Insert it with
      `SimInspiralWaveformParamsInsertModeArray`.
   3. Call `lalsimulation.SimInspiralChooseFDWaveformSequence` with these
      arguments, in this order:
      - `phi_ref`;
      - `m1 * lal.MSUN_SI`, `m2 * lal.MSUN_SI`;
      - the six spin components;
      - `f_ref`;
      - `dL * 1e6 * lal.PC_SI`;
      - `inclination`;
      - the dict;
      - `lalsimulation.IMRPhenomXPHM`;
      - the frequency vector.
3. Take `hp.data.data` and `hc.data.data`, and undo the sorting.
4. Conjugate both to get the `exp(+2 pi i f t)` convention.
5. Multiply each mode by `exp(i dPsi_lm(f)) * exp(2 pi i f tc)`.
6. Sum the modes. Fill zeros below `f_low`. With `return_modes`, return the
   per-mode pieces; they must sum to the total.

`f_peak` default: scored cases always pass `f_peak`, so the default only has to
be reasonable. The nonspinning fit `M omega_22 = 0.2733 + 0.2316 nu + 0.4463 nu^2`,
with `f_peak = omega_22 / (2 pi)`, is enough. Do not tune it against the checks.

## 8. Checks, with expected scales

Record each result in `notes.md`. The values below come from a correct
implementation. A correct build reproduces them closely. If one disagrees,
find the bug; do not change a model input, such as the default `f_peak`, to
make it agree.

1. **GR limit.** At `lambda = 1` the output equals the LAL per-mode sum to machine
   precision, for a precessing case with `tc = 0`.
2. **Phase parts.** Example: `Mc = 20`, `eta = 0.2`, `lambda = 0`,
   `f_low = 20`, `f_peak = 150 Hz`. At Fourier frequency `f` (Hz), each mode's
   orbital part `(m/2) dPsi_orb(2f/m)` and direct part `dArg_lm(2f/m)` after
   FTI, in rad, should agree to about 1e-3 rad:

   | Mode | Part | f = 25 | 40 | 60 | 100 |
   |---|---|---|---|---|---|
   | (2,2) | orbital | -1.046 | -2.543 | -2.811 | -2.811 |
   | (2,2) | direct | 0.0032 | 0.0098 | 0.0116 | 0.0116 |
   | (3,3) | orbital | 1.450 | -1.971 | -3.814 | -4.216 |
   | (3,3) | direct | -0.0030 | 0.0051 | 0.0119 | 0.0141 |
   | (4,4) | orbital | 5.280 | 0.000 | -3.514 | -5.602 |
   | (2,1) | orbital | -1.401 | -1.405 | -1.405 | -1.405 |

   - The (3,3) and (4,4) entries at 25 Hz sit below `f22 = f_low`. They test
     the continuation of the double integral below `f_low`.
   - At `lambda = 2` the (2,2) orbital part is +1.001, +2.411, +2.657, +2.657
     at the same frequencies.
   - An orbital part near zero means the flux chain is missing.
3. **Waveform scale.** Mismatch between the deformed and GR `h_plus`. Setup:
   - `eta = 0.2`, no spins, `inclination = 0.5`, `phi_ref = 0`, `tc = 0`,
     `f_low = 20`;
   - `f_peak` as listed;
   - noise `lalsimulation.SimNoisePSDaLIGOZeroDetHighPower` on a uniform grid
     from 20 to 990 Hz, `df = 0.125`.

   Expected values (agree within a few percent):

   | Mc | f_peak (Hz) | lambda = 0 | 0.5 | 2 |
   |---|---|---|---|---|
   | 12 | 344.1 | 0.426 | 0.255 | 0.404 |
   | 20 | 206.5 | 0.293 | 0.089 | 0.242 |
   | 30 | 137.6 | 0.074 | 0.019 | 0.061 |

   How to compute it, maximized over time and phase:
   - `(a, b) = 4 df sum(a conj(b) / S)`, so `(h, h) = 4 df sum(|h|^2 / S)`;
   - `c = h1 conj(h2) / S`;
   - the overlap is `max over t of |4 df sum(c exp(2 pi i f t))| / sqrt((h1,h1) (h2,h2))`;
   - for all `t` at once, use `n * numpy.fft.ifft(c, n)` with zero padding
     `n >= 8 len(c)`; `ifft` already divides by `n`;
   - the mismatch is 1 minus the overlap. It must lie in [0, 1]; a value
     outside that range means the normalization is wrong.

   The table takes a few minutes. Compute one `Mc` row per Python call so
   that each call stays well under the 600 s tool limit.
4. **Flux values.** Reproduce the Newtonian ratios and the flux change of
   section 3.4.
5. **Mode subsets.** The sum of single-mode outputs equals the all-mode output,
   and the flux does not depend on `modes`.
6. **Interface.** Test scalar, empty and 2-D `f`; values below `f_low`;
   shuffled, repeated and batched calls. Invalid inputs must raise
   `ValueError`.
7. **Convergence.** Doubling `n_grid` changes the deviation by much less than
   1e-3 rad.

## 9. Mistakes seen in earlier attempts

- Applying only `arg T_lm` and skipping the flux. This is the most common
  failure; the result is as far from the target as plain GR.
- Treating the FTI parameter `n_gw_tape` (`N_GW`) as the place where the flux
  enters. It only sets the window width.
- Tabulating `gamma` only for the output modes (`ell <= 4`). The flux needs
  `ell <= 8`.
- Truncating `gamma` at strict 4PN instead of keeping every printed term.
- A GR limit that misses the baseline: wrong conjugation, `f_ref` not passed, or
  spins or angles permuted.
- Per-mode pieces from `return_modes` that do not sum to the total.
- Dropping `nu` from `J`.
- Truncating the `ell <= 5` residual amplitudes at `x^4`, or retyping long
  coefficients by hand and inventing digits where the text is garbled.
- Changing a model input, such as the default `f_peak`, to make a check agree.
- Ending the session with a plan but no files. Write `candidate_waveform.py` and
  `notes.md` early and refine them.

When PDF text extraction garbles a table, render the page as an image and read it
again.
