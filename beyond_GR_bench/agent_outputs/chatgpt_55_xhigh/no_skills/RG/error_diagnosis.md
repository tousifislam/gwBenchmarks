# Error diagnosis for `chatgpt_55_xhigh/no_skills`

This note diagnoses the current
`benchmarks/chatgpt_55_xhigh/no_skills/candidate_waveform.py` result against
the project reference waveform in `src/rg_tail/waveform.py`.

The short answer is:

**No clear physics-level formula error was found.**  The candidate implements
the same factorized `(2,2)` waveform ingredients, the same 22-only flux, the
same balance-law amplitude, and the same Fourier sign convention as the
reference.  The remaining mismatch is small and is consistent with a numerical
quadrature/convention difference in how the balance-law phase integral is
evaluated.

## Benchmark result being diagnosed

Score file:

```text
benchmarks/chatgpt_55_xhigh/no_skills/score_level13.json
```

Summary:

```text
n_evaluations                         = 576
n_failed_evaluations                  = 0
mean optimized mismatch               = 7.2422255795e-05
median optimized mismatch             = 2.0683329021e-05
p90 optimized mismatch                = 3.6246028041e-04
max optimized mismatch                = 4.1340775933e-04
mean phase mismatch                   = 7.8112260709e-05
median |log10(SNR_candidate/SNR_ref)| = 1.0269687324e-10
mean |bias(lambda_RG)| / sigma        = 3.4887191251
median |bias(lambda_RG)| / sigma      = 1.4707130136
```

The SNR ratio is essentially exact, so the amplitude is not the issue.  The
remaining mismatch is a very small phase-shape residual.

## Direct diagnostic

For a GW150914-like point,

```text
Mc = 28.3
eta = 0.247
dL = 410 Mpc
lambda_RG = 1
f_low = 20 Hz
fmax_over_fisco = 1
sigma_taper_over_fisco = 0.01
```

the candidate/reference comparison gives:

```text
median |h_candidate|/|h_reference| = 0.9999999998
min    |h_candidate|/|h_reference| = 0.9999999998
max    |h_candidate|/|h_reference| = 0.9999999998
```

After removing the best-fit constant and linear phase terms, which correspond
to `phic` and `tc`, the residual phase is:

```text
phase residual RMS    = 1.09e-4 rad
max phase residual    = 4.74e-4 rad
```

This is the scale expected from a numerical-integration mismatch, not a
wrong-sign or wrong-flux error.

## Lines that match the reference

### Full factorized mode

Candidate:

```text
candidate_waveform.py:97-163
```

The candidate builds

```text
hhat_22 = H_eff * T_22 * rho_22^2 * exp(i delta_22).
```

The coefficient blocks match the reference formulas:

```text
src/rg_tail/waveform.py:263-326
src/rg_tail/waveform.py:467-495
```

### Full 22-mode flux

Candidate:

```python
166 def _flux_22(x, eta, lambda_RG):
167     hhat = _hhat_22(x, eta, lambda_RG)
168     return (32.0 / 5.0) * eta**2 * x**5 * np.abs(hhat) ** 2
```

Reference:

```text
src/rg_tail/waveform.py:597-612
```

```python
610     return (32.0 / 5.0) * nu ** 2 * x ** 5 * np.abs(
611         hhat_22(x, nu, lambda_RG=lambda_RG)
612     ) ** 2
```

So this candidate does **not** have the earlier tail-ratio-only flux bug.

### Fourier phase sign

Candidate:

```python
266     hhat = _hhat_22(x, eta, lambda_RG)
...
270     t_to_cut, phi_to_cut = _phase_integrals_to_cut(x, x_cut, M_sec, eta, lambda_RG)
271     spa_phase = (
272         2.0 * _PI * f_act * float(tc)
273         - float(phic)
274         - _PI / 4.0
275         + 2.0 * phi_to_cut
276         - 2.0 * _PI * f_act * t_to_cut
277     )
...
297     out[active] = taper * amp0 * amp_corr * mode_phase * np.exp(1.0j * spa_phase)
```

Here `mode_phase = hhat/|hhat|`, so line 297 is equivalent to:

```text
h = A exp[i(2 pi f tc - phic - pi/4 + Psi_bal + arg hhat_22)].
```

That matches the reference:

```text
src/rg_tail/waveform.py:785-798
```

```python
785     phase = 2.0 * PI * f_eval * tc - phic - PI / 4.0 + psi_orb + np.angle(hhat_running)
...
798     h[valid] = amp * np.exp(1j * phase) * taper
```

So this candidate does **not** have the Fourier complex-conjugation bug.

### Balance-law amplitude

Candidate:

```python
279     amp0 = (
...
289         _, denergy_dx = _energy_and_derivative(x, eta)
290         amp_corr = np.sqrt(np.maximum(-2.0 * denergy_dx / eta, 0.0))
...
297     out[active] = taper * amp0 * amp_corr * mode_phase * np.exp(1.0j * spa_phase)
```

Reference:

```text
src/rg_tail/waveform.py:787-789
```

```python
787     if amplitude_model == "balance":
788         amp = amp_newt * _balance_law_spa_amplitude_correction(x, nu)
```

So this candidate also gets the default balance-law amplitude right.

## Main residual source

The only meaningful difference I found is the numerical phase integration.

Candidate:

```text
candidate_waveform.py:176-216
```

The candidate uses a dense geometric grid and analytically subtracts Newtonian
singular pieces:

```python
186     # Subtract the Newtonian singular pieces analytically; the remaining
187     # integrands are much smoother on a logarithmic grid.
188     t_newt = lambda x: 5.0 * M_sec / (64.0 * eta) * x ** (-5.0)
189     phi_newt = lambda x: 5.0 / (64.0 * eta) * x ** (-3.5)
...
192     n_grid = int(max(12000, 9000 * log_span + 4 * x_eval.size))
...
198     dt = _dt_dx(grid, M_sec, eta, lambda_RG)
199     dphi = (grid**1.5 / M_sec) * dt
...
208     t_to_cut = (
209         5.0 * M_sec / (256.0 * eta) * (x_eval ** (-4.0) - x_cut ** (-4.0))
210         + t_corr_to_cut
211     )
212     phi_to_cut = (
213         1.0 / (32.0 * eta) * (x_eval ** (-2.5) - x_cut ** (-2.5))
214         + phi_corr_to_cut
215     )
```

Reference:

```text
src/rg_tail/waveform.py:628-676
```

The reference integrates directly over the evaluation nodes plus the endpoint:

```python
658     if append_ref:
659         x_nodes = np.concatenate((x, [x_ref]))
660     else:
661         x_nodes = x
...
663     dE_dx = d_energy_real_over_M_dx(x_nodes, nu)
664     F22 = flux_22(x_nodes, nu, lambda_RG=lambda_RG)
...
669     time_to_ref = _cumulative_integral_to_upper(x_nodes, dt_dx)
670     gw_phase_to_ref = _cumulative_integral_to_upper(x_nodes, dphi_gw_dx)
```

The candidate's method is arguably a more accurate quadrature scheme, but the
hidden benchmark compares to the reference implementation as written.  That
small discretization difference leaves a residual phase of order `1e-4 rad`,
which becomes the observed `~7e-5` mean mismatch.

## Ranking of issues

1. **No physics-level bug found.**
2. **Small benchmark-residual source:** `candidate_waveform.py:176-216` uses a
   different quadrature strategy from the reference.  This is the likely source
   of the remaining `~1e-4` phase residual.
3. **Minor interface difference:** `candidate_waveform.py:286-288` makes
   `phase_only=True` use `amp_corr = 1`.  The scored benchmark used
   `phase_only=False`, so this does not affect the current result.

If forced to point to a line for the residual mismatch, the best answer is not
"wrong physics" but:

```text
benchmarks/chatgpt_55_xhigh/no_skills/candidate_waveform.py:186-216
```

because the candidate uses a different phase-quadrature convention from the
hidden reference.
