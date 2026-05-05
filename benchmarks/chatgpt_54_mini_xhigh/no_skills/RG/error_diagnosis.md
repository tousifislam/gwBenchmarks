# Error diagnosis for `chatgpt_54_mini_xhigh/no_skills`

This note diagnoses why
`benchmarks/chatgpt_54_mini_xhigh/no_skills/candidate_waveform.py` still has a
non-negligible waveform mismatch against the project reference waveform in
`src/rg_tail/waveform.py`.

The short answer is:

**The main error is the amplitude model.**  The candidate uses a restricted
Newtonian SPA amplitude only, while the reference default uses the balance-law
SPA amplitude correction

```text
sqrt[-2 d(E_real/M)/dx / eta].
```

The phase convention and factorized 22-mode flux are mostly correct.  The
remaining phase residual is small and similar to the GPT-5.5 no-skill case,
consistent with a numerical quadrature difference.

## Benchmark result being diagnosed

Score file:

```text
benchmarks/chatgpt_54_mini_xhigh/no_skills/score_level13.json
```

Summary:

```text
n_evaluations                         = 576
n_failed_evaluations                  = 0
mean optimized mismatch               = 4.4289683693e-03
median optimized mismatch             = 3.7414320889e-03
p90 optimized mismatch                = 7.6209116870e-03
max optimized mismatch                = 8.8954402746e-03
mean phase mismatch                   = 4.4335852671e-03
median |log10(SNR_candidate/SNR_ref)| = 9.0718054572e-02
mean |bias(lambda_RG)| / sigma        = 44.07254473
median |bias(lambda_RG)| / sigma      = 16.87132201
```

The median log-SNR ratio is not close to zero, which points to an amplitude
shape/normalization difference.  The direct diagnostic below confirms that.

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

the candidate/reference amplitude ratio is:

```text
median |h_candidate|/|h_reference| = 1.2246
min    |h_candidate|/|h_reference| = 1.0933
max    |h_candidate|/|h_reference| = 1.4233
```

After removing the best-fit constant and linear phase terms, the residual phase
is only:

```text
phase residual RMS = 1.09e-4 rad
max phase residual = 4.75e-4 rad
```

If I multiply the candidate by the missing balance-law amplitude correction,
the amplitude ratio becomes:

```text
median |h_fixed|/|h_reference| = 0.9999999998
min    |h_fixed|/|h_reference| = 0.9999999998
max    |h_fixed|/|h_reference| = 0.9999999998
```

So the dominant error is definitely the missing amplitude correction.

## Primary wrong lines

### Candidate lines 250-254

Candidate code:

```python
250     amp0 = np.sqrt(5.0 / 24.0) * Mc_sec ** (5.0 / 6.0) / (np.pi ** (2.0 / 3.0) * dL_sec)
251     taper = 1.0 / (1.0 + np.exp((f_flat[valid] - f_cut) / sigma))
252     amp = amp0 * f_flat[valid] ** (-7.0 / 6.0) * taper
253
254     out_flat[valid] = amp * np.exp(1.0j * phase)
```

The wrong line is **line 252**.

It uses only the restricted Newtonian TaylorF2 amplitude:

```text
A_N(f) = sqrt(5/24) Mc_sec^(5/6) f^(-7/6) / [dL_sec pi^(2/3)].
```

The reference default is:

```text
A_ref(f) = A_N(f) * sqrt[-2 d(E_real/M)/dx / eta].
```

Reference:

```text
src/rg_tail/waveform.py:787-789
```

```python
787     if amplitude_model == "balance":
788         amp = amp_newt * _balance_law_spa_amplitude_correction(x, nu)
```

and:

```text
src/rg_tail/waveform.py:86-106
```

```python
104     dE_dx = d_energy_real_over_M_dx(x, nu)
105     correction_sq = -2.0 * dE_dx / nu
106     return np.sqrt(np.maximum(correction_sq, 0.0))
```

The candidate even has access to the needed derivative through
`_pn_energy_and_derivative`, but does not use it in the amplitude.

The minimal benchmark-aligned change would be:

```python
_, de_dx = _pn_energy_and_derivative(x_valid, nu)
amp_corr = np.sqrt(np.maximum(-2.0 * de_dx / nu, 0.0))
amp = amp0 * f_flat[valid] ** (-7.0 / 6.0) * amp_corr * taper
```

## Why this happened

The candidate's docstring states the choice explicitly:

```text
candidate_waveform.py:3-11
```

```python
3  The implementation follows the source packet's dominant (2,2) factorized
4  ingredients, but uses a restricted stationary-phase amplitude convention:
...
10 * the Fourier amplitude uses the standard Newtonian TaylorF2 scaling
11   ``f^(-7/6)`` with the requested smooth taper.
```

That is a plausible waveform choice, but it is not the benchmark reference
choice.  The benchmark reference uses the balance-law amplitude implied by the
same conservative energy derivative used in the phase.

## Lines that are mostly correct

### Full factorized mode

Candidate:

```text
candidate_waveform.py:88-152
```

The candidate builds:

```text
hhat_22 = H_eff * T_22 * rho_22^2 * exp(i delta_22).
```

The coefficient blocks match the reference:

```text
src/rg_tail/waveform.py:263-326
src/rg_tail/waveform.py:467-495
```

So this candidate does not have the earlier `rho_22` typo or 4PN-energy sign
typo seen in another mini run.

### Full 22-mode flux

Candidate:

```python
168     _, de_over_m_dx = _pn_energy_and_derivative(x_grid, nu)
169     hhat = _hhat_22(x_grid, nu, lambda_RG)
170
171     flux = (32.0 / 5.0) * nu**2 * x_grid**5 * np.abs(hhat) ** 2
172     dt_dx = -M_sec * de_over_m_dx / flux
```

Reference:

```text
src/rg_tail/waveform.py:597-612
src/rg_tail/waveform.py:663-667
```

So this candidate also does not have the tail-ratio-only flux bug.

### Fourier phase convention

Candidate:

```python
178     t_grid = tc - (int_t[-1] - int_t)
179     phi_orb_grid = 0.5 * phic - (int_phi[-1] - int_phi)
...
244     phase = 2.0 * np.pi * f_flat[valid] * t_valid - 2.0 * phi_valid - np.pi / 4.0 + hhat_phase_valid
...
254     out_flat[valid] = amp * np.exp(1.0j * phase)
```

Since

```text
t_valid = tc - Delta t
phi_valid = phic/2 - Delta phi_orb
```

line 244 is equivalent to:

```text
phase = 2 pi f tc - phic - pi/4 + 2 Delta phi_orb - 2 pi f Delta t + arg hhat_22.
```

That matches the reference sign convention.

## Secondary residual source

Like the GPT-5.5 no-skill candidate, this candidate uses a different numerical
phase-integration grid from the reference.

Candidate:

```text
candidate_waveform.py:155-182
```

Reference:

```text
src/rg_tail/waveform.py:628-676
```

The residual phase after time/phase alignment is only `~1e-4 rad`, so this is
not the reason for the `4e-3` mismatch.  The dominant reason is the missing
amplitude correction.

## Ranking of issues

1. **Main error:** `candidate_waveform.py:252` uses only Newtonian restricted
   SPA amplitude and omits the balance-law correction
   `sqrt[-2 d(E_real/M)/dx / eta]`.
2. **Secondary benchmark residual:** `candidate_waveform.py:155-182` uses a
   different phase-integration grid/convention from the reference, producing a
   small `~1e-4 rad` residual phase.
3. **No major flux/sign/coefficient error found** in the current file.

If we had to point to only one wrong line, it is:

```text
benchmarks/chatgpt_54_mini_xhigh/no_skills/candidate_waveform.py:252
```

because that line omits the reference amplitude correction.
