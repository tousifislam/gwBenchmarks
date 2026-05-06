# Error diagnosis for `chatgpt_53_codex_xhigh/no_skills`

This note diagnoses why
`benchmarks/chatgpt_53_codex_xhigh/no_skills/candidate_waveform.py` has a
non-negligible mismatch against the project reference waveform in
`src/rg_tail/waveform.py`.

The short answer is:

**The main error is the detector-domain amplitude model.**  The candidate
multiplies the frequency-domain waveform by the full complex factor
`hhat_22`, which includes `|hhat_22|` in the amplitude.  The reference default
uses only `arg(hhat_22)` in the phase and uses the balance-law SPA amplitude
correction

```text
sqrt[-2 d(E_real/M)/dx / nu].
```

The candidate's phase is actually very good.  The mismatch and RG-bias score
are dominated by the amplitude-shape difference.

## Benchmark result being diagnosed

Score file:

```text
benchmarks/chatgpt_53_codex_xhigh/no_skills/score_level13.json
```

Summary:

```text
n_evaluations                         = 576
n_failed_evaluations                  = 0
mean optimized mismatch               = 3.8667589843e-03
median optimized mismatch             = 3.2297217765e-03
p90 optimized mismatch                = 6.8362439564e-03
max optimized mismatch                = 8.8587280815e-03
mean phase mismatch                   = 3.8869283765e-03
median |log10(SNR_candidate/SNR_ref)| = 5.5866822183e-02
mean |bias(lambda_RG)| / sigma        = 57.71889726
median |bias(lambda_RG)| / sigma      = 22.09377136
```

The nonzero SNR-ratio metric points to an amplitude-model issue.  The direct
diagnostic below confirms this.

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
median |h_candidate|/|h_reference| = 1.1381
min    |h_candidate|/|h_reference| = 1.0263
max    |h_candidate|/|h_reference| = 1.3108
```

The residual phase after removing best-fit constant and linear phase terms is
tiny:

```text
phase residual RMS = 5.34e-6 rad
max phase residual = 2.36e-5 rad
```

If the candidate is rescaled by

```text
sqrt[-2 d(E_real/M)/dx / nu] / |hhat_22|,
```

the amplitude ratio becomes:

```text
median |h_fixed|/|h_reference| = 0.9999999998
min    |h_fixed|/|h_reference| = 0.9999999998
max    |h_fixed|/|h_reference| = 0.9999999998
```

So the main discrepancy is not phase, flux, or PN coefficients.  It is the
choice of amplitude factor.

## Primary wrong lines

### Candidate lines 199-209

Candidate code:

```python
199     Mc_sec = float(Mc) * MSUN_SEC
200     amp0 = np.sqrt(5.0 / 24.0) * (np.pi ** (-2.0 / 3.0)) * (Mc_sec ** (5.0 / 6.0)) / dL_sec
201     amp = amp0 * fs ** (-7.0 / 6.0)
...
209     hs = amp * taper * np.exp(1j * psi_spa) * hhat
```

The wrong line is **line 209**, together with line **201**.

Line 209 multiplies by the full complex `hhat`, so the amplitude becomes:

```text
A_candidate = A_N(f) * |hhat_22(x; lambda_RG)|.
```

The reference default amplitude is:

```text
A_reference =
  A_N(f) * sqrt[-2 d(E_real/M)/dx / nu].
```

Reference:

```text
src/rg_tail/waveform.py:86-106
```

```python
104     dE_dx = d_energy_real_over_M_dx(x, nu)
105     correction_sq = -2.0 * dE_dx / nu
106     return np.sqrt(np.maximum(correction_sq, 0.0))
```

and:

```text
src/rg_tail/waveform.py:787-789
```

```python
787     if amplitude_model == "balance":
788         amp = amp_newt * _balance_law_spa_amplitude_correction(x, nu)
```

The candidate should use `hhat` only for its phase, not for its magnitude, in
the default benchmark amplitude.  A benchmark-aligned replacement would be:

```python
amp_corr = np.sqrt(np.maximum(-2.0 * dE_dx / nu, 0.0))
amp = amp0 * fs ** (-7.0 / 6.0) * amp_corr
hs = amp * taper * np.exp(1j * (psi_spa + np.angle(hhat)))
```

## Why this happened

The candidate notes describe exactly this amplitude choice:

```text
notes.md
```

```text
Base amplitude uses Newtonian inspiral scaling
~ Mc^(5/6) f^(-7/6) / dL, multiplied by hhat_22.
```

That is a plausible factorized-waveform proxy:

```text
h_proxy(f) = A_N(f) * hhat_22(x_f) * exp(i psi_spa).
```

But it is not the benchmark reference.  The reference uses a balance-law SPA
construction where the explicit `|hhat_22|` in the time-domain mode cancels
against the `|hhat_22|^2` entering `df/dt` through the flux.  The remaining
amplitude correction is the conservative-energy derivative factor.

## Lines that are mostly correct

### PN coefficients and factorized mode

Candidate:

```text
candidate_waveform.py:23-135
```

The conservative energy, circular angular momentum, `gamma_22`, tail factor,
`rho_22`, and `delta_22` blocks match the reference formulas.

Reference formula blocks:

```text
src/rg_tail/waveform.py:263-326
src/rg_tail/waveform.py:467-495
```

So this candidate does not have the coefficient-typo failures seen in some
other runs.

### Full 22-mode flux

Candidate:

```python
181     hhat, E, dE_dx = _hhat22(x, nu, float(lambda_RG))
183     flux = (32.0 / 5.0) * nu * nu * x**5 * np.abs(hhat) ** 2
184     flux = np.maximum(flux, np.finfo(float).tiny)
186     dt_dx = -M_sec * dE_dx / flux
```

Reference:

```text
src/rg_tail/waveform.py:597-612
src/rg_tail/waveform.py:663-667
```

So this candidate also does not have the tail-ratio-only flux bug.

### Fourier phase

Candidate:

```python
190     tau_to_high = _suffix_trapezoid(dt_df, fs)
191     t_of_f = float(tc) - tau_to_high
...
193     dphi_df = 2.0 * np.pi * fs * dt_df
194     dphi_to_high = _suffix_trapezoid(dphi_df, fs)
195     phi_gw = float(phic) - dphi_to_high
197     psi_spa = 2.0 * np.pi * fs * t_of_f - phi_gw - np.pi / 4.0
...
209     hs = amp * taper * np.exp(1j * psi_spa) * hhat
```

Because `hhat` contributes `arg(hhat)` and `t_of_f`, `phi_gw` are referenced to
the high-frequency endpoint, this is consistent with the reference phase up to
absorbed `tc`/`phic` constants.  The direct diagnostic found only a
`5e-6 rad` residual phase after alignment.

## Minor issue: `phase_only`

Candidate:

```python
211     if phase_only:
212         hs = np.exp(1j * np.angle(hs))
```

The full benchmark used `phase_only=False`, so this did not affect the current
score.  If a phase-only benchmark were used, this convention would differ from
the reference default behavior.

## Ranking of issues

1. **Main error:** `candidate_waveform.py:209` multiplies by full complex
   `hhat`, causing the amplitude to include `|hhat_22|`.
2. **Supporting amplitude issue:** `candidate_waveform.py:201` uses only the
   Newtonian amplitude and omits the balance-law correction
   `sqrt[-2 d(E_real/M)/dx / nu]`.
3. **No major phase, flux, or coefficient error found** in the current file.

If we had to point to only one wrong line, it is:

```text
benchmarks/chatgpt_53_codex_xhigh/no_skills/candidate_waveform.py:209
```

because that line applies the wrong detector-domain amplitude convention.
