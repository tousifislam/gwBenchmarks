# Error diagnosis for `gemini_31_pro/no_skills`

This note diagnoses why
`benchmarks/gemini_31_pro/no_skills/candidate_waveform.py` still has a
non-negligible mismatch against the project reference waveform in
`src/rg_tail/waveform.py`.

The short answer is:

**The phase convention is essentially correct, but the SPA amplitude
normalization is still off by a constant factor.**

This candidate gets the upper-anchor phase integration right, so the raw phase
overlap is already extremely close to one. The remaining mismatch comes from a
uniform amplitude scale error, which also explains the inflated Fisher bias.

## Benchmark result being diagnosed

Score file:

```text
benchmarks/gemini_31_pro/no_skills/score_level13.json
```

Summary:

```text
n_evaluations                         = 576
n_failed_evaluations                  = 0
mean optimized mismatch               = 7.2124508583e-05
median optimized mismatch             = 2.0594770571e-05
mean phase mismatch                   = 7.7733709536e-05
median phase mismatch                 = 2.3998924806e-05
median |log10(SNR_candidate/SNR_ref)| = 6.5166492324e-01
mean |bias(lambda_RG)| / sigma        = 3.48983050
median |bias(lambda_RG)| / sigma      = 1.47096775
```

The overlap is excellent, including the unoptimized phase overlap. The
remaining problem is a constant amplitude scale mismatch of about
`10^0.6517 ≈ 4.48`.

## Direct diagnostic

For a representative point,

```text
Mc = 28.3
eta = 0.247
dL = 410 Mpc
lambda_RG = 1
f_low = 20 Hz
fmax_over_fisco = 1
sigma_taper_over_fisco = 0.01
```

the candidate/reference amplitude ratio is exactly constant across frequency:

```text
median |h_candidate|/|h_reference| = 4.483992972058022
min    |h_candidate|/|h_reference| = 4.483992972058020
max    |h_candidate|/|h_reference| = 4.483992972058026
```

After removing the best-fit constant and linear phase terms, the residual phase
is tiny:

```text
phase residual RMS = 1.09e-4 rad
max phase residual = 4.73e-4 rad
```

So the phase shape is basically correct; the score is being driven by the
overall amplitude normalization.

## Primary wrong line

### Candidate line 176

The benchmark reference uses the Newtonian SPA amplitude times the balance-law
correction:

```text
src/rg_tail/waveform.py:741-748
src/rg_tail/waveform.py:787-798
```

In other words, the benchmark keeps the standard Newtonian prefactor and then
multiplies by the balance-law correction. The missing normalization is exactly
what produces the constant SNR ratio `4.483992972...`.

## Why this happened

The candidate imported the right physical ingredients from the source packet:

- the 4PN conservative sector,
- the factorized `hhat_22` tail structure,
- the 22-only flux,
- and the upper-anchor phase integration.

What it did not preserve is the benchmark's exact SPA amplitude convention.
That leaves the waveform shape nearly unchanged, but scales the strain by a
constant factor, which is enough to move the Fisher bias score.

## Lines that are mostly correct

- The upper-anchor time and phase integrals match the reference convention.
- The factorized `hhat_22` sector is structurally correct.
- The flux and tapering behavior are in the right place.

## Minimal benchmark-aligned fix

Replace the custom amplitude with the benchmark SPA normalization. Keeping the
current phase construction should be fine; the amplitude normalization is the
remaining issue.
