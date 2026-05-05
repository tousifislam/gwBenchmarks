# Error diagnosis for `gemini_31_flash/no_skills`

This note diagnoses why
`benchmarks/gemini_31_flash/no_skills/candidate_waveform.py` still has a
noticeable mismatch against the project reference waveform in
`src/rg_tail/waveform.py`.

The short answer is:

**The candidate has two benchmark-visible issues: the raw phase is anchored at
the low-frequency edge instead of the benchmark's upper-reference convention,
and the SPA amplitude is missing the benchmark normalization.**

The factorized `(2,2)` sector, the 22-only flux, and the Fermi taper are mostly
right, so the optimized overlap stays high. But the raw phase overlap is poor
because the phase integrals are accumulated forward from `f_low`, while the
reference integrates to an upper reference point. On top of that, the
frequency-domain amplitude is too large by a constant factor.

## Benchmark result being diagnosed

Score file:

```text
benchmarks/gemini_31_flash/no_skills/score_level13.json
```

Summary:

```text
n_evaluations                         = 576
n_failed_evaluations                  = 0
mean optimized mismatch               = 6.4413834175e-04
median optimized mismatch             = 1.8954976160e-04
mean phase mismatch                   = 9.8131428814e-01
median phase mismatch                 = 9.9954007468e-01
median |log10(SNR_candidate/SNR_ref)| = 5.0114992540e-01
mean |bias(lambda_RG)| / sigma        = 14.11174927
median |bias(lambda_RG)| / sigma      = 1.90595231
```

The optimized mismatch is only moderate, but the raw phase mismatch is almost
maximal and the candidate SNR is systematically off by a factor of about
`10^0.5011 ≈ 3.17`.

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
median |h_candidate|/|h_reference| = 3.170661837335049
min    |h_candidate|/|h_reference| = 3.1706618373350466
max    |h_candidate|/|h_reference| = 3.1706618373350515
```

After removing the best-fit constant and linear phase terms, the residual phase
is tiny:

```text
phase residual RMS = 3.38e-4 rad
max phase residual = 1.47e-3 rad
```

So the raw phase problem is mostly a convention/anchoring issue, not a broken
physical phasing model.

## Primary wrong lines

### Candidate lines 143-149

These lines integrate forward from `f_low` and then add the accumulated orbital
phase directly. The benchmark reference instead builds the SPA phase relative
to an upper reference point:

```text
src/rg_tail/waveform.py:628-676
```

with both the time and orbital-phase integrals taken to the upper reference
cutoff. That difference is largely a constant plus linear term in `f`, which is
why the optimized overlap is still fine, but the raw phase overlap is not.

### Candidate line 156

This is not the benchmark amplitude normalization. The reference uses the
Newtonian SPA amplitude times the balance-law correction:

```text
src/rg_tail/waveform.py:741-748
src/rg_tail/waveform.py:787-798
```

In other words, the benchmark keeps the standard Newtonian prefactor and then
multiplies by the balance-law correction. The missing normalization is exactly
why the SNR ratio is a constant `3.170661837...`.

## Why this happened

The candidate followed the right physical ingredients from the source packet,
but it mixed two different waveform conventions:

- the phase was accumulated from the low-frequency boundary instead of the
  benchmark's upper-reference anchor;
- the amplitude was written as a self-contained closed form, but the benchmark
  scorer expects the standard Newtonian SPA normalization combined with the
  balance-law correction.

That combination gives a waveform that is physically plausible, but not the one
the hidden scorer is comparing against.

## Lines that are mostly correct

- The 4PN conservative energy and angular momentum blocks are broadly correct.
- The factorized `hhat_22 = H_eff * T_22 * rho_22^2 * exp(i delta_22)` sector
  matches the intended dominant-mode structure.
- The 22-only flux expression and the smooth Fermi taper are in the right
  spirit.

## Minimal benchmark-aligned fix

Use the benchmark's upper-anchor phase convention and the standard SPA
amplitude. That would remove both the raw phase mismatch and the constant
amplitude scale error.
