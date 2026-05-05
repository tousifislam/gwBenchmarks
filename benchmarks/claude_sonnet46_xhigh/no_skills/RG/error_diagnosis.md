# Error Diagnosis: `claude_sonnet46_xhigh/no_skills`

## Summary

This candidate is an RG-tail waveform candidate, not a finite-size waveform
candidate. It should therefore be judged with the RG-tail scorer:

- mean optimized mismatch: `0.25600514336908936`
- median optimized mismatch: `0.23459920061852346`
- p90 optimized mismatch: `0.4351603007616271`
- maximum optimized mismatch: `0.5472388831731552`
- mean `|Delta_sys lambda_RG| / sigma_lambda_RG`: `15.322290161435111`
- failed RG evaluations: `0`

The finite-size score in this folder is not meaningful. It fails because this
candidate does not implement the finite-size API and does not accept `chi1`.

## Main RG Waveform Error

The main error is the final stationary-phase amplitude normalization.

The problematic line is

- `candidate_waveform.py:267`

```python
amp_spa = amp_newt * np.abs(hhat_a) * np.sqrt(2.0*np.pi / np.abs(ddPsi_a))
```

This double-counts the amplitude/flux information. In the project reference
waveform, the balance-law amplitude is

```text
A_N(f) sqrt[-2 d(E_real/M)/dx / eta].
```

The reason is that the time-domain 22-mode amplitude contains `|hhat_22|`, while
the 22-only flux contains `|hhat_22|^2`. In the stationary-phase amplitude,
the `sqrt(df/dt)` factor contains the same `|hhat_22|`, so the explicit
`|hhat_22|` cancels. The reference implementation documents and applies this at

- `src/rg_tail/waveform.py:86-106`
- `src/rg_tail/waveform.py:787-789`

The candidate instead multiplies the Newtonian restricted amplitude by
`|hhat_22| sqrt(2 pi / |ddPsi/df^2|)`, producing a large frequency-dependent
amplitude error.

## Numerical Evidence

For a GW150914-like check case,

```text
Mc = 28.3 Msun,
eta = 0.247,
dL = 410 Mpc,
lambda_RG = 1,
f_low = 20 Hz,
fmax_over_fisco = 1.
```

The candidate/reference amplitude ratio is

```text
median: 16.172880701608825
min:     3.156124878554853
max:    47.39497203830885
rms(|ratio - 1|): 22.104244000997877
```

The same check shows that the RG 22-mode ingredients themselves are correct:

```text
hhat amplitude ratio median: 1.0
hhat amplitude ratio min:    0.9999999999999999
hhat amplitude ratio max:    1.0000000000000002
hhat phase RMS difference:   3.8918836124009064e-17 rad
```

The extra amplitude factor used by the candidate, compared to the reference
balance-law amplitude correction, is

```text
median: 16.120036962448946
min:     3.156124879301177
max:    47.1525301405985
```

This is enough to explain the large mismatch.

## Phase Behavior

The candidate's phase is not the dominant issue. After fitting out the arbitrary
constant and linear phase terms, the residual phase error in the same
GW150914-like case is

```text
phase residual RMS: 0.0018167074875866172 rad
phase residual max: 0.0044256019826096615 rad
```

That residual is not perfect, but it is small compared with the amplitude error.

The candidate also uses `-2.0 * phic` at

- `candidate_waveform.py:242`

while the project reference uses `-phic` at

- `src/rg_tail/waveform.py:785`

This does not affect the zero-`phic` benchmark cases strongly, but it is still a
phase-convention mismatch.

## Minimal Fix

Replace the final amplitude model with the project balance-law amplitude:

```python
dE_dx = _dEdx(x_a, eta)
amp_balance = amp_newt * np.sqrt(np.maximum(-2.0 * dE_dx / eta, 0.0))
h_a = amp_balance * np.exp(1.0j * phase) * W_a
```

Do not multiply by `abs(hhat_a)` and do not multiply by
`sqrt(2*pi / abs(ddPsi_a))` on top of `amp_newt`.

The RG phase should still include the complex tail phase through
`angle(hhat_22)`, consistent with the reference convention.

