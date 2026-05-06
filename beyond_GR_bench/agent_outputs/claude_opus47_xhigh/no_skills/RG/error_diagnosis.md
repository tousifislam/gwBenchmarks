# Error Diagnosis: `claude_opus47_xhigh/no_skills`

## Summary

This candidate is an RG-tail waveform candidate, not a finite-size waveform
candidate. It should therefore be judged with the RG-tail scorer:

- mean optimized mismatch: `0.7196371546342618`
- median optimized mismatch: `0.7709899642273503`
- p90 optimized mismatch: `0.8856939756948858`
- maximum optimized mismatch: `0.9124813134287639`
- mean `|Delta_sys lambda_RG| / sigma_lambda_RG`: `237.96283424204563`
- failed RG evaluations: `0`

The finite-size score in this folder is not meaningful. It fails because this
candidate does not implement the finite-size API and does not accept `chi1`.

## Main RG Waveform Error

The main error is a phase-convention and conjugation mismatch.

The problematic phase construction is

- `candidate_waveform.py:257-263`

```python
phase_natural = (
    2.0 * phi_rel_at_xf
    - 2.0 * np.pi * f_in * t_rel_at_xf
    + np.pi / 4.0
    + phic
    - 2.0 * np.pi * f_in * tc
)
```

and the problematic complex carrier is

- `candidate_waveform.py:265`

```python
common_factor = np.conj(hhat22_at_xf) * np.exp(1j * phase_natural)
```

The project reference uses

- `src/rg_tail/waveform.py:785`

```python
phase = 2.0 * PI * f_eval * tc - phic - PI / 4.0 + psi_orb + np.angle(hhat_running)
```

and then returns

- `src/rg_tail/waveform.py:798`

```python
h[valid] = amp * np.exp(1j * phase) * taper
```

So the reference adds `angle(hhat_22)` to the Fourier phase. The candidate uses
`conj(hhat_22)`, which contributes `-angle(hhat_22)` instead. It also reverses
the signs of the extrinsic pieces and the `pi/4` term relative to the project
benchmark convention.

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

The amplitude is essentially correct:

```text
candidate/reference amplitude ratio median: 0.9999999997635322
candidate/reference amplitude ratio min:    0.9999999997635312
candidate/reference amplitude ratio max:    0.9999999997635329
rms(|ratio - 1|):                            2.3646785734709425e-10
```

The RG 22-mode ingredients are also correct:

```text
hhat amplitude ratio median: 1.0
hhat amplitude ratio min:    0.9999999999999999
hhat amplitude ratio max:    1.0000000000000002
hhat phase RMS difference:   3.8918836124009064e-17 rad
```

But the phase mismatch is large. After fitting out the arbitrary constant and
linear phase terms, the residual is still

```text
phase residual RMS: 9.164736878651981 rad
phase residual max: 33.065920551406435 rad
phase residual p90: 11.4594811732841 rad
```

This is not a harmless coalescence-time or coalescence-phase offset. It is a
frequency-dependent phase error.

## Why Time/Phase Optimization Does Not Rescue It

The optimized mismatch allows a constant phase shift and a time shift. Those can
remove terms of the form

```text
Delta phase = constant + 2 pi f Delta t.
```

Here, after removing exactly that best-fit constant-plus-linear trend, the
nonlinear residual phase remains many radians. This is why the median optimized
mismatch is still `0.7709899642273503`.

## Minimal Fix

Keep the correct amplitude logic, but replace the final phase assembly with the
reference convention:

```python
psi_orb = phi_gw_to_ref - 2.0 * np.pi * f_in * time_to_ref
phase = 2.0 * np.pi * f_in * tc - phic - np.pi / 4.0 + psi_orb + np.angle(hhat22_at_xf)
h[in_band] = amp * np.exp(1.0j * phase) * W
```

Do not use `np.conj(hhat22_at_xf)` in the carrier. If the complex tail factor is
kept multiplicatively, use a convention that contributes `+angle(hhat_22)` to
the final phase, not `-angle(hhat_22)`.

