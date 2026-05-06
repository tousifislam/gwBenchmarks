# Error Diagnosis: kimi_K26 RG Waveform

## Score Summary

This diagnosis uses the current 1000-case single-detector RG benchmark.

| quantity | value |
|---|---:|
| mean optimized mismatch | 0.7197067575 |
| median optimized mismatch | 0.7179113030 |
| p90 optimized mismatch | 0.8375000384 |
| max optimized mismatch | 0.8868852035 |
| failed evaluations | 0 |
| median SNR ratio | 1.0318e4 |

The candidate waveform has SHA256
`65d38849343cdeb2b67bf8062958321da44c261912cb89485aa9cdb920d67d7b`.

## Main Conclusion

The candidate reconstructs several pieces of the factorized 22-mode correction
reasonably well, but it does not implement the benchmark frequency-domain
waveform wrapper. The large mismatch is mainly caused by an incorrect
stationary-phase amplitude and an incorrect stationary-phase phase.

## Error 1: Wrong SPA Amplitude

Problem lines: `candidate_waveform.py:258` to `candidate_waveform.py:264`.

The candidate uses

```python
A = sqrt(5/24) * M_sec**(5/6) / (pi**(1/6) * dL_sec)
    * x_f**(7/6) / sqrt(F_22 * dt_dx)
```

This is not the benchmark amplitude. Since the candidate also defines

```python
dt_dx = -M_sec * dE_dx / F_22
```

the product `F_22 * dt_dx` cancels the flux and leaves an amplitude with the
wrong mass scaling and the wrong frequency dependence. It does not reproduce
the restricted Newtonian SPA scaling `A_N(f) proportional to Mc^(5/6)
f^(-7/6) / dL`.

The reference benchmark uses the balance-law amplitude

```text
A(f) = A_N(f) * sqrt[-2 d(E/M)/dx / nu].
```

This amplitude mistake explains the very large SNR normalization error:
the median candidate/reference SNR ratio is about `1.0e4`.

## Error 2: Wrong SPA Phase

Problem lines: `candidate_waveform.py:273` to `candidate_waveform.py:281`.

The candidate uses only a Newtonian TaylorF2 phase plus the complex phase of
the factorized mode:

```python
psi_N = 3/(128*nu) * (pi*Mc*f)^(-5/3)
psi = 2*pi*f*tc - phic + psi_N + arg(T_22) + delta_22
```

This is not the benchmark phase. The reference waveform computes the orbital
SPA phase by integrating the 22-only balance law,

```text
dt/dx = -M d(E/M)/dx / F_22(x),
dPhi_GW/dx = 2 x^(3/2) dt/dx / M,
Psi = integral dPhi_GW - 2 pi f integral dt,
```

and then adds the phase of `hhat_22`. The candidate computes `F_22` and
`dt_dx`, but never uses them to build the phase. This missing balance-law phase
is not removable by optimizing over coalescence time and phase, so it produces
a large waveform mismatch.

## Recommended Fix

Keep the factorized `hhat_22` ingredients, but replace the frequency-domain
wrapper:

1. Use the Newtonian SPA amplitude `A_N(f)` multiplied by the balance-law
   correction `sqrt[-2 d(E/M)/dx / nu]`.
2. Build the phase by integrating `dt/dx = -M d(E/M)/dx / F_22(x)` up to the
   cutoff reference point.
3. Add `angle(hhat_22)` to the integrated balance-law phase.
