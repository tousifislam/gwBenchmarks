# Error Diagnosis: `chatgpt_52_xhigh/no_skills`

## Summary

This candidate is essentially correct. The benchmark score is

- mean optimized mismatch: `1.0922443989381925e-06`
- median optimized mismatch: `4.6065720743504457e-07`
- maximum optimized mismatch: `6.4164193780413825e-06`
- mean `|Delta_sys lambda_RG| / sigma_lambda_RG`: `0.07637791191900058`
- failed waveform evaluations: `0`

The remaining mismatch is not caused by a wrong RG-tail formula, flux formula, amplitude model, or taper. It is dominated by a small phase anchoring convention difference.

## Main Benchmark-Visible Issue

The candidate anchors the balance-law integrals at the largest sampled frequency in the input grid:

- `candidate_waveform.py:288`: `It = _cumtrapz(dt_dx, x)`
- `candidate_waveform.py:289`: `Iphi = _cumtrapz(dphi_gw_dx, x)`
- `candidate_waveform.py:290`: `total_t = It[-1]`
- `candidate_waveform.py:291`: `total_phi = Iphi[-1]`
- `candidate_waveform.py:294`: `t_of_x = tc - (total_t - It)`
- `candidate_waveform.py:295`: `phi_gw = phic - (total_phi - Iphi)`

This means the candidate effectively sets

```text
t(f_last_sample) = tc,
phi_gw(f_last_sample) = phic.
```

The project reference instead anchors the integrals at the exact cutoff frequency, not the last sampled frequency:

- `src/rg_tail/waveform.py:777`: `x_ref = (PI * M_sec * f_cut) ** (2.0 / 3.0)`
- `src/rg_tail/waveform.py:778`: calls `_spa_phase_from_22_flux(..., x_ref=x_ref)`
- `src/rg_tail/waveform.py:656-661`: appends `x_ref` to the quadrature grid when it is above the last waveform sample
- `src/rg_tail/waveform.py:669-676`: integrates to that reference point

Because the frequency grid generally stops below `f_cut`, the candidate and the reference differ by a small constant-plus-linear phase:

```text
Delta Psi(f) = Delta phi_c + 2 pi f Delta t_c.
```

For the worst-scoring benchmark case, `Mc = 12`, `eta = 0.12`, `dL = 200 Mpc`, `lambda_RG = 0.8`, `CE_40km`, the fitted difference is

```text
Delta Psi(f) = -0.017106355486254214 + 0.0001666404749947431 f.
```

The equivalent time shift is

```text
Delta t_c = 0.0001666404749947431 / (2 pi)
          = 2.6521655314594746e-05 s.
```

After subtracting this constant and linear phase, the residual phase RMS is

```text
4.986029121807609e-14 rad.
```

So the physical phasing is the same; the benchmark mismatch is mostly seeing a different arbitrary coalescence-time/coalescence-phase anchor.

## What Is Correct

The candidate correctly implements the RG-tail 22-mode structure

```text
hhat_22 = H_eff T_22 rho_22^2 exp(i delta_22).
```

The candidate also correctly uses the 22-only flux

```text
F_22 = (32 / 5) nu^2 x^5 |hhat_22|^2.
```

Relevant lines:

- `candidate_waveform.py:122-196`: constructs `hhat_22`
- `candidate_waveform.py:279-283`: computes `dE/dx`, `F_22`, and `dt/dx`
- `candidate_waveform.py:297-306`: constructs the SPA amplitude and phase
- `candidate_waveform.py:268` and `candidate_waveform.py:312`: applies the logistic taper

Direct numerical checks against the reference for the representative `GW150914`-like case give

```text
max relative hhat amplitude difference: 1.9984014443252818e-15
hhat phase RMS difference:             8.380162580032784e-17 rad
max relative dE/dx difference:          2.220446049250313e-16
max relative flux difference:           3.9968028886505635e-15
```

This is floating-point agreement.

## Minor Amplitude Difference

There is a negligible constant amplitude offset from the luminosity-distance conversion constant:

- candidate: `candidate_waveform.py:20`, `MPC_SEC = 3.0856775814913673e22 / 299792458.0`
- reference: `src/rg_tail/waveform.py:37`, `MPC_SEC = 1.0292712503e14`

The candidate/reference amplitude ratio is

```text
0.9999999997635317.
```

This is far too small to explain the benchmark mismatch.

## If We Wanted Exact Reference Agreement

The minimal reference-alignment change would be to integrate the candidate phase to the exact cutoff point `x_cut = (pi M f_cut)^(2/3)` rather than to `x[-1]`.

In practice, this means replacing the current sampled-grid anchor around `candidate_waveform.py:288-295` with the reference strategy:

```text
1. Compute x_cut = (pi M_sec f_cut)^(2/3).
2. If x_cut > x[-1], append x_cut to the quadrature grid.
3. Compute the upper-tail integrals from each x_i to x_cut.
4. Drop the appended reference point before returning h(f).
```

This would remove the small constant-plus-linear phase ramp and should push the mismatch down to numerical roundoff.

## Conclusion

There is no substantive waveform error in this candidate. The only benchmark-visible issue is the endpoint anchoring convention for `tc` and `phic`. The candidate is physically consistent with the project reference, but it does not exactly reproduce the reference's cutoff-anchored phase convention.
