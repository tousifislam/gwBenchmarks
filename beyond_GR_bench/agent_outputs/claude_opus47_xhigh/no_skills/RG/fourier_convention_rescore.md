# Fourier-Convention Rescore: `claude_opus47_xhigh/no_skills/RG`

This is a post-hoc diagnostic rescore, not a replacement for the raw agent
submission. The original candidate used the opposite Fourier carrier
convention:

- it multiplied by `conj(hhat_22)`, contributing `-arg(hhat_22)` instead of
  `+arg(hhat_22)`;
- it used the opposite signs for the extrinsic `tc`, `phic`, and `pi/4` pieces.

The corrected diagnostic copy is
`candidate_waveform_fourier_fixed.py`. Only the final Fourier carrier assembly
was changed; the RG ingredients, balance-law integration, taper, and amplitude
logic were left unchanged.

## Score Comparison

Both scores use the 1000-case single-detector RG benchmark with the
PyCBC/LALSimulation `aLIGOZeroDetHighPower` PSD.

| file | mean optimized mismatch | median optimized mismatch | p90 optimized mismatch | max optimized mismatch |
|---|---:|---:|---:|---:|
| `score_level13.json` | `6.9489506385e-01` | `6.8821086669e-01` | `8.6289560539e-01` | `8.9716089409e-01` |
| `score_level13_fourier_fixed.json` | `1.7173044762e-07` | `9.6257122384e-08` | `4.4456986836e-07` | `8.6410556177e-07` |

The Fourier-fixed score confirms that the original Claude Opus 4.7 RG waveform
was otherwise very close to the reference implementation. Its catastrophic raw
mismatch was caused by the Fourier convention, not by a wrong RG tail, flux,
taper, or SPA amplitude model.
