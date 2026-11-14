# Template Bank Benchmark Dataset

Template banks are how gravitational-wave searches find signals in noisy data:
the data are filtered against a finite set of waveforms, and a detection
statistic computed from the filter outputs is compared with a threshold set by
the tolerated false-alarm rate. This benchmark asks an agent to design both
parts, the filters and the statistic, for binary-black-hole signals with
strong higher harmonics in stationary Gaussian noise.

> **Version note (2026-11).** This is the second version of the benchmark. The
> first, noiseless version scored how few waveforms were needed to reconstruct
> hidden signals with match >= 0.97. It saturated: a weighted POD (SVD) basis
> with orthogonal projection is essentially optimal there, and agents found it.
> With noise, every extra filter dimension also adds noise, so the statistic
> matters and there is no closed-form answer. Results under
> `llm_agents/results/*/template_bank/` dated before 2026-11 belong to the
> noiseless version and are not comparable.

## Task in one paragraph

Submit at most 32 filter waveforms `b_i` on the frequency grid below and a
function `detection_statistic(z)` of the complex filter outputs
`z_i = <d, b_i>`. The evaluator draws 2,000,000 noise-only trials and 10,000
signal trials `d = rho * h + n` (1,000 hidden targets x 10), with
`p(rho) ~ rho^-4` for `rho >= 5` (sources uniform in volume). The threshold is
set at false-alarm probability 1e-4 from the noise trials. The **loss** is

```text
loss = 1 - detection_fraction / oracle_detection_fraction = 1 - VT / VT_oracle
```

where the oracle is a single exactly known template (detection fraction 0.948).
Lower is better. The full agent-facing statement is
`llm_agents/agent_prompts/template_bank.md`.

## Conventions

- Inner product: `<a, b> = sum(a * conj(b) * weights)` with `weights` from
  `Aref_weights.npy`. Waveforms are divided by `A_ref` and normalized to unit norm.
- Noise: circular complex Gaussian with `E|n_f|^2 = 2 / weights_f`, so the
  filter outputs have covariance `E[z z^H] = 2 conj(G)`, `G` the Gram matrix of
  the filters, and `|z|^2` is chi-squared with 2 degrees of freedom for one
  unit-norm filter. The evaluator samples `z` directly from this distribution.
- Signals: IMRPhenomXHM, `h_plus` only, edge-on (`iota = pi/2`, where the
  higher harmonics are strongest), aligned spins, random reference phase,
  modes `(2,±2), (2,±1), (3,±3), (3,±2), (4,±4)`.
- Parameter rows: `[m1, m2, s1z, s2z]` (detector-frame solar masses,
  dimensionless aligned spins).

## Files

Public (copied into every agent workspace):

| File | Content |
|---|---|
| `f_amp.npy` | frequency grid in Hz |
| `Aref_weights.npy` | `(2, n_freq)`: amplitude normalization `A_ref` and quadrature weights |
| `bank_wf_params.npy` | public parameter pool for the agent's own train/eval splits |

Evaluator-only (never give these to a running agent):

| File | Content |
|---|---|
| `bank_wf_params_test.npy` | 1,000 hidden target parameter rows |
| `calpha_grid_params.npy` / `.txt` | intrinsic points of the literature mode-filter bank (reference solution only) |

The current `f_amp.npy` and `Aref_weights.npy` are the corrected grid of
2026-11-13 (md5 `0b9ff31c…` and `21eaa5e0…`). Make sure the Hugging Face copy
matches before running agents.

## Code

- `gwbenchmarks/benchmarks/template_bank_core.py`: waveform generation, noise
  model and `score_detection`, which implements the hidden scoring exactly.
- `llm_agents/make_template_bank_workspace.py`: builds an isolated agent workspace.
- `llm_agents/evaluate_template_bank.py`: hidden evaluator.
- `llm_agents/template_bank_reference_scores.json`: scores of reference solutions.

## Reference scores (hidden set)

| Solution | Rows | Detection fraction | 1 - VT/VT_oracle |
|---|---|---|---|
| Starter: farthest-first templates, max \|z\|² | 16 | 0.721 | 0.240 |
| Literature mode filters + marginalized likelihood ratio | 27 | 0.834 | 0.121 |
| POD basis + library-marginalized likelihood ratio | 16 | 0.857 | 0.097 |
| Oracle (exact template) | 1 | 0.948 | 0 |

Statistical uncertainty on a detection fraction is about 0.004.

## Regeneration

The parameter pools, the grid and the weights come from the benchmark design
notebook (not part of this repository). Record exact generation commands and checksums
here when the hosted dataset is finalized.
