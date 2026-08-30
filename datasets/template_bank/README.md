# Template Bank Benchmark Dataset

Template banks are a core computational object in gravitational-wave searches:
they discretize a continuous compact-binary parameter space into a finite set
of waveform templates, so that incoming or simulated signals can be matched
against a manageable bank rather than an intractable continuum. This benchmark
tests whether an agent can design such a bank under an explicitly numerical
coverage objective, balancing physical waveform fidelity against the practical
cost of overlap evaluations.

The task gives the agent a public pool of binary-black-hole parameters and asks
it to construct an ordered waveform bank plus code that reconstructs a best-fit
waveform from complex bank overlaps. The hidden evaluator independently
normalizes the reconstruction and the physical target before calculating their
match. The primary score is the smallest prefix length with match at least 0.97
for strictly more than 90% of hidden test waveforms.

The Template Bank Bench accepts arbitrary finite waveform rows on the supplied
frequency grid. Public parameter rows are:

```text
[m1, m2, s1z, s2z]
```

## Files

Public files expected in this directory:

- `f_amp.npy` - frequency grid in Hz
- `Aref_weights.npy` - amplitude normalization and quadrature weights
- `bank_wf_params.npy` - public 4D parameter pool for agent train/eval splits

Hidden/local evaluation files:

- `bank_wf_params_test.npy` - hidden 4D parameter pool for final scoring
- `calpha_grid_params.txt` - 29 tracked literature-reference intrinsic points

The public dataset will be hosted at:

```text
https://huggingface.co/datasets/GWagents/gwBenchmarks
```

## Evaluation

The hidden evaluator appends evaluator-private deterministic phases to the
hidden 4D test parameters and generates full IMRPhenomXHM h-plus waveforms. For
each target it computes complex overlaps, calls the submitted reconstruction
function, and normalizes both the reconstructed and target waveforms. A prefix
passes only if strictly more than 90% of targets have match at least 0.97 (at
least 901 of the 1,000 hidden waveforms).

Hidden validation/test waveforms use the full higher-mode set:

```text
[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4),
 (2, -2), (2, -1), (3, -3), (3, -2), (4, -4)]
```

Agents may use any subset of these modes while constructing cheap public
proxies, but final scoring uses the full set above.

The literature comparison uses all five positive/negative mode-pair filters at
each of its 29 intrinsic points, for an evaluation cost of 145 waveforms. The
29-point grid was selected greedily using public data and reaches 93.2% hidden
coverage at the 0.97 threshold.

## Regeneration

The prototype data-generation notebook/script lives outside the merged package
for now. When the hosted dataset is finalized, record its exact generation
command and checksums here.
