# Template Bank Benchmark Dataset

Template banks are a core computational object in gravitational-wave searches:
they discretize a continuous compact-binary parameter space into a finite set
of waveform templates, so that incoming or simulated signals can be matched
against a manageable bank rather than an intractable continuum. This benchmark
tests whether an agent can design such a bank under an explicitly numerical
coverage objective, balancing physical waveform fidelity against the practical
cost of overlap evaluations.

The task gives the agent a public pool of binary-black-hole parameters and asks
it to construct an ordered bank of templates. The hidden evaluator measures how
well prefixes of this ordered bank cover an unseen test distribution using
IMRPhenomXHM frequency-domain waveforms. The primary score is the smallest
prefix length whose best-template overlaps reach match at least 0.97 for 50% of
hidden test waveforms, so better submissions achieve the target with fewer
templates and fewer overlap calculations.

The Template Bank Bench asks agents to construct an ordered bank of
frequency-domain gravitational-wave templates. Each submitted template row is

```text
[m1, m2, s1z, s2z, phi_ref]
```

where `phi_ref` is chosen by the agent. Public parameter rows omit phase:

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
- `calpha_grid_params.npy` - literature-reference template parameters

The public dataset will be hosted at:

```text
https://huggingface.co/datasets/GWagents/gwBenchmarks
```

## Evaluation

The hidden evaluator appends deterministic phases to the hidden 4D test
parameters and generates full IMRPhenomXHM h-plus waveforms. The main score is
the smallest submitted-bank prefix for which at least 50% of hidden test
waveforms have best overlap at least 0.97.

## Regeneration

The prototype data-generation notebook/script lives outside the merged package
for now. When the hosted dataset is finalized, record its exact generation
command and checksums here.
