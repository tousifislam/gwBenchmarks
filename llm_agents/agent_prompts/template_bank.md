# Template Bank Bench — {AGENT_LABEL} Agent

You are an autonomous agent building a compact ordered waveform bank for
frequency-domain gravitational-wave coverage.
Your work directory is: `llm_agents/results/{AGENT}/template_bank/`

## Task

Use the public parameter pool in `datasets/template_bank/bank_wf_params.npy` to
construct:

1. a public training subset
2. a public evaluation subset
3. an ordered waveform bank
4. a function that reconstructs a best-fit waveform from complex bank overlaps

Your objective is to pass the hidden coverage requirement with as few submitted
waveforms as possible. Hidden-test evaluation cost scales with the number of
submitted waveforms; training-time overlap calculations are diagnostic only.

A literature-motivated reference has an evaluation cost equivalent to 145
submitted waveforms and passes the target. Try to pass with fewer waveforms,
then minimize the smallest successful prefix. This reference number is a cost
comparison, not a restriction on the waveform construction method.

Do not stop at the first bank that reaches the public target. Perform at least
one evidence-based refinement of its waveform content, ordering, or
reconstruction method, and retain the best robust public-evaluation result.

## Completion Criteria

Before declaring DONE, verify:

- [ ] `bank_waveforms.npy` has shape `(n_waveforms, len(datasets/template_bank/f_amp.npy))`
- [ ] every bank row has finite values and nonzero norm
- [ ] `waveform_indices.npy` has shape `(n_waveforms,)`
- [ ] `reconstruction_function.py` defines `reconstruct_from_overlaps(complex_overlaps, bank_waveforms, indices)`
- [ ] `train_indices.npy` indexes the public parameter pool
- [ ] `eval_indices.npy` indexes the public parameter pool
- [ ] `run_summary.json` reports:
  - [ ] `method`
  - [ ] `n_waveforms`
  - [ ] `overlap_evaluations` (diagnostic only)
  - [ ] `public_train_size`
  - [ ] `public_eval_size`
  - [ ] `public_eval_threshold`
  - [ ] `public_eval_target_coverage`
  - [ ] `public_eval_coverage_fraction`
  - [ ] `public_eval_median_match`
  - [ ] `public_eval_prefix_length_to_target`
- [ ] the bank is ordered from highest-priority waveform to lowest-priority waveform
- [ ] the public evaluation result is explicitly reported in `run_summary.json`
- [ ] the reconstruction function returns one finite waveform with the same frequency-grid shape as a bank row

When all criteria are met, print `TEMPLATE_BANK_BENCH_COMPLETE` on its own line.

## Scoring Target

For each hidden-test waveform and each ordered bank prefix, the evaluator:

1. computes the complex inner products between the target and the submitted bank waveforms
2. calls `reconstruct_from_overlaps(...)` to obtain a best-fit waveform
3. normalizes the returned waveform and the physical target waveform
4. computes their absolute inner product as the final match

The submission never returns the final scalar match. A prefix passes only when
strictly more than 90% of hidden-test waveforms have evaluator-computed match
greater than or equal to `0.97`. For the 1,000-waveform hidden set, this means
at least 901 waveforms. The primary score is the smallest passing prefix length.
A submission that never passes has infinite loss.

## Public Data

Expected files:

- `datasets/template_bank/f_amp.npy`
- `datasets/template_bank/Aref_weights.npy`
- `datasets/template_bank/bank_wf_params.npy`

The public dataset is hosted at:

```text
https://huggingface.co/datasets/GWagents/gwBenchmarks
```

## Waveform Convention

The hidden evaluator generates validation/test waveforms with IMRPhenomXHM,
`h_plus`, and the full higher-mode set:

```text
[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4),
 (2, -2), (2, -1), (3, -3), (3, -2), (4, -4)]
```

During your own search, you may query any subset of these modes, or the full
set, as a speed/accuracy tradeoff. The helper function
`make_reference_waveform(..., harmonic_modes=...)` supports custom mode lists;
`make_full_reference_waveform(...)` uses the final hidden-evaluation convention.

## Required Output

Write all outputs directly under `llm_agents/results/{AGENT}/template_bank/`:

```text
bank_waveforms.npy
waveform_indices.npy
reconstruction_function.py
train_indices.npy
eval_indices.npy
run_summary.json
```

`bank_waveforms.npy` may contain any physically useful real or complex waveform
rows. Normalize each row with the benchmark inner product for your own public
evaluation; the evaluator also normalizes every row defensively.

`waveform_indices.npy` contains one integer identifier per bank row. The
evaluator passes the corresponding prefix of these identifiers to the
reconstruction function. The identifiers may encode public-pool indices,
groups, basis labels, or another convention used by your method.

`reconstruction_function.py` must define:

```python
def reconstruct_from_overlaps(complex_overlaps, bank_waveforms, indices=None):
    ...
```

`complex_overlaps` is a one-dimensional complex NumPy array with entries
`sum(target * conj(bank_waveform) * weights)`. `bank_waveforms` contains the
corresponding rows in the current prefix, and `indices` contains their submitted
identifiers. Return one complex or real best-fit waveform with shape
`(len(datasets/template_bank/f_amp.npy),)`. You may select a row, form a linear
combination, project onto a reduced basis, or use another method justified by
the public data. The evaluator assigns match zero to a zero-norm reconstruction
and rejects malformed or non-finite output.

## Rules

- Treat `datasets/` and `gwbenchmarks/` as read-only.
- Do not use hidden files such as `bank_wf_params_test.npy`.
- Do not assume all public parameters belong in the final bank.
- Choose your own public training/evaluation split and optimize for hidden generalization.
- When practical, check more than one deterministic public split.
- Use any public-data optimization strategy; the objective is the smallest passing waveform-bank prefix.
- `reconstruct_from_overlaps()` must return a best-fit waveform, not a scalar score.
