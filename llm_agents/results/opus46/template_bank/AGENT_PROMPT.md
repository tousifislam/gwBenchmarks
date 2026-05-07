# Template Bank Bench — Opus 4.6 Agent

You are an autonomous agent building a compact ordered template bank for
frequency-domain gravitational-wave waveform coverage.
Your work directory is: `llm_agents/results/opus46/template_bank/`

## Task

Use the public parameter pool in `datasets/template_bank/bank_wf_params.npy` to
construct:

1. a public training subset
2. a public evaluation subset
3. an ordered template bank

Each submitted bank row must be:

```text
[m1, m2, s1z, s2z, phi_ref]
```

Your objective is to reach the hidden-test target with as few ordered templates
and as few overlap calculations as possible.

## Completion Criteria

Before declaring DONE, verify:

- [ ] `bank_params.npy` exists and has shape `(n_bank, 5)`
- [ ] `train_indices.npy` exists and indexes `datasets/template_bank/bank_wf_params.npy`
- [ ] `eval_indices.npy` exists and indexes `datasets/template_bank/bank_wf_params.npy`
- [ ] `run_summary.json` exists and reports:
  - [ ] `method`
  - [ ] `n_bank`
  - [ ] `overlap_evaluations`
  - [ ] `public_train_size`
  - [ ] `public_eval_size`
  - [ ] `public_eval_threshold`
  - [ ] `public_eval_target_coverage`
  - [ ] `public_eval_coverage_fraction`
  - [ ] `public_eval_median_best_overlap`
  - [ ] `public_eval_prefix_length_to_50pct`
- [ ] the bank is ordered from highest-priority template to lowest-priority template
- [ ] the public evaluation result is explicitly reported in `run_summary.json`

When all criteria are met, print `TEMPLATE_BANK_BENCH_COMPLETE` on its own line.

## Scoring Target

The hidden evaluator scores the smallest prefix length whose templates give at
least 50% of hidden-test waveforms a best overlap of at least `0.97`.

Equivalently, the median of the hidden-test best-overlap distribution should be
at least `0.97`.

## Public Data

Expected files:

- `datasets/template_bank/f_amp.npy`
- `datasets/template_bank/Aref_weights.npy`
- `datasets/template_bank/bank_wf_params.npy`

The public dataset is hosted at:

```text
https://huggingface.co/datasets/GWagents/gwBenchmarks
```

## Required Output

Write all outputs directly under `llm_agents/results/opus46/template_bank/`:

```text
bank_params.npy
train_indices.npy
eval_indices.npy
run_summary.json
```

## Rules

- Treat `datasets/` and `gwbenchmarks/` as read-only.
- Do not use hidden files such as `bank_wf_params_test.npy`.
- Do not assume all public parameters belong in the final bank.
- Choose your own public training/evaluation split and optimize for hidden generalization.
- Prefer reasoning-driven search strategies over exhaustive overlap evaluation when possible.
