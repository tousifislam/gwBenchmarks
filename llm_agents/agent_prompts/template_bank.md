# Template Bank Detection Benchmark — {AGENT_LABEL} Agent

You are an autonomous agent designing a gravitational-wave search: a bank of
filter waveforms plus a detection statistic computed from the filter outputs.
Your work directory is the folder containing this file.

## Task

Using the public parameter pool in `data/bank_wf_params.npy`, construct:

1. an ordered set of at most 32 filter waveforms (`output/bank_waveforms.npy`)
2. a function `detection_statistic` that maps the complex filter outputs of a
   stretch of data to one real number (`output/detection_statistic.py`)

so that, at a fixed false-alarm probability, as many hidden signals as possible
are detected. Do not stop at the first working solution. Perform at least one
evidence-based refinement of the filters, the statistic, or both, and retain
the best robust public-evaluation result.

## Data model

All waveforms live on the frequency grid `data/f_amp.npy` and use the inner
product

```python
<a, b> = sum(a * conj(b) * weights)        # weights from data/Aref_weights.npy
```

Every filter row is normalized to `<b, b> = 1` by the evaluator.

Each trial is a data stretch `d`. A trial is either

- **noise only**: `d = n`, or
- **signal plus noise**: `d = rho * h + n`, where `h` is a unit-norm hidden
  IMRPhenomXHM waveform (full higher-mode set, random reference phase) and
  `rho` is its optimal signal-to-noise ratio.

The noise `n` is stationary Gaussian: independent circular complex Gaussian
values with `E|n_f|^2 = 2 / weights_f`. Consequently, for filter rows `b_i`,
the filter outputs `z_i = <d, b_i>` are jointly complex Gaussian with
`E[z_i conj(z_j)] = 2 <b_j, b_i>` under noise, and for one unit-norm filter
`|z|^2` is chi-squared with 2 degrees of freedom. Signal SNRs are drawn from
`p(rho) ∝ rho^-4` for `rho >= 5` (sources uniform in volume).

## Scoring

The hidden evaluator:

1. draws 2,000,000 noise-only trials and 10 signal trials for each of 1,000
   hidden target waveforms (10,000 signal trials), with private random seeds;
2. computes the complex filter outputs `z = <d, b_i>` for every trial and
   calls your `detection_statistic` on shuffled batches of 50,000 trials
   that mix noise-only and signal trials;
3. sets the threshold so that exactly a fraction `1e-4` of the noise-only
   trials exceed it;
4. reports the **detection fraction**: the fraction of signal trials whose
   statistic exceeds the threshold. Higher is better.

It also reports the detection fraction an exactly known single template would
achieve on the same SNRs; that number is an upper bound, not a target.

`template_bank_core.score_detection(...)` implements this procedure exactly.
Use it, with your own public splits, fewer trials and your own seeds, to
evaluate candidate solutions.

## Required output

Write all outputs under `output/`:

```text
output/
|-- bank_waveforms.npy        complex array, shape (n_rows, len(data/f_amp.npy)), 1 <= n_rows <= 32
|-- waveform_indices.npy      int array, shape (n_rows,), passed to detection_statistic
|-- detection_statistic.py    defines detection_statistic(complex_overlaps, bank_waveforms, indices)
|-- train_indices.npy         indices into data/bank_wf_params.npy
|-- eval_indices.npy          indices into data/bank_wf_params.npy
`-- run_summary.json
```

`detection_statistic(complex_overlaps, bank_waveforms, indices)`:

- `complex_overlaps`: complex array of shape `(n_trials, n_rows)`, entry
  `[t, i] = <d_t, b_i>`
- `bank_waveforms`: your normalized rows; `indices`: your `waveform_indices`
- returns a finite real array of shape `(n_trials,)`; larger means more
  signal-like. Only the ordering of values matters, because the threshold is
  set from the noise-only trials.

`detection_statistic.py` may load additional files that you place in
`output/` (locate them relative to `__file__`). It must be deterministic, use
only the trial data it is given, and must not read anything outside `output/`.
The hidden evaluation makes about 2,010,000 trial evaluations; keep the total
statistic runtime under 30 minutes on 5 cores.

`run_summary.json` must report at least: `method`, `n_waveforms`,
`public_train_size`, `public_eval_size`, `public_eval_noise_trials`,
`public_eval_signal_trials`, `public_eval_false_alarm_probability`,
`public_eval_detection_fraction`, `public_eval_detection_fraction_stderr`,
`public_eval_oracle_detection_fraction`, and `notes`.

## Completion criteria

Before declaring DONE, verify:

- [ ] all six output files exist with the shapes above
- [ ] `detection_statistic` runs on a batch of 50,000 trials and returns finite values
- [ ] the public evaluation result in `run_summary.json` was produced with
      `score_detection` on public data not used for construction
- [ ] the result was checked on more than one deterministic public split or seed

When all criteria are met, print `TEMPLATE_BANK_BENCH_COMPLETE` on its own line.

## Public files

- `data/f_amp.npy`, `data/Aref_weights.npy`, `data/bank_wf_params.npy`
- `template_bank_core.py`: waveform generation (`make_full_reference_waveform`,
  `generate_normalized_waveforms`), inner products, the noise model, and
  `score_detection`
- `detection_statistic.py`: starter statistic (maximum of `|z_i|^2`)
- `create_submission.py`: starter submission you may replace or improve

## Rules

- `data/` is read-only; keep all writes inside your work directory
- Use only the public data and public helper code in this workspace
- Use five or fewer CPU cores for any parallel work
- Use the Python interpreter given in `AGENTS.md` for every Python command;
  do not create another environment or install packages
