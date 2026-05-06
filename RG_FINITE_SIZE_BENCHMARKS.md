# RG and Finite-Size Waveform Benchmarks

This branch adds two agentic waveform-generation benchmarks intended to
complement the original `gwBenchmarks` waveform task.

The main summary of what is being tested is in:

- `beyond_GR_bench/README.md`

## Benchmarks

- `beyond_GR_bench/rg_waveform_benchmark`: candidates implement an RG-tail modified
  frequency-domain inspiral waveform.
- `beyond_GR_bench/finite_size_waveform_benchmark`: candidates implement a
  balance-law SPA waveform with finite-size corrections.

Each benchmark compares a candidate `candidate_waveform.py` against the hidden
project reference implementation in `src/rg_tail/waveform.py` or
`src/finite_size/waveform.py`.

## Scoring Convention

Both scorers now use the same single-detector convention as the original
`gwBenchmarks` waveform benchmark:

- PSD: PyCBC `aLIGOZeroDetHighPower`
- `f_low = 15 Hz`
- `f_high = 990 Hz`
- `df = 0.125 Hz`
- Metric: optimized frequency-domain mismatch, maximized over time and phase

The old multi-detector scores are not included in this branch.

## Source Packets

The no-skill benchmark prompts live in:

- `beyond_GR_bench/rg_waveform_source_packet`
- `beyond_GR_bench/finite_size_waveform_source_packet`

These packets are what an agent should receive in a fresh workspace. They
include a compact formula sheet and prompt, but not the reference
implementation, old candidates, score files, or skill-specific hints.

## Stored Results

This branch keeps the generated agent code and score files in:

- `beyond_GR_bench/agent_outputs`

Each model folder contains the candidate waveform implementation and the
machine-readable score JSON for the RG and finite-size tasks. A compact score
table is also stored in:

- `beyond_GR_bench/results_summary.md`

The RG folder also includes the current paper-style violin plot:

- `beyond_GR_bench/rg_waveform_benchmark/rg_waveform_all_models_violin.pdf`
- `beyond_GR_bench/rg_waveform_benchmark/rg_waveform_all_models_violin.png`

For candidates with a documented pure Fourier-convention mismatch, the raw
agent output is kept unchanged and a separate
`score_level13_fourier_fixed.json` diagnostic rescore is used in the summary
plot.

## Rerun One Candidate

From the repository root:

```bash
python beyond_GR_bench/rg_waveform_benchmark/score_candidate.py \
  --repo-root "$PWD" \
  --candidate /path/to/candidate_waveform.py \
  --label model_name \
  --output /path/to/score_level13.json \
  --skip-bias

python beyond_GR_bench/finite_size_waveform_benchmark/score_candidate.py \
  --repo-root "$PWD" \
  --candidate /path/to/candidate_waveform.py \
  --label model_name \
  --output /path/to/score_finite_size.json \
  --skip-bias
```

Use `--smoke` for a fast single-case check. The default full benchmark uses
1000 deterministic source cases for each task.

## Physics Notes

Detailed waveform notes are included in:

- `docs/rg_tail`
- `docs/finite_size`

These notes document the waveform conventions and the approximations used by
the reference implementations.
