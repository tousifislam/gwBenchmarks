# Finite-Size Waveform Benchmark

This folder contains the scorer for candidate implementations of the
finite-size balance-law SPA benchmark.

Example:

```bash
python benchmarks/finite_size_waveform_benchmark/score_candidate.py \
  --repo-root /path/to/repo \
  --candidate /path/to/candidate_waveform.py \
  --label model_name \
  --output /path/to/score_finite_size.json
```

Use `--smoke` for a fast single-case check, or `--skip-bias` to compute only
waveform mismatches.

By default the scorer uses 200 deterministic source cases with the
single-detector gwBenchmarks waveform convention: PyCBC
`aLIGOZeroDetHighPower`, `f_low = 15 Hz`, `f_high = 990 Hz`, and
`df = 0.125 Hz`. Use `--n-cases` to change the number of source cases.
