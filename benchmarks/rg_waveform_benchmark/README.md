# RG Waveform Benchmark

This folder contains the reproducible scorer for RG-tail waveform candidates.

Example:

```bash
python benchmarks/rg_waveform_benchmark/score_candidate.py \
  --repo-root /path/to/repo \
  --candidate /path/to/candidate_waveform.py \
  --label model_name \
  --output /path/to/score_level13.json
```

Use `--smoke` for a fast small-case check, and `--skip-bias` if you only want
waveform mismatches without the Fisher bias calculation.

By default the scorer uses the single-detector gwBenchmarks waveform convention:
PyCBC `aLIGOZeroDetHighPower` with `f_low = 15 Hz`, `f_high = 990 Hz`, and
`df = 0.125 Hz`. The full RG benchmark uses 144 deterministic source cases.
