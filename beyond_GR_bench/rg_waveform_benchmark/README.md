# RG Waveform Benchmark

This folder contains the reproducible scorer for RG-tail waveform candidates.
It also contains `plot_all_models_violin.py`, which regenerates
`rg_waveform_all_models_violin.pdf/png` from the stored agent scores.

Example:

```bash
python benchmarks/rg_waveform_benchmark/score_candidate.py \
  --repo-root /path/to/repo \
  --candidate /path/to/repo/benchmarks/chatgpt_52_xhigh/no_skills/RG/candidate_waveform.py \
  --label chatgpt_52_xhigh_no_skills \
  --output /path/to/repo/benchmarks/chatgpt_52_xhigh/no_skills/RG/score_level13.json
```

Use `--smoke` for a fast small-case check, and `--skip-bias` if you only want
waveform mismatches without the Fisher bias calculation.

By default the scorer uses 1000 deterministic source cases with the
single-detector gwBenchmarks waveform convention: PyCBC
`aLIGOZeroDetHighPower`, `f_low = 15 Hz`, `f_high = 990 Hz`, and
`df = 0.125 Hz`. Use `--n-cases` to change the number of source cases.

If a candidate has a documented pure Fourier-convention error, the raw score is
kept as `score_level13.json` and the convention-normalized diagnostic rescore is
stored separately as `score_level13_fourier_fixed.json`. The plotting script
prefers the Fourier-fixed score when present and marks that model with a dagger.
