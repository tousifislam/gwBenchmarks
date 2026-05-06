# Datasets

Curated datasets for each benchmark. Binary data files are not tracked in git;
download released artifacts from Hugging Face or use the curation scripts to
regenerate them from source:

```text
https://huggingface.co/datasets/GWagents/gwBenchmarks
```

| Benchmark | Directory | Source |
|-----------|-----------|--------|
| Ringdown (QNM) | `ringdown/` | Cook & Zalutskiy (2014), [Zenodo 2650358](https://zenodo.org/records/2650358) |
| Waveform | `waveform/` | SXS catalog v3.0.0, coprecessing h22 from precessing BBH (250+250) |
| Remnant | `remnant/` | SXS catalog v3.0.0, precessing BBH remnant properties (1000+1000) |
| Dynamics | `dynamics/` | SEOBNRv5EHM eccentric orbital dynamics (250+250, LHS) |
| Analytic | `analytic/` | SXS catalog v3.0.0, non-spinning BBH h22 (q = 1–20) |
| Validity | `validity/` | SXS catalog + NRHybSur3dq8, aligned-spin mismatch |
| Template Bank | `template_bank/` | IMRPhenomXHM frequency-domain template-bank parameter pools |

Each subdirectory contains:
- `README.md` — data source, format, and coverage documentation
- `scripts/` — curation scripts to download and process raw data
- `*.h5`, `*.npy` — curated benchmark data (gitignored)

The Hugging Face repository mirrors this directory convention, with one
top-level folder per benchmark.
