# Datasets

Curated datasets for each benchmark. HDF5 data files are not tracked in git (too large); use the curation scripts to regenerate from source.

| Benchmark | Directory | Source |
|-----------|-----------|--------|
| Ringdown (QNM) | `ringdown/` | Cook & Zalutskiy (2014), [Zenodo 2650358](https://zenodo.org/records/2650358) |
| Waveform | `waveform/` | TBD |
| Remnant | `remnant/` | TBD |
| Dynamics | `dynamics/` | TBD |
| Analytic | `analytic/` | TBD |
| Validity | `validity/` | TBD |

Each subdirectory contains:
- `README.md` — data source, format, and coverage documentation
- `scripts/` — curation scripts to download and process raw data
- `*.h5` — curated benchmark data (gitignored)
