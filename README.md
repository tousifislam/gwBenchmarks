[![arXiv](https://img.shields.io/badge/arXiv-2605.11269-b31b1b.svg)](https://arxiv.org/abs/2605.11269)
[![Website](https://img.shields.io/badge/Website-gwBenchmarks-blue.svg)](https://tousifislam.com/gwBenchmarks/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow.svg)](https://huggingface.co/GWagents)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=tousifislam.gwBenchmarks)
![Last Updated](https://img.shields.io/github/last-commit/tousifislam/gwBenchmarks/main?label=last%20updated)

![alt text](logo.png)

## **gwBenchmarks**

Benchmark suite for evaluating LLM-based gravitational wave (GW) modelling using fully numeric, physically motivated metrics. All tasks avoid human scoring and rely on standard loss functions from GW astronomy.

**Paper:** [arXiv:2605.11269](https://arxiv.org/abs/2605.11269)

**Investigators**: Tousif Islam (KITP, UC Santa Barbara) · Digvijay Wadekar (UT Austin) · Zihan Zhou (Princeton)

## Benchmarks

### 1. Waveform Bench (Co-precessing h₂₂)

| | |
|---|---|
| **Input** | q, spin vectors chi1, chi2, time grid t_i |
| **Output** | Re(h22(t_i)), Im(h22(t_i)) |
| **Loss** | Mean frequency-domain mismatch over total masses [40, 80, 120, 160, 200] M☉ |

### 2. Remnant Bench (Kick velocity)

| | |
|---|---|
| **Input** | q, spin vectors chi1, chi2 |
| **Output** | kick velocity magnitude v_k |
| **Loss** | NRMSE(v_k) |

### 3. Dynamics Bench (Eccentric spinning orbital dynamics)

| | |
|---|---|
| **Input** | q, chi1, chi2, initial conditions e0, x0, time grid t_i |
| **Output** | PN frequency parameter x(t_i) |
| **Loss** | Pointwise RMS relative error on x(t) |

### 4. Ringdown Bench (Quasi-normal modes)

| | |
|---|---|
| **Input** | final spin chi_f, mode indices (l, m, n) |
| **Output** | omega_real, omega_imag |
| **Loss** | Mean of relative errors on Re(ω) and Im(ω) |

### 5. Analytic Bench (Non-spinning BBH, q ∈ [1, 20])

| | |
|---|---|
| **Input** | q, time grid t_i |
| **Output** | Re(h22(t_i)), Im(h22(t_i)) |
| **Loss** | Mean frequency-domain mismatch over total masses [40, 80, 120, 160, 200] M☉ |

### 6. Validity Bench (NRHybSur3dq8 extrapolation)

| | |
|---|---|
| **Input** | q, chi1, chi2 |
| **Output** | predicted mismatch M̂ |
| **Loss** | RMSE(log M̂, log M*) |

### 7. Template Bank Bench (Detection in Gaussian noise)

| | |
|---|---|
| **Input** | public pool of `[m1, m2, s1z, s2z]` parameters; IMRPhenomXHM higher-mode waveforms; Gaussian noise model |
| **Output** | ≤ 32 filter waveforms plus a `detection_statistic(z)` of their complex outputs |
| **Loss** | 1 − VT/VT_oracle: fraction of hidden signals (SNR ≥ 5, p(ρ) ∝ ρ⁻⁴) missed at false-alarm probability 10⁻⁴, relative to an exactly known template |

The first, noiseless version of this benchmark (smallest bank reconstructing hidden waveforms at match ≥ 0.97) saturated: a weighted SVD basis is near-optimal. With noise, every extra filter dimension also raises the noise background, so the choice of statistic matters. Agents run in an isolated workspace built by `llm_agents/make_template_bank_workspace.py`; see `datasets/template_bank/README.md`.

### 8. New Physics Bench (RG-tail inspiral)

| | |
|---|---|
| **Input** | Physics formulas from arXiv:2602.08833 (source packet) |
| **Output** | `h_of_f(f, Mc, eta, dL, lambda_RG, ...)` implementation |
| **Loss** | Mean frequency-domain mismatch over 144 test cases (4 Mc × 4 eta × 3 dL × 3 lambda_RG) |

This benchmark is **formula-driven**: the agent receives post-Newtonian formulas for the dominant (2,2) mode with RG-tail corrections and must implement a frequency-domain waveform from first principles. The beyond-GR parameter `lambda_RG` deforms the radiative tail propagation; `lambda_RG = 1` recovers GR.

## Frequency-domain mismatch

The FD mismatch is computed via PyCBC using the aLIGO `aLIGOZeroDetHighPower` PSD, maximized over time and phase shifts:

```
mismatch = 1 - max_{t,phi} <h_pred, h_ref> / sqrt(<h_pred, h_pred> <h_ref, h_ref>)
```

with `f_low = 15 Hz`, `f_high = 990 Hz`. PyCBC is required for the waveform, analytic, and new physics benchmarks. The template bank benchmark uses its own inner product (see `datasets/template_bank/README.md`) and needs `lalsuite` instead (`pip install -e .[template_bank]`).

## Datasets

Binary dataset files are **not** stored in this repository due to size. They are hosted on Hugging Face:

**[🤗 GWagents/gwBenchmarks](https://huggingface.co/datasets/GWagents/gwBenchmarks)**

Each benchmark directory under `datasets/` contains:
- `README.md` — dataset description, parameter ranges, train/val split
- `scripts/` — curation and plotting scripts
- `plots/` — reference plots of the dataset

| Benchmark | Training | Validation |
|---|---|---|
| waveform | `waveform_training.h5` | `waveform_validation.h5` |
| remnant | `remnant_training.h5` | `remnant_validation.h5` |
| dynamics | `dynamics_training.h5` | `dynamics_validation.h5` |
| ringdown | `ringdown_training.h5` | `ringdown_validation.h5` |
| analytic | `analytic_training.h5` | `analytic_validation.h5` |
| validity | `validity_training.h5` | `validity_validation.h5` |
| template_bank | `bank_wf_params.npy` (+ `f_amp.npy`, `Aref_weights.npy`) | `bank_wf_params_test.npy` (hidden) |
| new_physics | — (formula-driven) | Reference in `gwbenchmarks/rg_tail_reference.py` |

## Rules

- No brute-force optimization at evaluation time — all outputs must be direct model predictions.
- Metrics are fully numeric and reproducible.

## Installation

```bash
pip install -e .
```

## Usage

```python
from gwbenchmarks import WaveformBench

bench = WaveformBench(config_path="configs/waveform.yaml")
result = bench.evaluate(predictions, targets, runtime=0.005)
print(f"Loss: {result.loss:.6f}")
```

## Project Structure

```
gwBenchmarks/
├── gwbenchmarks/                     # the Python package (pip install -e .)
│   ├── metrics.py                    # FD mismatch, RMS relative error, NRMSE
│   ├── runner.py                     # evaluate models across benchmarks
│   ├── rg_tail_reference.py          # reference waveform for New Physics Bench
│   ├── models/                       # API wrappers for LLM providers
│   └── benchmarks/
│       ├── base.py                   # abstract Benchmark class (compute_loss)
│       ├── waveform.py  remnant.py  dynamics.py  ringdown.py
│       ├── analytic.py  validity.py  new_physics.py
│       ├── template_bank.py          # TemplateBankBench (suite interface)
│       └── template_bank_core.py     # self-contained helpers + exact scoring, copied into agent workspaces
├── configs/                          # one YAML per benchmark
├── datasets/                         # per-benchmark README, scripts, plots (binary data on Hugging Face)
├── llm_agents/
│   ├── README.md                     # how to run agents through the suite
│   ├── agent_prompts/                # one prompt template per benchmark
│   ├── generate_prompt.py            # fills a prompt for an agent ID
│   ├── make_template_bank_workspace.py   # isolated workspace for Template Bank Bench
│   ├── evaluate_template_bank.py     # hidden evaluator for Template Bank Bench
│   ├── template_bank_reference_scores.json
│   ├── starters/template_bank/       # starter submission copied into each workspace
│   └── results/<agent>/<benchmark>/  # agent work and scores
├── llm_behaviours/                   # notes on observed agent behaviour
├── docs/                             # project website (GitHub Pages)
└── tests/
```
