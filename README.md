<p align="center">
  <img src="logo.png" alt="gwBenchmarks" width="200">
</p>

# gwBenchmarks

Benchmark suite for evaluating LLM-based gravitational wave (GW) modelling using fully numeric, physically motivated metrics. All tasks avoid human scoring and rely on standard loss functions from GW astronomy.

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

## Frequency-domain mismatch

The FD mismatch is computed via PyCBC using the aLIGO `aLIGOZeroDetHighPower` PSD, maximized over time and phase shifts:

```
mismatch = 1 - max_{t,phi} <h_pred, h_ref> / sqrt(<h_pred, h_pred> <h_ref, h_ref>)
```

with `f_low = 15 Hz`, `f_high = 990 Hz`. PyCBC is required for the waveform and analytic benchmarks.

## Datasets

HDF5 dataset files are **not** stored in this repository due to size. Each benchmark directory under `datasets/` contains:
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
├── gwbenchmarks/
│   ├── __init__.py
│   ├── metrics.py          # FD mismatch, RMS relative error, NRMSE
│   ├── runner.py           # Benchmark runner
│   └── benchmarks/
│       ├── base.py         # Abstract benchmark class
│       ├── waveform.py
│       ├── remnant.py
│       ├── dynamics.py
│       ├── ringdown.py
│       ├── analytic.py
│       └── validity.py
├── configs/                # YAML configs per benchmark
└── datasets/               # READMEs, scripts, plots (HDF5 files hosted separately)
```
