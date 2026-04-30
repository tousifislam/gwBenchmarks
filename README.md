# gwBenchmarks

Benchmark suite for evaluating LLM-based models for gravitational wave (GW) modelling using fully numeric, automatable metrics. All tasks avoid human scoring and instead rely on physically meaningful loss functions.

## Scoring

Each benchmark *b* is scored using:

```
S_b = L_b * [1 + alpha_b * log(1 + t_b / t0_b)]
```

where:
- **L_b** = accuracy loss (benchmark-specific)
- **t_b** = evaluation runtime
- **t0_b** = reference runtime scale
- **alpha_b** = cost penalty weight

Lower scores are better.

## Benchmarks

### 1. Waveform Bench (Co-precessing h22)

| | |
|---|---|
| **Input** | q, spin vectors chi1, chi2, time grid t_i |
| **Output** | Re(h22_copr(t_i)), Im(h22_copr(t_i)) |
| **Loss** | L = w1\*M + w2\*RMSE(phi) + w3\*RMSE(log A) |
| **t0** | 10 ms per waveform |
| **alpha** | 0.10 |

### 2. Remnant Bench

| | |
|---|---|
| **Input** | q, spin vectors chi1, chi2 |
| **Output** | final mass Mf/M, final spin vector chi_f, kick velocity vector v_k |
| **Loss** | L = NRMSE(Mf) + NRMSE(chi_f) + NRMSE(v_k) |
| **t0** | 0.1 ms per point |
| **alpha** | 0.05 |

### 3. Dynamics Bench (Eccentric Spinning)

| | |
|---|---|
| **Input** | q, chi1, chi2, initial conditions e0, x0, zeta0, time grid t_i |
| **Output** | e(t_i), x(t_i), zeta(t_i) |
| **Loss** | L = RMSE(e) + RMSE(x) + mean(1 - cos(zeta - zeta\*)) |
| **t0** | 10 ms per trajectory |
| **alpha** | 0.10 |

### 4. Ringdown Bench (QNM)

| | |
|---|---|
| **Input** | final mass Mf, final spin chi_f, mode indices (l, m, n) |
| **Output** | omega_real, omega_imag |
| **Loss** | L = \|d_omega_R / omega_R\*\| + \|d_omega_I / omega_I\*\| |
| **t0** | 0.01 ms per query |
| **alpha** | 0.03 |

### 5. Analytic Bench (Non-spinning BBH, q in [1, 20])

| | |
|---|---|
| **Input** | q, time grid t_i |
| **Output** | analytic surrogate waveform |
| **Loss** | L = mismatch + lambda \* RMSE(coefficients) |
| **t0** | 1 ms per waveform |
| **alpha** | 0.10 |

Requirements:
- Correct equal-mass limit
- Smooth behavior in q
- Correct test-mass trend near q=20
- No unphysical oscillations

### 6. Validity Bench (NRHybSur3dq8)

| | |
|---|---|
| **Input** | q, chi1, chi2 |
| **Output** | predicted mismatch M_hat |
| **Loss** | L = RMSE(log M_hat, log M\*) + ECE |
| **t0** | 1 ms per point |
| **alpha** | 0.05 |

Goal: evaluate extrapolation awareness and reliability prediction.

## Rules

- All datasets must include held-out parameter regions.
- No brute-force optimization allowed at evaluation time.
- All outputs must be direct predictions.
- Scores are fully numeric and reproducible.

## Installation

```bash
pip install -e .
```

## Usage

```python
from gwbenchmarks import WaveformBench

bench = WaveformBench(config_path="configs/waveform.yaml")
result = bench.evaluate(predictions, targets, runtime=0.005)
print(f"Loss: {result.loss:.6f}, Score: {result.score:.6f}")
```

## Project Structure

```
gwBenchmarks/
├── gwbenchmarks/
│   ├── __init__.py
│   ├── scoring.py          # Cost-penalized scoring formula
│   ├── metrics.py          # Mismatch, RMSE, NRMSE, ECE, etc.
│   ├── benchmarks/
│   │   ├── base.py         # Abstract benchmark class
│   │   ├── waveform.py     # Benchmark 1
│   │   ├── remnant.py      # Benchmark 2
│   │   ├── dynamics.py     # Benchmark 3
│   │   ├── ringdown.py     # Benchmark 4
│   │   ├── analytic.py     # Benchmark 5
│   │   └── validity.py     # Benchmark 6
│   └── data/
├── configs/                # YAML configs per benchmark
├── tests/
└── examples/
```
