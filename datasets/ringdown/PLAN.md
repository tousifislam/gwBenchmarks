# Ringdown Bench — Agent Plan

## Objective

Build models to predict quasi-normal mode (QNM) frequencies of Kerr black holes: the real part omega_R (oscillation frequency) and imaginary part omega_I (damping rate) as functions of the dimensionless spin a/M.

## Data

- **Training**: `ringdown_training.h5` — QNM data from Cook & Zalutskiy (2014)
- **Validation**: `ringdown_validation.h5` — held-out spin values
- **Input**: dimensionless spin a/M in [0, ~0.9999], mode indices (l, m, n)
- **Outputs**: omega_R (real frequency), omega_I (imaginary frequency / damping rate)
- **Modes covered**: l = 2–6, various m and overtone n values

### Loading data

```python
import h5py, numpy as np

with h5py.File("datasets/ringdown/ringdown_training.h5", "r") as f:
    # Navigate: l{l}/m{sign}{m}/n{n}
    g = f["l2/m+2/n0"]
    spin = g["spin"][:]
    omega_r = g["omega_real"][:]
    omega_i = g["omega_imag"][:]
```

## Loss Function

```
L = mean(|delta_omega_R / omega_R*|) + mean(|delta_omega_I / omega_I*|)
```

Mean relative error on both real and imaginary parts.

**Scoring**: t0 = 0.00001s, alpha = 0.03

## Key Properties

1. **1D input** (spin) per mode — this is a curve-fitting problem
2. **Smooth functions**: omega_R(a) and omega_I(a) are smooth, well-behaved
3. **Known asymptotics**: Schwarzschild limit (a=0) and extremal Kerr limit (a->1)
4. **Multiple modes**: need models for each (l, m, n) combination
5. **High precision available**: reference data is computed to ~12 digits
6. **Very fast evaluation required**: t0 = 10 microseconds

## Parameter Reparameterization

The input is the dimensionless spin a/M in [0, ~0.9999]. The agent should explore whether transforming the spin variable improves fit accuracy, especially near the extremal limit:

- **Raw spin**: a (direct)
- **Compactified**: a / (1 - a) — spreads out the near-extremal region
- **Logarithmic**: -log(1 - a) — natural for quantities that diverge at a = 1
- **Square root**: sqrt(1 - a^2) — relates to the irreducible mass
- **Chebyshev mapping**: map [0, 1] to [-1, 1] for Chebyshev polynomial fits
- **chi = 1 - a**: small near extremal, may simplify expansions

Test at least 2–3 reparameterizations and report which gives the best accuracy per approach.

## Approaches to Try (10–15)

### Polynomial and rational approximations
1. **Polynomial fit**: Fit omega(a) with polynomials of varying degree (5–20), use cross-validation.
2. **Chebyshev polynomials**: Chebyshev expansion on [0, 1] for numerical stability.
3. **Rational approximation (Padé)**: P_m(a) / Q_n(a) with optimized degrees.
4. **Continued fraction**: Iterative rational approximation.

### Symbolic and analytic
5. **Symbolic regression (PySR)**: Find compact closed-form expressions.
6. **Physics-inspired ansatz**: Fit omega = omega_Schw + a * f(a) with structured f(a).
7. **Power series in (1-a)**: Expansion near extremal Kerr limit.
8. **Levin-type approximation**: Rational fit designed for oscillatory QNM structure.

### Interpolation
9. **Cubic spline**: Natural cubic spline through training points.
10. **Chebyshev interpolation**: Interpolation at Chebyshev nodes.
11. **RBF interpolation**: Radial basis functions in 1D.

### Machine learning
12. **GPR**: Gaussian process with optimized kernel.
13. **Neural network**: Small MLP (overkill for 1D, but worth benchmarking).
14. **Random forest**: Ensemble of decision trees.

### Minimax
15. **Minimax polynomial**: Polynomial minimizing the maximum error (Remez algorithm).

## Evaluation Checklist

For each approach and each mode (l, m, n):
- [ ] Fit on training spin values
- [ ] Predict on validation spin values
- [ ] Compute relative errors on omega_R and omega_I
- [ ] Compute combined loss L
- [ ] Time per-evaluation (should be microseconds)
- [ ] Save scorecard.json

## Final Deliverables

1. **Progress plot** (updated after every approach): loss and eval time vs. approach number
2. **Violin plot**: per-approach distribution of relative errors across all modes
3. **Pareto plot**: loss vs. evaluation time (note: most methods will be very fast here; the differentiation is on accuracy)
4. **Summary table**: ranked approaches
5. **Residual plots**: (predicted - true) / true vs. spin for the best model
6. **Mode comparison**: accuracy across different (l, m, n) modes

All outputs go in `results/<agent>/ringdown/`. See `BENCHMARK_PLAN.md` for full directory structure and changelog format.
