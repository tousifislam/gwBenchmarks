# Validity Bench — Agent Plan

## Objective

Build models to predict the time-domain mismatch between SXS NR waveforms and NRHybSur3dq8 surrogate model for aligned-spin BBH systems. This benchmark tests the agent's ability to assess surrogate model reliability — predicting where a model fails before running an expensive NR comparison.

## Data

- **Training**: `validity_training.h5` — 393 SXS simulations
- **Validation**: `validity_validation.h5` — 393 SXS simulations
- **Input parameters** (4D): q, chi1z, chi2z, omega0
- **Output** (scalar): mm_td (time-domain mismatch between NR and NRHybSur3dq8)

### Loading data

```python
import h5py, numpy as np

with h5py.File("datasets/validity/validity_training.h5", "r") as f:
    q = f["q"][:]
    chi1z = f["chi1z"][:]
    chi2z = f["chi2z"][:]
    omega0 = f["omega0"][:]
    mm_td = f["mm_td"][:]
```

## Loss Function

```
L = RMSE(log10(mm_pred), log10(mm_true)) + ECE
```

- **Log-RMSE**: RMSE in log10 space (since mismatches span orders of magnitude)
- **ECE**: Expected calibration error — how well-calibrated are the predictions?

**Scoring**: t0 = 0.001s, alpha = 0.05

## Key Properties

1. **4D input**: relatively low-dimensional, but the output spans many orders of magnitude
2. **Log-space modeling**: mismatches range from ~1e-07 to ~1e-01; modeling in log space is essential
3. **Non-trivial structure**: mismatch depends on parameter-space distance from the surrogate's training region
4. **Extrapolation detection**: high mismatch regions correspond to where the surrogate extrapolates
5. **Calibration matters**: the ECE term penalizes over/under-confident predictions

## Parameter Reparameterization

The raw parameters are (q, chi1z, chi2z, omega0). The agent should systematically explore reparameterizations:

- **Mass ratio**: eta = q / (1+q)^2, delta_m = (q-1)/(q+1), log(q)
- **Spin combinations**: chi_eff = (q*chi1z + chi2z) / (1+q), chi_a = (chi1z - chi2z) / 2
- **Frequency**: log(omega0), or normalized (omega0 - min) / (max - min)
- **Interaction terms**: q * chi_eff, eta * chi_a — may capture nonlinear dependencies
- **Distance to training boundary**: features measuring proximity to NRHybSur3dq8's known valid region (q <= 8, |chi| <= 0.8)

Test at least 2–3 reparameterizations and report which gives the best results.

## Approaches to Try (10–15)

### Gaussian processes and kernel methods
1. **GPR (RBF kernel)**: On log10(mm), with optimized lengthscales.
2. **GPR (Matern kernel)**: Matern-5/2 or Matern-3/2.
3. **Kernel ridge regression**: Fast kernel method.
4. **Support vector regression**: With RBF kernel.

### Neural networks
5. **MLP**: Feedforward network predicting log10(mm).
6. **Bayesian neural network**: For uncertainty quantification (improves ECE).
7. **Deep ensemble**: Multiple MLPs for calibrated predictions.

### Tree-based methods
8. **Random forest**: Ensemble of trees on log10(mm).
9. **Gradient boosting (XGBoost)**: With careful tuning.
10. **LightGBM**: Fast gradient boosting.
11. **Quantile regression forest**: For calibrated prediction intervals.

### Classical and symbolic
12. **Polynomial regression**: In log space.
13. **Symbolic regression (PySR)**: Find analytic expression for log10(mm) = f(q, chi1z, chi2z, omega0).
14. **Physics-informed**: Based on parameter-space distance to NRHybSur3dq8 training set boundary.

### Other
15. **Ensemble stacking**: Combine top models via a meta-learner for better calibration.

## Calibration Tips

- Train models in log10 space to handle the wide range
- Use predicted variance (from GPR, Bayesian NN, or ensembles) to assess calibration
- Post-hoc calibration: Platt scaling or isotonic regression on validation set
- ECE is computed by binning predictions and comparing mean predicted vs. actual values

## Evaluation Checklist

For each approach:
- [ ] Train on training set (393 samples)
- [ ] Predict on validation set (393 samples)
- [ ] Compute log-RMSE and ECE
- [ ] Compute combined loss L
- [ ] Time per-prediction evaluation
- [ ] Save scorecard.json
- [ ] Plot predicted vs. true mismatch (log-log)

## Final Deliverables

1. **Progress plot** (updated after every approach): loss and eval time vs. approach number
2. **Violin plot**: per-approach distribution of |log10(mm_pred) - log10(mm_true)|
3. **Pareto plot**: combined loss vs. evaluation time
4. **Summary table**: ranked approaches
5. **Calibration plot**: predicted vs. actual mismatch in log-log space for the best model
6. **Reliability diagram**: ECE binned calibration curve

All outputs go in `results/<agent>/validity/`. See `BENCHMARK_PLAN.md` for full directory structure and changelog format.
