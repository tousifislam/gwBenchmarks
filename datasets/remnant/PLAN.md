# Remnant Bench — Agent Plan

## Objective

Build models to predict the remnant black hole properties (final mass Mf, spin magnitude |chif|, kick magnitude |vf|) from pre-merger binary parameters.

## Data

- **Training**: `remnant_training.h5` — 1000 SXS simulations
- **Validation**: `remnant_validation.h5` — 1000 SXS simulations
- **Input parameters** (7D): q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z
- **Outputs** (3 scalars): Mf (final mass / total mass), |chif| (spin magnitude), |vf| (kick speed / c)

### Loading data

```python
import h5py, numpy as np

with h5py.File("datasets/remnant/remnant_training.h5", "r") as f:
    meta = f["metadata"]
    q = meta["q"][:]
    chi1x, chi1y, chi1z = meta["chi1x"][:], meta["chi1y"][:], meta["chi1z"][:]
    chi2x, chi2y, chi2z = meta["chi2x"][:], meta["chi2y"][:], meta["chi2z"][:]
    Mf = meta["Mf"][:]
    chif = meta["chif_mag"][:]
    vf = meta["vf_mag"][:]
```

## Loss Function

```
L = NRMSE(Mf) + NRMSE(chif) + NRMSE(vf)
```

NRMSE = RMSE / range(true values). Equal weight on all three quantities.

**Scoring**: t0 = 0.0001s, alpha = 0.05

## NR Error Floor

From Lev-to-Lev comparison:
- delta_Mf: median ~ 6.6e-05
- delta_chif: median ~ 1.9e-04
- Stored per-sim in the HDF5

**Target**: match NR error floor while keeping evaluation < 0.1 ms per prediction.

## Parameter Reparameterization

The raw parameters are (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z). The agent should systematically explore whether reparameterizations improve accuracy:

**Mass ratio transforms**:
- eta = q / (1+q)^2 (symmetric mass ratio — appears naturally in PN expressions)
- delta_m = (q - 1) / (q + 1) (mass difference)

**Spin transforms**:
- chi_eff = (q * chi1z + chi2z) / (1 + q) — approximately conserved, dominant spin effect
- chi_p — precessing spin parameter, controls precession amplitude
- |chi1|, |chi2| — spin magnitudes, approximately conserved
- Spherical: (|chi1|, theta1, phi1, |chi2|, theta2, phi2)
- chi_a = (chi1z - chi2z) / 2 — anti-symmetric spin combination

**Combined / derived features**:
- eta * chi_eff, delta_m * chi_a — products that appear in PN remnant formulas
- Total spin S = q * chi1 + chi2 (in natural units)

**The agent should test** at least 2–3 reparameterizations and report which gives the best results for each approach.

## Approaches to Try (10–15)

### Gaussian processes and kernel methods
1. **GPR (RBF kernel)**: Separate GPR for each of Mf, chif, vf.
2. **GPR (Matern kernel)**: Same with Matern-5/2 kernel.
3. **Kernel ridge regression**: With optimized kernel choice.
4. **Support vector regression**: With RBF kernel.

### Neural networks
5. **MLP**: Feedforward network with batch normalization.
6. **Multi-task MLP**: Single network predicting all three quantities jointly.
7. **Physics-informed neural network**: Enforce mass/energy conservation constraints.

### Tree-based methods
8. **Random forest**: Separate model per output.
9. **Gradient boosting (XGBoost)**: With careful hyperparameter tuning.
10. **LightGBM**: Fast gradient boosting variant.

### Classical and symbolic
11. **Polynomial regression**: Multivariate polynomials with cross-validated degree.
12. **Symbolic regression (PySR)**: Find closed-form expressions for each output.
13. **Phenomenological fits**: Physics-inspired functional forms (e.g., Jiménez-Forteza et al. style for Mf and chif).

### Interpolation and other
14. **RBF interpolation**: Radial basis function interpolation in 7D.
15. **Ensemble stacking**: Combine top 3–5 models via a meta-learner.

## Evaluation Checklist

For each approach:
- [ ] Train on training set (1000 samples)
- [ ] Predict on validation set (1000 samples)
- [ ] Compute NRMSE for each of Mf, chif, vf
- [ ] Compute combined loss L = sum of NRMSEs
- [ ] Time 1000 evaluations, report median per-prediction time
- [ ] Compare errors to NR error floor per sample
- [ ] Save scorecard.json

## Final Deliverables

1. **Progress plot** (updated after every approach): loss and eval time vs. approach number, NR error floor marked
2. **Violin plot**: per-approach error distributions for each output (Mf, chif, vf)
3. **Pareto plot**: combined loss vs. evaluation time, NR error floor marked
4. **Summary table**: all approaches ranked by combined score
5. **Scatter plots**: predicted vs. true for the best model (Mf, chif, vf)

All outputs go in `results/<agent>/remnant/`. See `BENCHMARK_PLAN.md` for full directory structure and changelog format.
