# Ringdown Bench - opus48 CHANGELOG
Target: Kerr (l=2,m=2,n=0) QNM (omega_R, omega_I) vs spin a. Loss = 0.5*(mean|dwR/wR| + mean|dwI/wI|). Smooth 1D functions; the a->1 (near-extremal) region and omega_I->0 drive relative error. Log-compactification -log(1-a) resolves the near-extremal regime.

## Approaches (23)

### 1. poly10_raw [analytical]
- **Observed**: mean rel err val 9.794e-01, train 1.168e+00. omega_I is harder (rel err wR=5.62e-03, wI=1.95e+00); train/val gap 0.8x.
- **Hypothesis/Change**: Degree-10 polynomial in raw a (baseline).
- **Result**: raw_a reparam, 0.0001 ms/spin.

### 2. poly15_logcompact [analytical]
- **Observed**: mean rel err val 8.176e-03, train 1.889e-03. omega_I is harder (rel err wR=6.69e-06, wI=1.63e-02); train/val gap 4.3x.
- **Hypothesis/Change**: Degree-15 polynomial in -log(1-a): resolves the a->1 region.
- **Result**: log_compact reparam, 0.0001 ms/spin.

### 3. cheby14_raw [analytical]
- **Observed**: mean rel err val 6.342e-01, train 7.575e-01. omega_I is harder (rel err wR=3.08e-03, wI=1.27e+00); train/val gap 0.8x.
- **Hypothesis/Change**: Chebyshev expansion (avoids Runge oscillation).
- **Result**: raw_a reparam, 0.0001 ms/spin.

### 4. cheby20_chebymap [analytical]
- **Observed**: mean rel err val 4.194e-01, train 5.019e-01. omega_I is harder (rel err wR=1.72e-03, wI=8.37e-01); train/val gap 0.8x.
- **Hypothesis/Change**: High-order Chebyshev on [-1,1]-mapped spin.
- **Result**: cheby_map reparam, 0.0002 ms/spin.

### 5. rational66_raw [analytical]
- **Observed**: mean rel err val 2.934e-02, train 2.993e-02. omega_I is harder (rel err wR=1.19e-04, wI=5.86e-02); train/val gap 1.0x.
- **Hypothesis/Change**: Rational [6,6] Pade with rel-error LM refinement.
- **Result**: raw_a reparam, 0.0002 ms/spin.

### 6. rational88_logcompact [analytical]
- **Observed**: mean rel err val 8.775e-06, train 8.733e-06. omega_I is harder (rel err wR=5.65e-06, wI=1.19e-05); train/val gap 1.0x.
- **Hypothesis/Change**: Rational [8,8] on log-compactified spin.
- **Result**: log_compact reparam, 0.0003 ms/spin.

### 7. poly12_sqrtirr [analytical]
- **Observed**: mean rel err val 1.347e-03, train 1.499e-03. omega_I is harder (rel err wR=1.11e-03, wI=1.59e-03); train/val gap 0.9x.
- **Hypothesis/Change**: Degree-12 polynomial in sqrt(1-a^2) (third reparam).
- **Result**: sqrt_irr reparam, 0.0001 ms/spin.

### 8. pysr_raw [symbolic]
- omega_R ~ `sqrt(square(cube((-4.787172 - (-0.0968742 * ((cube(cube(square(-0.8262566) + (x0 / -1.0028`
- **Observed**: mean rel err val 5.109e-01, train 6.106e-01. omega_I is harder (rel err wR=5.26e-03, wI=1.02e+00); train/val gap 0.8x.
- **Hypothesis/Change**: PySR on omega_R and omega_I (raw a).
- **Result**: raw_a reparam, 0.0120 ms/spin.

### 9. gplearn_raw [symbolic]
- omega_R ~ `inv(sqrt(div(sub(div(X0, X0), mul(sqrt(0.590), mul(X0, X0))), 0.191)))`
- **Observed**: mean rel err val 2.065e+01, train 2.456e+01. omega_I is harder (rel err wR=7.23e-02, wI=4.12e+01); train/val gap 0.8x.
- **Hypothesis/Change**: gplearn SymbolicRegressor on omega_R, omega_I.
- **Result**: raw_a reparam, 0.0004 ms/spin.

### 10. pysr_logcompact [symbolic]
- omega_R ~ `((0.48289332 / (((((x0 / (sqrt(exp(x0)) + sqrt(x0))) + ((x0 + 0.71445537) * 0.9595109)) / `
- **Observed**: mean rel err val 1.177e-02, train 1.184e-02. omega_I is harder (rel err wR=3.63e-03, wI=1.99e-02); train/val gap 1.0x.
- **Hypothesis/Change**: PySR, second reparam (log-compactified).
- **Result**: log_compact reparam, 0.0091 ms/spin.

### 11. gplearn_sqrtirr [symbolic]
- omega_R ~ `sub(0.968, mul(X0, 0.600))`
- **Observed**: mean rel err val 2.063e+01, train 2.455e+01. omega_I is harder (rel err wR=3.64e-02, wI=4.12e+01); train/val gap 0.8x.
- **Hypothesis/Change**: gplearn on sqrt(1-a^2).
- **Result**: sqrt_irr reparam, 0.0003 ms/spin.

### 12. cubic_spline_raw [interpolation]
- **Observed**: mean rel err val 1.130e-05, train 6.854e-17. omega_I is harder (rel err wR=8.35e-08, wI=2.25e-05); train/val gap 11299381.3x.
- **Hypothesis/Change**: Cubic interpolating spline.
- **Result**: raw_a reparam, 0.0001 ms/spin.

### 13. spline_smooth_logcompact [interpolation]
- **Observed**: mean rel err val 1.333e-04, train 1.318e-04. omega_I is harder (rel err wR=4.93e-07, wI=2.66e-04); train/val gap 1.0x.
- **Hypothesis/Change**: Smoothing spline on log-compact.
- **Result**: log_compact reparam, 0.0001 ms/spin.

### 14. rbf_thinplate_raw [interpolation]
- **Observed**: mean rel err val 6.842e-03, train 7.229e-03. omega_I is harder (rel err wR=1.69e-04, wI=1.35e-02); train/val gap 0.9x.
- **Hypothesis/Change**: Thin-plate-spline RBF.
- **Result**: raw_a reparam, 0.0031 ms/spin.

### 15. rbf_multiquadric_raw [interpolation]
- **Observed**: mean rel err val 3.074e+01, train 2.690e+01. omega_I is harder (rel err wR=1.04e+00, wI=6.05e+01); train/val gap 1.1x.
- **Hypothesis/Change**: Multiquadric RBF.
- **Result**: raw_a reparam, 0.0019 ms/spin.

### 16. rbf_cubic_logcompact [interpolation]
- **Observed**: mean rel err val 6.338e-06, train 9.878e-11. omega_I is harder (rel err wR=3.79e-09, wI=1.27e-05); train/val gap 64160.7x.
- **Hypothesis/Change**: Cubic RBF on log-compact.
- **Result**: log_compact reparam, 0.0019 ms/spin.

### 17. gpr_rbf_raw [ml]
- **Observed**: mean rel err val 3.301e-01, train 3.958e-01. omega_I is harder (rel err wR=5.22e-04, wI=6.60e-01); train/val gap 0.8x.
- **Hypothesis/Change**: Gaussian process (RBF).
- **Result**: raw_a reparam, 0.0034 ms/spin.

### 18. gpr_rbf_logcompact [ml]
- **Observed**: mean rel err val 1.459e-03, train 1.248e-03. omega_I is harder (rel err wR=3.37e-05, wI=2.88e-03); train/val gap 1.2x.
- **Hypothesis/Change**: GPR on log-compactified spin.
- **Result**: log_compact reparam, 0.0039 ms/spin.

### 19. mlp_raw [ml]
- **Observed**: mean rel err val 2.329e+00, train 2.774e+00. omega_I is harder (rel err wR=8.99e-03, wI=4.65e+00); train/val gap 0.8x.
- **Hypothesis/Change**: MLP (tanh, 128x128).
- **Result**: raw_a reparam, 0.0015 ms/spin.

### 20. rf_raw [ml]
- **Observed**: mean rel err val 6.349e-03, train 8.426e-03. omega_I is harder (rel err wR=4.80e-04, wI=1.22e-02); train/val gap 0.8x.
- **Hypothesis/Change**: Random forest.
- **Result**: raw_a reparam, 0.0622 ms/spin.

### 21. histgbm_raw [ml]
- **Observed**: mean rel err val 8.606e-02, train 1.037e-01. omega_I is harder (rel err wR=1.24e-03, wI=1.71e-01); train/val gap 0.8x.
- **Hypothesis/Change**: Hist gradient boosting.
- **Result**: raw_a reparam, 0.0671 ms/spin.

### 22. krr_rbf_logcompact [ml]
- **Observed**: mean rel err val 8.275e-05, train 5.092e-05. omega_I is harder (rel err wR=3.07e-06, wI=1.62e-04); train/val gap 1.6x.
- **Hypothesis/Change**: Kernel ridge (RBF) on log-compact.
- **Result**: log_compact reparam, 0.0030 ms/spin.

### 23. rational1010_logcompact [analytical]
- **Observed**: mean rel err val 5.330e-06, train 5.262e-06. omega_I is harder (rel err wR=5.19e-07, wI=1.01e-05); train/val gap 1.0x.
- **Hypothesis/Change**: Higher-order rational [10,10]: reasoned push on a->1 accuracy.
- **Result**: log_compact reparam, 0.0004 ms/spin.

## Ranking (by mean rel err)

| rank | approach | category | loss | wR | wI | train |
|---|---|---|---|---|---|---|
| 1 | rational1010_logcompact | analytical | 5.33e-06 | 5.19e-07 | 1.01e-05 | 5.26e-06 |
| 2 | rbf_cubic_logcompact | interpolation | 6.34e-06 | 3.79e-09 | 1.27e-05 | 9.88e-11 |
| 3 | rational88_logcompact | analytical | 8.78e-06 | 5.65e-06 | 1.19e-05 | 8.73e-06 |
| 4 | cubic_spline_raw | interpolation | 1.13e-05 | 8.35e-08 | 2.25e-05 | 6.85e-17 |
| 5 | krr_rbf_logcompact | ml | 8.27e-05 | 3.07e-06 | 1.62e-04 | 5.09e-05 |
| 6 | spline_smooth_logcompact | interpolation | 1.33e-04 | 4.93e-07 | 2.66e-04 | 1.32e-04 |
| 7 | poly12_sqrtirr | analytical | 1.35e-03 | 1.11e-03 | 1.59e-03 | 1.50e-03 |
| 8 | gpr_rbf_logcompact | ml | 1.46e-03 | 3.37e-05 | 2.88e-03 | 1.25e-03 |
| 9 | rf_raw | ml | 6.35e-03 | 4.80e-04 | 1.22e-02 | 8.43e-03 |
| 10 | rbf_thinplate_raw | interpolation | 6.84e-03 | 1.69e-04 | 1.35e-02 | 7.23e-03 |
| 11 | poly15_logcompact | analytical | 8.18e-03 | 6.69e-06 | 1.63e-02 | 1.89e-03 |
| 12 | pysr_logcompact | symbolic | 1.18e-02 | 3.63e-03 | 1.99e-02 | 1.18e-02 |
| 13 | rational66_raw | analytical | 2.93e-02 | 1.19e-04 | 5.86e-02 | 2.99e-02 |
| 14 | histgbm_raw | ml | 8.61e-02 | 1.24e-03 | 1.71e-01 | 1.04e-01 |
| 15 | gpr_rbf_raw | ml | 3.30e-01 | 5.22e-04 | 6.60e-01 | 3.96e-01 |
| 16 | cheby20_chebymap | analytical | 4.19e-01 | 1.72e-03 | 8.37e-01 | 5.02e-01 |
| 17 | pysr_raw | symbolic | 5.11e-01 | 5.26e-03 | 1.02e+00 | 6.11e-01 |
| 18 | cheby14_raw | analytical | 6.34e-01 | 3.08e-03 | 1.27e+00 | 7.57e-01 |
| 19 | poly10_raw | analytical | 9.79e-01 | 5.62e-03 | 1.95e+00 | 1.17e+00 |
| 20 | mlp_raw | ml | 2.33e+00 | 8.99e-03 | 4.65e+00 | 2.77e+00 |
| 21 | gplearn_sqrtirr | symbolic | 2.06e+01 | 3.64e-02 | 4.12e+01 | 2.45e+01 |
| 22 | gplearn_raw | symbolic | 2.07e+01 | 7.23e-02 | 4.12e+01 | 2.46e+01 |
| 23 | rbf_multiquadric_raw | interpolation | 3.07e+01 | 1.04e+00 | 6.05e+01 | 2.69e+01 |
