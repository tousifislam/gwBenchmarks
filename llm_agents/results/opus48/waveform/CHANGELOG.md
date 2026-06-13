# Waveform Bench - opus48 CHANGELOG
Co-precessing h22 surrogate. Representation: per-waveform duration-normalised amplitude/phase on a common tau in [0,1] grid (N_tau=2000), SVD/EIM bases for log|h| and the (sign-canonicalised) phase, parameter->coefficient regression. Evaluation grid is given, so t0/tend need not be modelled. Loss = mean aLIGO FD mismatch over M in {40,80,120,160,200} Msun. NR floor median ~1.4e-3.
## Key findings
- Amplitude regresses to ~6e-3 mismatch (easy); phase dominates error.
- The co-precessing h22 phase has an arbitrary global sign that flips between simulations; since the metric scores only Re(h)=A cos(phi), canonicalising the sign (phi->-phi) makes total accumulated phase regressable (corr with duration 0.08 -> 0.88).
- Newtonian cycle features (omega0^-5/3/eta) further improve phase.
- GPR needs a FIXED broad kernel (ML-optimised length scale overfits with only 250 samples); RBF thin-plate generalises best.

## Approaches (22 total), evaluated on n_val=120

### 1. svd_gpr_rbf_raw [decomposition]
- **Observed**: val loss 1.378e-01 (median 1.130e-01), train 4.510e-02. train and val losses are close: the approach generalises; residual error is dominated by phase-shape regression.
- **Hypothesis/Change**: Baseline SVD + GPR (RBF) on raw params + Newtonian features.
- **Result**: rank(A=25,P=35), raw_newt reparam, 1.2 ms/waveform.

### 2. svd_gpr_matern_eff [decomposition]
- **Observed**: val loss 1.264e-01 (median 1.022e-01), train 2.467e-02. train and val losses are close: the approach generalises; residual error is dominated by phase-shape regression.
- **Hypothesis/Change**: SVD + GPR (Matern 5/2) on effective-spin + Newtonian reparam.
- **Result**: rank(A=25,P=35), eff_newt reparam, 1.2 ms/waveform.

### 3. svd_poly3_raw [decomposition]
- **Observed**: val loss 2.290e-01 (median 2.206e-01), train 7.124e-02. train and val losses are close: the approach generalises; residual error is dominated by phase-shape regression.
- **Hypothesis/Change**: SVD + cubic polynomial ridge regression.
- **Result**: rank(A=25,P=30), raw_omega reparam, 1.2 ms/waveform.

### 4. svd_mlp_eff [decomposition]
- **Observed**: val loss 2.030e-01 (median 2.009e-01), train 9.102e-02. train and val losses are close: the approach generalises; residual error is dominated by phase-shape regression.
- **Hypothesis/Change**: SVD + MLP (scaled targets, 128x128) on effective spins.
- **Result**: rank(A=25,P=30), eff_spin_omega reparam, 1.2 ms/waveform.

### 5. eim_gpr_raw [decomposition]
- **Observed**: val loss 1.554e-01 (median 1.373e-01), train 2.783e-02. train and val losses are close: the approach generalises; residual error is dominated by phase-shape regression.
- **Hypothesis/Change**: Empirical Interpolation Method nodes + GPR on node values.
- **Result**: rank(A=25,P=30), raw_omega reparam, 1.2 ms/waveform.

### 6. svd_pysr_eff [symbolic]
- **Observed**: val loss 2.373e-01 (median 2.103e-01), train 1.994e-01. high train loss too: the regressor underfits the SVD coefficient map; more capacity or better features needed.
- **Hypothesis/Change**: PySR on leading SVD coeffs (eff spins), ridge tail.
- **Result**: rank(A=12,P=14), eff_spin_omega reparam, 1.4 ms/waveform.

### 7. svd_gplearn_raw [symbolic]
- **Observed**: val loss 4.512e-01 (median 4.728e-01), train 2.924e-01. high train loss too: the regressor underfits the SVD coefficient map; more capacity or better features needed.
- **Hypothesis/Change**: gplearn SymbolicRegressor on leading SVD coeffs (raw).
- **Result**: rank(A=12,P=14), raw_omega reparam, 1.2 ms/waveform.

### 8. svd_pysr_spherical [symbolic]
- **Observed**: val loss 5.530e-01 (median 6.453e-01), train 4.390e-01. high train loss too: the regressor underfits the SVD coefficient map; more capacity or better features needed.
- **Hypothesis/Change**: PySR, second reparam (spherical spins).
- **Result**: rank(A=12,P=14), spherical reparam, 1.3 ms/waveform.

### 9. rbf_thinplate_raw [interp_kernel]
- **Observed**: val loss 1.235e-01 (median 9.387e-02), train 8.307e-04. large train/val gap (149x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Thin-plate-spline RBF interpolation on SVD coeffs.
- **Result**: rank(A=25,P=35), raw_newt reparam, 1.2 ms/waveform.

### 10. rbf_multiquadric_eff [interp_kernel]
- **Observed**: val loss 1.650e-01 (median 1.555e-01), train 2.393e-03. large train/val gap (69x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Multiquadric RBF interpolation, eff-spin reparam.
- **Result**: rank(A=25,P=30), eff_spin_omega reparam, 1.2 ms/waveform.

### 11. krr_rbf_eff [interp_kernel]
- **Observed**: val loss 1.760e-01 (median 1.585e-01), train 1.421e-02. large train/val gap (12x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Kernel ridge (RBF) on SVD coeffs.
- **Result**: rank(A=25,P=35), eff_newt reparam, 1.2 ms/waveform.

### 12. knn_correction_raw [interp_kernel]
- **Observed**: val loss 3.897e-01 (median 3.950e-01), train 8.467e-04. large train/val gap (460x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Distance-weighted kNN on SVD coeffs.
- **Result**: rank(A=25,P=30), raw_omega reparam, 1.2 ms/waveform.

### 13. mlp_deep_eff [ml]
- **Observed**: val loss 1.934e-01 (median 1.718e-01), train 1.026e-01. train and val losses are close: the approach generalises; residual error is dominated by phase-shape regression.
- **Hypothesis/Change**: Deeper MLP (256-256-128).
- **Result**: rank(A=25,P=35), eff_newt reparam, 1.1 ms/waveform.

### 14. rf_spherical [ml]
- **Observed**: val loss 4.303e-01 (median 4.486e-01), train 1.834e-01. high train loss too: the regressor underfits the SVD coefficient map; more capacity or better features needed.
- **Hypothesis/Change**: Random forest on spherical-spin reparam.
- **Result**: rank(A=25,P=30), spherical reparam, 1.8 ms/waveform.

### 15. gbr_raw [ml]
- **Observed**: val loss 2.401e-01 (median 2.232e-01), train 3.607e-02. train and val losses are close: the approach generalises; residual error is dominated by phase-shape regression.
- **Hypothesis/Change**: Gradient boosting (per-coeff) on raw params.
- **Result**: rank(A=20,P=24), raw_omega reparam, 1.3 ms/waveform.

### 16. extratrees_eff [ml]
- **Observed**: val loss 1.882e-01 (median 1.694e-01), train 8.467e-04. large train/val gap (222x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Extremely randomised trees.
- **Result**: rank(A=25,P=30), eff_spin_omega reparam, 1.8 ms/waveform.

### 17. svd_gpr_highrank_raw [decomposition]
- **Observed**: val loss 1.444e-01 (median 1.136e-01), train 1.061e-02. large train/val gap (14x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Higher-rank GPR: more basis for precessing modulations.
- **Result**: rank(A=35,P=45), raw_newt reparam, 1.2 ms/waveform.

### 18. svd_poly5_eff [decomposition]
- **Observed**: val loss 3.137e-01 (median 2.815e-01), train 3.092e-02. large train/val gap (10x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Quintic polynomial ridge.
- **Result**: rank(A=25,P=30), eff_spin_omega reparam, 1.1 ms/waveform.

### 19. amp_phase_asymrank_eff [decomposition]
- **Observed**: val loss 1.264e-01 (median 1.011e-01), train 2.475e-02. train and val losses are close: the approach generalises; residual error is dominated by phase-shape regression.
- **Hypothesis/Change**: Asymmetric ranks: amplitude is smoother than phase, so fewer amp basis, more phase basis.
- **Result**: rank(A=15,P=45), eff_newt reparam, 1.1 ms/waveform.

### 20. mlp_massdiff [ml]
- **Observed**: val loss 4.708e-01 (median 4.659e-01), train 1.514e-01. high train loss too: the regressor underfits the SVD coefficient map; more capacity or better features needed.
- **Hypothesis/Change**: MLP (scaled targets) on mass-difference reparam (third+ reparam).
- **Result**: rank(A=25,P=30), massdiff reparam, 1.2 ms/waveform.

### 21. rbf_linear_spherical [interp_kernel]
- **Observed**: val loss 4.609e-01 (median 5.118e-01), train 3.477e-03. large train/val gap (133x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Linear-kernel RBF on spherical spins.
- **Result**: rank(A=25,P=30), spherical reparam, 1.2 ms/waveform.

### 22. gpr_rbf_eff_tuned [decomposition]
- **Observed**: val loss 4.602e-01 (median 4.275e-01), train 9.974e-04. large train/val gap (461x): the coefficient regressor interpolates training but extrapolates poorly in the sparse 7D parameter space.
- **Hypothesis/Change**: Reasoned GPR retune: longer length scale + lower noise.
- **Result**: rank(A=30,P=40), eff_newt reparam, 1.2 ms/waveform.

## Ranking (by val loss)

| rank | approach | category | loss | median | train | ms |
|---|---|---|---|---|---|---|
| 1 | rbf_thinplate_raw | interp_kernel | 1.235e-01 | 9.387e-02 | 8.307e-04 | 1.2 |
| 2 | amp_phase_asymrank_eff | decomposition | 1.264e-01 | 1.011e-01 | 2.475e-02 | 1.1 |
| 3 | svd_gpr_matern_eff | decomposition | 1.264e-01 | 1.022e-01 | 2.467e-02 | 1.2 |
| 4 | svd_gpr_rbf_raw | decomposition | 1.378e-01 | 1.130e-01 | 4.510e-02 | 1.2 |
| 5 | svd_gpr_highrank_raw | decomposition | 1.444e-01 | 1.136e-01 | 1.061e-02 | 1.2 |
| 6 | eim_gpr_raw | decomposition | 1.554e-01 | 1.373e-01 | 2.783e-02 | 1.2 |
| 7 | rbf_multiquadric_eff | interp_kernel | 1.650e-01 | 1.555e-01 | 2.393e-03 | 1.2 |
| 8 | krr_rbf_eff | interp_kernel | 1.760e-01 | 1.585e-01 | 1.421e-02 | 1.2 |
| 9 | extratrees_eff | ml | 1.882e-01 | 1.694e-01 | 8.467e-04 | 1.8 |
| 10 | mlp_deep_eff | ml | 1.934e-01 | 1.718e-01 | 1.026e-01 | 1.1 |
| 11 | svd_mlp_eff | decomposition | 2.030e-01 | 2.009e-01 | 9.102e-02 | 1.2 |
| 12 | svd_poly3_raw | decomposition | 2.290e-01 | 2.206e-01 | 7.124e-02 | 1.2 |
| 13 | svd_pysr_eff | symbolic | 2.373e-01 | 2.103e-01 | 1.994e-01 | 1.4 |
| 14 | gbr_raw | ml | 2.401e-01 | 2.232e-01 | 3.607e-02 | 1.3 |
| 15 | svd_poly5_eff | decomposition | 3.137e-01 | 2.815e-01 | 3.092e-02 | 1.1 |
| 16 | knn_correction_raw | interp_kernel | 3.897e-01 | 3.950e-01 | 8.467e-04 | 1.2 |
| 17 | rf_spherical | ml | 4.303e-01 | 4.486e-01 | 1.834e-01 | 1.8 |
| 18 | svd_gplearn_raw | symbolic | 4.512e-01 | 4.728e-01 | 2.924e-01 | 1.2 |
| 19 | gpr_rbf_eff_tuned | decomposition | 4.602e-01 | 4.275e-01 | 9.974e-04 | 1.2 |
| 20 | rbf_linear_spherical | interp_kernel | 4.609e-01 | 5.118e-01 | 3.477e-03 | 1.2 |
| 21 | mlp_massdiff | ml | 4.708e-01 | 4.659e-01 | 1.514e-01 | 1.2 |
| 22 | svd_pysr_spherical | symbolic | 5.530e-01 | 6.453e-01 | 4.390e-01 | 1.3 |
