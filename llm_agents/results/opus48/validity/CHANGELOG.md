# Validity Bench - opus48 CHANGELOG
Target: log(mismatch) between SXS NR and NRHybSur3dq8 (aligned spin). Loss = natural-log RMSE. NRHybSur3dq8 is valid for q<=8, |chi|<=0.8; beyond that the mismatch saturates toward 1. Boundary-distance features (max(0,q-8), max(0,|chi|-0.8)) make the validity edge explicit.

## Approaches (23)

### 1. gpr_rbf_raw [kernel_gp]
- **Observed**: log RMSE val 1.8276, train 1.9049. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: Baseline GPR (RBF) on raw params.
- **Result**: raw_4d reparam, 0.002 ms/sample.

### 2. gpr_matern_eff [kernel_gp]
- **Observed**: log RMSE val 1.8151, train 1.8671. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: GPR Matern on effective spins.
- **Result**: eff_spin reparam, 0.003 ms/sample.

### 3. krr_rbf_logq [kernel_gp]
- **Observed**: log RMSE val 1.8832, train 1.6355. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: Kernel ridge on log(q)+log(omega0).
- **Result**: log_q reparam, 0.002 ms/sample.

### 4. svr_rbf_interactions [kernel_gp]
- **Observed**: log RMSE val 1.9162, train 1.8423. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: SVR with interaction features.
- **Result**: interactions reparam, 0.010 ms/sample.

### 5. gpr_rbf_boundary [kernel_gp]
- **Observed**: log RMSE val 1.8101, train 1.7459. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: GPR with boundary-distance features (extrapolation awareness).
- **Result**: boundary reparam, 0.002 ms/sample.

### 6. pysr_raw [symbolic]
- **Observed**: log RMSE val 1.9890, train 2.0998. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: PySR on log(mm), raw params.
- **Result**: raw_4d reparam, 0.011 ms/sample.

### 7. gplearn_raw [symbolic]
- **Observed**: log RMSE val 2.0295, train 1.9460. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: gplearn on log(mm), raw params.
- **Result**: raw_4d reparam, 0.001 ms/sample.

### 8. pysr_boundary [symbolic]
- **Observed**: log RMSE val 2.0486, train 2.1864. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: PySR with boundary features (second reparam).
- **Result**: boundary reparam, 0.005 ms/sample.

### 9. gplearn_eff [symbolic]
- **Observed**: log RMSE val 1.9149, train 2.0452. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: gplearn on effective spins.
- **Result**: eff_spin reparam, 0.000 ms/sample.

### 10. rbf_thinplate_eff [interpolation]
- **Observed**: log RMSE val 2.0230, train 0.2265. val-train log-RMSE gap 1.80: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: Thin-plate RBF interpolation.
- **Result**: eff_spin reparam, 0.003 ms/sample.

### 11. rbf_multiquadric_interactions [interpolation]
- **Observed**: log RMSE val 2.1235, train 0.7730. val-train log-RMSE gap 1.35: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: Multiquadric RBF.
- **Result**: interactions reparam, 0.002 ms/sample.

### 12. knn_raw [interpolation]
- **Observed**: log RMSE val 1.8283, train 0.0000. val-train log-RMSE gap 1.83: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: Distance-weighted kNN.
- **Result**: raw_4d reparam, 0.003 ms/sample.

### 13. rbf_linear_boundary [interpolation]
- **Observed**: log RMSE val 1.7363, train 0.3900. val-train log-RMSE gap 1.35: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: Linear RBF on boundary features.
- **Result**: boundary reparam, 0.002 ms/sample.

### 14. mlp_eff [ml]
- **Observed**: log RMSE val 2.0377, train 0.7993. val-train log-RMSE gap 1.24: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: MLP (scaled target).
- **Result**: eff_spin reparam, 0.001 ms/sample.

### 15. deep_ensemble_boundary [ml]
- **Observed**: log RMSE val 1.7898, train 1.0011. val-train log-RMSE gap 0.79: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: Deep ensemble of MLPs (boundary features).
- **Result**: boundary reparam, 0.002 ms/sample.

### 16. rf_raw [ml]
- **Observed**: log RMSE val 1.6669, train 0.6633. val-train log-RMSE gap 1.00: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: Random forest.
- **Result**: raw_4d reparam, 0.091 ms/sample.

### 17. extratrees_eff [ml]
- **Observed**: log RMSE val 1.7097, train 0.0000. val-train log-RMSE gap 1.71: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: Extra trees.
- **Result**: eff_spin reparam, 0.118 ms/sample.

### 18. xgboost_boundary [ml]
- **Observed**: log RMSE val 1.6301, train 0.5803. val-train log-RMSE gap 1.05: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: XGBoost with boundary features.
- **Result**: boundary reparam, 0.004 ms/sample.

### 19. lightgbm_eff [ml]
- **Observed**: log RMSE val 1.7951, train 0.6974. val-train log-RMSE gap 1.10: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: LightGBM.
- **Result**: eff_spin reparam, 0.009 ms/sample.

### 20. poly3_eff [ml]
- **Observed**: log RMSE val 1.8713, train 1.8656. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: Cubic polynomial ridge.
- **Result**: eff_spin reparam, 0.000 ms/sample.

### 21. poly2_interactions [ml]
- **Observed**: log RMSE val 1.8769, train 1.9074. high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region.
- **Hypothesis/Change**: Quadratic polynomial on interaction features.
- **Result**: interactions reparam, 0.000 ms/sample.

### 22. xgboost_logq [ml]
- **Observed**: log RMSE val 1.7161, train 0.6122. val-train log-RMSE gap 1.10: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: XGBoost on log(q) reparam.
- **Result**: log_q reparam, 0.002 ms/sample.

### 23. mlp_deep_boundary [ml]
- **Observed**: log RMSE val 1.7209, train 0.9885. val-train log-RMSE gap 0.73: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled.
- **Hypothesis/Change**: Deeper MLP on boundary features (reasoned: capture sharp validity edge).
- **Result**: boundary reparam, 0.002 ms/sample.

## Ranking (by log RMSE)

| rank | approach | category | log RMSE | train | ms |
|---|---|---|---|---|---|
| 1 | xgboost_boundary | ml | 1.6301 | 0.5803 | 0.004 |
| 2 | rf_raw | ml | 1.6669 | 0.6633 | 0.091 |
| 3 | extratrees_eff | ml | 1.7097 | 0.0000 | 0.118 |
| 4 | xgboost_logq | ml | 1.7161 | 0.6122 | 0.002 |
| 5 | mlp_deep_boundary | ml | 1.7209 | 0.9885 | 0.002 |
| 6 | rbf_linear_boundary | interpolation | 1.7363 | 0.3900 | 0.002 |
| 7 | deep_ensemble_boundary | ml | 1.7898 | 1.0011 | 0.002 |
| 8 | lightgbm_eff | ml | 1.7951 | 0.6974 | 0.009 |
| 9 | gpr_rbf_boundary | kernel_gp | 1.8101 | 1.7459 | 0.002 |
| 10 | gpr_matern_eff | kernel_gp | 1.8151 | 1.8671 | 0.003 |
| 11 | gpr_rbf_raw | kernel_gp | 1.8276 | 1.9049 | 0.002 |
| 12 | knn_raw | interpolation | 1.8283 | 0.0000 | 0.003 |
| 13 | poly3_eff | ml | 1.8713 | 1.8656 | 0.000 |
| 14 | poly2_interactions | ml | 1.8769 | 1.9074 | 0.000 |
| 15 | krr_rbf_logq | kernel_gp | 1.8832 | 1.6355 | 0.002 |
| 16 | gplearn_eff | symbolic | 1.9149 | 2.0452 | 0.000 |
| 17 | svr_rbf_interactions | kernel_gp | 1.9162 | 1.8423 | 0.010 |
| 18 | pysr_raw | symbolic | 1.9890 | 2.0998 | 0.011 |
| 19 | rbf_thinplate_eff | interpolation | 2.0230 | 0.2265 | 0.003 |
| 20 | gplearn_raw | symbolic | 2.0295 | 1.9460 | 0.001 |
| 21 | mlp_eff | ml | 2.0377 | 0.7993 | 0.001 |
| 22 | pysr_boundary | symbolic | 2.0486 | 2.1864 | 0.005 |
| 23 | rbf_multiquadric_interactions | interpolation | 2.1235 | 0.7730 | 0.002 |
