# Dynamics Bench - opus48 CHANGELOG
Target: PN frequency parameter x(t) for eccentric spinning BBH. Loss = pointwise RMS relative error. Representation: duration-normalised tau in [0,1], SVD/EIM of log(x)(tau), param->coeff regression; eval grid given so endpoints need no modelling.
## Key findings
- x(t) = smooth inspiral growth + eccentric oscillations (~30-50 cycles).
- The secular trend dominates the relative error and regresses well (oracle rank-20 ~0.013); the eccentric oscillation modes have varying phase in tau and do not regress, so adding high-rank modes plateaus.
- cos/sin(zeta0) embedding (trig/full reparam) captures initial anomaly.

## Approaches (22)

### 1. svd_gpr_raw [decomposition]
- **Observed**: RMS-rel-err val 0.0249 (median 0.0204), train 0.0149. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Baseline SVD + GPR (RBF) on raw params.
- **Result**: rank 25, raw_6d reparam, 0.05 ms/evolution.

### 2. svd_gpr_matern_full [decomposition]
- **Observed**: RMS-rel-err val 0.0287 (median 0.0244), train 0.0126. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: SVD + GPR Matern on fully-transformed reparam.
- **Result**: rank 25, full_transform reparam, 0.05 ms/evolution.

### 3. svd_poly3_eff [decomposition]
- **Observed**: RMS-rel-err val 0.0271 (median 0.0224), train 0.0201. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: SVD + cubic polynomial ridge.
- **Result**: rank 20, eff_loge reparam, 0.05 ms/evolution.

### 4. svd_mlp_full [decomposition]
- **Observed**: RMS-rel-err val 0.0278 (median 0.0243), train 0.0152. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: SVD + MLP (scaled targets).
- **Result**: rank 25, full_transform reparam, 0.05 ms/evolution.

### 5. eim_gpr_full [decomposition]
- **Observed**: RMS-rel-err val 0.0312 (median 0.0269), train 0.0154. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: EIM nodes + GPR on node values.
- **Result**: rank 25, full_transform reparam, 0.05 ms/evolution.

### 6. svd_pysr_full [symbolic]
- **Observed**: RMS-rel-err val 0.0269 (median 0.0235), train 0.0264. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: PySR on leading SVD coeffs (full transform).
- **Result**: rank 12, full_transform reparam, 0.09 ms/evolution.

### 7. svd_gplearn_eff [symbolic]
- **Observed**: RMS-rel-err val 0.0394 (median 0.0355), train 0.0377. high train loss: underfits the eccentric oscillation modes.
- **Hypothesis/Change**: gplearn on leading SVD coeffs (eff+log e0).
- **Result**: rank 12, eff_loge reparam, 0.05 ms/evolution.

### 8. svd_pysr_trig [symbolic]
- **Observed**: RMS-rel-err val 0.0246 (median 0.0214), train 0.0237. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: PySR, second reparam (trig anomaly).
- **Result**: rank 12, trig_anom reparam, 0.08 ms/evolution.

### 9. rbf_thinplate_full [interp_kernel]
- **Observed**: RMS-rel-err val 0.0256 (median 0.0217), train 0.0126. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Thin-plate RBF interpolation.
- **Result**: rank 25, full_transform reparam, 0.05 ms/evolution.

### 10. rbf_multiquadric_eff [interp_kernel]
- **Observed**: RMS-rel-err val 0.0255 (median 0.0207), train 0.0126. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Multiquadric RBF.
- **Result**: rank 25, eff_loge reparam, 0.05 ms/evolution.

### 11. krr_full [interp_kernel]
- **Observed**: RMS-rel-err val 0.0295 (median 0.0251), train 0.0142. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Kernel ridge (RBF).
- **Result**: rank 25, full_transform reparam, 0.05 ms/evolution.

### 12. knn_raw [interp_kernel]
- **Observed**: RMS-rel-err val 0.0296 (median 0.0256), train 0.0126. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Distance-weighted kNN.
- **Result**: rank 25, raw_6d reparam, 0.05 ms/evolution.

### 13. mlp_deep_full [ml]
- **Observed**: RMS-rel-err val 0.0263 (median 0.0234), train 0.0137. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Deeper MLP.
- **Result**: rank 25, full_transform reparam, 0.05 ms/evolution.

### 14. rf_eff [ml]
- **Observed**: RMS-rel-err val 0.0324 (median 0.0280), train 0.0167. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Random forest.
- **Result**: rank 25, eff_loge reparam, 0.19 ms/evolution.

### 15. extratrees_full [ml]
- **Observed**: RMS-rel-err val 0.0305 (median 0.0266), train 0.0126. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Extremely randomised trees.
- **Result**: rank 25, full_transform reparam, 0.23 ms/evolution.

### 16. xgboost_eff [ml]
- **Observed**: RMS-rel-err val 0.0263 (median 0.0235), train 0.0129. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: XGBoost (per coeff).
- **Result**: rank 25, eff_loge reparam, 0.11 ms/evolution.

### 17. mlp_trig [ml]
- **Observed**: RMS-rel-err val 0.0267 (median 0.0220), train 0.0142. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: MLP on trig-anomaly reparam (third reparam).
- **Result**: rank 25, trig_anom reparam, 0.05 ms/evolution.

### 18. svd_gpr_lowrank_full [decomposition]
- **Observed**: RMS-rel-err val 0.0302 (median 0.0249), train 0.0172. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Lower rank: drop noisy eccentric-oscillation modes that do not regress.
- **Result**: rank 15, full_transform reparam, 0.05 ms/evolution.

### 19. svd_gpr_highrank_full [decomposition]
- **Observed**: RMS-rel-err val 0.0299 (median 0.0256), train 0.0067. train/val gap 4.5x: coeff regression extrapolates poorly.
- **Hypothesis/Change**: Higher rank: attempt to capture eccentric oscillations.
- **Result**: rank 60, full_transform reparam, 0.05 ms/evolution.

### 20. rbf_thinplate_logfreq [interp_kernel]
- **Observed**: RMS-rel-err val 0.0224 (median 0.0179), train 0.0130. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: RBF on log-frequency reparam.
- **Result**: rank 22, log_freq reparam, 0.05 ms/evolution.

### 21. svd_poly4_full [decomposition]
- **Observed**: RMS-rel-err val 0.0452 (median 0.0289), train 0.0154. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Quartic polynomial ridge.
- **Result**: rank 20, full_transform reparam, 0.05 ms/evolution.

### 22. gpr_rbf_tuned_full [decomposition]
- **Observed**: RMS-rel-err val 0.0306 (median 0.0242), train 0.0159. generalises; residual set by unmodelled eccentric oscillation phase.
- **Hypothesis/Change**: Reasoned GPR retune (broad kernel).
- **Result**: rank 22, full_transform reparam, 0.05 ms/evolution.

## Ranking (by RMS rel err)

| rank | approach | category | loss | median | train | ms |
|---|---|---|---|---|---|---|
| 1 | rbf_thinplate_logfreq | interp_kernel | 0.0224 | 0.0179 | 0.0130 | 0.05 |
| 2 | svd_pysr_trig | symbolic | 0.0246 | 0.0214 | 0.0237 | 0.08 |
| 3 | svd_gpr_raw | decomposition | 0.0249 | 0.0204 | 0.0149 | 0.05 |
| 4 | rbf_multiquadric_eff | interp_kernel | 0.0255 | 0.0207 | 0.0126 | 0.05 |
| 5 | rbf_thinplate_full | interp_kernel | 0.0256 | 0.0217 | 0.0126 | 0.05 |
| 6 | mlp_deep_full | ml | 0.0263 | 0.0234 | 0.0137 | 0.05 |
| 7 | xgboost_eff | ml | 0.0263 | 0.0235 | 0.0129 | 0.11 |
| 8 | mlp_trig | ml | 0.0267 | 0.0220 | 0.0142 | 0.05 |
| 9 | svd_pysr_full | symbolic | 0.0269 | 0.0235 | 0.0264 | 0.09 |
| 10 | svd_poly3_eff | decomposition | 0.0271 | 0.0224 | 0.0201 | 0.05 |
| 11 | svd_mlp_full | decomposition | 0.0278 | 0.0243 | 0.0152 | 0.05 |
| 12 | svd_gpr_matern_full | decomposition | 0.0287 | 0.0244 | 0.0126 | 0.05 |
| 13 | krr_full | interp_kernel | 0.0295 | 0.0251 | 0.0142 | 0.05 |
| 14 | knn_raw | interp_kernel | 0.0296 | 0.0256 | 0.0126 | 0.05 |
| 15 | svd_gpr_highrank_full | decomposition | 0.0299 | 0.0256 | 0.0067 | 0.05 |
| 16 | svd_gpr_lowrank_full | decomposition | 0.0302 | 0.0249 | 0.0172 | 0.05 |
| 17 | extratrees_full | ml | 0.0305 | 0.0266 | 0.0126 | 0.23 |
| 18 | gpr_rbf_tuned_full | decomposition | 0.0306 | 0.0242 | 0.0159 | 0.05 |
| 19 | eim_gpr_full | decomposition | 0.0312 | 0.0269 | 0.0154 | 0.05 |
| 20 | rf_eff | ml | 0.0324 | 0.0280 | 0.0167 | 0.19 |
| 21 | svd_gplearn_eff | symbolic | 0.0394 | 0.0355 | 0.0377 | 0.05 |
| 22 | svd_poly4_full | decomposition | 0.0452 | 0.0289 | 0.0154 | 0.05 |
