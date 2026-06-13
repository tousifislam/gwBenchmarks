# Remnant Bench - opus48 CHANGELOG
Target: remnant kick magnitude v_k. Loss = NRMSE(v_k). Kicks are notoriously hard: superkick configurations (q~1, anti-aligned in-plane spins) produce sharp peaks. The in-plane spin difference and PN product features (eta, chi_eff, delta*chi_a) are physically motivated.

## Approaches

### 1. gpr_rbf_raw [kernel_gp] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1163, train 0.0006. train/val gap 193.0x -> overfitting in the sparse kick landscape.
- **Hypothesis/Change**: Baseline GPR (RBF) on raw params.
- **Result**: raw_7d reparam, target vf_mag, 0.007 ms/sample.

### 2. gpr_matern_eff [kernel_gp] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1154, train 0.0001. train/val gap 1330.5x -> overfitting in the sparse kick landscape.
- **Hypothesis/Change**: GPR Matern 5/2 on effective spins.
- **Result**: eff_spin reparam, target vf_mag, 0.010 ms/sample.

### 3. krr_rbf_eff [kernel_gp] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.0967, train 0.0800. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: Kernel ridge (RBF).
- **Result**: eff_spin reparam, target vf_mag, 0.004 ms/sample.

### 4. svr_rbf_pn [kernel_gp] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1305, train 0.0813. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: Support-vector regression on PN-product features.
- **Result**: pn_products reparam, target vf_mag, 0.011 ms/sample.

### 5. gpr_rbf_pn_logt [kernel_gp] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.0959, train 0.0988. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: GPR on log-target (kick distribution is heavy-tailed).
- **Result**: pn_products reparam, target vf_mag, 0.007 ms/sample.

### 6. pysr_vf_raw [symbolic] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1035, train 0.1146. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: PySR on vf, raw params.
- **Result**: raw_7d reparam, target vf_mag, 0.003 ms/sample.

### 7. gplearn_vf_raw [symbolic] (target vf_mag)
- **Observed**: NRMSE(v_k) val 1.7416, train 1.7393. high train loss -> underfits the sharp superkick features.
- **Hypothesis/Change**: gplearn SymbolicRegressor on vf, raw params.
- **Result**: raw_7d reparam, target vf_mag, 0.000 ms/sample.

### 8. pysr_vf_pn [symbolic] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1043, train 0.1109. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: PySR on vf with PN-product reparam (second reparam).
- **Result**: pn_products reparam, target vf_mag, 0.002 ms/sample.

### 9. gplearn_vf_eff [symbolic] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.3284, train 0.3331. high train loss -> underfits the sharp superkick features.
- **Hypothesis/Change**: gplearn on vf, effective spins.
- **Result**: eff_spin reparam, target vf_mag, 0.000 ms/sample.

### 10. rbf_thinplate_eff [interpolation] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1163, train 0.0067. train/val gap 17.3x -> overfitting in the sparse kick landscape.
- **Hypothesis/Change**: Thin-plate-spline RBF interpolation.
- **Result**: eff_spin reparam, target vf_mag, 0.008 ms/sample.

### 11. rbf_multiquadric_pn [interpolation] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1317, train 0.0260. train/val gap 5.1x -> overfitting in the sparse kick landscape.
- **Hypothesis/Change**: Multiquadric RBF interpolation.
- **Result**: pn_products reparam, target vf_mag, 0.005 ms/sample.

### 12. knn_raw [interpolation] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.0947, train 0.0000. train/val gap 94687542.7x -> overfitting in the sparse kick landscape.
- **Hypothesis/Change**: Distance-weighted k-nearest-neighbour.
- **Result**: raw_7d reparam, target vf_mag, 0.006 ms/sample.

### 13. rbf_linear_spherical [interpolation] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.0991, train 0.0015. train/val gap 67.7x -> overfitting in the sparse kick landscape.
- **Hypothesis/Change**: Linear-kernel RBF on spherical spins.
- **Result**: spherical reparam, target vf_mag, 0.005 ms/sample.

### 14. mlp_eff [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1011, train 0.0717. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: MLP (128x128, scaled target).
- **Result**: eff_spin reparam, target vf_mag, 0.000 ms/sample.

### 15. mlp_deep_pn [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1020, train 0.0687. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: Deeper MLP on PN products.
- **Result**: pn_products reparam, target vf_mag, 0.001 ms/sample.

### 16. rf_raw [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.0929, train 0.0372. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: Random forest, raw params.
- **Result**: raw_7d reparam, target vf_mag, 0.049 ms/sample.

### 17. extratrees_eff [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.0948, train 0.0000. train/val gap 6717183.4x -> overfitting in the sparse kick landscape.
- **Hypothesis/Change**: Extremely randomised trees.
- **Result**: eff_spin reparam, target vf_mag, 0.048 ms/sample.

### 18. xgboost_pn [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.0950, train 0.0452. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: XGBoost gradient boosting.
- **Result**: pn_products reparam, target vf_mag, 0.001 ms/sample.

### 19. lightgbm_eff [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1021, train 0.0337. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: LightGBM gradient boosting.
- **Result**: eff_spin reparam, target vf_mag, 0.013 ms/sample.

### 20. poly3_eff [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.0906, train 0.0915. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: Cubic polynomial ridge regression.
- **Result**: eff_spin reparam, target vf_mag, 0.000 ms/sample.

### 21. poly5_pn [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1067, train 0.0770. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: Quintic polynomial ridge.
- **Result**: pn_products reparam, target vf_mag, 0.001 ms/sample.

### 22. xgboost_logt_massdiff [ml] (target vf_mag)
- **Observed**: NRMSE(v_k) val 0.1014, train 0.0451. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: XGBoost on log-target, mass-difference+antisymmetric reparam.
- **Result**: massdiff_antisym reparam, target vf_mag, 0.001 ms/sample.

### 23. pysr_Mf_eff [symbolic] (target Mf)
- **Observed**: NRMSE(v_k) val 0.0173, train 0.0178. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: PySR on remnant mass Mf (auxiliary).
- **Result**: eff_spin reparam, target Mf, 0.003 ms/sample.

### 24. pysr_chif_eff [symbolic] (target chif_mag)
- **Observed**: NRMSE(v_k) val 0.0720, train 0.0749. generalises well; error set by intrinsic kick scatter.
- **Hypothesis/Change**: PySR on remnant spin chif (auxiliary).
- **Result**: eff_spin reparam, target chif_mag, 0.002 ms/sample.

## Ranking (v_k approaches, by NRMSE)

| rank | approach | category | NRMSE | train | ms |
|---|---|---|---|---|---|
| 1 | poly3_eff | ml | 0.0906 | 0.0915 | 0.000 |
| 2 | rf_raw | ml | 0.0929 | 0.0372 | 0.049 |
| 3 | knn_raw | interpolation | 0.0947 | 0.0000 | 0.006 |
| 4 | extratrees_eff | ml | 0.0948 | 0.0000 | 0.048 |
| 5 | xgboost_pn | ml | 0.0950 | 0.0452 | 0.001 |
| 6 | gpr_rbf_pn_logt | kernel_gp | 0.0959 | 0.0988 | 0.007 |
| 7 | krr_rbf_eff | kernel_gp | 0.0967 | 0.0800 | 0.004 |
| 8 | rbf_linear_spherical | interpolation | 0.0991 | 0.0015 | 0.005 |
| 9 | mlp_eff | ml | 0.1011 | 0.0717 | 0.000 |
| 10 | xgboost_logt_massdiff | ml | 0.1014 | 0.0451 | 0.001 |
| 11 | mlp_deep_pn | ml | 0.1020 | 0.0687 | 0.001 |
| 12 | lightgbm_eff | ml | 0.1021 | 0.0337 | 0.013 |
| 13 | pysr_vf_raw | symbolic | 0.1035 | 0.1146 | 0.003 |
| 14 | pysr_vf_pn | symbolic | 0.1043 | 0.1109 | 0.002 |
| 15 | poly5_pn | ml | 0.1067 | 0.0770 | 0.001 |
| 16 | gpr_matern_eff | kernel_gp | 0.1154 | 0.0001 | 0.010 |
| 17 | gpr_rbf_raw | kernel_gp | 0.1163 | 0.0006 | 0.007 |
| 18 | rbf_thinplate_eff | interpolation | 0.1163 | 0.0067 | 0.008 |
| 19 | svr_rbf_pn | kernel_gp | 0.1305 | 0.0813 | 0.011 |
| 20 | rbf_multiquadric_pn | interpolation | 0.1317 | 0.0260 | 0.005 |
| 21 | gplearn_vf_eff | symbolic | 0.3284 | 0.3331 | 0.000 |
| 22 | gplearn_vf_raw | symbolic | 1.7416 | 1.7393 | 0.000 |
