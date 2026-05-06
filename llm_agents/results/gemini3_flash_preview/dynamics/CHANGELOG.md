## Approach 1: svd_gpr_raw
- **Parameterization**: raw
- **Loss**: 0.050042
- **Notes**: SVD + GPR with RBF kernel on raw params.

## Approach 2: svd_gpr_eff_log_e
- **Parameterization**: eff_log_e
- **Loss**: 0.030427
- **Notes**: SVD + GPR with Matern 1.5 on effective spin + log eccentricity.

## Approach 3: svd_poly2_raw
- **Parameterization**: raw
- **Loss**: 0.009826
- **Notes**: SVD + Polynomial Regression (degree 2) on raw params.

## Approach 4: svd_mlp_raw
- **Parameterization**: raw
- **Loss**: 0.109050
- **Notes**: SVD + MLP on raw params.

## Approach 5: svd_rf_raw
- **Parameterization**: raw
- **Loss**: 0.013269
- **Notes**: SVD + Random Forest on raw params.

## Approach 6: svd_xgb_raw
- **Parameterization**: raw
- **Loss**: 0.015055
- **Notes**: SVD + XGBoost on raw params.

## Approach 7: svd_lgbm_raw
- **Parameterization**: raw
- **Loss**: 0.012817
- **Notes**: SVD + LightGBM on raw params.

## Approach 8: svd_cat_raw
- **Parameterization**: raw
- **Loss**: 0.011565
- **Notes**: SVD + CatBoost on raw params.

## Approach 9: svd_poly3_raw
- **Parameterization**: raw
- **Loss**: 0.010966
- **Notes**: SVD + Polynomial Regression (degree 3) on raw params.

## Approach 10: svd_poly2_trig
- **Parameterization**: trig_anomaly
- **Loss**: 0.009895
- **Notes**: SVD + Polynomial Regression (degree 2) on trig anomaly params.

## Approach 11: rbf_raw
- **Parameterization**: raw
- **Loss**: 0.014100
- **Notes**: RBF Interpolation on raw params.

## Approach 12: knn_raw
- **Parameterization**: raw
- **Loss**: 0.033064
- **Notes**: K-Nearest Neighbors on raw params.

## Approach 13: mlp_eff_log_e
- **Parameterization**: eff_log_e
- **Loss**: 0.084848
- **Notes**: MLP on effective spin + log eccentricity.

## Approach 14: rf_eff_log_e
- **Parameterization**: eff_log_e
- **Loss**: 0.013043
- **Notes**: Random Forest on effective spin + log eccentricity.

## Approach 15: gpr_matern25_fully
- **Parameterization**: fully_transformed
- **Loss**: 0.014617
- **Notes**: GPR with Matern 2.5 on fully transformed parameters.

## Approach 16: gplearn_raw
- **Parameterization**: raw
- **Loss**: 0.038489
- **Notes**: gplearn on raw (first 5 coeffs).

## Approach 17: gplearn_trig
- **Parameterization**: trig_anomaly
- **Loss**: 0.039984
- **Notes**: gplearn on trig_anomaly (first 5 coeffs).

## Approach 19: lr_raw
- **Parameterization**: raw
- **Loss**: 0.013424
- **Notes**: Linear Regression on raw params.

## Approach 20: ridge_raw
- **Parameterization**: raw
- **Loss**: 0.025049
- **Notes**: Ridge Regression on raw params.

## Approach 21: lasso_raw
- **Parameterization**: raw
- **Loss**: 0.037441
- **Notes**: Lasso Regression on raw params.

