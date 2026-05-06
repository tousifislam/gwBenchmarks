## Approach 1: gpr_rbf_raw
- **Parameterization**: raw
- **Loss**: 5.608293
- **Notes**: GPR with RBF kernel on raw params.

## Approach 2: gpr_matern15_eff
- **Parameterization**: effective_spins
- **Loss**: 3.683035
- **Notes**: GPR with Matern 1.5 on effective spins.

## Approach 3: rf_raw
- **Parameterization**: raw
- **Loss**: 0.729669
- **Notes**: Random Forest on raw params.

## Approach 4: xgb_raw
- **Parameterization**: raw
- **Loss**: 0.777293
- **Notes**: XGBoost on raw params.

## Approach 5: mlp_raw
- **Parameterization**: raw
- **Loss**: 0.871594
- **Notes**: MLP on raw params.

## Approach 6: krr_rbf_raw
- **Parameterization**: raw
- **Loss**: 0.870312
- **Notes**: Kernel Ridge Regression with RBF kernel.

## Approach 7: poly3_raw
- **Parameterization**: raw
- **Loss**: 0.826856
- **Notes**: Polynomial Regression (degree 3) on raw params.

## Approach 9: mlp_eff
- **Parameterization**: effective_spins
- **Loss**: 0.825028
- **Notes**: MLP on effective spins.

## Approach 10: gpr_matern25_interaction
- **Parameterization**: interaction
- **Loss**: 3.703114
- **Notes**: GPR with Matern 2.5 on interaction terms.

## Approach 11: rbf_raw
- **Parameterization**: raw
- **Loss**: 19.477683
- **Notes**: RBF Interpolation on raw params.

## Approach 12: knn_raw
- **Parameterization**: raw
- **Loss**: 0.943470
- **Notes**: K-Nearest Neighbors on raw params.

## Approach 13: ridge_boundary
- **Parameterization**: boundary
- **Loss**: 0.866488
- **Notes**: Ridge Regression on boundary distance features.

## Approach 14: rf_interaction
- **Parameterization**: interaction
- **Loss**: 0.767476
- **Notes**: Random Forest on interaction terms.

## Approach 15: xgb_log_q
- **Parameterization**: log_q
- **Loss**: 0.788453
- **Notes**: XGBoost on log mass ratio parameters.

## Approach 16: gplearn_raw
- **Parameterization**: raw
- **Loss**: 0.894662
- **Notes**: gplearn on raw.

## Approach 17: gplearn_eff
- **Parameterization**: effective_spins
- **Loss**: 0.908426
- **Notes**: gplearn on effective_spins.

## Approach 18: lr_raw
- **Parameterization**: raw
- **Loss**: 0.814900
- **Notes**: Linear Regression on raw params.

## Approach 19: ridge_raw
- **Parameterization**: raw
- **Loss**: 0.868758
- **Notes**: Ridge Regression on raw params.

## Approach 20: lasso_raw
- **Parameterization**: raw
- **Loss**: 0.892216
- **Notes**: Lasso Regression on raw params.

## Approach 21: cat_raw
- **Parameterization**: raw
- **Loss**: 0.724168
- **Notes**: CatBoost on raw params.

