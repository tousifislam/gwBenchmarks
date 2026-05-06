## Approach 1: poly10_raw
- **Parameterization**: raw
- **Loss**: 9.793524e-01
- **Notes**: Polynomial degree 10 on raw spin.

## Approach 2: poly15_log
- **Parameterization**: log_compact
- **Loss**: 5.829330e+00
- **Notes**: Polynomial degree 15 on log-compactified spin.

## Approach 3: spline_raw
- **Parameterization**: raw
- **Loss**: 1.129938e-05
- **Notes**: Cubic spline interpolation on raw spin.

## Approach 1: poly10_raw
- **Parameterization**: raw
- **Loss**: 9.793524e-01
- **Notes**: Polynomial degree 10 on raw spin.

## Approach 4: poly10_raw_logtarget
- **Parameterization**: raw
- **Loss**: 2.366153e-01
- **Notes**: Polynomial degree 10 on raw spin with log-targets.

## Approach 5: spline_raw_logtarget
- **Parameterization**: raw
- **Loss**: 3.453794e-05
- **Notes**: Cubic spline on raw spin with log-targets.

## Approach 6: poly15_cheb
- **Parameterization**: chebyshev
- **Loss**: 5.840023e-01
- **Notes**: Polynomial degree 15 on Chebyshev mapped spin.

## Approach 8: poly15_log_logt
- **Parameterization**: log_compact
- **Loss**: 1.879114e-01
- **Notes**: Polynomial degree 15 on log-compactified spin and log targets.

## Approach 9: gpr_rbf_raw
- **Parameterization**: raw
- **Loss**: 9.811886e-01
- **Notes**: GPR with RBF kernel on raw spin.

## Approach 10: gpr_matern25_logt
- **Parameterization**: raw
- **Loss**: 7.037484e+00
- **Notes**: GPR with Matern 2.5 on raw spin and log targets.

## Approach 11: rbf_raw
- **Parameterization**: raw
- **Loss**: 1.857741e-03
- **Notes**: RBF Interpolation on raw spin.

## Approach 12: knn_raw
- **Parameterization**: raw
- **Loss**: 4.018069e-03
- **Notes**: K-Nearest Neighbors on raw spin.

## Approach 13: mlp_raw_logt
- **Parameterization**: raw
- **Loss**: 5.370341e-01
- **Notes**: MLP on raw spin with log targets.

## Approach 14: rf_raw
- **Parameterization**: raw
- **Loss**: 4.122905e-03
- **Notes**: Random Forest on raw spin.

## Approach 16: gplearn_raw_logt
- **Parameterization**: raw
- **Loss**: 2.463972e-01
- **Notes**: gplearn on raw with log targets.

## Approach 17: gplearn_log_logt
- **Parameterization**: log_compact
- **Loss**: 6.908603e-02
- **Notes**: gplearn on log_compact with log targets.

## Approach 15: poly20_raw
- **Parameterization**: raw
- **Loss**: 4.559630e-01
- **Notes**: Polynomial degree 20 on raw spin.

## Approach 19: xgb_raw
- **Parameterization**: raw
- **Loss**: 7.832065e-02
- **Notes**: XGBoost on raw spin.

## Approach 20: lgbm_raw
- **Parameterization**: raw
- **Loss**: 7.880925e-02
- **Notes**: LightGBM on raw spin.

## Approach 21: cat_raw
- **Parameterization**: raw
- **Loss**: 1.254564e-02
- **Notes**: CatBoost on raw spin.

## Approach 22: ridge_poly10
- **Parameterization**: raw
- **Loss**: 3.597594e+00
- **Notes**: Ridge Regression (deg 10) on raw spin.

## Approach 23: lasso_poly10
- **Parameterization**: raw
- **Loss**: 2.074069e+01
- **Notes**: Lasso Regression (deg 10) on raw spin.

## Approach 24: svr_raw
- **Parameterization**: raw
- **Loss**: 1.236820e+01
- **Notes**: SVR with RBF kernel on raw spin.

## Approach 25: lr_log_logt
- **Parameterization**: log_compact
- **Loss**: 1.587160e-01
- **Notes**: Linear Regression on log-compactified spin and log targets.

