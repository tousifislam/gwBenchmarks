## Approach 1: svd_gpr_raw
- **Parameterization**: raw_7d
- **Loss**: 0.219693
- **Notes**: Baseline SVD+GPR with raw parameters.

## Approach 2: svd_gpr_eff
- **Parameterization**: effective_spins
- **Loss**: 0.188348
- **Notes**: SVD+GPR with effective spins and Matern kernel.

## Approach 3: svd_poly2_raw
- **Parameterization**: raw
- **Loss**: 0.235692
- **Notes**: SVD + Polynomial Regression (degree 2) on raw params.

## Approach 4: svd_mlp_raw
- **Parameterization**: raw
- **Loss**: 0.231519
- **Notes**: SVD + MLP on raw params.

## Approach 6: svd_kr_raw
- **Parameterization**: raw
- **Loss**: 0.227798
- **Notes**: SVD + Kernel Ridge on raw params.

## Approach 2: svd_gpr_eff
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD+GPR with effective spins and Matern kernel.

## Approach 3: svd_poly2_raw
- **Parameterization**: raw
- **Loss**: 0.235692
- **Notes**: SVD + Polynomial Regression (degree 2) on raw params.

## Approach 7: svd_xgb_raw
- **Parameterization**: raw
- **Loss**: 0.243403
- **Notes**: SVD + XGBoost on raw params.

## Approach 11: eim_gpr_raw
- **Parameterization**: raw
- **Loss**: 0.233376
- **Notes**: EIM + GPR on raw parameters.

## Approach 4: svd_mlp_raw
- **Parameterization**: raw
- **Loss**: 0.231519
- **Notes**: SVD + MLP on raw params.

## Approach 8: svd_gpr_sph
- **Parameterization**: spherical
- **Loss**: nan
- **Notes**: SVD + GPR with spherical spin parameters.

## Approach 5: svd_rf_raw
- **Parameterization**: raw
- **Loss**: 0.202467
- **Notes**: SVD + Random Forest on raw params.

## Approach 10: svd_gpr_eff_matern25
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD + GPR with effective spins and Matern 2.5 kernel.

## Approach 2: svd_gpr_eff
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD+GPR with effective spins and Matern kernel.

## Approach 3: svd_poly2_raw
- **Parameterization**: raw
- **Loss**: 0.235692
- **Notes**: SVD + Polynomial Regression (degree 2) on raw params.

## Approach 4: svd_mlp_raw
- **Parameterization**: raw
- **Loss**: 0.231519
- **Notes**: SVD + MLP on raw params.

## Approach 5: svd_rf_raw
- **Parameterization**: raw
- **Loss**: 0.202467
- **Notes**: SVD + Random Forest on raw params.

## Approach 6: svd_kr_raw
- **Parameterization**: raw
- **Loss**: 0.227798
- **Notes**: SVD + Kernel Ridge on raw params.

## Approach 7: svd_xgb_raw
- **Parameterization**: raw
- **Loss**: 0.243403
- **Notes**: SVD + XGBoost on raw params.

## Approach 8: svd_gpr_sph
- **Parameterization**: spherical
- **Loss**: nan
- **Notes**: SVD + GPR with spherical spin parameters.

## Approach 10: svd_gpr_eff_matern25
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD + GPR with effective spins and Matern 2.5 kernel.

## Approach 15: svd_gpr_raw_tstart
- **Parameterization**: raw
- **Loss**: nan
- **Notes**: SVD + GPR on raw params with t=0 at start.

## Approach 14: amp_phase_svd_gpr_raw
- **Parameterization**: raw
- **Loss**: 0.646043
- **Notes**: Separate SVD+GPR for log-amplitude and phase.

## Approach 15: svd_gpr_raw_tstart
- **Parameterization**: raw
- **Loss**: nan
- **Notes**: SVD + GPR on raw params with t=0 at start.

## Approach 17: svd_poly3_raw
- **Parameterization**: raw
- **Loss**: 0.269018
- **Notes**: SVD + Poly (deg 3) on raw params.

## Approach 16: svd_gpr_eff_tstart
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD + GPR on effective spins with t=0 at start.

## Approach 12: svd_lgbm_raw
- **Parameterization**: raw
- **Loss**: 0.227950
- **Notes**: SVD + LightGBM on raw params.

## Approach 16: svd_gpr_eff_tstart
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD + GPR on effective spins with t=0 at start.

## Approach 18: svd_poly2_eff
- **Parameterization**: effective_spins
- **Loss**: 0.420028
- **Notes**: SVD + Poly (deg 2) on effective spins.

## Approach 13: svd_cat_raw
- **Parameterization**: raw
- **Loss**: 0.224698
- **Notes**: SVD + CatBoost on raw params.

## Approach 11: eim_gpr_raw
- **Parameterization**: raw
- **Loss**: 0.233376
- **Notes**: EIM + GPR on raw parameters.

## Approach 19: svd_mlp_eff
- **Parameterization**: effective_spins
- **Loss**: 0.366999
- **Notes**: SVD + MLP on effective spins.

## Approach 22: svd_rbf_raw
- **Parameterization**: raw
- **Loss**: 0.223455
- **Notes**: SVD + RBF Interpolation on raw parameters.

## Approach 17: svd_poly3_raw
- **Parameterization**: raw
- **Loss**: 0.269018
- **Notes**: SVD + Poly (deg 3) on raw params.

## Approach 20: svd_rf_eff
- **Parameterization**: effective_spins
- **Loss**: 0.354631
- **Notes**: SVD + Random Forest on effective spins.

## Approach 18: svd_poly2_eff
- **Parameterization**: effective_spins
- **Loss**: 0.420028
- **Notes**: SVD + Poly (deg 2) on effective spins.

## Approach 2: svd_gpr_eff
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD+GPR with effective spins and Matern kernel.

## Approach 10: svd_gpr_eff_matern25
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD + GPR with effective spins and Matern 2.5 kernel.

## Approach 16: svd_gpr_eff_tstart
- **Parameterization**: effective_spins
- **Loss**: nan
- **Notes**: SVD + GPR on effective spins with t=0 at start.

## Approach 15: svd_gpr_raw_tstart
- **Parameterization**: raw
- **Loss**: nan
- **Notes**: SVD + GPR on raw params with t=0 at start.

## Approach 8: svd_gpr_sph
- **Parameterization**: spherical
- **Loss**: nan
- **Notes**: SVD + GPR with spherical spin parameters.

