# Validity Bench — CHANGELOG

## Approach 01: gpr_rbf_raw
- Category: kernel
- Parameterization: raw
- Val Loss: 0.9026
- Notes: gpr_rbf_raw with raw parameterization

## Approach 02: gpr_matern_eta
- Category: kernel
- Parameterization: eta_chieff
- Val Loss: 0.9026
- Notes: gpr_matern_eta with eta_chieff parameterization

## Approach 03: gpr_matern_logq
- Category: kernel
- Parameterization: logq
- Val Loss: 0.7553
- Notes: gpr_matern_logq with logq parameterization

## Approach 04: krr_raw
- Category: kernel
- Parameterization: raw
- Val Loss: 0.8085
- Notes: krr_raw with raw parameterization

## Approach 05: svr_eta
- Category: kernel
- Parameterization: eta_chieff
- Val Loss: 0.8489
- Notes: svr_eta with eta_chieff parameterization

## Approach 06: krr_interact
- Category: kernel
- Parameterization: interact
- Val Loss: 0.8152
- Notes: krr_interact with interact parameterization

## Approach 07: poly3_raw
- Category: symbolic
- Parameterization: raw
- Val Loss: 0.8256
- Notes: poly3_raw with raw parameterization

## Approach 08: poly4_eta
- Category: symbolic
- Parameterization: eta_chieff
- Val Loss: 0.8576
- Notes: poly4_eta with eta_chieff parameterization

## Approach 09: poly5_logq
- Category: symbolic
- Parameterization: logq
- Val Loss: 1.0568
- Notes: poly5_logq with logq parameterization

## Approach 10: lasso_interact
- Category: symbolic
- Parameterization: interact
- Val Loss: 0.7918
- Notes: lasso_interact with interact parameterization

## Approach 11: elasticnet_boundary
- Category: symbolic
- Parameterization: boundary
- Val Loss: 0.8094
- Notes: elasticnet_boundary with boundary parameterization

## Approach 12: bayesian_ridge_eta
- Category: symbolic
- Parameterization: eta_chieff
- Val Loss: 0.8093
- Notes: bayesian_ridge_eta with eta_chieff parameterization

## Approach 13: rbf_tps_raw
- Category: interpolation
- Parameterization: raw
- Val Loss: 0.8555
- Notes: rbf_tps_raw with raw parameterization

## Approach 14: rbf_cubic_eta
- Category: interpolation
- Parameterization: eta_chieff
- Val Loss: 0.9287
- Notes: rbf_cubic_eta with eta_chieff parameterization

## Approach 15: rbf_linear_logq
- Category: interpolation
- Parameterization: logq
- Val Loss: 0.7674
- Notes: rbf_linear_logq with logq parameterization

## Approach 16: knn5_raw
- Category: interpolation
- Parameterization: raw
- Val Loss: 0.8012
- Notes: knn5_raw with raw parameterization

## Approach 17: knn10_eta
- Category: interpolation
- Parameterization: eta_chieff
- Val Loss: 0.8101
- Notes: knn10_eta with eta_chieff parameterization

## Approach 18: mlp_raw
- Category: ml
- Parameterization: raw
- Val Loss: 0.7996
- Notes: mlp_raw with raw parameterization

## Approach 19: mlp_eta
- Category: ml
- Parameterization: eta_chieff
- Val Loss: 0.7975
- Notes: mlp_eta with eta_chieff parameterization

## Approach 20: rf_raw
- Category: ml
- Parameterization: raw
- Val Loss: 0.7221
- Notes: rf_raw with raw parameterization

## Approach 21: gbr_eta
- Category: ml
- Parameterization: eta_chieff
- Val Loss: 0.7853
- Notes: gbr_eta with eta_chieff parameterization

## Approach 22: et_raw
- Category: ml
- Parameterization: raw
- Val Loss: 0.7317
- Notes: et_raw with raw parameterization

## Approach 23: ada_eta
- Category: ml
- Parameterization: eta_chieff
- Val Loss: 0.7969
- Notes: ada_eta with eta_chieff parameterization

## Approach 24: bagging_logq
- Category: ml
- Parameterization: logq
- Val Loss: 0.7319
- Notes: bagging_logq with logq parameterization

## Approach 25: pysr_eta
- Category: symbolic
- Parameterization: eta_chieff
- Val Loss: 0.8788
- Expression: `-4.7658687 + 0.016878607/x3`
- Notes: PySR symbolic regression on eta_chieff params

## Approach 26: gplearn_raw
- Category: symbolic
- Parameterization: raw
- Val Loss: 0.9022
- Expression: `sub(log(0.061), neg(-0.845))`
- Notes: gplearn symbolic regression on raw params
