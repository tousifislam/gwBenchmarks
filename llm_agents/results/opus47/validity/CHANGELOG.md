# Validity Benchmark — CHANGELOG

Each entry: observation -> hypothesis -> action -> outcome.

## 03 — rf_raw4
- Category: **ml**, parameterization: **raw4**
- Loss (RMSE log10 mm): **0.7220**, runtime: 0.081 ms
- Notes: Random Forest 300 trees, raw features.

## 12 — extra_trees_log_q
- Category: **ml**, parameterization: **log_q**
- Loss (RMSE log10 mm): **0.7385**, runtime: 0.095 ms
- Notes: ExtraTrees with log(q) reparam.

## 18 — stacked_gbm_gpr_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.7631**, runtime: 0.161 ms
- Notes: Stacked GBM + GPR-on-residuals.

## 24 — rf_interaction_tuned
- Category: **ml**, parameterization: **interaction**
- Loss (RMSE log10 mm): **0.7666**, runtime: 0.073 ms
- Notes: Tuned RF with interaction features (deeper, sqrt features).

## 04 — gbm_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.7782**, runtime: 0.003 ms
- Notes: GBM (sklearn fallback).

## 17 — gbm_boundary_features
- Category: **ml**, parameterization: **boundary**
- Loss (RMSE log10 mm): **0.7869**, runtime: 0.004 ms
- Notes: GBM with NRHybSur3dq8 boundary distance features (q-8, |chi|-0.8 clipped).
- **Reasoning**: Mismatch grows sharply at boundary q=8, |chi|=0.8 (NRHybSur3dq8 trained on q<=8, |chi|<=0.8). Boundary-distance features encode this physics.

## 10 — hgbr_interaction
- Category: **ml**, parameterization: **interaction**
- Loss (RMSE log10 mm): **0.7948**, runtime: 0.036 ms
- Notes: HistGradientBoosting with interaction features.

## 11 — mlp_deep_ensemble_boundary
- Category: **ml**, parameterization: **boundary**
- Loss (RMSE log10 mm): **0.8017**, runtime: 0.125 ms
- Notes: Deep ensemble of 5 MLPs with boundary-distance features.

## 23 — poly2_raw4
- Category: **symbolic**, parameterization: **raw4**
- Loss (RMSE log10 mm): **0.8139**, runtime: 0.000 ms
- Notes: Polynomial deg-2 ridge baseline (raw features).

## 07 — poly3_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.8219**, runtime: 0.058 ms
- Notes: Polynomial deg-3 ridge.

## 05 — mlp_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.8273**, runtime: 0.007 ms
- Notes: MLP 64-64.

## 16 — lasso_poly3_interaction
- Category: **symbolic**, parameterization: **interaction**
- Loss (RMSE log10 mm): **0.8476**, runtime: 0.029 ms
- Notes: Lasso poly-3 with interaction features.

## 20 — bayes_ridge_poly2_eta
- Category: **ml**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.8732**, runtime: 0.000 ms
- Notes: Bayesian Ridge poly-2.

## 21 — mlp_deep_interaction
- Category: **ml**, parameterization: **interaction**
- Loss (RMSE log10 mm): **0.8734**, runtime: 0.023 ms
- Notes: Deep MLP with interaction features.

## 08 — pysr_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.8788**, runtime: 0.003 ms
- Notes: PySR symbolic regression on log10(mm).

## 19 — pysr_log_q
- Category: **symbolic**, parameterization: **log_q**
- Loss (RMSE log10 mm): **0.8794**, runtime: 0.002 ms
- Notes: PySR (log_q reparameterization, second symbolic run).

## 15 — svr_rbf_eta_chieff
- Category: **kernel_gp**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.8948**, runtime: 0.009 ms
- Notes: SVR RBF.

## 02 — gpr_matern_eta_chieff
- Category: **kernel_gp**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.8981**, runtime: 0.046 ms
- Notes: GPR Matern-5/2 with eta+chi_eff reparam.

## 01 — gpr_rbf_raw4
- Category: **kernel_gp**, parameterization: **raw4**
- Loss (RMSE log10 mm): **0.8982**, runtime: 0.041 ms
- Notes: GPR RBF baseline.

## 22 — gpr_compound_boundary
- Category: **kernel_gp**, parameterization: **boundary**
- Loss (RMSE log10 mm): **0.9010**, runtime: 0.132 ms
- Notes: GPR with compound RBF*Matern kernel + boundary features.

## 09 — gplearn_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.9029**, runtime: 0.000 ms
- Notes: gplearn SymbolicRegressor.

## 06 — krr_rbf_raw4
- Category: **kernel_gp**, parameterization: **raw4**
- Loss (RMSE log10 mm): **0.9040**, runtime: 0.111 ms
- Notes: Kernel Ridge RBF.

## 14 — knn_eta_chieff
- Category: **interpolation**, parameterization: **eta_chieff**
- Loss (RMSE log10 mm): **0.9278**, runtime: 0.003 ms
- Notes: KNN k=5 distance-weighted.

## 13 — rbf_interp_raw4
- Category: **interpolation**, parameterization: **raw4**
- Loss (RMSE log10 mm): **0.9281**, runtime: 0.043 ms
- Notes: Thin-plate-spline RBF interpolation.

## Summary

- Best: **rf_raw4** (loss=0.7220)
- Total approaches: 24
- Categories: ['interpolation', 'kernel_gp', 'ml', 'symbolic']
- Parameterizations: ['boundary', 'eta_chieff', 'interaction', 'log_q', 'raw4']