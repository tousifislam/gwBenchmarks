# Waveform Benchmark — CHANGELOG

Each entry records: observation → hypothesis → action → outcome.

## 10 — svd_knn_raw7
- Category: **interpolation**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1253**, proxy L2: 0.6453, runtime: 0.002 ms
- Notes: KNN k=5 with distance weighting on raw7.

## 12 — svd_rbfinterp_raw7
- Category: **interpolation**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1466**, proxy L2: 0.6369, runtime: 0.547 ms
- Notes: Thin-plate-spline RBF interpolation on SVD coeffs.

## 07 — svd_rf_raw7
- Category: **ml**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1475**, proxy L2: 0.6115, runtime: 0.062 ms
- Notes: Random forest with 200 trees, depth 15.
- **Reasoning**: Trees handle non-linearity natively without needing explicit polynomial features. Major improvement.

## 09 — svd_mlp_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1546**, proxy L2: 0.6242, runtime: 0.002 ms
- Notes: MLP 128-128-64 tanh on eta+chi_eff reparam.
- **Reasoning**: MLP can fit smooth nonlinearities; eta+chi_eff reparam reduces effective dim.

## 08 — svd_gbm_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1570**, proxy L2: 0.6084, runtime: 0.226 ms
- Notes: Gradient boosting (per output) with 100 trees.

## 24 — svd_lasso_poly3_eta
- Category: **svd_decomp**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1662**, proxy L2: 0.6797, runtime: 0.250 ms
- Notes: Lasso poly-3 on eta_chieff (sparsity-inducing).

## 11 — svd_kernelridge_eta_chieff
- Category: **interpolation**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1783**, proxy L2: 0.6544, runtime: 0.429 ms
- Notes: Kernel ridge regression with RBF kernel.

## 13 — svd_extratrees_spherical
- Category: **ml**, parameterization: **spherical**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1824**, proxy L2: 0.6577, runtime: 0.120 ms
- Notes: ExtraTrees on spherical spin parameterization.

## 18 — svd_poly2_log_q
- Category: **svd_decomp**, parameterization: **log_q**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1854**, proxy L2: 0.7066, runtime: 0.172 ms
- Notes: Polynomial deg-2 ridge with log(q) reparam.

## 02 — svd_poly2_eta_chieff
- Category: **svd_decomp**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.1929**, proxy L2: 0.6847, runtime: 0.233 ms
- Notes: Polynomial degree-2 features on eta+chi_eff reparam.
- **Reasoning**: Observed: linear can only fit 7-D subspace of 30-D coeffs. Hypothesis: poly features expand effective rank. Outcome: improved.

## 20 — svd_gplearn_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2055**, proxy L2: 0.7182, runtime: 0.003 ms
- Notes: gplearn SymbolicRegressor on top 5 SVD coeffs; linear for rest.
- **Reasoning**: gplearn alternative symbolic regression — comparing two SR engines.

## 01 — svd_linear_raw7
- Category: **svd_decomp**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2086**, proxy L2: 0.7408, runtime: 0.000 ms
- Notes: Linear regression on SVD coefficients with raw 7D parameters.
- **Reasoning**: First baseline. Linear regression on raw 7D parameters - establishes the floor.

## 04 — svd_ridge_raw7
- Category: **svd_decomp**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2086**, proxy L2: 0.7408, runtime: 0.000 ms
- Notes: Ridge regression on raw7 — baseline regularized linear.

## 21 — svd_pysr_raw7
- Category: **symbolic**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2093**, proxy L2: 0.7316, runtime: 0.011 ms
- Notes: PySR (raw7 inputs) on top 3 SVD coeffs; second reparameterization for symbolic.

## 06 — svd_gpr_matern_eta_chieff
- Category: **svd_decomp**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2141**, proxy L2: 0.7501, runtime: 0.350 ms
- Notes: GPR with Matern-5/2 kernel on eta+chi_eff reparam.

## 19 — svd_pysr_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2198**, proxy L2: 0.7352, runtime: 0.019 ms
- Notes: PySR symbolic regression on top 5 SVD coeffs; linear regression for remainder.
- **Reasoning**: PySR gives interpretable expressions for top SVD coefficients; remaining coeffs use linear fallback for stability.

## 05 — svd_gpr_rbf_raw7
- Category: **svd_decomp**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2266**, proxy L2: 0.7662, runtime: 0.352 ms
- Notes: GPR with RBF kernel (one per SVD coefficient).

## 23 — svd_mlp_raw8_omega0
- Category: **ml**, parameterization: **raw8**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2292**, proxy L2: 0.7199, runtime: 0.005 ms
- Notes: MLP 256-256-128 with omega0 included as 8th input feature.

## 22 — svd_huber_raw7
- Category: **ml**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2307**, proxy L2: 0.7576, runtime: 0.051 ms
- Notes: Huber regression (robust to outliers) on raw7.

## 14 — eim_linear_raw7
- Category: **svd_decomp**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2514**, proxy L2: 0.7918, runtime: 0.000 ms
- Notes: EIM with linear regression at K=30 selected nodes.

## 03 — svd_poly3_spherical
- Category: **svd_decomp**, parameterization: **spherical**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.2623**, proxy L2: 0.7417, runtime: 0.253 ms
- Notes: Polynomial degree-3 ridge on spherical spin parameterization.

## 16 — ap_svd_rf_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.4581**, proxy L2: 0.8856, runtime: 0.125 ms
- Notes: Amp/phase SVD + random forest.

## 17 — ap_svd_mlp_spherical
- Category: **ml**, parameterization: **spherical**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.4654**, proxy L2: 0.9433, runtime: 0.007 ms
- Notes: Amp/phase SVD + MLP on spherical reparam.

## 15 — ap_svd_linear_raw7
- Category: **svd_decomp**, parameterization: **raw7**, time conv: **t0_at_peak**
- Loss (FD mismatch, subset): **0.5315**, proxy L2: 0.9732, runtime: 0.000 ms
- Notes: Amp/phase SVD + linear regression. Time convention: t0=peak.

## Summary

- Best: **svd_knn_raw7** with loss 0.1253
- Total approaches: 24
- Categories covered: ['interpolation', 'ml', 'svd_decomp', 'symbolic']
- Parameterizations: ['eta_chieff', 'log_q', 'raw7', 'raw8', 'spherical']