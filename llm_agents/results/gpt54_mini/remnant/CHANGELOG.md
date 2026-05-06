# Remnant Benchmark — CHANGELOG

Each entry: observation -> hypothesis -> action -> outcome.

## 15 — mlp_multitask_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **8.8678e-02**, runtime: 0.001 ms
- Notes: Multi-task MLP jointly predicts Mf, chif, vf; here we report vf NRMSE.

## 02 — gpr_matern_eta_chieff
- Category: **kernel_gp**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **8.9546e-02**, runtime: 0.039 ms
- Notes: GPR Matern-5/2 with eta+chi_eff reparameterization.

## 23 — bayes_ridge_poly2_eta
- Category: **ml**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **8.9757e-02**, runtime: 0.017 ms
- Notes: Bayesian Ridge regression with poly-2 features.

## 08 — poly3_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **9.0429e-02**, runtime: 0.028 ms
- Notes: Polynomial degree-3 regression.

## 16 — lasso_poly3_pn_products
- Category: **symbolic**, parameterization: **pn_products**
- Loss (NRMSE v_k): **9.0793e-02**, runtime: 0.013 ms
- Notes: Lasso poly-3 on PN-inspired products: eta, chi_eff, eta*chi_eff, delta*chi_a, chi_p.

## 21 — gpr_rbf_pn_products
- Category: **kernel_gp**, parameterization: **pn_products**
- Loss (NRMSE v_k): **9.1066e-02**, runtime: 0.049 ms
- Notes: GPR RBF on PN-inspired product features (5D).

## 07 — gbm_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **9.1337e-02**, runtime: 0.003 ms
- Notes: Sklearn GradientBoosting fallback (XGBoost not available).

## 20 — stacked_gbm_gpr_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **9.1549e-02**, runtime: 0.044 ms
- Notes: Stacked ensemble: GBM trained, then GPR fit residuals.
- **Reasoning**: Stacking GBM (captures bulk trend) + GPR-on-residuals (captures smooth correlations).

## 18 — krr_polynomial_delta_chia
- Category: **kernel_gp**, parameterization: **delta_chia**
- Loss (NRMSE v_k): **9.1999e-02**, runtime: 0.072 ms
- Notes: KernelRidge with polynomial degree-4 kernel on delta_m+chi_a parameterization.

## 06 — rf_raw7
- Category: **ml**, parameterization: **raw7**
- Loss (NRMSE v_k): **9.2829e-02**, runtime: 0.026 ms
- Notes: Random forest 300 trees, depth 15.

## 13 — extra_trees_spherical
- Category: **ml**, parameterization: **spherical**
- Loss (NRMSE v_k): **9.4744e-02**, runtime: 0.038 ms
- Notes: ExtraTrees on spherical spin parameterization.

## 12 — knn_raw7
- Category: **interpolation**, parameterization: **raw7**
- Loss (NRMSE v_k): **9.5181e-02**, runtime: 0.003 ms
- Notes: KNN k=5 with distance weighting.

## 19 — hgbr_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **9.5266e-02**, runtime: 0.019 ms
- Notes: HistGradientBoosting (400 trees) — sklearn fallback for LightGBM.

## 09 — pysr_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **9.7451e-02**, runtime: 0.001 ms
- Notes: PySR symbolic regression on eta+chi_eff parameterization.
- **Reasoning**: PySR finds analytic forms. Top equation: see saved_model/expressions.json.

## 17 — phen_lousto_zlochower
- Category: **symbolic**, parameterization: **raw7**
- Loss (NRMSE v_k): **9.8071e-02**, runtime: 0.000 ms
- Notes: Lousto-Zlochower-inspired phenomenological features: eta^2*delta, eta^2*chi_a, etc.
- **Reasoning**: Lousto-Zlochower hand-crafted features encode the leading-order kick contributions: eta^2*delta_m, eta^2*chi_a, eta^2*chi_p.

## 11 — rbf_interp_raw7
- Category: **interpolation**, parameterization: **raw7**
- Loss (NRMSE v_k): **9.9703e-02**, runtime: 0.029 ms
- Notes: Thin-plate-spline RBF interpolation.

## 22 — pysr_raw7
- Category: **symbolic**, parameterization: **raw7**
- Loss (NRMSE v_k): **1.2507e-01**, runtime: 0.001 ms
- Notes: PySR with raw 7D parameters as input (second reparameterization).

## 01 — gpr_rbf_raw7
- Category: **kernel_gp**, parameterization: **raw7**
- Loss (NRMSE v_k): **1.2525e-01**, runtime: 0.075 ms
- Notes: Gaussian Process Regression with RBF + White noise.
- **Reasoning**: GPR baseline. Kicks have a sharp 'superkick' regime around q~1, antialigned spins. RBF kernel may smooth over this.

## 03 — krr_rbf_raw7
- Category: **kernel_gp**, parameterization: **raw7**
- Loss (NRMSE v_k): **1.3816e-01**, runtime: 0.084 ms
- Notes: Kernel ridge regression with RBF kernel on raw 7D parameters.

## 04 — svr_rbf_eta_chieff
- Category: **kernel_gp**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **3.9254e-01**, runtime: 0.000 ms
- Notes: Support Vector Regression with RBF kernel.

## 24 — mlp_deep_pn_products
- Category: **ml**, parameterization: **pn_products**
- Loss (NRMSE v_k): **4.2358e-01**, runtime: 0.006 ms
- Notes: Deeper MLP 256-128-64-32 on PN-product features.

## 10 — gplearn_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **1.1438e+00**, runtime: 0.000 ms
- Notes: gplearn SymbolicRegressor on eta+chi_eff.

## 14 — mlp_deep_spherical
- Category: **ml**, parameterization: **spherical**
- Loss (NRMSE v_k): **1.3948e+00**, runtime: 0.002 ms
- Notes: Deeper MLP 128-128-64 on spherical reparam.

## 05 — mlp_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**
- Loss (NRMSE v_k): **1.7662e+00**, runtime: 0.001 ms
- Notes: MLP with 64-64 tanh layers.
- **Reasoning**: MLP can capture sharp features that kernel methods over-smooth. Standardized inputs help convergence.

## Summary

- Best: **mlp_multitask_eta_chieff** with NRMSE 8.8678e-02
- Total approaches: 24
- Categories: ['interpolation', 'kernel_gp', 'ml', 'symbolic']
- Parameterizations: ['delta_chia', 'eta_chieff', 'pn_products', 'raw7', 'spherical']