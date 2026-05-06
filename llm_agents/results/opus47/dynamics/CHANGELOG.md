# Dynamics Benchmark — CHANGELOG

Each entry: observation -> hypothesis -> action -> outcome.

## 24 — svd_bayes_ridge_log_omega
- Category: **ml**, parameterization: **log_omega**, time: tau-normalized
- Loss (RMS rel err x): **9.8441e-03**, runtime: 0.065 ms
- Notes: SVD + Bayesian ridge poly-2 on log_omega.

## 18 — svd_lasso_poly3_log_omega
- Category: **svd_decomp**, parameterization: **log_omega**, time: tau-normalized
- Loss (RMS rel err x): **9.8510e-03**, runtime: 0.253 ms
- Notes: SVD + Lasso poly-3 on log_omega reparam.

## 19 — svd_extra_trees_trig_zeta
- Category: **ml**, parameterization: **trig_zeta**, time: tau-normalized
- Loss (RMS rel err x): **1.2456e-02**, runtime: 0.098 ms
- Notes: SVD + ExtraTrees on trig_zeta parameterization.

## 22 — direct_rf_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **1.2629e-02**, runtime: 0.106 ms
- Notes: Random Forest predicting log(x) on full 256-point grid (no SVD).

## 08 — svd_gbm_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **1.3023e-02**, runtime: 0.149 ms
- Notes: SVD + GBM (per output).

## 12 — logsvd_linear_raw6
- Category: **svd_decomp**, parameterization: **raw6**, time: tau-normalized
- Loss (RMS rel err x): **1.3125e-02**, runtime: 0.001 ms
- Notes: SVD on log(x) + linear; x grows monotonically so log scale stabilizes fits.
- **Reasoning**: x(t) grows monotonically from chirp evolution. log(x) is more linear in time → SVD coefficients become smoother → easier to fit.

## 01 — svd_linear_raw6
- Category: **svd_decomp**, parameterization: **raw6**, time: tau-normalized
- Loss (RMS rel err x): **1.3583e-02**, runtime: 0.000 ms
- Notes: SVD + linear regression (baseline).

## 14 — svd_pysr_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **1.3922e-02**, runtime: 0.020 ms
- Notes: PySR on top 5 log-SVD coeffs; linear regression for rest.
- **Reasoning**: PySR symbolic regression on log-SVD coefficients. Equations saved in expressions.json.

## 11 — svd_rbfinterp_raw6
- Category: **interpolation**, parameterization: **raw6**, time: tau-normalized
- Loss (RMS rel err x): **1.4203e-02**, runtime: 0.183 ms
- Notes: SVD + thin-plate-spline RBF interpolation.

## 06 — svd_rf_raw6
- Category: **ml**, parameterization: **raw6**, time: tau-normalized
- Loss (RMS rel err x): **1.4650e-02**, runtime: 0.189 ms
- Notes: SVD + Random Forest.

## 23 — svd_huber_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **2.0515e-02**, runtime: 0.052 ms
- Notes: SVD + Huber regression (robust).

## 21 — svd_pysr_raw6
- Category: **symbolic**, parameterization: **raw6**, time: tau-normalized
- Loss (RMS rel err x): **2.1166e-02**, runtime: 0.011 ms
- Notes: PySR (raw6) on top 3 log-SVD coeffs.

## 15 — svd_gplearn_eta_chieff
- Category: **symbolic**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **2.2778e-02**, runtime: 0.001 ms
- Notes: gplearn on top 5 log-SVD coeffs.

## 13 — logsvd_poly2_eta_chieff
- Category: **svd_decomp**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **2.4082e-02**, runtime: 0.001 ms
- Notes: SVD on log(x) + polynomial-2 ridge.

## 02 — svd_poly2_eta_chieff
- Category: **svd_decomp**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **2.4378e-02**, runtime: 0.001 ms
- Notes: SVD + polynomial-2 ridge regression on eta+chi_eff+log(e0).

## 03 — svd_poly3_trig_zeta
- Category: **svd_decomp**, parameterization: **trig_zeta**, time: tau-normalized
- Loss (RMS rel err x): **2.4427e-02**, runtime: 0.144 ms
- Notes: SVD + polynomial-3 ridge with cos/sin(zeta0) reparameterization.

## 09 — svd_krr_rbf_eta_chieff
- Category: **interpolation**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **3.1777e-02**, runtime: 0.306 ms
- Notes: SVD + Kernel Ridge RBF.

## 20 — logsvd_gpr_rbf_eta_chieff
- Category: **svd_decomp**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **3.2567e-02**, runtime: 0.605 ms
- Notes: log-SVD + GPR-RBF (one per coefficient).

## 10 — svd_knn_raw6
- Category: **interpolation**, parameterization: **raw6**, time: tau-normalized
- Loss (RMS rel err x): **3.2996e-02**, runtime: 0.002 ms
- Notes: SVD + KNN k=5 distance-weighted.

## 05 — svd_gpr_matern_eta_chieff
- Category: **svd_decomp**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **3.8818e-02**, runtime: 0.686 ms
- Notes: SVD + GPR-Matern-5/2 with eta_chieff reparam.

## 17 — direct_mlp_dense_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **3.9616e-02**, runtime: 0.006 ms
- Notes: Direct MLP predicting log(x) on the full 256-point grid (no SVD).
- **Reasoning**: Skip SVD entirely; let MLP learn the basis. Useful when SVD truncation is the bottleneck.

## 04 — svd_gpr_rbf_raw6
- Category: **svd_decomp**, parameterization: **raw6**, time: tau-normalized
- Loss (RMS rel err x): **3.9713e-02**, runtime: 0.436 ms
- Notes: SVD + GPR-RBF (one per coefficient).

## 07 — svd_mlp_eta_chieff
- Category: **ml**, parameterization: **eta_chieff**, time: tau-normalized
- Loss (RMS rel err x): **1.0981e-01**, runtime: 0.002 ms
- Notes: SVD + MLP 128-128 tanh.

## 16 — eim_linear_raw6
- Category: **svd_decomp**, parameterization: **raw6**, time: tau-normalized
- Loss (RMS rel err x): **7.6674e-01**, runtime: 0.000 ms
- Notes: EIM at 20 nodes with linear regression.

## Summary

- Best: **svd_bayes_ridge_log_omega** (loss=9.8441e-03)
- Total approaches: 24
- Categories: ['interpolation', 'ml', 'svd_decomp', 'symbolic']
- Parameterizations: ['eta_chieff', 'log_omega', 'raw6', 'trig_zeta']