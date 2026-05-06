# CHANGELOG

## Approach 1_svd_gpr_raw
- **Hypothesis/Reasoning:** Testing SVD with raw_6d reparameterization and 5 PCA components.
- **Loss (RMS rel error x):** 0.01323
- **Runtime:** 0.00 ms

## Approach 2_svd_gpr_eta
- **Hypothesis/Reasoning:** Testing SVD with eta_chi_eff_log_e0 reparameterization and 5 PCA components.
- **Loss (RMS rel error x):** 0.01419
- **Runtime:** 0.00 ms

## Approach 3_svd_poly_trig
- **Hypothesis/Reasoning:** Testing SVD with trig_anomaly reparameterization and 5 PCA components.
- **Loss (RMS rel error x):** 0.00959
- **Runtime:** 0.00 ms

## Approach 4_svd_mlp_full
- **Hypothesis/Reasoning:** Testing SVD with fully_transformed reparameterization and 10 PCA components.
- **Loss (RMS rel error x):** 0.10389
- **Runtime:** 0.00 ms

## Approach 5_svd_rf_raw
- **Hypothesis/Reasoning:** Testing SVD with raw_6d reparameterization and 5 PCA components.
- **Loss (RMS rel error x):** 0.01361
- **Runtime:** 0.01 ms

## Approach 6_knn_raw
- **Hypothesis/Reasoning:** Testing Interpolation with raw_6d reparameterization and 10 PCA components.
- **Loss (RMS rel error x):** 0.01560
- **Runtime:** 0.00 ms

## Approach 7_knn_eta
- **Hypothesis/Reasoning:** Testing Interpolation with eta_chi_eff_log_e0 reparameterization and 10 PCA components.
- **Loss (RMS rel error x):** 0.02089
- **Runtime:** 0.00 ms

## Approach 8_knn_trig
- **Hypothesis/Reasoning:** Testing Interpolation with trig_anomaly reparameterization and 10 PCA components.
- **Loss (RMS rel error x):** 0.01586
- **Runtime:** 0.00 ms

## Approach 9_krr_logw
- **Hypothesis/Reasoning:** Testing Interpolation with log_omega reparameterization and 5 PCA components.
- **Loss (RMS rel error x):** 0.01217
- **Runtime:** 0.01 ms

## Approach 10_svr_full
- **Hypothesis/Reasoning:** Testing Interpolation with fully_transformed reparameterization and 5 PCA components.
- **Loss (RMS rel error x):** 0.03847
- **Runtime:** 0.00 ms

## Approach 11_mlp_raw
- **Hypothesis/Reasoning:** Testing ML with raw_6d reparameterization and 20 PCA components.
- **Loss (RMS rel error x):** 0.09375
- **Runtime:** 0.00 ms

## Approach 12_mlp_eta
- **Hypothesis/Reasoning:** Testing ML with eta_chi_eff_log_e0 reparameterization and 20 PCA components.
- **Loss (RMS rel error x):** 0.09759
- **Runtime:** 0.00 ms

## Approach 13_gb_trig
- **Hypothesis/Reasoning:** Testing ML with trig_anomaly reparameterization and 10 PCA components.
- **Loss (RMS rel error x):** 0.01161
- **Runtime:** 0.01 ms

## Approach 14_rf_full
- **Hypothesis/Reasoning:** Testing ML with fully_transformed reparameterization and 20 PCA components.
- **Loss (RMS rel error x):** 0.01352
- **Runtime:** 0.03 ms

## Approach 15_mlp_deep_trig
- **Hypothesis/Reasoning:** Testing ML with trig_anomaly reparameterization and 15 PCA components.
- **Loss (RMS rel error x):** 0.06373
- **Runtime:** 0.00 ms

## Approach 16_gplearn_raw
- **Hypothesis/Reasoning:** Testing Symbolic with raw_6d reparameterization and 2 PCA components.
- **Loss (RMS rel error x):** 0.03989
- **Runtime:** 0.00 ms

## Approach 17_gplearn_eta
- **Hypothesis/Reasoning:** Testing Symbolic with eta_chi_eff_log_e0 reparameterization and 2 PCA components.
- **Loss (RMS rel error x):** 0.03672
- **Runtime:** 0.00 ms

## Approach 18_pysr_raw
- **Hypothesis/Reasoning:** Testing Symbolic with raw_6d reparameterization and 2 PCA components.
- **Loss (RMS rel error x):** 0.02781
- **Runtime:** 0.01 ms

## Approach 19_pysr_trig
- **Hypothesis/Reasoning:** Testing Symbolic with trig_anomaly reparameterization and 2 PCA components.
- **Loss (RMS rel error x):** 0.02939
- **Runtime:** 0.01 ms

## Approach 20_pysr_full
- **Hypothesis/Reasoning:** Testing Symbolic with fully_transformed reparameterization and 2 PCA components.
- **Loss (RMS rel error x):** 0.02567
- **Runtime:** 0.01 ms

