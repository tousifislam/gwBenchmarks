# Waveform Benchmark - CHANGELOG

## Summary

- Total approaches: 24
- Best loss: 0.4044
- Parameterizations tested: raw, effective, mass_diff, spherical
- Categories: SVD/decomposition, Symbolic/analytical, Interpolation/kernel, Machine learning

## Approaches

### 01_svd_gpr_rbf_raw
- Loss: 0.4616
- Parameterization: raw
- Notes: Model: gpr_rbf, n_basis: 15

### 02_svd_gpr_matern_effective
- Loss: 0.4559
- Parameterization: effective
- Notes: Model: gpr_matern, n_basis: 15

### 03_svd_polynomial_raw
- Loss: 0.4206
- Parameterization: raw
- Notes: Model: poly, n_basis: 15

### 04_svd_mlp_raw
- Loss: 0.4747
- Parameterization: raw
- Notes: Model: mlp, n_basis: 15

### 05_svd_rf_raw
- Loss: 0.4088
- Parameterization: raw
- Notes: Model: rf, n_basis: 15

### 06_eim_gpr_rbf_raw
- Loss: 0.4586
- Parameterization: raw
- Notes: Model: gpr_rbf, n_basis: 25

### 07_svd_gpr_rbf_mass_diff
- Loss: 0.4790
- Parameterization: mass_diff
- Notes: Model: gpr_rbf, n_basis: 15

### 08_svd_mlp_effective
- Loss: 0.4695
- Parameterization: effective
- Notes: Model: mlp, n_basis: 15

### 09_svd_kr_rbf_raw
- Loss: 0.4327
- Parameterization: raw
- Notes: Model: kr, n_basis: 15

### 10_svd_knn_raw
- Loss: 0.4205
- Parameterization: raw
- Notes: Model: knn, n_basis: 15

### 11_svd_svr_raw
- Loss: 0.4393
- Parameterization: raw
- Notes: Model: svr, n_basis: 15

### 12_svd_kr_rbf_effective
- Loss: 0.4329
- Parameterization: effective
- Notes: Model: kr, n_basis: 15

### 13_svd_gbr_raw
- Loss: 0.4104
- Parameterization: raw
- Notes: Model: gbr, n_basis: 15

### 14_svd_extra_trees_raw
- Loss: 0.4044
- Parameterization: raw
- Notes: Model: extra_trees, n_basis: 15

### 15_svd_mlp_large_raw
- Loss: 0.4719
- Parameterization: raw
- Notes: Model: mlp_large, n_basis: 20

### 16_svd_ridge_raw
- Loss: 0.4621
- Parameterization: raw
- Notes: Model: ridge, n_basis: 15

### 17_svd_elastic_net_raw
- Loss: 0.4724
- Parameterization: raw
- Notes: Model: elastic_net, n_basis: 15

### 18_svd_lasso_raw
- Loss: 0.4794
- Parameterization: raw
- Notes: Model: lasso, n_basis: 15

### 19_svd_huber_raw
- Loss: 0.4722
- Parameterization: raw
- Notes: Model: huber, n_basis: 15

### 20_svd_linear_raw
- Loss: 0.4620
- Parameterization: raw
- Notes: Model: linear, n_basis: 15

### 21_svd_bayesian_ridge_raw
- Loss: 0.4792
- Parameterization: raw
- Notes: Model: bayesian_ridge, n_basis: 15

### 22_svd_mlp_small_raw
- Loss: 0.4440
- Parameterization: raw
- Notes: Model: mlp_small, n_basis: 15

### 23_svd_pysr_raw
- Loss: 0.4850
- Parameterization: raw
- Notes: PySR on top 5 SVD coefficients, Ridge for rest. PySR expressions saved in saved_model/expressions.json

### 24_svd_gplearn_raw
- Loss: 0.4640
- Parameterization: raw
- Notes: gplearn on top 3 SVD coefficients, Ridge for rest

