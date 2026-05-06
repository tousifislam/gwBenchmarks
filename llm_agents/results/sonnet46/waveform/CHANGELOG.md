# Waveform Benchmark CHANGELOG (sonnet46)

## Setup
- Grid: t=-2847..100, n=2048
- SVD 50 basis, recon error=0.0193
- Truncated (50) error=0.0193

## 01: svd_gpr_rbf_raw
- cat=svd_decomp, reparam=raw
- train=4.4774e-01, val=9.9047e-01, rt=23.6ms
- SVD+GPR RBF, raw 7D params, 15 coeff

## 02: svd_gpr_matern_eff
- cat=svd_decomp, reparam=eff
- train=9.3219e-01, val=9.8997e-01, rt=36.1ms
- SVD+GPR Matern-2.5, eff_spins, 15 coeff

## 03: svd_poly3_raw
- cat=svd_decomp, reparam=raw
- train=7.1378e-01, val=1.2301e+00, rt=38.7ms
- SVD+poly3 Ridge, raw 7D

## 04: svd_mlp_raw
- cat=ml, reparam=raw
- train=9.5931e-01, val=9.8798e-01, rt=0.2ms
- SVD+MLP(128,128,64) relu, raw 7D

## 05: svd_rf_raw
- cat=ml, reparam=raw
- train=5.6086e-01, val=9.5646e-01, rt=17.3ms
- SVD+RF(100) depth 12, raw 7D

## 06: svd_gbr_eff
- cat=ml, reparam=eff
- train=6.2890e-01, val=1.0089e+00, rt=38.0ms
- SVD+GBR(80) eff_spins

## 07: amp_phase_gpr_raw
- cat=svd_decomp, reparam=raw
- train=6.7430e-01, val=1.4110e+00, rt=2.9ms
- Amp+Phase SVD+GPR, raw 7D

## 08: eim_gpr_raw
- cat=svd_decomp, reparam=raw
- train=1.0804e+00, val=1.0834e+00, rt=23.3ms
- EIM (12 nodes)+GPR, raw 7D

## 09: rbf_interp_sph
- cat=kernel_interp, reparam=sph
- train=5.3296e-01, val=1.0684e+00, rt=30.1ms
- TPS-RBF on spherical spins

## 10: krr_eff
- cat=kernel_interp, reparam=eff
- train=2.5451e-01, val=1.2221e+00, rt=59.9ms
- KRR(RBF) on eff_spins

## 11: svd_gpr_sph
- cat=svd_decomp, reparam=sph
- train=3.7915e-01, val=9.9335e-01, rt=44.4ms
- SVD+GPR RBF, spherical, N_C_GPR coeff

## 12: knn_raw
- cat=kernel_interp, reparam=raw
- train=2.2275e-02, val=9.8485e-01, rt=0.7ms
- kNN(7, distance) on raw 7D

## 13: svd_mlp_large_eff
- cat=ml, reparam=eff
- train=9.6574e-01, val=9.9342e-01, rt=0.5ms
- MLP(256,256,128,64) tanh eff

## 14: svd_et_md
- cat=ml, reparam=md
- train=6.9510e-02, val=9.6236e-01, rt=15.1ms
- ExtraTrees(150,d14) mass_diff

## 15: gplearn_svd_raw
- cat=symbolic, reparam=raw
- train=9.9132e-01, val=9.8917e-01, rt=0.3ms
- gplearn SR on 3 SVD coeff (re+im), raw 7D

## 16: pysr_svd_raw
- cat=symbolic, reparam=raw
- train=9.7902e-01, val=9.9087e-01, rt=5.7ms
- PySR on 2 SVD coeff, raw 7D

## 17: pysr_svd_eff
- cat=symbolic, reparam=eff
- train=9.9700e-01, val=1.0011e+00, rt=8.9ms
- PySR on 2 SVD coeff, eff_spins

## 18: amp_phase_gpr_md
- cat=svd_decomp, reparam=md
- train=1.4210e+00, val=1.4104e+00, rt=120.1ms
- Amp+Phase GPR, mass_diff

## 19: svd_poly2_eff
- cat=svd_decomp, reparam=eff
- train=9.0358e-01, val=1.0199e+00, rt=22.9ms
- SVD+poly2 Ridge, eff_spins

## 20: svd_gpr_matern_md
- cat=svd_decomp, reparam=md
- train=3.7915e-01, val=1.0046e+00, rt=28.4ms
- SVD+GPR Matern-1.5, mass_diff

## 21: gplearn_svd_eff
- cat=symbolic, reparam=eff
- train=9.9482e-01, val=1.0008e+00, rt=0.3ms
- gplearn SR on 2 SVD coeff, eff_spins

## 22: svd_gpr_t0start
- cat=svd_decomp, reparam=raw
- train=1.0003e+00, val=1.0024e+00, rt=22.4ms
- SVD+GPR t=0 at start

## 23: svd_mlp_md
- cat=ml, reparam=md
- train=9.7290e-01, val=9.8753e-01, rt=0.1ms
- SVD+MLP(128,64,32) relu mass_diff

## 24: svd_poly2_sph
- cat=svd_decomp, reparam=sph
- train=9.0779e-01, val=1.0549e+00, rt=28.3ms
- SVD+poly2 Ridge, spherical

