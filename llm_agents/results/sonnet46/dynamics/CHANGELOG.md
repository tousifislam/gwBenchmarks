# Dynamics Benchmark CHANGELOG (sonnet46)

## Setup
- Grid: normalized tau [0,1], 512 pts
- SVD 30 basis, recon error=0.0055
- Truncated (20) error=0.0068

## 01: svd_gpr_rbf_raw
- cat=svd_decomp, reparam=raw
- train=9.0282e-03, val=9.0083e-03, rt=21.6ms
- SVD+GPR RBF, raw 6D

## 02: svd_gpr_matern_eff
- cat=svd_decomp, reparam=eff
- train=1.0694e-02, val=1.0678e-02, rt=10.5ms
- SVD+GPR Matern, eff_spins+log_e

## 03: svd_poly2_raw
- cat=svd_decomp, reparam=raw
- train=7.8871e-03, val=8.4450e-03, rt=0.2ms
- SVD+poly2 Ridge, raw 6D

## 04: svd_mlp_raw
- cat=ml, reparam=raw
- train=1.9934e-02, val=3.2622e-02, rt=0.1ms
- SVD+MLP(128,128,64) relu, raw

## 05: svd_rf_raw
- cat=ml, reparam=raw
- train=6.9949e-03, val=1.1628e-02, rt=14.2ms
- SVD+RF(100,d12), raw 6D

## 06: eim_gpr_raw
- cat=svd_decomp, reparam=raw
- train=8.7575e-03, val=9.0616e-03, rt=25.9ms
- EIM(8 nodes)+GPR, raw

## 07: svd_gpr_trig
- cat=svd_decomp, reparam=trig
- train=8.9438e-03, val=8.9703e-03, rt=7.6ms
- SVD+GPR RBF, trig anomaly reparam

## 08: rbf_interp_raw
- cat=kernel_interp, reparam=raw
- train=6.8611e-03, val=8.3082e-03, rt=27.3ms
- TPS-RBF interp, raw 6D

## 09: knn_raw
- cat=kernel_interp, reparam=raw
- train=5.5622e-03, val=1.3290e-02, rt=0.4ms
- kNN(5, distance), raw 6D

## 10: krr_eff
- cat=kernel_interp, reparam=eff
- train=6.6659e-03, val=1.1871e-01, rt=31.5ms
- KRR(RBF), eff_spins

## 11: svd_gbr_eff
- cat=ml, reparam=eff
- train=7.1941e-03, val=1.0298e-02, rt=676.1ms
- SVD+GBR(80,d4), eff_spins

## 12: svd_et_trig
- cat=ml, reparam=trig
- train=5.5620e-03, val=1.0079e-02, rt=13.6ms
- SVD+ET(150,d14), trig anomaly

## 13: svd_mlp_large_eff
- cat=ml, reparam=eff
- train=3.5165e-02, val=3.7092e-02, rt=0.2ms
- SVD+MLP(256,256,128,64) tanh, eff

## 14: svd_poly3_eff
- cat=svd_decomp, reparam=eff
- train=8.5683e-03, val=1.1627e-02, rt=21.9ms
- SVD+poly3 Ridge, eff_spins

## 15: gplearn_svd_raw
- cat=symbolic, reparam=raw
- train=3.1509e-02, val=3.1239e-02, rt=0.3ms
- gplearn SR on 3 SVD coeff, raw

## 16: gplearn_svd_eff
- cat=symbolic, reparam=eff
- train=3.3355e-02, val=3.2666e-02, rt=0.5ms
- gplearn SR on 2 SVD coeff, eff

## 17: pysr_svd_raw
- cat=symbolic, reparam=raw
- train=2.4164e-02, val=2.4692e-02, rt=5.4ms
- PySR on 2 SVD coeff, raw

## 18: pysr_svd_eff
- cat=symbolic, reparam=eff
- train=2.8232e-02, val=2.7332e-02, rt=28.4ms
- PySR on 2 SVD coeff, eff

## 19: rbf_interp_eff
- cat=kernel_interp, reparam=eff
- train=7.5455e-03, val=1.1391e-02, rt=7.4ms
- Linear-RBF interp, eff_spins

## 20: svd_gpr_lf
- cat=svd_decomp, reparam=lf
- train=6.7745e-03, val=1.5411e-02, rt=12.3ms
- SVD+GPR Matern-1.5, log_freq reparam

## 21: knn10_eff
- cat=kernel_interp, reparam=eff
- train=5.5622e-03, val=1.7564e-02, rt=0.6ms
- kNN(10, distance), eff_spins

## 22: svd_mlp_lf
- cat=ml, reparam=lf
- train=2.2529e-02, val=3.3146e-02, rt=0.2ms
- SVD+MLP(128,64,32) relu, log_freq

## 23: svd_poly2_trig
- cat=svd_decomp, reparam=trig
- train=7.8386e-03, val=8.5051e-03, rt=0.2ms
- SVD+poly2 Ridge, trig anomaly

## 24: svd_rf_trig
- cat=ml, reparam=trig
- train=8.3976e-03, val=1.1590e-02, rt=13.9ms
- SVD+RF(150), trig anomaly

