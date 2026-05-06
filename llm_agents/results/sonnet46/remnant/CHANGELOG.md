# Remnant Benchmark CHANGELOG (sonnet46)

## Setup
- n_train=1000, n_val=1000
- vf range=0.013464
- Loss: NRMSE(vf_mag)

## 01: gpr_rbf_raw
- cat=kernel_gp, reparam=raw
- train=3.8151e-04, val=7.2274e-02, rt=0.0ms
- GPR RBF, raw 7D params

## 02: gpr_matern_eff
- cat=kernel_gp, reparam=eff
- train=4.3612e-02, val=5.6254e-02, rt=0.0ms
- GPR Matern-2.5, eff_spins

## 03: gpr_rbf_sph
- cat=kernel_gp, reparam=sph
- train=3.9368e-04, val=7.7142e-02, rt=0.0ms
- GPR RBF, spherical spins

## 04: krr_eff
- cat=kernel_gp, reparam=eff
- train=1.8292e-02, val=9.0405e-02, rt=0.1ms
- KRR(RBF), eff_spins

## 05: svr_eff
- cat=kernel_gp, reparam=eff
- train=3.5607e-02, val=9.0936e-02, rt=0.0ms
- SVR(RBF), eff_spins

## 06: poly2_raw
- cat=symbolic, reparam=raw
- train=7.0765e-02, val=7.1950e-02, rt=0.0ms
- Poly-2 Ridge, raw 7D

## 07: poly3_eff
- cat=symbolic, reparam=eff
- train=5.4479e-02, val=5.7846e-02, rt=0.0ms
- Poly-3 Ridge, eff_spins

## 08: gplearn_raw
- cat=symbolic, reparam=raw
- train=1.3310e+00, val=1.3263e+00, rt=0.0ms
- gplearn SR, raw 7D

## 09: gplearn_eff
- cat=symbolic, reparam=eff
- train=3.1371e-01, val=3.0871e-01, rt=0.0ms
- gplearn SR, eff_spins

## 10: pysr_raw
- cat=symbolic, reparam=raw
- train=1.2047e-01, val=1.1668e-01, rt=0.0ms
- PySR, raw 7D

## 11: pysr_eff
- cat=symbolic, reparam=eff
- train=1.2789e-01, val=1.2234e-01, rt=0.0ms
- PySR, eff_spins

## 12: rbf_interp_raw
- cat=interpolation, reparam=raw
- train=2.1252e-02, val=6.0824e-02, rt=0.0ms
- TPS-RBF interp, raw 7D

## 13: rbf_interp_eff
- cat=interpolation, reparam=eff
- train=2.8287e-02, val=5.5241e-02, rt=0.0ms
- Linear-RBF interp, eff_spins

## 14: knn5_raw
- cat=interpolation, reparam=raw
- train=0.0000e+00, val=5.8205e-02, rt=0.0ms
- kNN(5, distance), raw 7D

## 15: knn10_sph
- cat=interpolation, reparam=sph
- train=0.0000e+00, val=6.5689e-02, rt=0.0ms
- kNN(10, distance), spherical

## 16: mlp_raw
- cat=ml, reparam=raw
- train=1.0870e-01, val=5.4058e-01, rt=0.0ms
- MLP(128,128,64) relu, raw 7D

## 17: mlp_large_eff
- cat=ml, reparam=eff
- train=2.4905e-01, val=2.7029e-01, rt=0.0ms
- MLP(256,256,128,64) tanh, eff_spins

## 18: rf_raw
- cat=ml, reparam=raw
- train=2.6278e-02, val=5.7519e-02, rt=0.0ms
- RF(200, min_leaf=2), raw 7D

## 19: et_md
- cat=ml, reparam=md
- train=2.0586e-02, val=5.3785e-02, rt=0.0ms
- ExtraTrees(200), mass_diff

## 20: gbr_eff
- cat=ml, reparam=eff
- train=2.9589e-02, val=5.5798e-02, rt=0.0ms
- GBR(200,d4), eff_spins

## 21: gpr_matern_md
- cat=kernel_gp, reparam=md
- train=5.8820e-05, val=7.7550e-02, rt=0.0ms
- GPR Matern-1.5, mass_diff

## 22: poly4_md
- cat=symbolic, reparam=md
- train=4.9395e-02, val=5.8549e-02, rt=0.0ms
- Poly-4 Ridge, mass_diff

## 23: mlp_md
- cat=ml, reparam=md
- train=1.9388e-01, val=5.1653e-01, rt=0.0ms
- MLP(128,64,32) relu, mass_diff

## 24: krr_poly_md
- cat=kernel_gp, reparam=md
- train=5.2804e-02, val=5.8669e-02, rt=0.1ms
- KRR(poly3), mass_diff

