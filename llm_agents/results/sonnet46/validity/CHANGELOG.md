## 01: gpr_rbf_raw
- cat=kernel_gp, rep=raw
- train=6.8834e-01, val=6.1702e-01, rt=0.42ms
- GPR(RBF), raw 4D

## 02: gpr_matern_eff
- cat=kernel_gp, rep=eff
- train=3.8738e-02, val=6.1702e-01, rt=0.20ms
- GPR(Matern-2.5), eta+chi_eff+chi_a

## 03: rf_raw
- cat=ml, rep=raw
- train=2.0985e-01, val=5.3429e-01, rt=14.53ms
- RF(200), raw 4D, log10(mm) target

## 04: gbr_logq
- cat=ml, rep=logq
- train=1.1025e-01, val=5.7544e-01, rt=0.19ms
- GBR(300,d5), log(q)+chi_eff+chi_a+log(omega0)

## 05: mlp_eff
- cat=ml, rep=eff
- train=5.7002e-01, val=5.7033e-01, rt=0.12ms
- MLP(128,128,64 relu), eff_spins

## 06: krr_eff
- cat=kernel_gp, rep=eff
- train=3.4013e-01, val=6.3058e-01, rt=0.29ms
- KRR(RBF gamma=0.5), eff_spins

## 07: poly2_raw
- cat=symbolic, rep=raw
- train=6.1345e-01, val=5.7774e-01, rt=0.12ms
- Poly-2 Ridge, raw 4D, log10 target

## 08: rbf_tps_raw
- cat=interp, rep=raw
- train=3.5261e-01, val=5.6332e-01, rt=0.23ms
- RBF(TPS, s=0.5) interp, raw 4D

## 09: knn5_eff
- cat=interp, rep=eff
- train=0.0000e+00, val=5.7877e-01, rt=0.29ms
- kNN(5, distance), eff_spins

## 10: gpr_matern_logq
- cat=kernel_gp, rep=logq
- train=4.7401e-01, val=5.6870e-01, rt=0.54ms
- GPR(Matern-1.5), log(q)+chi_eff+chi_a+log(omega0)

## 11: et_inter
- cat=ml, rep=inter
- train=8.5079e-15, val=5.6514e-01, rt=13.95ms
- ET(200), interaction features eta*chi_eff

## 12: mlp_deep_logq
- cat=ml, rep=logq
- train=6.5658e-01, val=6.3759e-01, rt=0.14ms
- MLP(256,256,128,64 tanh), log(q)+chi_eff+log(omega0)

## 13: rbf_mq_eff
- cat=interp, rep=eff
- train=5.0552e-01, val=5.5948e-01, rt=0.26ms
- RBF(multiquadric) interp, eff_spins

## 14: poly3_logq
- cat=symbolic, rep=logq
- train=5.8372e-01, val=5.7667e-01, rt=0.12ms
- Poly-3 Ridge, log(q)+chi_eff+chi_a+log(omega0)

## 15: gplearn_raw
- cat=symbolic, rep=raw
- train=6.9423e-01, val=6.2036e-01, rt=0.16ms
- gplearn SR: inv(-0.276)

## 16: gplearn_eff
- cat=symbolic, rep=eff
- train=6.5495e-01, val=6.1378e-01, rt=0.12ms
- gplearn SR, eff_spins reparam

## 17: pysr_raw
- cat=symbolic, rep=raw
- train=6.8025e-01, val=6.3863e-01, rt=1.52ms
- PySR symbolic regression, raw 4D

## 18: pysr_logq
- cat=symbolic, rep=logq
- train=6.8516e-01, val=6.3285e-01, rt=1.38ms
- PySR, log(q)+chi_eff+chi_a+log(omega0)

## 19: krr_matern_logq
- cat=kernel_gp, rep=logq
- train=1.9050e-01, val=5.6928e-01, rt=0.29ms
- KRR(Laplacian gamma=0.5), logq

## 20: rf_logq
- cat=ml, rep=logq
- train=2.6815e-01, val=5.3320e-01, rt=25.78ms
- RF(300, leaf=2), log(q)+chi_eff+chi_a+log(omega0)

## 21: gpr_bnd
- cat=kernel_gp, rep=bnd
- train=2.9343e-01, val=6.2661e-01, rt=0.17ms
- GPR(Matern-2.5), boundary distance features

## 22: mlp_inter
- cat=ml, rep=inter
- train=6.3037e-01, val=5.8534e-01, rt=0.11ms
- MLP(64,64 relu), interaction features

## 23: knn10_logq
- cat=interp, rep=logq
- train=0.0000e+00, val=5.6362e-01, rt=0.28ms
- kNN(10, distance), log(q)+chi_eff+log(omega0)

## 24: et_bnd
- cat=ml, rep=bnd
- train=1.2241e-14, val=5.5271e-01, rt=25.72ms
- ET(300), boundary distance features

