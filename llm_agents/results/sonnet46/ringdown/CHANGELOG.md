## 01: poly10_raw
- cat=analytical, reparam=raw
- train=1.1678e+00, val=9.7935e-01, rt=0.03ms
- Poly(10), raw spin a

## 02: poly15_log
- cat=analytical, reparam=log
- train=2.0632e-03, val=1.3058e-03, rt=0.04ms
- Poly(15), log-compactified -log(1-a)

## 03: cheb20_cheb
- cat=analytical, reparam=cheb
- train=5.0191e-01, val=4.1942e-01, rt=0.05ms
- Chebyshev(20), 2a-1 mapped to [-1,1]

## 04: spline_raw
- cat=interp, reparam=raw
- train=1.0455e-19, val=1.1299e-05, rt=0.22ms
- Cubic spline on raw spin

## 05: spline_cheb_nodes
- cat=interp, reparam=cheb
- train=1.9569e-02, val=1.9639e-02, rt=0.03ms
- Cubic spline on 50 Chebyshev-Gauss-Lobatto nodes

## 06: pade_10_10_raw
- cat=analytical, reparam=raw
- train=9.9657e-01, val=9.9676e-01, rt=0.06ms
- Pade rational [10,10], raw spin

## 07: rbf_tps_raw
- cat=interp, reparam=raw
- train=2.2700e-03, val=1.5541e-03, rt=0.06ms
- RBF(TPS) interpolation, raw spin

## 01: poly10_raw
- cat=analytical, reparam=raw
- train=1.1678e+00, val=9.7935e-01, rt=0.03ms
- Poly(10), raw spin a

## 02: poly15_log
- cat=analytical, reparam=log
- train=2.0632e-03, val=1.3058e-03, rt=0.04ms
- Poly(15), log-compactified -log(1-a)

## 03: cheb20_cheb
- cat=analytical, reparam=cheb
- train=5.0191e-01, val=4.1942e-01, rt=0.05ms
- Chebyshev(20), 2a-1 mapped to [-1,1]

## 04: spline_raw
- cat=interp, reparam=raw
- train=1.0455e-19, val=1.1299e-05, rt=0.03ms
- Cubic spline on raw spin

## 05: spline_cheb_nodes
- cat=interp, reparam=cheb
- train=1.9569e-02, val=1.9639e-02, rt=0.03ms
- Cubic spline on 50 Chebyshev-Gauss-Lobatto nodes

## 06: pade_10_10_raw
- cat=analytical, reparam=raw
- train=9.9657e-01, val=9.9676e-01, rt=0.06ms
- Pade rational [10,10], raw spin

## 07: rbf_tps_raw
- cat=interp, reparam=raw
- train=2.2700e-03, val=1.5541e-03, rt=0.06ms
- RBF(TPS) interpolation, raw spin

## 08: rbf_mq_log
- cat=interp, reparam=log
- train=1.4252e-03, val=1.1479e-03, rt=0.05ms
- RBF(multiquadric) interpolation, log-compactified

## 09: gpr_rbf_raw
- cat=ml, reparam=raw
- train=3.9578e-01, val=3.3011e-01, rt=0.18ms
- GPR (RBF kernel), raw spin

## 10: gpr_matern_log
- cat=ml, reparam=log
- train=1.8483e-03, val=1.0919e-03, rt=0.42ms
- GPR (Matern-2.5), log-compactified

## 11: mlp_small_raw
- cat=ml, reparam=raw
- train=8.5115e+00, val=7.1562e+00, rt=0.11ms
- MLP(64,64,32 tanh), raw spin

## 12: mlp_deep_log
- cat=ml, reparam=log
- train=5.6293e-01, val=4.5735e-01, rt=0.13ms
- MLP(128,128,64,32 relu), log-compactified

## 13: rf_raw
- cat=ml, reparam=raw
- train=1.4367e-02, val=1.1394e-02, rt=12.63ms
- Random Forest(200), raw spin

## 14: gbr_log
- cat=ml, reparam=log
- train=1.9868e-03, val=4.8059e-03, rt=12.01ms
- GBR(200, d5) multitask, log-compactified

## 15: gplearn_raw
- cat=symbolic, reparam=raw
- train=2.2737e+01, val=1.9133e+01, rt=0.12ms
- gplearn SR, raw spin. omega_r: 0.535

## 16: gplearn_log
- cat=symbolic, reparam=log
- train=2.5681e+01, val=2.1591e+01, rt=0.15ms
- gplearn SR, log-compactified spin

## 17: pysr_raw
- cat=symbolic, reparam=raw
- train=3.8671e-02, val=nan, rt=2.79ms
- PySR, raw spin a

## 18: pysr_log
- cat=symbolic, reparam=log
- train=8.7649e-03, val=7.3394e-03, rt=2.82ms
- PySR, log-compactified spin

## 19: krr_raw
- cat=ml, reparam=raw
- train=5.8240e-01, val=4.8691e-01, rt=0.65ms
- Kernel Ridge (RBF gamma=10), raw spin

## 20: poly20_sqrt
- cat=analytical, reparam=sqrt
- train=6.7209e-04, val=6.0725e-04, rt=0.05ms
- Poly(20), sqrt(1-a^2) irreducible mass

## 21: knn5_log
- cat=ml, reparam=log
- train=1.4252e-03, val=2.8370e-03, rt=0.37ms
- kNN(5, distance), log-compactified

## 22: cheb30_log
- cat=analytical, reparam=log
- train=1.4252e-03, val=9.6019e-04, rt=0.07ms
- Chebyshev(30), log-compactified mapped to [-1,1]

## 23: rbf_cubic_sqrt
- cat=interp, reparam=sqrt
- train=5.1009e-05, val=4.6930e-05, rt=0.08ms
- RBF(cubic) interpolation, sqrt(1-a^2)

## 24: pade_15_5_log
- cat=analytical, reparam=log
- train=1.2760e+00, val=1.2224e+00, rt=0.07ms
- Pade rational [15,5], log-compactified

