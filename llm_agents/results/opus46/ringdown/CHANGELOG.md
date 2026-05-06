# Ringdown Benchmark CHANGELOG -- Opus 4.6

Mode: l2/m+2/n0 | Loss: mean relative error (omega_R, omega_I)

## 1: poly10_raw
- Param: raw, Loss: 0.97935239, RT: 0.00ms
- Polynomial degree 10, raw spin

## 2: poly15_raw
- Param: raw, Loss: 0.58400298, RT: 0.00ms
- Polynomial degree 15, raw spin

## 3: poly20_raw
- Param: raw, Loss: 0.45585528, RT: 0.00ms
- Polynomial degree 20, raw spin

## 4: chebyshev_neglog
- Param: neglog, Loss: 0.00473625, RT: 0.00ms
- Chebyshev deg 20, neglog reparameterization

## 5: pade_rational_raw
- Param: raw, Loss: 0.14314812, RT: 0.00ms
- Pade/rational approx [7,6]/[7,6], raw spin

## 6: ridge_poly_neglog
- Param: neglog, Loss: 0.07687986, RT: 0.00ms
- Ridge poly deg 20 + neglog reparam

## 7: pysr_raw
- Param: raw, Loss: 3.76567743, RT: 0.01ms
- PySR symbolic regression, raw spin, separate R/I fits

## 8: pysr_neglog
- Param: neglog, Loss: 0.02529811, RT: 0.01ms
- PySR symbolic regression, neglog reparam, separate R/I fits

## 9: gplearn_raw
- Param: raw, Loss: 19.85444741, RT: 0.00ms
- gplearn symbolic regression, raw spin, separate R/I fits

## 10: gplearn_neglog
- Param: neglog, Loss: 23.51599657, RT: 0.00ms
- gplearn symbolic regression, neglog reparam, separate R/I fits

## 11: cubic_spline_raw
- Param: raw, Loss: 0.00001130, RT: 0.00ms
- Cubic spline interpolation, raw spin (sorted)

## 12: rbf_tps_raw
- Param: raw, Loss: 0.47565686, RT: 0.07ms
- RBF thin_plate_spline interpolation, raw spin

## 13: rbf_linear_neglog
- Param: neglog, Loss: 0.00022098, RT: 0.09ms
- RBF linear interpolation, neglog reparam

## 14: knn_raw
- Param: raw, Loss: 0.00440786, RT: 0.00ms
- 5-NN distance-weighted, raw spin

## 15: gpr_rbf_neglog
- Param: neglog, Loss: 0.00198610, RT: 0.08ms
- GPR RBF kernel, neglog reparam

## 16: gpr_matern_neglog
- Param: neglog, Loss: 0.00045141, RT: 0.22ms
- GPR Matern-5/2 kernel, neglog reparam

## 17: krr_neglog
- Param: neglog, Loss: 0.00058162, RT: 0.11ms
- KRR RBF kernel, neglog reparam

## 18: mlp_neglog
- Param: neglog, Loss: 0.15205413, RT: 0.00ms
- MLP [256,128,64], neglog reparam, 2D output

## 19: mlp_sqrt1ma2
- Param: sqrt1ma2, Loss: 0.44402389, RT: 0.01ms
- MLP [512,256,128], sqrt(1-a^2) reparam

## 20: rf_raw
- Param: raw, Loss: 0.01263792, RT: 0.15ms
- Random Forest 500 trees, raw spin

## 21: rf_neglog
- Param: neglog, Loss: 0.00354979, RT: 0.17ms
- Random Forest 500 trees, neglog reparam

## 22: gbr_neglog
- Param: neglog, Loss: 0.00274236, RT: 0.00ms
- GBR 500 estimators, neglog reparam

## 23: svr_neglog
- Param: neglog, Loss: 0.12259984, RT: 0.00ms
- SVR RBF C=100, neglog reparam

## 24: extratrees_raw
- Param: raw, Loss: 1.04985333, RT: 0.17ms
- ExtraTrees 500, raw spin

## 25: extratrees_neglog
- Param: neglog, Loss: 1.08639741, RT: 0.16ms
- ExtraTrees 500, neglog reparam

## 26: gbr_sqrt1ma2
- Param: sqrt1ma2, Loss: 0.00228052, RT: 0.01ms
- GBR 500, sqrt(1-a^2) reparam

