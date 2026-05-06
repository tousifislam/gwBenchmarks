# Remnant Benchmark CHANGELOG — Opus 4.6

## 1: gpr_rbf_raw
- Param: raw, NRMSE: 0.116252, RT: 0.0ms
- GPR RBF kernel, raw params

## 2: gpr_matern_eta
- Param: eta_chieff, NRMSE: 0.091682, RT: 0.0ms
- GPR Matern-5/2, eta+chieff

## 3: krr_raw
- Param: raw, NRMSE: 0.107643, RT: 0.1ms
- KRR RBF

## 4: svr_raw
- Param: raw, NRMSE: 0.123008, RT: 0.0ms
- SVR RBF C=100

## 5: mlp_raw
- Param: raw, NRMSE: 0.098286, RT: 0.0ms
- MLP [256,128,64]

## 6: mlp_eta
- Param: eta_chieff, NRMSE: 0.095801, RT: 0.0ms
- MLP [512,256,128] eta

## 7: rf_raw
- Param: raw, NRMSE: 0.092233, RT: 0.0ms
- RF 500 trees

## 8: rf_eta
- Param: eta_chieff, NRMSE: 0.092554, RT: 0.0ms
- RF 500 trees eta

## 9: gbr_eta
- Param: eta_chieff, NRMSE: 0.095097, RT: 0.0ms
- GBR 500 est eta

## 10: poly3_raw
- Param: raw, NRMSE: 0.098719, RT: 0.0ms
- Poly-3 + Ridge

## 11: poly4_eta
- Param: eta_chieff, NRMSE: 0.093113, RT: 0.0ms
- Poly-4 + Ridge eta

## 12: knn_raw
- Param: raw, NRMSE: 0.095302, RT: 0.0ms
- 7-NN distance-weighted

## 13: rbf_interp_raw
- Param: raw, NRMSE: 0.106405, RT: 0.1ms
- RBF TPS interp

## 14: rbf_interp_eta
- Param: eta_chieff, NRMSE: 0.107351, RT: 0.0ms
- RBF TPS interp eta

## 15: et_raw
- Param: raw, NRMSE: 0.089878, RT: 0.0ms
- ExtraTrees 500

## 16: lasso_poly3_eta
- Param: eta_chieff, NRMSE: 0.091656, RT: 0.0ms
- Lasso poly-3 eta

## 17: bayridge_mdiff
- Param: mass_diff, NRMSE: 0.093115, RT: 0.0ms
- BayRidge poly-3 mass_diff

## 18: adaboost_eta
- Param: eta_chieff, NRMSE: 0.095422, RT: 0.0ms
- AdaBoost DT-8 eta

## 19: mlp_sph
- Param: spherical, NRMSE: 0.099959, RT: 0.0ms
- MLP [512,512,256,128] spherical

## 20: bagging_rf_eta
- Param: eta_chieff, NRMSE: 0.091240, RT: 0.1ms
- Bagging RF eta

## 21: enet_pn
- Param: pn_inspired, NRMSE: 0.099284, RT: 0.0ms
- ElasticNet poly-4 PN-inspired

## 22: gbr_mdiff
- Param: mass_diff, NRMSE: 0.098125, RT: 0.0ms
- GBR 500 mass_diff

## 23: pysr_raw
- Param: raw, NRMSE: 0.111408, RT: 0.0ms
- PySR symbolic regression raw

## 24: pysr_eta
- Param: eta_chieff, NRMSE: 0.105432, RT: 0.0ms
- PySR symbolic regression eta

## 25: gplearn_raw
- Param: raw, NRMSE: 1.143794, RT: 0.0ms
- gplearn symbolic regression raw

## 26: gplearn_eta
- Param: eta_chieff, NRMSE: 1.143794, RT: 0.0ms
- gplearn symbolic regression eta

