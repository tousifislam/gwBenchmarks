# Waveform Benchmark CHANGELOG — Opus 4.6

## 3: svd_poly3_raw
- Param: raw, Loss: 0.254667, RT: 0.1ms
- SVD(40)+deg-3 poly+Ridge

## 4: svd_mlp_raw
- Param: raw, Loss: 0.269045, RT: 0.0ms
- SVD(40)+MLP[256,128,64]

## 5: svd_rf_raw
- Param: raw, Loss: 0.185782, RT: 0.1ms
- SVD(40)+RF200

## 6: svd_gbr_eta
- Param: eta_chieff, Loss: 0.171054, RT: 1.0ms
- SVD(40)+GBR eta

## 7: svd_krr_raw
- Param: raw, Loss: 0.215631, RT: 0.4ms
- SVD(40)+KRR RBF

## 8: svd_rbf_interp_raw
- Param: raw, Loss: 0.202117, RT: 0.4ms
- SVD(40)+RBF TPS

## 9: svd_knn_raw
- Param: raw, Loss: 0.197130, RT: 0.0ms
- SVD(40)+5-NN dist-wt

## 10: svd_mlp_eta
- Param: eta_chieff, Loss: 0.254435, RT: 1.5ms
- SVD(40)+MLP[512,256,128] eta

## 11: svd_poly4_eta
- Param: eta_chieff, Loss: 0.207367, RT: 0.5ms
- SVD(40)+deg-4 poly eta

## 13: svd_rf_eta
- Param: eta_chieff, Loss: 0.166441, RT: 0.2ms
- SVD(40)+RF500 eta

## 14: svd_et_sph
- Param: spherical, Loss: 0.197005, RT: 0.3ms
- SVD(40)+ET500 sph

## 15: svd_svr_mdiff
- Param: mass_diff, Loss: 0.215888, RT: 0.2ms
- SVD(40)+SVR mdiff

## 16: svd_lasso_raw
- Param: raw, Loss: 0.254380, RT: 0.2ms
- SVD(40)+Lasso poly3

## 17: svd_adaboost_eta
- Param: eta_chieff, Loss: 0.171053, RT: 1.8ms
- SVD(40)+AdaBoost eta

## 18: svd_mlp_large_sph
- Param: spherical, Loss: 0.263794, RT: 0.1ms
- SVD(40)+MLP[512,512,256,128] sph

## 19: ampphase_rf_eta
- Param: eta_chieff, Loss: 0.478045, RT: 0.1ms
- AmpPhase SVD(40)+RF300 eta

## 20: svd_rbf_interp_eta
- Param: eta_chieff, Loss: 0.162560, RT: 0.3ms
- SVD(40)+RBF TPS eta

## 21: svd_bayridge_mdiff
- Param: mass_diff, Loss: 0.249380, RT: 0.1ms
- SVD(40)+BayRidge poly3 mdiff

## 22: svd_elasticnet_eta
- Param: eta_chieff, Loss: 0.238375, RT: 0.3ms
- SVD(40)+ElasticNet poly3 eta

## 1: svd_gpr_rbf_raw
- Param: raw, Loss: 0.263203, RT: 1.1ms
- SVD(5)+GPR RBF raw (no opt)

## 2: svd_gpr_matern_eta
- Param: eta_chieff, Loss: 0.267713, RT: 0.1ms
- SVD(5)+GPR Matern eta (no opt)

## 12: svd_gpr_matern_sph
- Param: spherical, Loss: 0.258616, RT: 0.1ms
- SVD(5)+GPR Matern sph (no opt)


## 23: pysr_raw
- Param: raw, Loss: 0.270198
- PySR on top-5 SVD coefficients, raw params

## 24: pysr_eta
- Param: eta_chieff, Loss: 0.250704
- PySR on top-5 SVD coefficients, eta+chieff

## 25: gplearn_raw
- Param: raw, Loss: 0.255429
- gplearn on top-5 SVD coefficients, raw params

## 26: gplearn_eta
- Param: eta_chieff, Loss: 0.238460
- gplearn on top-5 SVD coefficients, eta+chieff
