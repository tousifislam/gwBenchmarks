
## [V-01] GPR-RBF (raw)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: kernel/gp
- **Method**: gpr_rbf
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 8.9689e-01
- **Eval time**: 0.0037 ms
- **Reasoned optimization**:
  - Observed: train_rmse=3.389e-02, val_rmse=8.969e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-02] GPR-Matern-5/2 (eff)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: kernel/gp
- **Method**: gpr_matern
- **Parameterization**: effective_4d
- **Loss (RMSE log10(mm))**: 8.9532e-01
- **Eval time**: 0.0038 ms
- **Reasoned optimization**:
  - Observed: train_rmse=3.303e-02, val_rmse=8.953e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-03] KRR-RBF (log)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: kernel/gp
- **Method**: krr_rbf
- **Parameterization**: log_4d
- **Loss (RMSE log10(mm))**: 2.2845e+00
- **Eval time**: 0.0020 ms
- **Reasoned optimization**:
  - Observed: train_rmse=1.654e-01, val_rmse=4.262e+00 -> val_rmse=2.285e+00
  - Hypothesis: Validation error suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-05

## [V-04] SVR-RBF (raw)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: kernel/gp
- **Method**: svr_rbf
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 8.8773e-01
- **Eval time**: 0.0089 ms
- **Reasoned optimization**:
  - Observed: train_rmse=7.479e-01, val_rmse=8.877e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-05] RBFInterp (TPS, raw)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: interpolation
- **Method**: rbf_interp
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 1.9721e+01
- **Eval time**: 0.2556 ms
- **Reasoned optimization**:
  - Observed: train_rmse=3.783e-06, val_rmse=1.972e+01
  - Hypothesis: N/A
  - Change: N/A

## [V-06] kNN (eff)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: interpolation
- **Method**: knn
- **Parameterization**: effective_4d
- **Loss (RMSE log10(mm))**: 7.9147e-01
- **Eval time**: 0.0020 ms
- **Reasoned optimization**:
  - Observed: train_rmse=0.000e+00, val_rmse=7.915e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-07] Ridge (eff)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: symbolic/analytical
- **Method**: ridge
- **Parameterization**: effective_4d
- **Loss (RMSE log10(mm))**: 8.1468e-01
- **Eval time**: 0.0002 ms
- **Reasoned optimization**:
  - Observed: train_rmse=9.031e-01, val_rmse=8.147e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-08] Poly-3 Ridge (raw)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: symbolic/analytical
- **Method**: poly_ridge
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 8.2677e-01
- **Eval time**: 0.0004 ms
- **Reasoned optimization**:
  - Observed: train_rmse=8.118e-01, val_rmse=8.268e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-09] Lasso (raw)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: symbolic/analytical
- **Method**: lasso
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 8.1492e-01
- **Eval time**: 0.0002 ms
- **Reasoned optimization**:
  - Observed: train_rmse=9.052e-01, val_rmse=8.149e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-10] ElasticNet (raw)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: symbolic/analytical
- **Method**: elasticnet
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 8.1491e-01
- **Eval time**: 0.0002 ms
- **Reasoned optimization**:
  - Observed: train_rmse=9.052e-01, val_rmse=8.149e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-11] RandomForest (raw)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: machine_learning
- **Method**: rf
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 7.2092e-01
- **Eval time**: 0.1753 ms
- **Reasoned optimization**:
  - Observed: train_rmse=2.860e-01, val_rmse=7.231e-01 -> val_rmse=7.209e-01
  - Hypothesis: Trees overfitting → cap max_depth.
  - Change: max_depth -> 12

## [V-12] ExtraTrees (raw)
- **Time**: 2026-05-01 16:00
- **Benchmark**: validity
- **Category**: machine_learning
- **Method**: extratrees
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 7.1658e-01
- **Eval time**: 0.2609 ms
- **Reasoned optimization**:
  - Observed: train_rmse=8.250e-14, val_rmse=7.294e-01 -> val_rmse=7.166e-01
  - Hypothesis: Trees overfitting → cap max_depth.
  - Change: max_depth -> 12

## [V-13] HistGB (raw)
- **Time**: 2026-05-01 16:01
- **Benchmark**: validity
- **Category**: machine_learning
- **Method**: hgb
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 7.3501e-01
- **Eval time**: 0.0528 ms
- **Reasoned optimization**:
  - Observed: train_rmse=2.427e-01, val_rmse=7.350e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-14] MLP (log)
- **Time**: 2026-05-01 16:01
- **Benchmark**: validity
- **Category**: machine_learning
- **Method**: mlp
- **Parameterization**: log_4d
- **Loss (RMSE log10(mm))**: 8.0566e-01
- **Eval time**: 0.0004 ms
- **Reasoned optimization**:
  - Observed: train_rmse=7.854e-01, val_rmse=8.057e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-15] KRR-RBF (interaction)
- **Time**: 2026-05-01 16:01
- **Benchmark**: validity
- **Category**: kernel/gp
- **Method**: krr_rbf
- **Parameterization**: interaction_6d
- **Loss (RMSE log10(mm))**: 2.8393e+00
- **Eval time**: 0.0020 ms
- **Reasoned optimization**:
  - Observed: train_rmse=1.799e-01, val_rmse=6.356e+00 -> val_rmse=2.839e+00
  - Hypothesis: Validation error suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-05

## [V-16] SVR-RBF (log)
- **Time**: 2026-05-01 16:01
- **Benchmark**: validity
- **Category**: kernel/gp
- **Method**: svr_rbf
- **Parameterization**: log_4d
- **Loss (RMSE log10(mm))**: 9.0622e-01
- **Eval time**: 0.0093 ms
- **Reasoned optimization**:
  - Observed: train_rmse=7.488e-01, val_rmse=9.062e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-17] Ridge (boundary-distance)
- **Time**: 2026-05-01 16:01
- **Benchmark**: validity
- **Category**: symbolic/analytical
- **Method**: ridge
- **Parameterization**: boundary_distance_6d
- **Loss (RMSE log10(mm))**: 8.1234e-01
- **Eval time**: 0.0002 ms
- **Reasoned optimization**:
  - Observed: train_rmse=8.958e-01, val_rmse=8.123e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-18] HistGB (boundary-distance)
- **Time**: 2026-05-01 16:01
- **Benchmark**: validity
- **Category**: machine_learning
- **Method**: hgb
- **Parameterization**: boundary_distance_6d
- **Loss (RMSE log10(mm))**: 8.0293e-01
- **Eval time**: 0.0480 ms
- **Reasoned optimization**:
  - Observed: train_rmse=2.432e-01, val_rmse=8.029e-01
  - Hypothesis: N/A
  - Change: N/A

## [V-19] PySR (log features)
- **Time**: 2026-05-01 16:01
- **Benchmark**: validity
- **Category**: symbolic/analytical
- **Method**: PySR
- **Parameterization**: log_4d
- **Loss (RMSE log10(mm))**: 8.0058e-01

## [V-20] gplearn (raw)
- **Time**: 2026-05-01 16:02
- **Benchmark**: validity
- **Category**: symbolic/analytical
- **Method**: gplearn
- **Parameterization**: raw_4d
- **Loss (RMSE log10(mm))**: 8.5219e-01
