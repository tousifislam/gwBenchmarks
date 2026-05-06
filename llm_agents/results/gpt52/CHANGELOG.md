# Changelog — gpt52

<!--
Append a new entry after each approach. Do not edit previous entries.
Prefix codes: W=waveform, R=remnant, D=dynamics, Q=ringdown, V=validity, A=analytic
-->


## [W-01] SVD+Ridge (raw)
- **Time**: 2026-05-01 14:45
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + ridge
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.9740e-01
- **Eval time**: 0.10 ms
- **Reasoned optimization**:
  - Observed: train_loss=3.019e-01, val_loss=2.955e-01
  - Hypothesis: Both losses high → increase basis size to capture more variance.
  - Change: n_svd -> 50

## [W-02] SVD+GPR (RBF, raw)
- **Time**: 2026-05-01 14:45
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + gpr_rbf
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 3.2044e-01
- **Eval time**: 0.11 ms
- **Reasoned optimization**:
  - Observed: train_loss=2.036e-01, val_loss=3.204e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-03] SVD+GPR (Matern-5/2, eff)
- **Time**: 2026-05-01 14:46
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + gpr_matern
- **Parameterization**: effective_spins_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 3.2021e-01
- **Eval time**: 0.12 ms
- **Reasoned optimization**:
  - Observed: train_loss=2.037e-01, val_loss=3.202e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-04] SVD+Poly-3 Ridge (raw)
- **Time**: 2026-05-01 14:46
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + poly_ridge
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.9086e-01
- **Eval time**: 0.12 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.727e-01, val_loss=2.910e-01
  - Hypothesis: Validation loss suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-02

## [W-05] SVD+KRR (RBF, spherical)
- **Time**: 2026-05-01 14:47
- **Benchmark**: waveform
- **Category**: interpolation/kernel
- **Method**: SVD + krr_rbf
- **Parameterization**: spherical_spins_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.8616e-01
- **Eval time**: 0.13 ms
- **Reasoned optimization**:
  - Observed: train_loss=9.420e-02, val_loss=3.030e-01
  - Hypothesis: Validation loss suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-02

## [W-06] SVD+RBFInterpolator (eff)
- **Time**: 2026-05-01 14:48
- **Benchmark**: waveform
- **Category**: interpolation/kernel
- **Method**: SVD + rbf_interp
- **Parameterization**: effective_spins_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.6052e-01
- **Eval time**: 0.27 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.780e-01, val_loss=2.605e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-07] SVD+kNN (eff)
- **Time**: 2026-05-01 14:48
- **Benchmark**: waveform
- **Category**: interpolation/kernel
- **Method**: SVD + knn
- **Parameterization**: effective_spins_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.2351e-01
- **Eval time**: 0.17 ms
- **Reasoned optimization**:
  - Observed: train_loss=9.054e-02, val_loss=2.235e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-08] EIM+KRR (raw)
- **Time**: 2026-05-01 14:49
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: EIM + krr_rbf
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.9911e-01
- **Eval time**: 0.12 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.769e-01, val_loss=3.068e-01
  - Hypothesis: Validation loss suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-02

## [W-09] SVD+MLP (sklearn, eff)
- **Time**: 2026-05-01 14:49
- **Benchmark**: waveform
- **Category**: machine_learning
- **Method**: SVD + mlp_sklearn
- **Parameterization**: effective_spins_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.9645e-01
- **Eval time**: 0.13 ms
- **Reasoned optimization**:
  - Observed: train_loss=2.806e-01, val_loss=2.964e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-10] SVD+RandomForest (raw)
- **Time**: 2026-05-01 14:50
- **Benchmark**: waveform
- **Category**: machine_learning
- **Method**: SVD + rf
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.4027e-01
- **Eval time**: 0.24 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.013e-01, val_loss=2.403e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-11] SVD+ExtraTrees (raw)
- **Time**: 2026-05-01 14:50
- **Benchmark**: waveform
- **Category**: machine_learning
- **Method**: SVD + extratrees
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.1992e-01
- **Eval time**: 0.66 ms
- **Reasoned optimization**:
  - Observed: train_loss=9.054e-02, val_loss=2.199e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-12] SVD+HistGB (raw)
- **Time**: 2026-05-01 14:51
- **Benchmark**: waveform
- **Category**: machine_learning
- **Method**: SVD + hgb
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.6227e-01
- **Eval time**: 1.25 ms
- **Reasoned optimization**:
  - Observed: train_loss=9.452e-02, val_loss=2.623e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-13] SVD+SVR (RBF, raw)
- **Time**: 2026-05-01 14:51
- **Benchmark**: waveform
- **Category**: interpolation/kernel
- **Method**: SVD + svr_rbf
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.8129e-01
- **Eval time**: 0.17 ms
- **Reasoned optimization**:
  - Observed: train_loss=2.068e-01, val_loss=2.813e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-14] SVD+Lasso (raw)
- **Time**: 2026-05-01 14:52
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + lasso
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.9543e-01
- **Eval time**: 0.11 ms
- **Reasoned optimization**:
  - Observed: train_loss=3.020e-01, val_loss=2.954e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-15] SVD+ElasticNet (raw)
- **Time**: 2026-05-01 14:52
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + elasticnet
- **Parameterization**: raw_7d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 2.9547e-01
- **Eval time**: 0.10 ms
- **Reasoned optimization**:
  - Observed: train_loss=3.020e-01, val_loss=2.955e-01
  - Hypothesis: N/A
  - Change: N/A

## [W-16] SVD+Ridge (raw+omega0)
- **Time**: 2026-05-01 14:53
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + ridge
- **Parameterization**: raw_plus_omega0_8d
- **Time convention**: t0_at_peak
- **Representation**: real_imag
- **Loss**: 3.0040e-01
- **Eval time**: 0.10 ms
- **Reasoned optimization**:
  - Observed: train_loss=3.017e-01, val_loss=2.987e-01
  - Hypothesis: Both losses high → increase basis size to capture more variance.
  - Change: n_svd -> 50

## [W-17] Amp/Phase SVD+Ridge (eff)
- **Time**: 2026-05-01 14:53
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + ridge
- **Parameterization**: effective_spins_7d
- **Time convention**: t0_at_peak
- **Representation**: amp_phase
- **Loss**: 5.3301e-01
- **Eval time**: 0.23 ms
- **Reasoned optimization**:
  - Observed: train_loss=5.353e-01, val_loss=5.330e-01
  - Hypothesis: Both losses high → increase basis size to capture more variance.
  - Change: n_svd -> 50

## [W-18] Reversed-time SVD+Ridge (raw)
- **Time**: 2026-05-01 14:54
- **Benchmark**: waveform
- **Category**: svd/decomposition
- **Method**: SVD + ridge
- **Parameterization**: raw_7d
- **Time convention**: reversed_time
- **Representation**: real_imag
- **Loss**: 2.9740e-01
- **Eval time**: 0.10 ms
- **Reasoned optimization**:
  - Observed: train_loss=3.019e-01, val_loss=2.955e-01
  - Hypothesis: Both losses high → increase basis size to capture more variance.
  - Change: n_svd -> 50

## [W-19] PySR (SVD coeffs, eff)
- **Time**: 2026-05-01 15:03
- **Benchmark**: waveform
- **Category**: symbolic/analytical
- **Method**: PySR on first 3 SVD coefficients
- **Parameterization**: effective_spins_7d
- **Loss**: 3.0436e-01
- **Eval time**: 0.05 ms

## [W-20] gplearn (SVD coeffs, raw)
- **Time**: 2026-05-01 15:10
- **Benchmark**: waveform
- **Category**: symbolic/analytical
- **Method**: gplearn SymbolicRegressor on 6 SVD coefficients
- **Parameterization**: raw_7d
- **Loss**: 3.2122e-01
- **Eval time**: 0.04 ms

## [R-01] GPR (RBF, raw)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: kernel/gp
- **Method**: gpr_rbf
- **Parameterization**: raw_7d
- **Loss (NRMSE)**: 1.1625e-01
- **Eval time**: 0.012 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=6.022e-04, val_nrmse=1.163e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-02] GPR (Matern-5/2, eff)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: kernel/gp
- **Method**: gpr_matern
- **Parameterization**: effective_spins_7d
- **Loss (NRMSE)**: 1.1547e-01
- **Eval time**: 0.017 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=3.299e-06, val_nrmse=1.155e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-03] KRR (RBF, pn)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: kernel/gp
- **Method**: krr_rbf
- **Parameterization**: pn_products_5d
- **Loss (NRMSE)**: 1.0310e-01
- **Eval time**: 0.005 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=9.856e-02, val_nrmse=1.031e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-04] SVR (RBF, raw)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: kernel/gp
- **Method**: svr_rbf
- **Parameterization**: raw_7d
- **Loss (NRMSE)**: 1.2348e-01
- **Eval time**: 0.018 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=2.216e-02, val_nrmse=1.235e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-05] RBFInterpolator (eff)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: interpolation
- **Method**: rbf_interp
- **Parameterization**: effective_spins_7d
- **Loss (NRMSE)**: 1.2551e-01
- **Eval time**: 0.253 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=6.545e-06, val_nrmse=1.255e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-06] kNN (eff)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: interpolation
- **Method**: knn
- **Parameterization**: effective_spins_7d
- **Loss (NRMSE)**: 9.5126e-02
- **Eval time**: 0.004 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=0.000e+00, val_nrmse=9.513e-02
  - Hypothesis: N/A
  - Change: N/A

## [R-07] MLP (eff)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: machine_learning
- **Method**: mlp
- **Parameterization**: effective_spins_7d
- **Loss (NRMSE)**: 6.7894e-01
- **Eval time**: 0.000 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=2.969e-01, val_nrmse=6.789e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-08] RandomForest (raw)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: machine_learning
- **Method**: rf
- **Parameterization**: raw_7d
- **Loss (NRMSE)**: 9.2814e-02
- **Eval time**: 0.059 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=3.727e-02, val_nrmse=9.281e-02
  - Hypothesis: Trees overfitting → cap max_depth.
  - Change: max_depth -> 12

## [R-09] ExtraTrees (raw)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: machine_learning
- **Method**: extratrees
- **Parameterization**: raw_7d
- **Loss (NRMSE)**: 9.1409e-02
- **Eval time**: 0.060 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=1.790e-08, val_nrmse=9.141e-02
  - Hypothesis: Trees overfitting → cap max_depth.
  - Change: max_depth -> 12

## [R-10] HistGB (pn)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: machine_learning
- **Method**: hgb
- **Parameterization**: pn_products_5d
- **Loss (NRMSE)**: 1.0430e-01
- **Eval time**: 0.011 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=4.780e-02, val_nrmse=1.043e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-11] Ridge (massdiff+chia)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: symbolic/analytical
- **Method**: ridge
- **Parameterization**: massdiff_chia_5d
- **Loss (NRMSE)**: 1.0784e-01
- **Eval time**: 0.000 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=1.178e-01, val_nrmse=1.078e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-12] Poly-3 Ridge (raw)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: symbolic/analytical
- **Method**: poly_ridge
- **Parameterization**: raw_7d
- **Loss (NRMSE)**: 9.8759e-02
- **Eval time**: 0.001 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=8.377e-02, val_nrmse=9.876e-02
  - Hypothesis: N/A
  - Change: N/A

## [R-13] Poly-5 Ridge (pn)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: symbolic/analytical
- **Method**: poly_ridge
- **Parameterization**: pn_products_5d
- **Loss (NRMSE)**: 1.0231e-01
- **Eval time**: 0.001 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=1.015e-01, val_nrmse=1.023e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-14] Ridge (spherical)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: symbolic/analytical
- **Method**: ridge
- **Parameterization**: spherical_spins_7d
- **Loss (NRMSE)**: 1.0902e-01
- **Eval time**: 0.000 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=1.193e-01, val_nrmse=1.090e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-15] KRR (RBF, raw, alpha=1e-2)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: kernel/gp
- **Method**: krr_rbf
- **Parameterization**: raw_7d
- **Loss (NRMSE)**: 1.0017e-01
- **Eval time**: 0.005 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=2.778e-02, val_nrmse=1.136e-01
  - Hypothesis: Validation error suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-01

## [R-16] SVR (RBF, pn)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: kernel/gp
- **Method**: svr_rbf
- **Parameterization**: pn_products_5d
- **Loss (NRMSE)**: 1.0972e-01
- **Eval time**: 0.026 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=1.008e-01, val_nrmse=1.097e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-17] MLP (raw)
- **Time**: 2026-05-01 15:16
- **Benchmark**: remnant
- **Category**: machine_learning
- **Method**: mlp
- **Parameterization**: raw_7d
- **Loss (NRMSE)**: 9.0633e-01
- **Eval time**: 0.001 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=4.055e-01, val_nrmse=9.063e-01
  - Hypothesis: N/A
  - Change: N/A

## [R-18] HistGB (eff)
- **Time**: 2026-05-01 15:17
- **Benchmark**: remnant
- **Category**: machine_learning
- **Method**: hgb
- **Parameterization**: effective_spins_7d
- **Loss (NRMSE)**: 9.8153e-02
- **Eval time**: 0.015 ms
- **Reasoned optimization**:
  - Observed: train_nrmse=4.392e-02, val_nrmse=9.815e-02
  - Hypothesis: N/A
  - Change: N/A

## [R-19] PySR (eff)
- **Time**: 2026-05-01 15:18
- **Benchmark**: remnant
- **Category**: symbolic/analytical
- **Method**: PySR
- **Parameterization**: effective_spins_7d
- **Loss (NRMSE)**: 9.2608e-02

## [R-20] gplearn (raw)
- **Time**: 2026-05-01 15:19
- **Benchmark**: remnant
- **Category**: symbolic/analytical
- **Method**: gplearn SymbolicRegressor
- **Parameterization**: raw_7d
- **Loss (NRMSE)**: 8.0478e-01

## [D-01] SVD+Ridge (raw, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + ridge
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 1.3500e-02
- **Eval time**: 0.01 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.344e-02, val_loss=1.350e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-02] SVD+GPR (RBF, raw, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + gpr_rbf
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 1.0032e-02
- **Eval time**: 0.08 ms
- **Reasoned optimization**:
  - Observed: train_loss=8.288e-03, val_loss=1.003e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-03] SVD+GPR (Matern, eff, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + gpr_matern
- **Parameterization**: eff_loge0_6d
- **Time convention**: normalized_time
- **Loss**: 1.1235e-02
- **Eval time**: 0.03 ms
- **Reasoned optimization**:
  - Observed: train_loss=8.341e-03, val_loss=1.123e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-04] SVD+Poly-3 Ridge (raw, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + poly_ridge
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 1.0826e-02
- **Eval time**: 0.02 ms
- **Reasoned optimization**:
  - Observed: train_loss=8.638e-03, val_loss=1.115e-02
  - Hypothesis: Validation loss suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-02

## [D-05] SVD+KRR (trig, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: interpolation/kernel
- **Method**: SVD + krr_rbf
- **Parameterization**: trig_anomaly_7d
- **Time convention**: normalized_time
- **Loss**: 1.2173e-02
- **Eval time**: 0.04 ms
- **Reasoned optimization**:
  - Observed: train_loss=6.861e-03, val_loss=1.346e-02
  - Hypothesis: Validation loss suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-02

## [D-06] SVD+RBFInterpolator (eff, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: interpolation/kernel
- **Method**: SVD + rbf_interp
- **Parameterization**: eff_loge0_6d
- **Time convention**: normalized_time
- **Loss**: 1.0421e-02
- **Eval time**: 0.65 ms
- **Reasoned optimization**:
  - Observed: train_loss=7.644e-03, val_loss=1.042e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-07] SVD+kNN (eff, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: interpolation/kernel
- **Method**: SVD + knn
- **Parameterization**: eff_loge0_6d
- **Time convention**: normalized_time
- **Loss**: 2.0457e-02
- **Eval time**: 0.09 ms
- **Reasoned optimization**:
  - Observed: train_loss=6.850e-03, val_loss=2.046e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-08] EIM+KRR (raw, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: EIM + krr_rbf
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 3.3914e-02
- **Eval time**: 0.06 ms
- **Reasoned optimization**:
  - Observed: train_loss=9.085e-03, val_loss=3.246e-02
  - Hypothesis: Validation loss suggests overfitting → increase regularization (alpha).
  - Change: alpha -> 1.00e-02

## [D-09] SVD+MLP (eff, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: machine_learning
- **Method**: SVD + mlp
- **Parameterization**: eff_loge0_6d
- **Time convention**: normalized_time
- **Loss**: 1.8031e-02
- **Eval time**: 0.06 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.413e-02, val_loss=1.803e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-10] SVD+RandomForest (raw, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: machine_learning
- **Method**: SVD + rf
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 1.5076e-02
- **Eval time**: 0.18 ms
- **Reasoned optimization**:
  - Observed: train_loss=8.796e-03, val_loss=1.508e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-11] SVD+ExtraTrees (raw, tau)
- **Time**: 2026-05-01 15:27
- **Benchmark**: dynamics
- **Category**: machine_learning
- **Method**: SVD + extratrees
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 1.2766e-02
- **Eval time**: 0.25 ms
- **Reasoned optimization**:
  - Observed: train_loss=6.850e-03, val_loss=1.277e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-12] SVD+HistGB (eff, tau)
- **Time**: 2026-05-01 15:28
- **Benchmark**: dynamics
- **Category**: machine_learning
- **Method**: SVD + hgb
- **Parameterization**: eff_loge0_6d
- **Time convention**: normalized_time
- **Loss**: 1.1747e-02
- **Eval time**: 2.24 ms
- **Reasoned optimization**:
  - Observed: train_loss=7.151e-03, val_loss=1.175e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-13] SVD+SVR (raw, tau)
- **Time**: 2026-05-01 15:28
- **Benchmark**: dynamics
- **Category**: interpolation/kernel
- **Method**: SVD + svr_rbf
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 1.1873e-02
- **Eval time**: 0.07 ms
- **Reasoned optimization**:
  - Observed: train_loss=8.228e-03, val_loss=1.187e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-14] SVD+Lasso (raw, tau)
- **Time**: 2026-05-01 15:28
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + lasso
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 1.3484e-02
- **Eval time**: 0.01 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.343e-02, val_loss=1.348e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-15] SVD+ElasticNet (raw, tau)
- **Time**: 2026-05-01 15:28
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + elasticnet
- **Parameterization**: raw_6d
- **Time convention**: normalized_time
- **Loss**: 1.3490e-02
- **Eval time**: 0.01 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.343e-02, val_loss=1.349e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-16] SVD+Ridge (fully, tau)
- **Time**: 2026-05-01 15:28
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + ridge
- **Parameterization**: fully_transformed_7d
- **Time convention**: normalized_time
- **Loss**: 2.1292e-02
- **Eval time**: 0.01 ms
- **Reasoned optimization**:
  - Observed: train_loss=2.102e-02, val_loss=2.129e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-17] SVD+Ridge (raw, t_end)
- **Time**: 2026-05-01 15:28
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + ridge
- **Parameterization**: raw_6d
- **Time convention**: t0_at_end
- **Loss**: 1.9402e-02
- **Eval time**: 0.01 ms
- **Reasoned optimization**:
  - Observed: train_loss=1.859e-02, val_loss=1.940e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-18] SVD+GPR (RBF, raw, t_end)
- **Time**: 2026-05-01 15:28
- **Benchmark**: dynamics
- **Category**: svd/decomposition
- **Method**: SVD + gpr_rbf
- **Parameterization**: raw_6d
- **Time convention**: t0_at_end
- **Loss**: 1.2989e-02
- **Eval time**: 0.03 ms
- **Reasoned optimization**:
  - Observed: train_loss=7.862e-03, val_loss=1.299e-02
  - Hypothesis: N/A
  - Change: N/A

## [D-19] PySR (SVD coeffs, eff)
- **Time**: 2026-05-01 15:30
- **Benchmark**: dynamics
- **Category**: symbolic/physics-informed
- **Method**: PySR on first 3 SVD coefficients
- **Parameterization**: eff_loge0_6d
- **Loss**: 1.2902e-02

## [D-20] gplearn (SVD coeffs, raw)
- **Time**: 2026-05-01 15:32
- **Benchmark**: dynamics
- **Category**: symbolic/physics-informed
- **Method**: gplearn SymbolicRegressor on 4 SVD coefficients
- **Parameterization**: raw_6d
- **Loss**: 3.7974e-02

## [Q-01] Poly-10 (raw)
- **Time**: 2026-05-01 15:45
- **Benchmark**: ringdown
- **Category**: analytical/classical
- **Method**: poly
- **Spin parameterization**: raw_a
- **Mode**: l2/m+2/n0
- **Loss**: 2.1220e+00
- **Eval time**: 0.0000 ms
- **Reasoned optimization**:
  - Observed: val_loss=9.794e-01 -> 2.122e+00
  - Hypothesis: High-degree polynomial likely oscillating near a→1 → reduce degree.
  - Change: degree -> 6

## [Q-02] Poly-15 (-log(1-a))
- **Time**: 2026-05-01 15:45
- **Benchmark**: ringdown
- **Category**: analytical/classical
- **Method**: poly
- **Spin parameterization**: log_compact
- **Mode**: l2/m+2/n0
- **Loss**: 2.8861e-03
- **Eval time**: 0.0001 ms
- **Reasoned optimization**:
  - Observed: val_loss=8.176e-03 -> 2.886e-03
  - Hypothesis: High-degree polynomial likely oscillating near a→1 → reduce degree.
  - Change: degree -> 11

## [Q-03] Chebyshev-25 (2a-1)
- **Time**: 2026-05-01 15:45
- **Benchmark**: ringdown
- **Category**: analytical/classical
- **Method**: chebyshev
- **Spin parameterization**: cheb_mapped
- **Mode**: l2/m+2/n0
- **Loss**: 3.2952e-01
- **Eval time**: 0.0001 ms
- **Reasoned optimization**:
  - Observed: val_loss=3.295e-01
  - Hypothesis: N/A
  - Change: N/A

## [Q-04] Chebyshev-35 (2a-1)
- **Time**: 2026-05-01 15:45
- **Benchmark**: ringdown
- **Category**: analytical/classical
- **Method**: chebyshev
- **Spin parameterization**: cheb_mapped
- **Mode**: l2/m+2/n0
- **Loss**: 2.3812e-01
- **Eval time**: 0.0002 ms
- **Reasoned optimization**:
  - Observed: val_loss=2.381e-01
  - Hypothesis: N/A
  - Change: N/A

## [Q-05] Rational [5,5] (log)
- **Time**: 2026-05-01 15:45
- **Benchmark**: ringdown
- **Category**: analytical/classical
- **Method**: rational
- **Spin parameterization**: log_compact
- **Mode**: l2/m+2/n0
- **Loss**: 3.6803e-04
- **Eval time**: 0.0001 ms
- **Reasoned optimization**:
  - Observed: val_loss=3.680e-04
  - Hypothesis: N/A
  - Change: N/A

## [Q-06] Rational [7,7] (raw)
- **Time**: 2026-05-01 15:45
- **Benchmark**: ringdown
- **Category**: analytical/classical
- **Method**: rational
- **Spin parameterization**: raw_a
- **Mode**: l2/m+2/n0
- **Loss**: 5.2696e+11
- **Eval time**: 0.0002 ms
- **Reasoned optimization**:
  - Observed: val_loss=5.270e+11
  - Hypothesis: N/A
  - Change: N/A

## [Q-07] CubicSpline (raw)
- **Time**: 2026-05-01 15:45
- **Benchmark**: ringdown
- **Category**: interpolation
- **Method**: spline
- **Spin parameterization**: raw_a
- **Mode**: l2/m+2/n0
- **Loss**: 2.6610e-05
- **Eval time**: 0.0000 ms
- **Reasoned optimization**:
  - Observed: val_loss=2.661e-05
  - Hypothesis: N/A
  - Change: N/A

## [Q-08] CubicSpline (sqrt(1-a^2))
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: interpolation
- **Method**: spline
- **Spin parameterization**: sqrt_irreducible
- **Mode**: l2/m+2/n0
- **Loss**: 3.9327e-07
- **Eval time**: 0.0000 ms
- **Reasoned optimization**:
  - Observed: val_loss=3.933e-07
  - Hypothesis: N/A
  - Change: N/A

## [Q-09] RBFInterp (TPS, raw)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: interpolation
- **Method**: rbf_interp
- **Spin parameterization**: raw_a
- **Mode**: l2/m+2/n0
- **Loss**: 9.2494e-04
- **Eval time**: 0.4279 ms
- **Reasoned optimization**:
  - Observed: val_loss=9.249e-04
  - Hypothesis: N/A
  - Change: N/A

## [Q-10] RBFInterp (cubic, log)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: interpolation
- **Method**: rbf_interp
- **Spin parameterization**: log_compact
- **Mode**: l2/m+2/n0
- **Loss**: 6.3379e-06
- **Eval time**: 0.4158 ms
- **Reasoned optimization**:
  - Observed: val_loss=6.338e-06
  - Hypothesis: N/A
  - Change: N/A

## [Q-11] GPR-RBF (raw)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: machine_learning
- **Method**: gpr_rbf
- **Spin parameterization**: raw_a
- **Mode**: l2/m+2/n0
- **Loss**: 4.1218e-01
- **Eval time**: 0.0097 ms
- **Reasoned optimization**:
  - Observed: val_loss=4.122e-01
  - Hypothesis: N/A
  - Change: N/A

## [Q-12] KRR-RBF (raw)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: machine_learning
- **Method**: krr_rbf
- **Spin parameterization**: raw_a
- **Mode**: l2/m+2/n0
- **Loss**: 6.5350e-01
- **Eval time**: 0.0053 ms
- **Reasoned optimization**:
  - Observed: val_loss=6.535e-01
  - Hypothesis: N/A
  - Change: N/A

## [Q-13] SVR-RBF (log)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: machine_learning
- **Method**: svr_rbf
- **Spin parameterization**: log_compact
- **Mode**: l2/m+2/n0
- **Loss**: 4.1928e-02
- **Eval time**: 0.0066 ms
- **Reasoned optimization**:
  - Observed: val_loss=4.193e-02
  - Hypothesis: N/A
  - Change: N/A

## [Q-14] MLP (log)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: machine_learning
- **Method**: mlp
- **Spin parameterization**: log_compact
- **Mode**: l2/m+2/n0
- **Loss**: 4.9303e-01
- **Eval time**: 0.0009 ms
- **Reasoned optimization**:
  - Observed: val_loss=4.930e-01
  - Hypothesis: N/A
  - Change: N/A

## [Q-15] RandomForest (raw)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: machine_learning
- **Method**: rf
- **Spin parameterization**: raw_a
- **Mode**: l2/m+2/n0
- **Loss**: 4.1533e-03
- **Eval time**: 0.1096 ms
- **Reasoned optimization**:
  - Observed: val_loss=4.153e-03
  - Hypothesis: N/A
  - Change: N/A

## [Q-16] HistGB (raw)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: machine_learning
- **Method**: hgb
- **Spin parameterization**: raw_a
- **Mode**: l2/m+2/n0
- **Loss**: 8.7176e-02
- **Eval time**: 0.1043 ms
- **Reasoned optimization**:
  - Observed: val_loss=8.718e-02
  - Hypothesis: N/A
  - Change: N/A

## [Q-17] Poly-12 (compactified)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: analytical/classical
- **Method**: poly
- **Spin parameterization**: compactified
- **Mode**: l2/m+2/n0
- **Loss**: 7.2941e+05
- **Eval time**: 0.0000 ms
- **Reasoned optimization**:
  - Observed: val_loss=6.146e+07 -> 7.294e+05
  - Hypothesis: High-degree polynomial likely oscillating near a→1 → reduce degree.
  - Change: degree -> 8

## [Q-18] Rational [9,9] (cheb)
- **Time**: 2026-05-01 15:46
- **Benchmark**: ringdown
- **Category**: analytical/classical
- **Method**: rational
- **Spin parameterization**: cheb_mapped
- **Mode**: l2/m+2/n0
- **Loss**: 1.4376e+09
- **Eval time**: 0.0003 ms
- **Reasoned optimization**:
  - Observed: val_loss=1.438e+09
  - Hypothesis: N/A
  - Change: N/A

## [Q-19] PySR (log)
- **Time**: 2026-05-01 15:49
- **Benchmark**: ringdown
- **Category**: symbolic regression
- **Mode**: l2/m+2/n0
- **Loss**: 4.0605e-03

## [Q-20] gplearn (raw)
- **Time**: 2026-05-01 15:51
- **Benchmark**: ringdown
- **Category**: symbolic regression
- **Mode**: l2/m+2/n0
- **Loss**: 2.0510e+01
