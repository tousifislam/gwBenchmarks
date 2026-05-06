# Changelog — gemini25_flash

<!-- 
This is a running log of every approach attempted across all benchmarks.
Append a new entry after each approach. Do not edit previous entries.
This serves as a lab notebook — record results, observations, and reasoning.
-->

<!-- Example entry:

## [W-01] SVD + GPR (waveform)
- **Time**: 2026-04-30 10:15
- **Benchmark**: waveform
- **Method**: SVD decomposition (50 basis vectors) + GPR with Matern-5/2 kernel on each coefficient
- **Parameterization**: eta, chi_eff, chi_p, |chi1|, |chi2|, theta1, theta2
- **Time convention**: t=0 at peak (default)
- **Loss**: 0.0234 (mismatch=0.018, phase_rmse=0.003, log_amp_rmse=0.002)
- **Eval time**: 12.3 ms
- **Score**: 0.0245
- **Key observations**:
  - Phase errors dominate, especially at high chi_p (>0.5)
  - GPR lengthscale analysis: chi_eff is the most informative feature (shortest lengthscale)
  - 50 SVD basis vectors capture 99.9% of variance
  - Training time: 45 min
- **Next idea**: Try neural network instead of GPR — should be faster at inference

-->

<!-- Use prefix codes: W=waveform, R=remnant, D=dynamics, Q=ringdown (QNM), V=validity, A=analytic -->

## [D-01] SVD_GPR_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + GPR
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0822 (rms_relative_error on x(t))
- **Eval time**: 6.25 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-02] SVD_GPR_EtaChiEff (eta_chieff_chia_loge0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + GPR
- **Parameterization**: eta_chieff_chia_loge0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0877 (rms_relative_error on x(t))
- **Eval time**: 6.05 ms
- **Key observations**:
  - Model trained with eta_chieff_chia_loge0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-03] SVD_Polynomial_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + Polynomial
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0833 (rms_relative_error on x(t))
- **Eval time**: 0.42 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-04] SVD_MLP_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + MLP
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.1383 (rms_relative_error on x(t))
- **Eval time**: 0.15 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-05] SVD_RandomForest_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + RandomForest
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0850 (rms_relative_error on x(t))
- **Eval time**: 14.23 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-06] RBFInterp_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + RBFInterp
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0724 (rms_relative_error on x(t))
- **Eval time**: 0.55 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-07] NNInterp_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + NNInterp
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0987 (rms_relative_error on x(t))
- **Eval time**: 25.94 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-08] SVD_XGBoost_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + XGBoost
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0863 (rms_relative_error on x(t))
- **Eval time**: 0.97 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-09] SVD_LightGBM_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + LightGBM
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0790 (rms_relative_error on x(t))
- **Eval time**: 1192.23 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-11] PhenomFit_Poly_EtaChiEff (eta_chieff_chia_loge0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + PhenomenologicalFit_Poly
- **Parameterization**: eta_chieff_chia_loge0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.1583 (rms_relative_error on x(t))
- **Eval time**: 0.73 ms
- **Key observations**:
  - Model trained with eta_chieff_chia_loge0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-12] SVD_KRR_M1M2S1S2 (m1m2_s1zs2z_e0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + KRR
- **Parameterization**: m1m2_s1zs2z_e0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.1193 (rms_relative_error on x(t))
- **Eval time**: 4.29 ms
- **Key observations**:
  - Model trained with m1m2_s1zs2z_e0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-13] RBFInterp_EtaChiEff (eta_chieff_chia_loge0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + RBFInterp
- **Parameterization**: eta_chieff_chia_loge0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0732 (rms_relative_error on x(t))
- **Eval time**: 0.52 ms
- **Key observations**:
  - Model trained with eta_chieff_chia_loge0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-14] SVD_KRR_Raw (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + KRR
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.1141 (rms_relative_error on x(t))
- **Eval time**: 4.28 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-15] SVD_KRR_EtaChiEff (eta_chieff_chia_loge0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + KRR
- **Parameterization**: eta_chieff_chia_loge0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.1119 (rms_relative_error on x(t))
- **Eval time**: 4.19 ms
- **Key observations**:
  - Model trained with eta_chieff_chia_loge0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-16] SVD_Polynomial_Raw_Deg3 (raw_6d parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + Polynomial_Deg3
- **Parameterization**: raw_6d
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0808 (rms_relative_error on x(t))
- **Eval time**: 0.59 ms
- **Key observations**:
  - Model trained with raw_6d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-17] SVD_MLP_EtaChiEff (eta_chieff_chia_loge0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + MLP
- **Parameterization**: eta_chieff_chia_loge0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.1552 (rms_relative_error on x(t))
- **Eval time**: 0.24 ms
- **Key observations**:
  - Model trained with eta_chieff_chia_loge0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-18] SVD_RandomForest_EtaChiEff (eta_chieff_chia_loge0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + RandomForest
- **Parameterization**: eta_chieff_chia_loge0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0797 (rms_relative_error on x(t))
- **Eval time**: 14.21 ms
- **Key observations**:
  - Model trained with eta_chieff_chia_loge0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-19] SVD_XGBoost_EtaChiEff (eta_chieff_chia_loge0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + XGBoost
- **Parameterization**: eta_chieff_chia_loge0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0834 (rms_relative_error on x(t))
- **Eval time**: 0.97 ms
- **Key observations**:
  - Model trained with eta_chieff_chia_loge0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-20] SVD_LightGBM_EtaChiEff (eta_chieff_chia_loge0_zeta0_omega0 parameterization)
- **Time**: 2026-05-02 19:31
- **Benchmark**: dynamics
- **Method**: SVD decomposition (10 basis vectors) + LightGBM
- **Parameterization**: eta_chieff_chia_loge0_zeta0_omega0
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: 0.0756 (rms_relative_error on x(t))
- **Eval time**: 1151.68 ms
- **Key observations**:
  - Model trained with eta_chieff_chia_loge0_zeta0_omega0 parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [D-10] SVD + PySR (Raw Params) - BLOCKED
- **Time**: 2026-05-02 19:31 (approx)
- **Benchmark**: dynamics
- **Method**: Attempted SVD decomposition + PySRRegressor.
- **Key observations**:
  - PySR installation failed due to persistent Julia dependency issues, specifically precompilation errors with the `PostNewtonian` package.
  - The error `MethodError: no method matching make_Expr` suggests incompatibility between `PostNewtonian` and `FastDifferentiation`.
  - This is a recurring issue across benchmarks, making direct use of PySR currently infeasible.
- **Next idea**: Due to blocked PySR, an alternative "Symbolic/analytical" approach will be implemented to fulfill the category requirement. This will likely involve a phenomenological fit or a simple analytical model, similar to what was done for the Waveform and Remnant benchmarks.
