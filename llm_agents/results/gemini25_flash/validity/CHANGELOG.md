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

## [V-01] GPR_RBF_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: GPR RBF Raw
- **Parameterization**: raw_4d
- **Loss**: 4.8620 (RMSE on log10(mismatch))
- **Eval time**: 2.10 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-02] GPR_Matern_EffectiveSpins (effective_spins parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: GPR Matern EffectiveSpins
- **Parameterization**: effective_spins
- **Loss**: 3.6370 (RMSE on log10(mismatch))
- **Eval time**: 2.23 ms
- **Key observations**:
  - Model trained with effective_spins parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-03] KRR_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: KRR Raw
- **Parameterization**: raw_4d
- **Loss**: 0.8029 (RMSE on log10(mismatch))
- **Eval time**: 0.99 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-04] SVR_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: SVR Raw
- **Parameterization**: raw_4d
- **Loss**: 0.8513 (RMSE on log10(mismatch))
- **Eval time**: 3.27 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-05] PhenomFit_Poly_EtaChiEff (effective_spins parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: Phenom Fit Poly EtaChiEff
- **Parameterization**: effective_spins
- **Loss**: 0.8066 (RMSE on log10(mismatch))
- **Eval time**: 0.26 ms
- **Key observations**:
  - Model trained with effective_spins parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-06] RBFInterp_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: RBFInterp Raw
- **Parameterization**: raw_4d
- **Loss**: 5.7588 (RMSE on log10(mismatch))
- **Eval time**: 1.32 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-07] NNInterp_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: Nearest Neighbor Interpolation Raw
- **Parameterization**: raw_4d
- **Loss**: 0.8071 (RMSE on log10(mismatch))
- **Eval time**: 21.28 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-08] MLP_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: MLP Raw
- **Parameterization**: raw_4d
- **Loss**: 0.8051 (RMSE on log10(mismatch))
- **Eval time**: 0.55 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-09] RandomForest_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: RandomForest Raw
- **Parameterization**: raw_4d
- **Loss**: 0.7248 (RMSE on log10(mismatch))
- **Eval time**: 15.71 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-10] XGBoost_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: XGBoost Raw
- **Parameterization**: raw_4d
- **Loss**: 0.7738 (RMSE on log10(mismatch))
- **Eval time**: 1.02 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-11] LightGBM_Raw (raw_4d parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: LightGBM Raw
- **Parameterization**: raw_4d
- **Loss**: 0.7353 (RMSE on log10(mismatch))
- **Eval time**: 1.57 ms
- **Key observations**:
  - Model trained with raw_4d parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-12] KRR_LogMassRatio (log_mass_ratio parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: KRR LogMassRatio
- **Parameterization**: log_mass_ratio
- **Loss**: 0.8081 (RMSE on log10(mismatch))
- **Eval time**: 1.09 ms
- **Key observations**:
  - Model trained with log_mass_ratio parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-13] XGBoost_LogMassRatio (log_mass_ratio parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: XGBoost LogMassRatio
- **Parameterization**: log_mass_ratio
- **Loss**: 0.7918 (RMSE on log10(mismatch))
- **Eval time**: 1.07 ms
- **Key observations**:
  - Model trained with log_mass_ratio parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-14] MLP_LogMassRatio (log_mass_ratio parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: MLP LogMassRatio
- **Parameterization**: log_mass_ratio
- **Loss**: 0.7990 (RMSE on log10(mismatch))
- **Eval time**: 0.57 ms
- **Key observations**:
  - Model trained with log_mass_ratio parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-15] GPR_RBF_InteractionTerms (interaction_terms parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: GPR RBF InteractionTerms
- **Parameterization**: interaction_terms
- **Loss**: 4.8615 (RMSE on log10(mismatch))
- **Eval time**: 2.10 ms
- **Key observations**:
  - Model trained with interaction_terms parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-16] RandomForest_InteractionTerms (interaction_terms parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: RandomForest InteractionTerms
- **Parameterization**: interaction_terms
- **Loss**: 0.7742 (RMSE on log10(mismatch))
- **Eval time**: 15.88 ms
- **Key observations**:
  - Model trained with interaction_terms parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-17] KRR_EffectiveSpins (effective_spins parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: KRR EffectiveSpins
- **Parameterization**: effective_spins
- **Loss**: 0.8064 (RMSE on log10(mismatch))
- **Eval time**: 1.17 ms
- **Key observations**:
  - Model trained with effective_spins parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-18] XGBoost_EffectiveSpins (effective_spins parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: XGBoost EffectiveSpins
- **Parameterization**: effective_spins
- **Loss**: 0.8022 (RMSE on log10(mismatch))
- **Eval time**: 1.10 ms
- **Key observations**:
  - Model trained with effective_spins parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-19] LightGBM_EffectiveSpins (effective_spins parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: LightGBM EffectiveSpins
- **Parameterization**: effective_spins
- **Loss**: 0.7805 (RMSE on log10(mismatch))
- **Eval time**: 1.54 ms
- **Key observations**:
  - Model trained with effective_spins parameters.
- **Next idea**: Continue implementing other models and reparameterizations.

## [V-20] Polynomial_Fit_InteractionTerms (interaction_terms parameterization)
- **Time**: 2026-05-02 20:19
- **Benchmark**: validity
- **Method**: Polynomial  Fit InteractionTerms
- **Parameterization**: interaction_terms
- **Loss**: 0.8139 (RMSE on log10(mismatch))
- **Eval time**: 0.23 ms
- **Key observations**:
  - Model trained with interaction_terms parameters.
- **Next idea**: Continue implementing other models and reparameterizations.
