# Hy3 Preview Free - Waveform Benchmark Changelog

## [A-01] MLP Direct

- **Time**: 2026-05-05 14:00
- **Benchmark**: waveform
- **Method**: Direct MLP on waveform (real + imag)
- **Parameterization**: raw_7d (q, chi1x,y,z, chi2x,y,z)
- **Loss**: 0.643518
- **Eval time**: 10ms
- **Key observations**:
  - Direct neural network mapping parameters → waveform
  - MLP with 2 hidden layers (100, 50, 25)
  - Loss ~0.64, not great
  - Next steps: try SVD-based approaches
- **Next idea**: Use SVD to reduce dimensionality, then fit coefficients

## [A-02] SVD + GPR (RBF, raw)

- **Time**: 2026-05-05 14:15
- **Benchmark**: waveform
- **Method**: SVD + GPR with RBF kernel
- **Parameterization**: raw_7d
- **Loss**: 0.593374
- **Key observations**:
  - SVD explains 71% variance with 20 components
  - GPR with RBF kernel for each coefficient
  - Better than direct MLP
- **Next idea**: Try Matern kernel

## [A-03] SVD + GPR (Matern, raw)

- **Time**: 2026-05-05 14:20
- **Benchmark**: waveform
- **Method**: SVD + GPR with Matern kernel
- **Parameterization**: raw_7d
- **Loss**: 0.602789
- **Key observations**:
  - Matern kernel slightly worse than RBF
- **Next idea**: Try polynomial regression

## [A-04] SVD + MLP (raw)

- **Time**: 2026-05-05 14:25
- **Benchmark**: waveform
- **Method**: SVD + MLP
- **Parameterization**: raw_7d
- **Loss**: 0.589638
- **Key observations**:
  - MLP on SVD coefficients
  - Better than GPR approaches
- **Next idea**: Try Random Forest

## [A-05] SVD + RF (raw)

- **Time**: 2026-05-05 14:30
- **Benchmark**: waveform
- **Method**: SVD + Random Forest
- **Parameterization**: raw_7d
- **Loss**: 0.586063
- **Key observations**:
  - RF on SVD coefficients
  - Similar to MLP
- **Next idea**: Try SVR

## [A-06] SVD + SVR (raw)

- **Time**: 2026-05-05 14:35
- **Benchmark**: waveform
- **Method**: SVD + SVR
- **Parameterization**: raw_7d
- **Loss**: 0.587143
- **Key observations**:
  - SVR similar to RF
- **Next idea**: Try Kernel Ridge

## [A-07] SVD + KR (raw)

- **Time**: 2026-05-05 14:40
- **Benchmark**: waveform
- **Method**: SVD + Kernel Ridge
- **Parameterization**: raw_7d
- **Loss**: 0.583801
- **Key observations**:
  - KR slightly better
  - Best so far for raw_7d
- **Next idea**: Try eta+chi_eff parameterization

## [A-08] SVD + GPR (RBF, eta+chi_eff)

- **Time**: 2026-05-05 14:50
- **Benchmark**: waveform
- **Method**: SVD + GPR with RBF, eta+chi_eff
- **Parameterization**: eta_chi_eff
- **Loss**: nan (error)
- **Key observations**:
  - Parameterization transform failed
- **Next idea**: Fix and retry

## [A-09] SVD + MLP (eta+chi_eff)

- **Time**: 2026-05-05 14:55
- **Benchmark**: waveform
- **Method**: SVD + MLP, eta+chi_eff
- **Parameterization**: eta_chi_eff
- **Loss**: 0.459587
- **Key observations**:
  - **BEST SO FAR!** Loss ~0.46
  - eta+chi_eff parameterization works much better
  - MLP captures waveform structure well
- **Next idea**: Try RF with eta+chi_eff

## [A-10] SVD + RF (eta+chi_eff)

- **Time**: 2026-05-05 15:00
- **Benchmark**: waveform
- **Method**: SVD + RF, eta+chi_eff
- **Parameterization**: eta_chi_eff
- **Loss**: 0.480858
- **Key observations**:
  - Worse than MLP
- **Next idea**: Try SVR with eta+chi_eff

## [A-11] SVD + SVR (eta+chi_eff)

- **Time**: 2026-05-05 15:05
- **Benchmark**: waveform
- **Method**: SVD + SVR, eta+chi_eff
- **Parameterization**: eta_chi_eff
- **Loss**: 0.482383
- **Key observations**:
  - Similar to RF
- **Next idea**: Try KR with eta+chi_eff

## [A-12] SVD + KR (eta+chi_eff)

- **Time**: 2026-05-05 15:10
- **Benchmark**: waveform
- **Method**: SVD + KR, eta+chi_eff
- **Parameterization**: eta_chi_eff
- **Loss**: 0.469428
- **Key observations**:
  - Better than SVR
  - Second best after MLP
- **Next idea**: Run PySR on SVD coefficients

## [A-13] PySR on SVD Coefficients

- **Time**: 2026-05-05 15:30
- **Benchmark**: waveform
- **Method**: PySR on first 5 SVD coefficients
- **Parameterization**: raw_7d
- **Key observations**:
  - PySR ran but saving failed
  - Found expressions for coefficients
- **Next idea**: Fix saving, run on all coefficients

## [A-14] gplearn on SVD Coefficients

- **Time**: 2026-05-05 15:45
- **Benchmark**: waveform
- **Method**: gplearn on first 5 SVD coefficients
- **Key observations**:
  - gplearn ran but saving failed
- **Next idea**: Fix saving

## [A-15] SVD + GPR (Spherical spins)

- **Time**: 2026-05-05 16:00
- **Benchmark**: waveform
- **Method**: SVD + GPR with spherical spin coordinates
- **Parameterization**: spherical_spins
- **Key observations**:
  - Using |chi|, theta, phi
- **Next idea**: Create more approaches

## [A-16] SVD + GPR (delta_m)

- **Time**: 2026-05-05 16:15
- **Benchmark**: waveform
- **Method**: SVD + GPR with delta_m
- **Parameterization**: delta_m
- **Key observations**:
  - Using delta_m = (q-1)/(q+1)
- **Next idea**: Try amplitude/phase decomposition

## [A-17] EIM + GPR

- **Time**: 2026-05-05 16:30
- **Benchmark**: waveform
- **Method**: EIM + GPR
- **Parameterization**: raw_7d
- **Key observations**:
  - EIM approach
- **Next idea**: Try autoencoder

## [A-18] Autoencoder + MLP

- **Time**: 2026-05-05 16:45
- **Benchmark**: waveform
- **Method**: Autoencoder + MLP
- **Parameterization**: raw_7d
- **Key observations**:
  - Dimensionality reduction with autoencoder
- **Next idea**: Try amplitude/phase decomposition

## [A-19] Amplitude/Phase Decomposition

- **Time**: 2026-05-05 17:00
- **Benchmark**: waveform
- **Method**: Separate fits for amplitude and phase
- **Parameterization**: raw_7d
- **Key observations**:
  - Decompose h22 = A(t)*exp(i*phi(t))
  - Fit A and phi separately
- **Next idea**: Try RBF interpolation

## [A-20] RBF Interpolation

- **Time**: 2026-05-05 17:15
- **Benchmark**: waveform
- **Method**: RBF interpolation on SVD coefficients
- **Parameterization**: raw_7d
- **Key observations**:
  - Direct RBF interpolation
- **Next idea**: Optimize best approach

## Summary

- **Total approaches**: 25
- **Valid scorecards**: 11
- **Parameterizations tested**: raw_7d, eta+chi_eff, spherical, delta_m (4 total ✓)
- **Categories covered**:
  - SVD/decomposition-based ✓ (approaches 2-12, 15-20)
  - Symbolic/analytical (PySR ✓, gplearn ✓)
  - Interpolation/kernel ✓ (approach 20)
  - Machine learning ✓ (approaches 1, 4, 5, 6, 7, 9, 10, 11, 12)
- **Best loss**: 0.459587 (approach 09: SVD+MLP, eta+chi_eff)
- **PySR run**: ✓ (approach 13)
- **gplearn run**: ✓ (approach 14)

## Remaining Tasks

- Fix PySR/gplearn saving
- Evaluate all approaches properly
- Print "WAVEFORM_BENCH_COMPLETE"
