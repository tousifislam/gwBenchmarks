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