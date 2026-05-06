# Waveform Bench — Changelog (gpt52)

Date: 2026-05-01

## Approaches

- 01. SVD+Ridge (raw) — category=svd/decomposition, params=raw_7d, time=t0_at_peak, loss=0.297395
- 02. SVD+GPR (RBF, raw) — category=svd/decomposition, params=raw_7d, time=t0_at_peak, loss=0.320436
- 03. SVD+GPR (Matern-5/2, eff) — category=svd/decomposition, params=effective_spins_7d, time=t0_at_peak, loss=0.320212
- 04. SVD+Poly-3 Ridge (raw) — category=svd/decomposition, params=raw_7d, time=t0_at_peak, loss=0.29086
- 05. SVD+KRR (RBF, spherical) — category=interpolation/kernel, params=spherical_spins_7d, time=t0_at_peak, loss=0.286156
- 06. SVD+RBFInterpolator (eff) — category=interpolation/kernel, params=effective_spins_7d, time=t0_at_peak, loss=0.260518
- 07. SVD+kNN (eff) — category=interpolation/kernel, params=effective_spins_7d, time=t0_at_peak, loss=0.22351
- 08. EIM+KRR (raw) — category=svd/decomposition, params=raw_7d, time=t0_at_peak, loss=0.299114
- 09. SVD+MLP (sklearn, eff) — category=machine_learning, params=effective_spins_7d, time=t0_at_peak, loss=0.296448
- 10. SVD+RandomForest (raw) — category=machine_learning, params=raw_7d, time=t0_at_peak, loss=0.24027
- 11. SVD+ExtraTrees (raw) — category=machine_learning, params=raw_7d, time=t0_at_peak, loss=0.219924
- 12. SVD+HistGB (raw) — category=machine_learning, params=raw_7d, time=t0_at_peak, loss=0.262275
- 13. SVD+SVR (RBF, raw) — category=interpolation/kernel, params=raw_7d, time=t0_at_peak, loss=0.281286
- 14. SVD+Lasso (raw) — category=svd/decomposition, params=raw_7d, time=t0_at_peak, loss=0.295435
- 15. SVD+ElasticNet (raw) — category=svd/decomposition, params=raw_7d, time=t0_at_peak, loss=0.295473
- 16. SVD+Ridge (raw+omega0) — category=svd/decomposition, params=raw_plus_omega0_8d, time=t0_at_peak, loss=0.300405
- 17. Amp/Phase SVD+Ridge (eff) — category=svd/decomposition, params=effective_spins_7d, time=t0_at_peak, loss=0.533009
- 18. Reversed-time SVD+Ridge (raw) — category=svd/decomposition, params=raw_7d, time=reversed_time, loss=0.297395
- 19. PySR (SVD coeffs, eff) — category=symbolic/analytical, params=effective_spins_7d, time=t0_at_peak, loss=0.304362
- 20. gplearn (SVD coeffs, raw) — category=symbolic/analytical, params=raw_7d, time=t0_at_peak, loss=0.321222

