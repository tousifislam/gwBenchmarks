# Dynamics benchmark changelog

## 2026-05-01T13:52:07 - SVD+Ridge (raw_6d)
- Category: SVD/decomposition-based; validation loss=0.0137164; train loss=0.013607.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:07 - SVD+Poly2 (effspin_loge)
- Category: SVD/decomposition-based; validation loss=0.0164555; train loss=0.0148924.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:10 - SVD+GPR-RBF (raw_6d)
- Category: SVD/decomposition-based; validation loss=0.0159205; train loss=1.92405e-06.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:15 - SVD+GPR-Matern (effspin_loge)
- Category: SVD/decomposition-based; validation loss=0.013369; train loss=0.00821899.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:15 - EIM+PLS (trig_anomaly)
- Category: SVD/decomposition-based; validation loss=0.015463; train loss=0.0148395.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:16 - PySR-Coefs (effspin_loge)
- Category: Symbolic/physics-informed; validation loss=0.0384728; train loss=0.0387551.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:16 - gplearn-Coefs (trig_anomaly)
- Category: Symbolic/physics-informed; validation loss=0.0136555; train loss=0.0134385.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:17 - PN+PolyPatch (log_frequency)
- Category: Symbolic/physics-informed; validation loss=0.012703; train loss=0.00810884.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:17 - Symbolic-Ridge (fully_transformed)
- Category: Symbolic/physics-informed; validation loss=0.0213882; train loss=0.0210642.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:18 - KRR-RBF (raw_6d)
- Category: Interpolation/kernel; validation loss=0.109671; train loss=3.65847e-05.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:18 - KRR-Poly (effspin_loge)
- Category: Interpolation/kernel; validation loss=0.0160342; train loss=0.00940909.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:18 - RBF-Interp (trig_anomaly)
- Category: Interpolation/kernel; validation loss=0.0117062; train loss=6.69038e-08.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:19 - KNN-Correction (log_frequency)
- Category: Interpolation/kernel; validation loss=0.0159524; train loss=0.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:19 - SVR-RBF (fully_transformed)
- Category: Interpolation/kernel; validation loss=0.0205983; train loss=0.0153099.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:20 - MLP-Small (raw_6d)
- Category: Machine learning; validation loss=1.14611; train loss=1.09926.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:20 - MLP-Deep (effspin_loge)
- Category: Machine learning; validation loss=0.607076; train loss=0.57186.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:21 - RandomForest (trig_anomaly)
- Category: Machine learning; validation loss=0.0152514; train loss=0.00980803.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:21 - ExtraTrees (log_frequency)
- Category: Machine learning; validation loss=0.0134109; train loss=1.24798e-15.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:30 - GradBoost (fully_transformed)
- Category: Machine learning; validation loss=0.0118485; train loss=0.004585.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

## 2026-05-01T13:52:30 - Ensemble-Avg (effspin_loge)
- Category: Machine learning; validation loss=0.038524; train loss=0.0388053.
- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.

