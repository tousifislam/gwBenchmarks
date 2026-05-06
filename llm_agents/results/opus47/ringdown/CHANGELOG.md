# Ringdown Benchmark — CHANGELOG

Each entry: observation -> hypothesis -> action -> outcome.

## 04 — cubic_spline_per_mode
- Category: **interpolation**, parameterization: **raw**, mode: **all_modes**
- Loss (rel err omega): **5.6343e-06**, runtime: 0.0051 ms
- Notes: Cubic spline per (l, m, n) mode in spin a.
- **Reasoning**: QNM frequencies are smooth in spin per mode. Cubic spline per (l,m,n) should be very accurate.

## 18 — per_mode_poly_log1ma
- Category: **symbolic**, parameterization: **log_1ma**, mode: **all_modes**
- Loss (rel err omega): **4.6191e-02**, runtime: 0.0039 ms
- Notes: Per-mode polynomial deg-8 in -log(1-a).
- **Reasoning**: At extremal spin a→1, frequencies have logarithmic singularity. The -log(1-a) parameterization stabilizes the polynomial fit near the boundary.

## 19 — per_mode_pade
- Category: **symbolic**, parameterization: **raw**, mode: **all_modes**
- Loss (rel err omega): **9.1364e-02**, runtime: 0.0027 ms
- Notes: Per-mode Padé rational approximant.

## 13 — pade_l2m2n0
- Category: **symbolic**, parameterization: **raw_a**, mode: **l2_m2_n0**
- Loss (rel err omega): **1.6844e-01**, runtime: 0.0010 ms
- Notes: Padé rational P(a)/(1+Q(a)(1-a)) for l2m2n0.

## 23 — rf_raw
- Category: **ml**, parameterization: **raw**, mode: **all_modes**
- Loss (rel err omega): **1.7564e-01**, runtime: 0.0003 ms
- Notes: RF with raw [a, l, m, n] features.

## 10 — rf_lm_diff
- Category: **ml**, parameterization: **lm_diff**, mode: **all_modes**
- Loss (rel err omega): **1.7784e-01**, runtime: 0.0003 ms
- Notes: Random Forest 80 trees on lm_diff (30k subsample).

## 12 — extra_trees_compact
- Category: **ml**, parameterization: **compact**, mode: **all_modes**
- Loss (rel err omega): **3.7185e-01**, runtime: 0.0005 ms
- Notes: ExtraTrees 100 trees on compactified spin.

## 11 — hgbr_all_normalized
- Category: **ml**, parameterization: **all_normalized**, mode: **all_modes**
- Loss (rel err omega): **4.6991e-01**, runtime: 0.0017 ms
- Notes: HistGradientBoosting 200 iters.

## 24 — per_mode_poly_chebyshev
- Category: **symbolic**, parameterization: **chebyshev**, mode: **all_modes**
- Loss (rel err omega): **5.0416e-01**, runtime: 0.0035 ms
- Notes: Per-mode polynomial deg-12 in Chebyshev variable.

## 17 — per_mode_poly10
- Category: **symbolic**, parameterization: **raw**, mode: **all_modes**
- Loss (rel err omega): **6.3998e-01**, runtime: 0.0043 ms
- Notes: Per-mode polynomial deg-10 fit in spin a.

## 21 — pysr_l3m3n0
- Category: **symbolic**, parameterization: **raw_a_only**, mode: **l3_m3_n0**
- Loss (rel err omega): **1.0789e+00**, runtime: 0.0010 ms
- Notes: PySR symbolic regression on l=3,m=3,n=0 mode (second mode reparameterization).

## 02 — poly8_log1ma
- Category: **symbolic**, parameterization: **log_1ma**, mode: **all_modes**
- Loss (rel err omega): **1.0875e+00**, runtime: 0.0021 ms
- Notes: Polynomial deg-8 ridge with log(1-a) reparam.

## 03 — poly8_chebyshev
- Category: **symbolic**, parameterization: **chebyshev**, mode: **all_modes**
- Loss (rel err omega): **1.3496e+00**, runtime: 0.0021 ms
- Notes: Polynomial deg-8 ridge with 2a-1 reparam.

## 14 — knn_all_normalized
- Category: **interpolation**, parameterization: **all_normalized**, mode: **all_modes**
- Loss (rel err omega): **1.4641e+00**, runtime: 0.0010 ms
- Notes: KNN k=3 distance-weighted with normalized features.

## 01 — poly8_raw
- Category: **symbolic**, parameterization: **raw**, mode: **all_modes**
- Loss (rel err omega): **2.1150e+00**, runtime: 0.0021 ms
- Notes: Polynomial deg-8 baseline (joint fit on all 1.2M points).

## 15 — krr_poly8_chebyshev
- Category: **kernel_gp**, parameterization: **chebyshev**, mode: **all_modes**
- Loss (rel err omega): **2.2054e+00**, runtime: 0.3209 ms
- Notes: Kernel Ridge polynomial deg-8 with Chebyshev mapping.

## 08 — pysr_l2m2n0
- Category: **symbolic**, parameterization: **raw_a_only**, mode: **l2_m2_n0**
- Loss (rel err omega): **2.5773e+00**, runtime: 0.0010 ms
- Notes: PySR symbolic regression on l=2,m=2,n=0 mode.

## 20 — mlp_deep_log1ma
- Category: **ml**, parameterization: **log_1ma**, mode: **all_modes**
- Loss (rel err omega): **2.6837e+00**, runtime: 0.0040 ms
- Notes: Deep MLP 256-128-64 with -log(1-a).

## 07 — mlp_lm_diff
- Category: **ml**, parameterization: **lm_diff**, mode: **all_modes**
- Loss (rel err omega): **3.7976e+00**, runtime: 0.0031 ms
- Notes: MLP 128-128-64 on lm_diff features (subsampled training).

## 22 — bayes_ridge_poly6_log1ma
- Category: **ml**, parameterization: **log_1ma**, mode: **all_modes**
- Loss (rel err omega): **4.2516e+00**, runtime: 0.0020 ms
- Notes: Bayesian Ridge poly-6 with log(1-a).

## 16 — mlp_raw
- Category: **ml**, parameterization: **raw**, mode: **all_modes**
- Loss (rel err omega): **5.2306e+00**, runtime: 0.0011 ms
- Notes: MLP 64-64 baseline.

## 05 — gpr_rbf_raw_subsample
- Category: **kernel_gp**, parameterization: **raw**, mode: **all_modes**
- Loss (rel err omega): **5.7735e+00**, runtime: 0.1626 ms
- Notes: GPR RBF with 1500 subsampled points.

## 09 — gplearn_l2m2n0
- Category: **symbolic**, parameterization: **raw_a_only**, mode: **l2_m2_n0**
- Loss (rel err omega): **1.7617e+01**, runtime: 0.0010 ms
- Notes: gplearn SymbolicRegressor on l=2,m=2,n=0 mode.

## 06 — rbf_interp_compact
- Category: **interpolation**, parameterization: **compact**, mode: **all_modes**
- Loss (rel err omega): **5.1369e+10**, runtime: 0.0327 ms
- Notes: Thin-plate-spline RBF on 3000 points.

## Summary

- Best: **cubic_spline_per_mode** (loss=5.6343e-06)
- Total approaches: 24
- Categories: ['interpolation', 'kernel_gp', 'ml', 'symbolic']
- Parameterizations: ['all_normalized', 'chebyshev', 'compact', 'lm_diff', 'log_1ma', 'raw', 'raw_a', 'raw_a_only']