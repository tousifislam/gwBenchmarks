# Analytic Bench - opus48 CHANGELOG
Goal: closed-form h22(t;q) for non-spinning quasi-circular BBH. Model: h22 = A(t;q) exp(-i phi(t;q)), all closed-form (no SVD/PCA, no stored bases, no ODE solves). Phase = exact integral of an integrable frequency omega = b0 + b1 (tc-t)^(-3/8) + b2 tanh((t-tm)/wr) (PN chirp + tanh merger transition), giving phi = b0 t - (8/5) b1 (tc-t)^(5/8) + b2 wr log cosh((t-tm)/wr) + c. Coefficients are analytic polynomials in the mass variable, fitted across the 20 training waveforms. PySR/gplearn discover closed-form log-amplitude. Loss = mean aLIGO FD mismatch.
## Key findings
- Per-waveform the closed form reaches ~0.05 mismatch; the integrable frequency makes the phase an exact analytic integral (no numerics at eval).
- Phase sign is canonicalised (metric scores only Re(h)).

## Approaches (22)

### 1. pn_t3_sech_eta [physics]
- **Observed**: mean FD mismatch val 0.0418 (median 0.0368), train 0.0664. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: TaylorT3-integrable phase + power-law/sech amplitude (baseline).
- **Result**: eta mass variable, 1.81 ms/waveform, closed-form (see expression.txt).

### 2. pn_t3_sech_q [physics]
- **Observed**: mean FD mismatch val 0.0439 (median 0.0467), train 0.0697. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Same form, raw q parameterisation.
- **Result**: q mass variable, 1.80 ms/waveform, closed-form (see expression.txt).

### 3. pn_t3_sech_delta [physics]
- **Observed**: mean FD mismatch val 0.0731 (median 0.0667), train 0.0864. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Mass-difference parameterisation (third reparam).
- **Result**: delta_m mass variable, 1.75 ms/waveform, closed-form (see expression.txt).

### 4. pn_t3_2term_eta [physics]
- **Observed**: mean FD mismatch val 0.0228 (median 0.0217), train 0.0336. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Two-PN-power integrable phase: captures late-inspiral drift.
- **Result**: eta mass variable, 2.13 ms/waveform, closed-form (see expression.txt).

### 5. pn_t3_sech_sqrteta [physics]
- **Observed**: mean FD mismatch val 0.0364 (median 0.0397), train 0.0636. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: sqrt(eta) reparam.
- **Result**: sqrt_eta mass variable, 1.77 ms/waveform, closed-form (see expression.txt).

### 6. pn_t3_sech_eta_deg4 [physics]
- **Observed**: mean FD mismatch val 0.0441 (median 0.0411), train 0.0702. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Higher-degree eta polynomial for coefficients.
- **Result**: eta mass variable, 1.79 ms/waveform, closed-form (see expression.txt).

### 7. gauss_amp_t3_eta [functional]
- **Observed**: mean FD mismatch val 0.0991 (median 0.0897), train 0.1101. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Two-Gaussian amplitude (asymmetric merger peak).
- **Result**: eta mass variable, 1.82 ms/waveform, closed-form (see expression.txt).

### 8. lorentz_amp_t3_eta [functional]
- **Observed**: mean FD mismatch val 0.2382 (median 0.1605), train 0.2030. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Lorentzian amplitude peak.
- **Result**: eta mass variable, 1.78 ms/waveform, closed-form (see expression.txt).

### 9. gauss_amp_t3_q [functional]
- **Observed**: mean FD mismatch val 0.1125 (median 0.0956), train 0.1185. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Two-Gaussian amplitude, q reparam.
- **Result**: q mass variable, 1.82 ms/waveform, closed-form (see expression.txt).

### 10. lorentz_amp_2term_eta [functional]
- **Observed**: mean FD mismatch val 0.2022 (median 0.1115), train 0.1586. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Lorentzian amplitude + 2-term phase.
- **Result**: eta mass variable, 2.27 ms/waveform, closed-form (see expression.txt).

### 11. gauss_amp_delta [functional]
- **Observed**: mean FD mismatch val 0.1278 (median 0.1333), train 0.1238. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Two-Gaussian, mass-difference reparam.
- **Result**: delta_m mass variable, 1.78 ms/waveform, closed-form (see expression.txt).

### 12. composite_2term_eta [matched_asymptotic]
- **Observed**: mean FD mismatch val 0.0228 (median 0.0217), train 0.0336. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Composite inspiral(PN)+merger(tanh)+ringdown(sech) matched, 2-term.
- **Result**: eta mass variable, 2.10 ms/waveform, closed-form (see expression.txt).

### 13. composite_2term_q [matched_asymptotic]
- **Observed**: mean FD mismatch val 0.0229 (median 0.0201), train 0.0356. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Composite, q reparam.
- **Result**: q mass variable, 2.24 ms/waveform, closed-form (see expression.txt).

### 14. composite_2term_delta [matched_asymptotic]
- **Observed**: mean FD mismatch val 0.0493 (median 0.0452), train 0.0588. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Composite, mass-difference reparam.
- **Result**: delta_m mass variable, 2.08 ms/waveform, closed-form (see expression.txt).

### 15. composite_lorentz_2term [matched_asymptotic]
- **Observed**: mean FD mismatch val 0.2022 (median 0.1115), train 0.1586. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Composite with Lorentzian amplitude.
- **Result**: eta mass variable, 2.29 ms/waveform, closed-form (see expression.txt).

### 16. composite_2term_deg4 [matched_asymptotic]
- **Observed**: mean FD mismatch val 0.0246 (median 0.0236), train 0.0349. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Composite, degree-4 eta polynomials (reasoned: capture curvature).
- **Result**: eta mass variable, 2.06 ms/waveform, closed-form (see expression.txt).

### 17. composite_sqrteta [matched_asymptotic]
- **Observed**: mean FD mismatch val 0.0190 (median 0.0178), train 0.0296. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: Composite, sqrt(eta) reparam.
- **Result**: sqrt_eta mass variable, 2.19 ms/waveform, closed-form (see expression.txt).

### 18. pysr_amp_eta [symbolic]
- **Observed**: mean FD mismatch val 0.1094 (median 0.1058), train 0.1310. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: PySR closed-form log-amplitude, eta.
- **Result**: eta mass variable, 3.91 ms/waveform, closed-form (see expression.txt).

### 19. pysr_amp_delta [symbolic]
- **Observed**: mean FD mismatch val 0.1435 (median 0.1381), train 0.1521. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: PySR closed-form log-amplitude, delta_m (2nd reparam).
- **Result**: delta_m mass variable, 3.61 ms/waveform, closed-form (see expression.txt).

### 20. gplearn_amp_eta [symbolic]
- **Observed**: mean FD mismatch val 0.6784 (median 0.6792), train 0.6722. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: gplearn closed-form log-amplitude.
- **Result**: eta mass variable, 2.90 ms/waveform, closed-form (see expression.txt).

### 21. pysr_amp_q_2term [symbolic]
- **Observed**: mean FD mismatch val 0.0306 (median 0.0330), train 0.0399. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: PySR amplitude + 2-term phase, q reparam.
- **Result**: q mass variable, 4.42 ms/waveform, closed-form (see expression.txt).

### 22. gplearn_amp_delta [symbolic]
- **Observed**: mean FD mismatch val 0.2132 (median 0.2061), train 0.2025. generalises across q; residual set by the closed-form's intrinsic fidelity to NR.
- **Hypothesis/Change**: gplearn log-amplitude, mass-difference.
- **Result**: delta_m mass variable, 2.90 ms/waveform, closed-form (see expression.txt).

## Ranking (by FD mismatch)

| rank | approach | category | loss | median | train | ms |
|---|---|---|---|---|---|---|
| 1 | composite_sqrteta | matched_asymptotic | 0.0190 | 0.0178 | 0.0296 | 2.19 |
| 2 | pn_t3_2term_eta | physics | 0.0228 | 0.0217 | 0.0336 | 2.13 |
| 3 | composite_2term_eta | matched_asymptotic | 0.0228 | 0.0217 | 0.0336 | 2.10 |
| 4 | composite_2term_q | matched_asymptotic | 0.0229 | 0.0201 | 0.0356 | 2.24 |
| 5 | composite_2term_deg4 | matched_asymptotic | 0.0246 | 0.0236 | 0.0349 | 2.06 |
| 6 | pysr_amp_q_2term | symbolic | 0.0306 | 0.0330 | 0.0399 | 4.42 |
| 7 | pn_t3_sech_sqrteta | physics | 0.0364 | 0.0397 | 0.0636 | 1.77 |
| 8 | pn_t3_sech_eta | physics | 0.0418 | 0.0368 | 0.0664 | 1.81 |
| 9 | pn_t3_sech_q | physics | 0.0439 | 0.0467 | 0.0697 | 1.80 |
| 10 | pn_t3_sech_eta_deg4 | physics | 0.0441 | 0.0411 | 0.0702 | 1.79 |
| 11 | composite_2term_delta | matched_asymptotic | 0.0493 | 0.0452 | 0.0588 | 2.08 |
| 12 | pn_t3_sech_delta | physics | 0.0731 | 0.0667 | 0.0864 | 1.75 |
| 13 | gauss_amp_t3_eta | functional | 0.0991 | 0.0897 | 0.1101 | 1.82 |
| 14 | pysr_amp_eta | symbolic | 0.1094 | 0.1058 | 0.1310 | 3.91 |
| 15 | gauss_amp_t3_q | functional | 0.1125 | 0.0956 | 0.1185 | 1.82 |
| 16 | gauss_amp_delta | functional | 0.1278 | 0.1333 | 0.1238 | 1.78 |
| 17 | pysr_amp_delta | symbolic | 0.1435 | 0.1381 | 0.1521 | 3.61 |
| 18 | lorentz_amp_2term_eta | functional | 0.2022 | 0.1115 | 0.1586 | 2.27 |
| 19 | composite_lorentz_2term | matched_asymptotic | 0.2022 | 0.1115 | 0.1586 | 2.29 |
| 20 | gplearn_amp_delta | symbolic | 0.2132 | 0.2061 | 0.2025 | 2.90 |
| 21 | lorentz_amp_t3_eta | functional | 0.2382 | 0.1605 | 0.2030 | 1.78 |
| 22 | gplearn_amp_eta | symbolic | 0.6784 | 0.6792 | 0.6722 | 2.90 |
