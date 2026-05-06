# Analytic Benchmark - CHANGELOG

## 01 - phys_poly_raw
- Category: physics
- Parameterization: raw
- Loss: 6.387178e-01
- Notes: Polynomial q-curves on a PN-like time basis.

## 02 - phys_poly_eta
- Category: physics
- Parameterization: eta
- Loss: 6.406213e-01
- Notes: Polynomial q-curves in eta on a PN-like time basis.

## 03 - phys_pade_delta
- Category: physics
- Parameterization: delta
- Loss: 6.386667e-01
- Notes: Padé q-curves on a PN-like time basis.

## 04 - phys_logq
- Category: physics
- Parameterization: log_q
- Loss: 6.387985e-01
- Notes: Polynomial q-curves in log(q) on a log-augmented time basis.

## 05 - composite_tanh_raw
- Category: matched_composite
- Parameterization: raw
- Loss: 6.387345e-01
- Notes: Smooth inspiral/merger/ringdown blending.

## 06 - composite_tanh_eta
- Category: matched_composite
- Parameterization: eta
- Loss: 6.386805e-01
- Notes: Smooth blended model in eta.

## 07 - composite_tanh_delta
- Category: matched_composite
- Parameterization: delta
- Loss: 6.388282e-01
- Notes: Smooth blended model in delta.

## 08 - composite_tanh_sqrt_eta
- Category: matched_composite
- Parameterization: sqrt_eta
- Loss: 6.390748e-01
- Notes: Smooth blended model in sqrt(eta).

## 09 - functional_gauss_raw
- Category: functional_optimization
- Parameterization: raw
- Loss: 6.386729e-01
- Notes: Gaussian basis on time with polynomial q-curves.

## 10 - functional_gauss_eta
- Category: functional_optimization
- Parameterization: eta
- Loss: 6.386462e-01
- Notes: Gaussian basis on time with eta q-curves.

## 11 - functional_lorentz_raw
- Category: functional_optimization
- Parameterization: raw
- Loss: 6.387066e-01
- Notes: Lorentzian basis on time with polynomial q-curves.

## 12 - functional_lorentz_delta
- Category: functional_optimization
- Parameterization: delta
- Loss: 6.386346e-01
- Notes: Lorentzian basis on time with delta q-curves.

## 13 - functional_damped_sin_raw
- Category: functional_optimization
- Parameterization: raw
- Loss: 6.386212e-01
- Notes: Damped oscillatory time basis with polynomial q-curves.

## 14 - functional_damped_sin_eta
- Category: functional_optimization
- Parameterization: eta
- Loss: 6.386195e-01
- Notes: Damped oscillatory time basis with eta q-curves.

## 15 - symbolic_pysr_amp_eta
- Category: symbolic
- Parameterization: eta
- Loss: 6.406213e-01
- Notes: PySR on the dominant real coefficient curve.

## 16 - symbolic_pysr_phase_raw
- Category: symbolic
- Parameterization: raw
- Loss: 6.387154e-01
- Notes: PySR on a phase-sensitive imaginary coefficient curve.

## 17 - symbolic_pysr_freq_delta
- Category: symbolic
- Parameterization: delta
- Loss: 6.386078e-01
- Notes: PySR on a frequency-sensitive imaginary coefficient curve.

## 18 - symbolic_gplearn_amp_raw
- Category: symbolic
- Parameterization: raw
- Loss: 6.387178e-01
- Notes: gplearn on the dominant real coefficient curve.

## 19 - symbolic_gplearn_phase_eta
- Category: symbolic
- Parameterization: eta
- Loss: 6.395177e-01
- Notes: gplearn on a phase-sensitive imaginary coefficient curve.

## 20 - symbolic_gplearn_freq_delta
- Category: symbolic
- Parameterization: delta
- Loss: 6.386078e-01
- Notes: gplearn on a frequency-sensitive imaginary coefficient curve.

## Summary
- Best: symbolic_pysr_freq_delta (6.386078e-01)
- Categories: ['functional_optimization', 'matched_composite', 'physics', 'symbolic']
- Approaches: 20
