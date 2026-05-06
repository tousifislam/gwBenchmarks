# Analytic Bench — CHANGELOG

## Approach 01: lorentzian_tanh_eta
- Category: physics
- Parameterization: eta
- Val Loss (FD mismatch): 0.3323
- Notes: lorentzian amp + tanh freq, eta param, deg 3

## Approach 02: gaussian_tanh_eta
- Category: physics
- Parameterization: eta
- Val Loss (FD mismatch): 0.3256
- Notes: gaussian amp + tanh freq, eta param, deg 3

## Approach 03: sech_sigmoid_q
- Category: physics
- Parameterization: q
- Val Loss (FD mismatch): 0.5529
- Notes: sech amp + sigmoid freq, q param, deg 3

## Approach 04: power_pn_eta
- Category: physics
- Parameterization: eta
- Val Loss (FD mismatch): 0.1682
- Notes: power_exp amp + pn_tanh freq, eta param, deg 3

## Approach 05: lorentzian_tanh_dm
- Category: physics
- Parameterization: delta_m
- Val Loss (FD mismatch): 0.3457
- Notes: lorentzian amp + tanh freq, delta_m param, deg 3

## Approach 06: lorentzian_tanh_eta4
- Category: physics
- Parameterization: eta
- Val Loss (FD mismatch): 0.3300
- Notes: lorentzian amp + tanh freq, eta param, deg 4

## Approach 07: gaussian_sigmoid_q4
- Category: physics
- Parameterization: q
- Val Loss (FD mismatch): 0.5633
- Notes: gaussian amp + sigmoid freq, q param, deg 4

## Approach 08: pysr_amp_eta
- Category: symbolic
- Parameterization: eta
- Val Loss (FD mismatch): 0.3323
- Notes: PySR on A_peak

## Approach 09: pysr_freq_q
- Category: symbolic
- Parameterization: q
- Val Loss (FD mismatch): 0.3337
- Notes: PySR on omega_rd

## Approach 10: pysr_merger_eta
- Category: symbolic
- Parameterization: eta
- Val Loss (FD mismatch): 0.3315
- Notes: PySR on width_l

## Approach 11: gplearn_amp_q
- Category: symbolic
- Parameterization: q
- Val Loss (FD mismatch): 0.3238
- Notes: gplearn on A_peak

## Approach 12: gplearn_freq_eta
- Category: symbolic
- Parameterization: eta
- Val Loss (FD mismatch): 0.3172
- Notes: gplearn on omega_rd

## Approach 13: pn_qnm_blend_eta
- Category: matched
- Parameterization: eta
- Val Loss (FD mismatch): 0.1675
- Notes: power_exp amp + pn_tanh freq, eta param, deg 4

## Approach 14: three_region_q
- Category: matched
- Parameterization: q
- Val Loss (FD mismatch): 0.3924
- Notes: tanh_power amp + pn_tanh freq, q param, deg 3

## Approach 15: window_blend_dm
- Category: matched
- Parameterization: delta_m
- Val Loss (FD mismatch): 0.4637
- Notes: power_exp amp + tanh freq, delta_m param, deg 3

## Approach 16: fermi_pn_eta
- Category: matched
- Parameterization: eta
- Val Loss (FD mismatch): 0.1730
- Notes: sech amp + pn_tanh freq, eta param, deg 3

## Approach 17: overlap_match_q4
- Category: matched
- Parameterization: q
- Val Loss (FD mismatch): 0.1726
- Notes: lorentzian amp + pn_tanh freq, q param, deg 4

## Approach 18: double_gauss_q
- Category: functional
- Parameterization: q
- Val Loss (FD mismatch): 0.3321
- Notes: double_gauss amp + tanh freq, q param, deg 3

## Approach 19: sum_lorentz_eta
- Category: functional
- Parameterization: eta
- Val Loss (FD mismatch): 0.3479
- Notes: sum_lorentz amp + tanh freq, eta param, deg 3

## Approach 20: asym_gauss_dm
- Category: functional
- Parameterization: delta_m
- Val Loss (FD mismatch): 0.5988
- Notes: asym_gauss amp + sigmoid freq, delta_m param, deg 3

## Approach 21: rational_tanh_eta
- Category: functional
- Parameterization: eta
- Val Loss (FD mismatch): 0.3264
- Notes: rational amp + tanh freq, eta param, deg 3

## Approach 22: super_lor_q
- Category: functional
- Parameterization: q
- Val Loss (FD mismatch): 0.3179
- Notes: super_lor amp + tanh freq, q param, deg 3

## Approach 23: cheby_tanh_q
- Category: functional
- Parameterization: q
- Val Loss (FD mismatch): 0.5314
- Notes: cheby amp + tanh freq, q param, deg 3

## Approach 24: exp_gauss_eta
- Category: functional
- Parameterization: eta
- Val Loss (FD mismatch): 0.6648
- Notes: exp_gauss amp + sigmoid freq, eta param, deg 3

## Approach 25: tanh_power_dm
- Category: functional
- Parameterization: delta_m
- Val Loss (FD mismatch): 0.6471
- Notes: tanh_power amp + tanh freq, delta_m param, deg 3

## Approach 26: lor_sigmoid_eta4
- Category: functional
- Parameterization: eta
- Val Loss (FD mismatch): 0.5595
- Notes: lorentzian amp + sigmoid freq, eta param, deg 4
