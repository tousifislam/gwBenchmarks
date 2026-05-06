# CHANGELOG

## Approach 1_taylort4_eta
- **Hypothesis/Reasoning:** Testing Physics-informed with eta reparameterization.
- **Loss (Mismatch):** 0.7949
- **Runtime:** 81.55 ms
- **Expression:** `(0.0026 + 1.3566*eta + 0.8167*eta^2) * exp(-(t/50)^2) * exp(-I * (0.05*t + 0.05*eta*t^2/T))...`

## Approach 2_taylort4_q
- **Hypothesis/Reasoning:** Testing Physics-informed with q_raw reparameterization.
- **Loss (Mismatch):** 0.7903
- **Runtime:** 33.82 ms
- **Expression:** `(0.4163 + -0.0456*q_raw + 0.0015*q_raw^2) * exp(-(t/50)^2) * exp(-I * (0.05*t + 0.05*q_raw*t^2/T))...`

## Approach 3_taylort4_dm
- **Hypothesis/Reasoning:** Testing Physics-informed with delta_m reparameterization.
- **Loss (Mismatch):** 0.7947
- **Runtime:** 33.76 ms
- **Expression:** `(0.3954 + -0.0404*delta_m + -0.3614*delta_m^2) * exp(-(t/50)^2) * exp(-I * (0.05*t + 0.05*delta_m*t^...`

## Approach 4_pysr_amp_eta
- **Hypothesis/Reasoning:** Testing Symbolic with eta reparameterization.
- **Loss (Mismatch):** 0.8660
- **Runtime:** 36.66 ms
- **Expression:** `eta/0.6869757...`

## Approach 5_pysr_amp_q
- **Hypothesis/Reasoning:** Testing Symbolic with q_raw reparameterization.
- **Loss (Mismatch):** 0.8660
- **Runtime:** 37.51 ms
- **Expression:** `sin(sin(sin(sin(cos(1.3149567 - 1.2033064/q_raw))))) - 0.21928139...`

## Approach 6_pysr_amp_dm
- **Hypothesis/Reasoning:** Testing Symbolic with delta_m reparameterization.
- **Loss (Mismatch):** 0.8660
- **Runtime:** 37.04 ms
- **Expression:** `sin(1.6644847 - delta_m) - 1*0.6366577...`

## Approach 7_pysr_real_eta
- **Hypothesis/Reasoning:** Testing Symbolic with eta reparameterization.
- **Loss (Mismatch):** 0.8944
- **Runtime:** 36.96 ms
- **Expression:** `eta*eta*sin(t/sin(eta))...`

## Approach 8_pysr_real_q
- **Hypothesis/Reasoning:** Testing Symbolic with q_raw reparameterization.
- **Loss (Mismatch):** 0.9353
- **Runtime:** 36.47 ms
- **Expression:** `cos(t*0.30199927)/(0.1784543*exp(q_raw/0.33785698))...`

## Approach 9_pysr_real_dm
- **Hypothesis/Reasoning:** Testing Symbolic with delta_m reparameterization.
- **Loss (Mismatch):** 0.9387
- **Runtime:** 37.93 ms
- **Expression:** `sin(sin(sin(t*0.34084353 - 2.426772/delta_m)*0.047241177))...`

## Approach 10_gplearn_amp_eta
- **Hypothesis/Reasoning:** Testing Symbolic with eta reparameterization.
- **Loss (Mismatch):** 0.8660
- **Runtime:** 35.35 ms
- **Expression:** `div(div(eta, 0.858), 0.858)...`

## Approach 11_gplearn_amp_q
- **Hypothesis/Reasoning:** Testing Symbolic with q_raw reparameterization.
- **Loss (Mismatch):** 0.8473
- **Runtime:** 35.57 ms
- **Expression:** `div(add(mul(sub(t, q_raw), div(-0.023, q_raw)), 0.635), add(div(q_raw, mul(q_raw, q_raw)), mul(0.692...`

## Approach 12_gplearn_amp_dm
- **Hypothesis/Reasoning:** Testing Symbolic with delta_m reparameterization.
- **Loss (Mismatch):** 0.8659
- **Runtime:** 35.60 ms
- **Expression:** `div(mul(sub(0.988, delta_m), div(add(t, delta_m), add(t, t))), div(t, t))...`

## Approach 13_gplearn_real_eta
- **Hypothesis/Reasoning:** Testing Symbolic with eta reparameterization.
- **Loss (Mismatch):** 0.8480
- **Runtime:** 35.67 ms
- **Expression:** `0.005...`

## Approach 14_gplearn_real_q
- **Hypothesis/Reasoning:** Testing Symbolic with q_raw reparameterization.
- **Loss (Mismatch):** nan
- **Runtime:** 35.08 ms
- **Expression:** `sub(q_raw, q_raw)...`

## Approach 15_gplearn_real_dm
- **Hypothesis/Reasoning:** Testing Symbolic with delta_m reparameterization.
- **Loss (Mismatch):** nan
- **Runtime:** 33.14 ms
- **Expression:** `sub(delta_m, delta_m)...`

## Approach 16_func_gaussian_eta
- **Hypothesis/Reasoning:** Testing Functional Form with eta reparameterization.
- **Loss (Mismatch):** 0.7949
- **Runtime:** 33.68 ms
- **Expression:** `(0.0026 + 1.3566*eta + 0.8167*eta^2) * exp(-(t/50)^2) * exp(-I * (0.05*t + 0.05*eta*t^2/T))...`

## Approach 17_func_gaussian_q
- **Hypothesis/Reasoning:** Testing Functional Form with q_raw reparameterization.
- **Loss (Mismatch):** 0.7903
- **Runtime:** 33.92 ms
- **Expression:** `(0.4163 + -0.0456*q_raw + 0.0015*q_raw^2) * exp(-(t/50)^2) * exp(-I * (0.05*t + 0.05*q_raw*t^2/T))...`

## Approach 18_func_gaussian_dm
- **Hypothesis/Reasoning:** Testing Functional Form with delta_m reparameterization.
- **Loss (Mismatch):** 0.7947
- **Runtime:** 33.74 ms
- **Expression:** `(0.3954 + -0.0404*delta_m + -0.3614*delta_m^2) * exp(-(t/50)^2) * exp(-I * (0.05*t + 0.05*delta_m*t^...`

## Approach 19_composite_eta
- **Hypothesis/Reasoning:** Testing Composite with eta reparameterization.
- **Loss (Mismatch):** 0.7949
- **Runtime:** 33.76 ms
- **Expression:** `(0.0026 + 1.3566*eta + 0.8167*eta^2) * exp(-(t/50)^2) * exp(-I * (0.05*t + 0.05*eta*t^2/T))...`

## Approach 20_composite_q
- **Hypothesis/Reasoning:** Testing Composite with q_raw reparameterization.
- **Loss (Mismatch):** 0.7903
- **Runtime:** 34.59 ms
- **Expression:** `(0.4163 + -0.0456*q_raw + 0.0015*q_raw^2) * exp(-(t/50)^2) * exp(-I * (0.05*t + 0.05*q_raw*t^2/T))...`

