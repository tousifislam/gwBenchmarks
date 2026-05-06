## Approach 1: phenom_simple
- **Parameterization**: eta
- **Loss**: 0.596076
- **Notes**: Lorentzian amplitude + quadratic phase.

## Approach 2: gplearn_t_eta
- **Parameterization**: eta
- **Loss**: 0.732497
- **Notes**: gplearn on (t, eta).

## Approach 11: poly5_q
- **Parameterization**: q
- **Loss**: 0.592695
- **Notes**: Poly deg 5 on mean waveform, scaled by eta.

## Approach 12: poly10_eta
- **Parameterization**: eta
- **Loss**: 0.616748
- **Notes**: Poly deg 10 on mean waveform, scaled by eta.

## Approach 13: poly15_deltam
- **Parameterization**: delta_m
- **Loss**: 0.671378
- **Notes**: Poly deg 15 on mean waveform, scaled by eta.

## Approach 14: log_poly_q
- **Parameterization**: q
- **Loss**: 0.685509
- **Notes**: Log-amplitude polynomial + phase polynomial.

## Approach 15: omega_poly_q
- **Parameterization**: q
- **Loss**: 0.700251
- **Notes**: Amplitude polynomial + frequency polynomial.

## Approach 16: poly7_sqrteta
- **Parameterization**: sqrt_eta
- **Loss**: 0.619727
- **Notes**: Poly deg 7 scaled by eta.

## Approach 17: tanh_merger
- **Parameterization**: eta
- **Loss**: 0.943628
- **Notes**: Simple tanh merger amplitude + linear phase.

## Approach 18: const_amp
- **Parameterization**: eta
- **Loss**: nan
- **Notes**: Constant amplitude, zero phase (baseline).

## Approach 19: pure_sin
- **Parameterization**: eta
- **Loss**: 0.879723
- **Notes**: Pure sinusoid.

## Approach 20: damped_sin
- **Parameterization**: eta
- **Loss**: 0.413900
- **Notes**: Damped sinusoid.

## Approach 21: poly20_q
- **Parameterization**: q
- **Loss**: 0.626388
- **Notes**: Poly deg 20.

## Approach 22: poly5_sqrteta
- **Parameterization**: sqrt_eta
- **Loss**: 0.592695
- **Notes**: Poly deg 5 scaled by eta.

## Approach 23: pn_like
- **Parameterization**: eta
- **Loss**: 0.457878
- **Notes**: PN-like power laws.

## Approach 24: poly3_deltam
- **Parameterization**: delta_m
- **Loss**: 0.631377
- **Notes**: Poly deg 3.

## Approach 25: poly12_eta
- **Parameterization**: eta
- **Loss**: 0.683458
- **Notes**: Poly deg 12.

## Approach 26: exp_a
- **Parameterization**: eta
- **Loss**: 0.476909
- **Notes**: Exponential amplitude.

## Approach 27: linear_all
- **Parameterization**: eta
- **Loss**: 0.901398
- **Notes**: Linear amplitude and phase.

## Approach 28: gaussian_a
- **Parameterization**: eta
- **Loss**: 0.595546
- **Notes**: Gaussian amplitude.

