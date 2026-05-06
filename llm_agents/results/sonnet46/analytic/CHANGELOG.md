## 01: pn0_qnm
- cat=physics, rep=eta
- train=6.6725e-01, val=6.3154e-01, rt=5.3ms
- Leading PN + QNM with tanh blend

## 02: pn0_qnm_ampcorr
- cat=physics, rep=eta
- train=6.4016e-01, val=5.7619e-01, rt=5.6ms
- PN0+QNM with log-polynomial amplitude correction

## 03: pn2_qnm
- cat=physics, rep=eta
- train=6.6159e-01, val=6.4971e-01, rt=8.3ms
- PN+phase correction in v-expansion + QNM

## 04: pn_gauss_merger
- cat=physics, rep=eta
- train=5.9209e-01, val=5.8294e-01, rt=6.6ms
- PN inspiral + Gaussian merger peak + QNM

## 05: imrphenom_style
- cat=physics, rep=eta
- train=5.6624e-01, val=5.5920e-01, rt=7.8ms
- IMRPhenom-style: fitted PN frequency ratio

## 06: pade_amp
- cat=physics, rep=eta
- train=7.2055e-01, val=6.9389e-01, rt=6.7ms
- Pade [2,2] amplitude in PN velocity v

## 07: pn_full_corr
- cat=physics, rep=eta
- train=6.8688e-01, val=6.5757e-01, rt=8.0ms
- Full PN+amp correction+phase correction+QNM

## 08: phenom_ampcorr
- cat=physics, rep=eta
- train=3.0757e-01, val=2.9592e-01, rt=8.2ms
- Phenom frequency + amplitude correction

## 09: gauss3_amp
- cat=functional, rep=eta
- train=5.6504e-01, val=5.0851e-01, rt=5.1ms
- 3-Gaussian amplitude + PN phase

## 10: lorentz_peak
- cat=functional, rep=eta
- train=5.2872e-01, val=4.5671e-01, rt=6.2ms
- Lorentzian peak amplitude + PN phase

## 11: sigmoid_blend
- cat=matched, rep=eta
- train=6.3711e-01, val=6.2716e-01, rt=7.3ms
- Sigmoid blend PN inspiral + QNM ringdown

## 12: gplearn_amp
- cat=symbolic, rep=eta
- train=5.7793e-01, val=5.1919e-01, rt=5.7ms
- gplearn: log(A)=f(log(tau),eta): log(X1)

## 13: gplearn_freq
- cat=symbolic, rep=eta
- train=6.3974e-01, val=6.3880e-01, rt=7.0ms
- gplearn: omega/omega_PN=f(v,eta): 0.758

## 14: pysr_amp
- cat=symbolic, rep=eta
- train=5.6351e-01, val=4.8026e-01, rt=7.6ms
- PySR symbolic: log(A)=f(log(tau),eta)

## 15: pysr_freq
- cat=symbolic, rep=eta
- train=5.2911e-01, val=5.2230e-01, rt=9.0ms
- PySR symbolic: omega/omega_PN=f(v,eta)

## 16: matched_3region
- cat=matched, rep=eta
- train=6.2318e-01, val=5.7585e-01, rt=9.3ms
- 3-region matched: PN+Gaussian merger+QNM

## 17: damp_sinusoid
- cat=functional, rep=eta
- train=6.6671e-01, val=6.2819e-01, rt=6.9ms
- 2-QNM overtones + PN inspiral

## 18: powerlaw_delta
- cat=functional, rep=delta_m
- train=5.4786e-01, val=4.7628e-01, rt=5.5ms
- Power-law amplitude in tau, delta_m reparam

## 19: pn_qnm_composite
- cat=matched, rep=eta
- train=6.6850e-01, val=6.5007e-01, rt=8.0ms
- PN+QNM composite at ISCO, delta_m=vary

## 20: sqrteta_model
- cat=physics, rep=sqrt_eta
- train=6.8688e-01, val=6.5757e-01, rt=8.0ms
- QNM+PN with sqrt(eta) reparameterization

## 21: poly_q_direct
- cat=physics, rep=log_q
- train=6.8689e-01, val=6.5750e-01, rt=7.9ms
- QNM+PN with log(q) reparameterization

## 22: phenom_pysr_freq
- cat=symbolic, rep=eta
- train=2.7866e-01, val=2.7208e-01, rt=10.0ms
- Phenom using PySR frequency + amp correction

