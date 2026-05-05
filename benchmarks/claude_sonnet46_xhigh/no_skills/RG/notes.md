# Waveform Convention and Checks

## Waveform Convention

Stationary phase approximation (SPA) for the dominant (2,2) nonspinning mode.

The frequency-domain strain is assembled as

```
h(f) = A_N(f) * |hhat_22(x_f)| * sqrt(2*pi / |d^2Psi/df^2|) * exp(i Psi(f)) * W(f)
```

where

- `A_N(f) = sqrt(5/24) / pi^(2/3) * Mc_sec^(5/6) / dL_sec * f^(-7/6)`  is the standard
  Newtonian SPA prefactor (face-on, optimal polarization);
- `hhat_22 = H_eff * T_22 * rho_22^2 * exp(i delta_22)` is the dimensionless
  factorized correction from arXiv:2602.08833 Sec. IV;
- `d^2Psi/df^2 = 2*pi * dt/df` is computed numerically from the energy balance
  `dt/dx = -M dE/dx / F_22`;
- `Psi(f) = 2*pi*f*tc - 2*phic - pi/4 + phi_orb(f)` with `phi_orb` built by
  double numerical integration of `dtdf`;
- `W(f) = 1/(1 + exp((f - f_cut)/sigma))` is the logistic taper.

The internal variable is `x_f = (pi M_sec f)^(2/3)`, detector-frame.

## Equation Checks

1. **Newtonian limit**: with `lambda_RG = 0` and `eta -> 0.25`, hhat -> ~1 and
   the amplitude reduces to the TaylorF2 Newtonian form.

2. **Mass/distance scaling**: for `Mc = 28 Msun`, `eta = 0.25`, `dL = 400 Mpc`
   the total mass M ≈ 64.4 Msun, f_isco ≈ 68 Hz, f_cut ≈ 89 Hz,
   max |h| ~ 1e-22 — consistent with GW150914-class events.

3. **lambda_RG sensitivity**: amplitude ratio (lRG=1)/(lRG=0) ≈ 1.14 and mean
   phase shift (lRG=2 vs 1) ≈ −0.24 rad, confirming the parameter enters both
   amplitude and phase through T_22.

4. **Zero conditions**: h is exactly zero for f < f_low = 20 Hz and f >= f_cut,
   enforced by Boolean masking before any computation.

## Known Limitations

- The SPA double integration uses the numerically integrated phase on the input
  frequency grid; denser f arrays give more accurate phases.
- The amplitude uses the face-on, optimal-orientation prefactor; angular
  factors (F+, Fx, inclination) are not included.
- Spin and tidal effects are not included (benchmark is nonspinning (2,2) only).
