# Notes on `candidate_waveform.py`

## Overall construction

Frequency-domain (2,2)-only inspiral built from the arXiv:2602.08833 formula
sheet through three steps:

1. **Conservative sector polynomials.** `E_real/M(x)` and
   `p_phi,circ/(mu M)(x)` from the `c_n`/`d_n` coefficients (with `log(16x)`
   pieces handled explicitly inside `_eM_and_prime` and `_pphi`).
2. **Factorized (2,2) correction**

       hhat_22 = H_eff * T_22 * rho_22^2 * exp(i delta_22)

   - `H_eff = ((E_real/M)^2 - 1)/(2 nu) + 1`
   - `T_22 = exp(logT_22)` with the source-packet expression for `logT_22`
     (uses `loggamma` for the complex Gamma functions; `phi0 = exp(17/12 -
     gamma_E)/4`; `2 k r_omega = 4 sqrt(x)`)
   - `ellhat_22 = 2 + lambda_RG * gamma_22^univ` deforms only the running
     part; `lambda_RG = 1` is GR.
   - `rho_22` and `delta_22` from the residual polynomials, with
     `eulerlog_2(x) = gamma_E + log(4 sqrt(x))` and `y = (eM x^{3/2})^{2/3}`.
3. **Energy balance ⇒ SPA frequency-domain template.**

       F_22  = (32/5) nu^2 x^5 |hhat_22|^2
       dt/dx = -M_sec dE/dx / F_22
       dphi_orb/dx = Omega dt/dx = -x^{3/2} dE/dx / F_22

   Both are integrated cumulatively on a log-spaced `x`-grid from
   `x_low = (pi M_sec f_low)^{2/3}` to `x_high = (pi M_sec f_cut)^{2/3}`
   (8192 points). Integration constants are fixed so that
   `phi_orb_rel(x_high) = 0` and `t_rel(x_high) = 0` (coalescence at the
   high-frequency edge).

## Stationary-phase template

Carrier convention `h_22(t) ∝ exp(-2 i phi_orb)`. The face-on circular
polarisation Fourier transform at `f > 0` picks up `hhat_22^*` because the
positive-frequency content lives in the conjugate carrier. With

    x_f       = (pi M_sec f)^{2/3}
    df/dt|_xf = -3 sqrt(x_f) F_22(x_f) / (2 pi M_sec^2 dE/dx(x_f))

(which is positive because `dE/dx < 0` in the inspiral) the SPA gives

    h(f) = -2 (M_sec nu / dL_sec) x_f hhat_22^*(x_f) (df/dt)^{-1/2}
           * exp(i Phi(f))

    Phi(f) = 2 phi_orb_rel(x_f) - 2 pi f t_rel(x_f) + pi/4
             + phic - 2 pi f tc.

This is the conjugate ("`+i`") sign convention. To match the LIGO LAL
convention `h ∝ exp(-i Psi_LAL)` take a complex conjugate of the output;
the Fisher matrix is unchanged either way.

## Cutoffs and taper

```
f_isco = 1 / (pi 6^{3/2} M_sec)
f_cut  = fmax_over_fisco * f_isco
sigma  = sigma_taper_over_fisco * f_isco
W(f)   = 1 / (1 + exp((f - f_cut)/sigma))
```

`h(f) = 0` strictly for `f < f_low` and `f >= f_cut`; the smooth Fermi
taper `W(f)` modulates the amplitude inside the band.

## `phase_only=True`

Returns the unit-modulus phase factor only,

    h(f) = hhat_22^*(x_f) / |hhat_22(x_f)| * exp(i Phi(f))

with strict zero outside `[f_low, f_cut)` and **no** Fermi taper applied
(taper is an amplitude effect). Inside the band the result has unit
modulus.

## Equation checks performed

- **Newtonian-amplitude limit.** At `Mc=1.2 Msun`, `eta=0.25`, `f=50 Hz`,
  `|h(f)|` agrees with the TF2 LO amplitude
  `sqrt(5/24)/pi^{2/3} Mc_sec^{5/6} / (dL_sec f^{7/6})`
  to better than 1.5 % (residual is `rho_22^2` correction at the tested x).
- **Newtonian-phase limit.** Reduction of the natural integrals at LO
  (`F_22 = (32/5) nu^2 x^5`, `eM' = -nu/2`) gives
  `-2 phi_orb_rel + 2 pi f t_rel = 3/(128 nu) (pi M_sec f)^{-5/3}`,
  matching the standard TaylorF2 leading-order chirp phase.
- **`tc` shift.** Numerical phase difference from `tc = 1e-3 s` matches
  `-2 pi f tc` to machine precision at f = 50 Hz.
- **`phic` shift.** Numerical phase difference from `phic = 0.5` is
  exactly `+0.5 rad`.
- **Hard cutoffs.** `h(f < f_low) = 0` and `h(f >= f_cut) = 0`.
- **`lambda_RG` deformation.** Non-trivial phase change from
  `lambda_RG = 0 / 1 / 2`, vanishing at low frequencies (because
  `gamma_22^univ ∝ x^3` at leading order).

## Implementation notes

- `loggamma` gives the principal branch of `log Gamma`; `T_22` is built by
  exponentiating the complex log so that small imaginary parts don't
  produce spurious branch jumps.
- `c4` and `d4` carry an `x`-dependent `log(16 x)` term; the polynomial
  derivative used in `dE/dx` correctly accounts for `d/dx[c4(x) x^4]`.
- All internal arithmetic uses geometric-second masses and distance
  (`MSUN_SEC`, `MPC_SEC`).
- The integration grid is logarithmic in `x` because the integrand
  scales like `x^{-7/2}` at low frequency.
