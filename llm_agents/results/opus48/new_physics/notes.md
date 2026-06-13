# New Physics Bench - opus48 notes

## What the model is
Dominant nonspinning (2,2) RG-tail waveform from the Section-IV factorized
ingredients (formula sheet). The complex correction is

    hhat_22 = H_eff * T_22 * rho_22^2 * exp(i * delta_22)

built exactly from the sheet:
- Conservative sector: E_real/M (with explicit log(16x) at 4PN) and
  p_phi,circ/(mu M); H_eff = ((E_real/M)^2 - 1)/(2 nu) + 1.
- Running tail T_22 = exp(logT_22) with the universal anomalous dimension
  gamma_22^univ and ellhat_22 = 2 + lambda_RG * gamma_22^univ. The complex
  log-Gammas use scipy.special.loggamma. Note 2 k r_omega = 4 sqrt(x),
  log(4 phi0) = 17/12 - gamma_E, and khat = (E_real/M) * m * x^(3/2) (since
  M*Omega = x^(3/2)), J = (p_phi/(mu M)) / (E_real/M)^2.
- Residual rho_22 (with eulerlog_2 in r4) and delta_22 (in y = (E*x^1.5)^(2/3)).

## Frequency-domain construction (SPA)
- f = Omega/pi for the (2,2), x = (pi M_sec f)^(2/3), M_sec = Mc*MSUN_SEC/eta^(3/5).
- Radiation reaction: F_22 = (32/5) nu^2 x^5 |hhat_22|^2,
  dt/dx = -M_sec * d(E_real/M)/dx / F_22  (dE/dx by finite differences, which
  correctly captures the explicit log(16x) term).
- t(f) and phi_orb(f) by cumulative trapezoidal integration of dt/df and
  dphi_orb/df = pi f dt/df over the in-band frequencies.
- Stationary-phase phase:
    Psi(f) = 2 pi f t(f) - 2 phi_orb(f) + arg(hhat_22(f)) - pi/4 (+ tc, phic).
  The RG/tail physics enters the phase through arg(hhat_22) (i.e. arg T_22 and
  delta_22), which is what makes the waveform sensitive to lambda_RG.
- SPA amplitude: |h(f)| proportional to nu * x * |hhat_22| * sqrt(dt/df) (times
  M_sec/dL_sec and the taper). This reproduces the standard f^(-7/6) inspiral
  amplitude (verified analytically: x * x^(-2.75) = x^(-1.75) ~ f^(-7/6)).

## Convention checks performed
- |hhat_22| ~ 0.90-0.96 across band (≈1 as expected; log(120) cancels
  loggamma(2 ellhat+2)=loggamma(6) at leading order).
- F_22 matches the Newtonian (32/5) nu^2 x^5 at low x.
- d(E/M)/dx ≈ -nu/2 at leading order.
- t(f) span over [20, 300] Hz for Mc=8, eta=0.24 is 6.90 s vs the Newtonian
  chirp time 6.86 s -> the phase evolution (dPsi/df = 2 pi t(f)) is correct.
- Fourier convention: PyCBC uses h~(f) = int h(t) exp(-2 pi i f t) dt, so the
  SPA strain carries exp(-i*Psi). With this sign the waveform matches a
  matched-mass TaylorF2 at ~0.89 on a long low-mass band (it is a *different*
  model + a Fermi taper near f_isco, so a perfect match is not expected); the
  opposite sign gives ~0.06, confirming the convention.
- lambda_RG sensitivity: mismatch(lambda=1, lambda=2) ~ 0.02-0.19 for
  low/moderate masses; ~0 for very high masses whose inspiral barely enters the
  [f_low, 1.3 f_isco] band (physically expected).

## Cutoffs / tapering
Exactly zero for f < f_low and f >= f_cut = fmax_over_fisco * f_isco; inside the
band a Fermi taper W(f) = 1/(1+exp((f - f_isco)/sigma)),
sigma = sigma_taper_over_fisco * f_isco. phase_only=True returns the tapered
unit-amplitude phase factor.
