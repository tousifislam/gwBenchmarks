## RG Waveform Notes

- Implemented the requested `h_of_f(...)` interface in `candidate_waveform.py`.
- Used detector-frame inputs with
  - `M_sec = Mc * MSUN_SEC / eta^(3/5)`
  - `dL_sec = dL * MPC_SEC`
  - `x = (pi M_sec f)^(2/3)`.
- Implemented all coefficients from the source packet for:
  - conservative `E_real/M`, `p_phi,circ/(mu M)`, and `H_eff`
  - running-tail `gamma_22^univ`, `ellhat_22`, `T_22`
  - residual `rho_22`, `delta_22`
  - `hhat_22 = H_eff T_22 rho_22^2 exp(i delta_22)`.
- Used `F_22 = (32/5) nu^2 x^5 |hhat_22|^2` and
  `dt/dx = -M d(E_real/M)/dx / F_22` to build `dt/df`.
- Built SPA-like phase with
  - `dPhi_gw/df = 2 pi f dt/df`
  - `Psi(f) = 2 pi f t(f) - Phi_gw(f) - pi/4`
  and then multiplied by complex `hhat_22`.
- Base amplitude uses Newtonian inspiral scaling
  `~ Mc^(5/6) f^(-7/6) / dL`, multiplied by `hhat_22`.
- Applied the requested cutoff/taper:
  - exact zero for `f < f_low`
  - exact zero for `f >= f_cut`
  - logistic taper `W(f)` inside the valid band.
