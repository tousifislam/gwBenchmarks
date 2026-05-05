# Notes (Finite-Size Balance-Law SPA, Source-Packet / No-Skills)

## What This Implements

`candidate_waveform.py:h_of_f` constructs an inspiral-only frequency-domain waveform using the balance law

`-F_infty(v) - dotM(v) = dE/dt`

with the source-packet expressions for:

- binding energy per total mass `e(v) = E(v)/M`
- flux at infinity `F_infty(v)`
- absorbed flux `dotM(v)`

The point-particle baseline is Newtonian, and the finite-size terms are added through the spin-induced quadrupole/cubic-spin pieces and the effective leading Love-number flux term (`Lambda_tilde`) as given in `2410.00294_relevant_formulas.md`.

## Phase Convention

I compute the SPA phase from time and GW phase functions derived from `dt/dv`:

- `dt/dv = -M_sec * d(e)/dv / (F_infty + dotM)`
- `dphi_gw/dt = 2*pi*f = 2*v^3/M_sec`
- `psi(f) = 2*pi*f*t(v) - phi_gw(v) - pi/4`
- `h(f) = A(f) * exp(-i * psi(f))`

Integration constants are fixed at the top of the in-band frequency grid:

- `t(v_max) = tc`
- `phi_gw(v_max) = phic`

Any additional constant and linear-in-`f` offsets are degenerate with `(phic, tc)` as noted in the prompt.

## Amplitude Convention

The restricted SPA amplitude is:

- `A_N(f) = sqrt(5/24) * Mc_sec^(5/6) * pi^(-2/3) * f^(-7/6) / dL_sec`
- `A(f) = A_N(f) * sqrt(dfdt_N / dfdt)`

where `dfdt` is obtained from `dt/dv` and `f(v) = v^3/(pi M_sec)`, and `dfdt_N` is the Newtonian chirp rate from the source packet.

## Cutoffs / Taper

Per the prompt:

- exactly zero for `f < f_low` and `f >= f_cut`
- logistic taper `W(f)` applied for `f < f_cut`

