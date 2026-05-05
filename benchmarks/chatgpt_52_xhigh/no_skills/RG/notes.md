# Notes (RG Waveform, Source-Packet / No-Skills)

## What This Implements

`candidate_waveform.py:h_of_f` builds a single-mode inspiral frequency-domain strain using the dominant nonspinning `(2,2)` RG-tail ingredients from the provided formula sheet:

`hhat_22 = H_eff * T_22 * rho_22^2 * exp(i * delta_22)`.

The inspiral phasing is obtained by energy balance using:

- `F_22 = (32/5) * nu^2 * x^5 * |hhat_22|^2`
- `dt/dx = -M * d(E_real/M)/dx / F_22`
- `f = Omega/pi`, `x = (pi M f)^(2/3)`

## Fourier / SPA Convention

I treat the frequency-domain strain as a stationary-phase approximation (SPA) for a chirp with Fourier kernel `exp(+2π i f t)` (matching the usual LAL-style SPA formula forms), so that:

`h(f) ~ [a(t_f)/sqrt(df/dt)] * exp(i [2π f t_f - phi(t_f) - π/4])`.

Here, I take the time-domain complex amplitude for the dominant mode to be proportional to:

`a(t) = (2 * nu * M / dL) * x * hhat_22(x)`,

which reproduces the standard Newtonian `f^{-7/6}` amplitude scaling when `hhat_22 -> 1`.

The integration constants are chosen so that near the high-frequency cutoff:

- `t(f_cut) = tc`
- `phi_gw(f_cut) = phic`

## Cutoffs / Taper

Per the prompt:

- Exactly zero for `f < f_low` and `f >= f_cut`
- Logistic taper `W(f) = 1 / (1 + exp((f - f_cut)/sigma))` applied for `f < f_cut`

## Quick Internal Checks

1. Verified `dt/dx > 0` (since `dE/dx < 0`) over typical inspiral `x`.
2. In the Newtonian limit (turning off higher PN structure and setting `hhat_22 -> 1`), the resulting SPA amplitude reduces to the familiar scaling `|h(f)| ∝ Mc^(5/6) f^(-7/6) / dL`.

