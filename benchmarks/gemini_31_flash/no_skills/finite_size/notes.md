# Finite-Size SPA Waveform Implementation Notes

## Convention

The frequency-domain strain is defined as:
```
h(f) = A(f) * exp(-1j * psi(f))
```
where `A(f)` is the amplitude and `psi(f)` is the SPA phase.

### Amplitude
The amplitude is calculated as:
```
A(f) = A_N(f) * sqrt((df/dt)_N / (df/dt))
```
where `A_N(f)` is the Newtonian amplitude and `(df/dt)_N` is the Newtonian chirp rate. `(df/dt)` is the total chirp rate derived from the balance law:
```
df/dt = [3 v^2 / (pi M_sec)] / [dt/dv]
dt/dv = -M_sec * (de/dv) / F_tot
```
`F_tot = F_infty + dotM`.

### Phase
The SPA phase `psi(f)` satisfies:
```
d^2 psi / df^2 = 2 * pi / (df/dt)
```
Integrating twice:
```
d psi / df = 2 * pi * tc + 2 * pi * \int^f (dt/df') df'
psi(f) = 2 * pi * f * tc - phic - pi/4 + 2 * pi * \int^f df' \int^{f'} df'' (dt/df'')
```
Equivalently, using integration by parts:
```
psi(f) = 2 * pi * f * tc - phic - pi/4 + 2 * pi * \int^f (f - f') (dt/df') df'
```
We use numerical integration (cumulative trapezoidal rule) to compute the phase.

### Cutoffs and Tapers
- The waveform is zero for `f < f_low` or `f >= f_cut`.
- A Planck-style taper `W(f) = 1 / [1 + exp((f - f_cut)/sigma)]` is applied for `f < f_cut`.

## Equation Checks
- Verified `dt/dv` and `df/dt` definitions from the balance law.
- Verified the SPA phase relation to the chirp rate.
- Constants `MSUN_SEC` and `MPC_SEC` are taken from the prompt.
