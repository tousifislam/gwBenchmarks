# arXiv:2602.08833 Relevant Formulas

This compact formula sheet collects the Section-IV ingredients needed for a
dominant nonspinning `(2,2)` RG-tail waveform prototype. It is not project
source code.

The physics idea is the following. The paper rewrites radiative multipole
moments in a factorized form, where the usual conservative circular dynamics
enter through an effective source, and long-distance propagation effects enter
through a tail factor. For a benchmark forecast we keep only the dominant
quadrupolar mode and use the Section-IV comparable-mass ingredients to build a
single complex correction factor

```text
hhat_22 = H_eff T_22 rho_22^2 exp(i delta_22).
```

The parameter `lambda_RG` is a simple phenomenological deformation of the
running part of the radiative tail. It does not change the conservative
energy or angular momentum. The value `lambda_RG = 1` corresponds to the GR
running predicted by the paper.

Use `G = c = 1`, `nu = eta`, and

```text
x = (M Omega)^(2/3),      f = Omega / pi,      x_f = (pi M f)^(2/3).
```

## Conservative Sector

This block describes the circular binary dynamics used as input to the
radiative waveform. `E_real/M` is the real two-body energy along circular
orbits, and `p_phi,circ/(mu M)` is the dimensionless circular angular momentum.
They are GR quantities here; the RG deformation is not applied to this
conservative sector.

```text
E_real/M = 1 - (nu x / 2) [c0 + c1 x + c2 x^2 + c3 x^3 + c4 x^4]
```

```text
c0 = 1
c1 = -3/4 - nu/12
c2 = -27/8 + 19 nu/8 - nu^2/24
c3 = -675/64
     + (34445/576 - 205 pi^2/96) nu
     - 155 nu^2/96
     - 35 nu^3/5184
c4 = -3969/128
     + (-123671/5760 + 9037 pi^2/1536 + 896 gamma_E/15
        + 448 log(16 x)/15) nu
     + (498449/3456 - 3157 pi^2/576) nu^2
     + 301 nu^3/1728
     + 77 nu^4/31104
```

```text
p_phi,circ/(mu M) = x^(-1/2) [d0 + d1 x + d2 x^2 + d3 x^3 + d4 x^4]
```

```text
d0 = 1
d1 = 3/2 + nu/6
d2 = 27/8 - 19 nu/8 + nu^2/24
d3 = 135/16
     + (-6889/144 + 41 pi^2/24) nu
     + 31 nu^2/24
     + 7 nu^3/1296
d4 = 2835/128
     + (98869/5760 - 128 gamma_E/3 - 6455 pi^2/1536
        - 64 log(16 x)/3) nu
     + (356035/3456 - 2255 pi^2/576) nu^2
     - 215 nu^3/1728
     - 55 nu^4/31104
```

For even parity,

```text
H_eff = [(E_real/M)^2 - 1]/(2 nu) + 1.
```

`H_eff` is the even-parity effective source. It is the conservative source
factor multiplying the radiative tail and residual waveform corrections.

## Running Tail

This block is the part sensitive to the RG-running idea. The paper defines a
universal anomalous dimension for the radiative multipoles. For the `(2,2)`
mode it depends on the dimensionless radiative frequency `khat` and on the
angular-momentum combination `J`.

For the `(2,2)` mode,

```text
m = 2
khat = (E_real/M) m Omega
J = [p_phi,circ/(mu M)] / (E_real/M)^2
```

```text
gamma_22^univ =
  -214 khat^2/105
  + 2 m J khat^3/3
  - 3390466 khat^4/1157625
  + 381863 m J khat^5/99225
```

Benchmark deformation:

```text
ellhat_22 = 2 + lambda_RG gamma_22^univ,       lambda_RG = 1 is GR.
```

Thus changing `lambda_RG` scales the running correction inside the tail factor.
It should be viewed as a one-parameter deformation of the radiative
propagation/running sector.

The external radiative tail factor is

```text
T_22 = exp(logT_22)
```

with

```text
logT_22 =
  log(120)
  + (ellhat_22 - 2) log(2 k r_omega)
  + 2 i khat log(4 phi0)
  + log Gamma(ellhat_22 - 1 - 2 i khat)
  - log Gamma(2 ellhat_22 + 2)
  + pi khat
  - i pi (ellhat_22 - 2)/2.
```

For circular orbits in this convention,

```text
k = 2 Omega,       r_omega = 1/x,       2 k r_omega = 4 sqrt(x),
phi0 = exp(17/12 - gamma_E)/4.
```

`T_22` is complex. Its magnitude affects the radiated flux, while its phase
contributes directly to the Fourier-domain waveform phase through
`arg(hhat_22)`.

## Residual Amplitude and Phase

The tail factor does not by itself reproduce the full factorized waveform
mode. The residual amplitude `rho_22` and residual phase `delta_22` collect
additional PN information for the `(2,2)` mode.

```text
eulerlog_2(x) = gamma_E + log(4 sqrt(x))
```

```text
rho_22 = 1 + r1 x + r2 x^2 + r3 x^3 + r4 x^4
```

```text
r1 = -43/42 + 55 nu/84
r2 = -20555/10584 - 33025 nu/21168 + 19583 nu^2/42336
r3 = -4296031/4889808
     + (41 pi^2/192 - 48993925/9779616) nu
     - 6292061 nu^2/3259872
     + 10620745 nu^3/39118464
r4 = 9228174993589/800950550400
     + (-2487107795131/145627372800
        + 464 eulerlog_2(x)/35
        - 9953 pi^2/21504) nu
     + (10815863492353/640760440320 - 3485 pi^2/5376) nu^2
     - 2088847783 nu^3/11650189824
     + 70134663541 nu^4/512608352256
```

```text
y = [(E_real/M) x^(3/2)]^(2/3)
```

```text
delta_22 =
  -17 y^(3/2)/3
  - 24 nu y^(5/2)
  + (30995 nu/1134 + 962 nu^2/135) y^(7/2)
  - 4976 pi nu y^4/105.
```

The dominant factorized correction is

```text
hhat_22 = H_eff T_22 rho_22^2 exp(i delta_22).
```

This object is dimensionless. It corrects the Newtonian `(2,2)` mode and is
also used below to define the dominant-mode flux.

## Frequency-Domain Use

The paper gives factorized radiative-mode ingredients, not a complete
detector-domain Fisher template. For this benchmark, use the formulas above to
construct a frequency-domain inspiral waveform with the interface and cutoff
rules specified in the prompt.

Useful standard relations are:

```text
F_22 = (32/5) nu^2 x^5 |hhat_22|^2,
dt/dx = -M d(E_real/M)/dx / F_22,
f = Omega/pi.
```

Document in `notes.md` how you choose the Fourier phase and SPA amplitude
conventions.
