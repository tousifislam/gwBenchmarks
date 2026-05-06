# New Physics Bench — {AGENT_LABEL} Agent

You are an autonomous agent implementing a frequency-domain gravitational waveform with beyond-GR corrections.
Your work directory is: `llm_agents/results/{AGENT}/new_physics/`

## Task

Implement a standalone Python file `candidate_waveform.py` that computes the dominant nonspinning (2,2) mode inspiral waveform with RG-tail corrections from arXiv:2602.08833.

## Function Signature

```python
def h_of_f(
    f,
    Mc,
    eta,
    dL,
    tc=0.0,
    phic=0.0,
    lambda_RG=1.0,
    f_low=20.0,
    fmax_over_fisco=1.3,
    sigma_taper_over_fisco=0.01,
    phase_only=False,
):
    """Return complex frequency-domain strain array with same shape as f."""
    ...
```

The function accepts a NumPy frequency array in Hz and returns a complex array. Use only `numpy` and `scipy`.

## Units and Conventions

Use detector-frame masses and geometric seconds:

```
MSUN_SEC = 4.925491025543576e-6
MPC_SEC  = 1.0292712503e14
M_sec    = Mc * MSUN_SEC / eta^(3/5)
dL_sec   = dL * MPC_SEC
x_f      = (pi * M_sec * f)^(2/3)
nu       = eta
```

ISCO and cutoffs:

```
f_isco = 1 / (pi * 6^(3/2) * M_sec)
f_cut  = fmax_over_fisco * f_isco
sigma  = sigma_taper_over_fisco * f_isco
W(f)   = 1 / [1 + exp((f - f_isco) / sigma)]
```

Set h(f) = 0 for f < f_low and f >= f_cut. Apply the Fermi taper W(f) within the valid band.

## Physics: Factorized (2,2) Mode

The key object is:

```
hhat_22 = H_eff * T_22 * rho_22^2 * exp(i * delta_22)
```

### Conservative Sector (E_real/M)

```
E_real/M = 1 - (nu * x / 2) * [c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4]
```

```
c0 = 1
c1 = -3/4 - nu/12
c2 = -27/8 + 19*nu/8 - nu^2/24
c3 = -675/64 + (34445/576 - 205*pi^2/96)*nu - 155*nu^2/96 - 35*nu^3/5184
c4 = -3969/128
     + (-123671/5760 + 9037*pi^2/1536 + 896*gamma_E/15 + 448*log(16*x)/15)*nu
     + (498449/3456 - 3157*pi^2/576)*nu^2
     + 301*nu^3/1728 + 77*nu^4/31104
```

### Angular Momentum (p_phi_circ / (mu*M))

```
p_phi/(mu*M) = x^(-1/2) * [d0 + d1*x + d2*x^2 + d3*x^3 + d4*x^4]
```

```
d0 = 1
d1 = 3/2 + nu/6
d2 = 27/8 - 19*nu/8 + nu^2/24
d3 = 135/16 + (-6889/144 + 41*pi^2/24)*nu + 31*nu^2/24 + 7*nu^3/1296
d4 = 2835/128
     + (98869/5760 - 128*gamma_E/3 - 6455*pi^2/1536 - 64*log(16*x)/3)*nu
     + (356035/3456 - 2255*pi^2/576)*nu^2
     - 215*nu^3/1728 - 55*nu^4/31104
```

### Effective Source

```
H_eff = [(E_real/M)^2 - 1] / (2*nu) + 1
```

### Running Tail (RG deformation)

For the (2,2) mode with m=2:

```
khat = (E_real/M) * m * Omega,    where Omega = x^(3/2)
J = [p_phi/(mu*M)] / (E_real/M)^2
```

Universal anomalous dimension:

```
gamma_22^univ = -214*khat^2/105 + 2*m*J*khat^3/3
               - 3390466*khat^4/1157625 + 381863*m*J*khat^5/99225
```

Running angular momentum:

```
ellhat_22 = 2 + lambda_RG * gamma_22^univ
```

`lambda_RG = 1` is GR. The parameter scales the running correction.

### Tail Factor T_22

```
T_22 = exp(logT_22)
```

```
logT_22 = log(120)
         + (ellhat_22 - 2) * log(2*k*r_omega)
         + 2i * khat * log(2*m*phi0)
         + log_Gamma(ellhat_22 - 1 - 2i*khat)
         - log_Gamma(2*ellhat_22 + 2)
         + pi*khat
         - i*pi*(ellhat_22 - 2)/2
```

where:

```
k = 2*Omega,   r_omega = 1/x,   2*k*r_omega = 4*sqrt(x)
phi0 = exp(17/12 - gamma_E) / 4
```

Use `scipy.special.loggamma` for the complex log-Gamma function.

### Residual Amplitude rho_22

```
eulerlog_2(x) = gamma_E + log(4*sqrt(x))
```

```
rho_22 = 1 + r1*x + r2*x^2 + r3*x^3 + r4*x^4
```

```
r1 = -43/42 + 55*nu/84
r2 = -20555/10584 - 33025*nu/21168 + 19583*nu^2/42336
r3 = -4296031/4889808 + (41*pi^2/192 - 48993925/9779616)*nu
     - 6292061*nu^2/3259872 + 10620745*nu^3/39118464
r4 = 9228174993589/800950550400
     + (-2487107795131/145627372800 + 464*eulerlog_2(x)/35 - 9953*pi^2/21504)*nu
     + (10815863492353/640760440320 - 3485*pi^2/5376)*nu^2
     - 2088847783*nu^3/11650189824 + 70134663541*nu^4/512608352256
```

### Residual Phase delta_22

```
y = [(E_real/M) * x^(3/2)]^(2/3)
```

```
delta_22 = -17*y^(3/2)/3 - 24*nu*y^(5/2)
           + (30995*nu/1134 + 962*nu^2/135)*y^(7/2)
           - 4976*pi*nu*y^4/105
```

## Frequency-Domain Waveform Construction

Use the balance-law SPA approach:

### Flux

```
F_22 = (32/5) * nu^2 * x^5 * |hhat_22|^2
```

### Chirp Rate

```
dt/dx = -M_sec * d(E_real/M)/dx / F_22
```

### SPA Phase

Integrate from x to x_ref (at f_cut):

```
Psi_orb(f) = integral_x^x_ref [2*(x'^(3/2)/M_sec) * dt/dx'] dx'
           - 2*pi*f * integral_x^x_ref [dt/dx'] dx'
```

The total FD phase is:

```
phase(f) = 2*pi*f*tc - phic - pi/4 + Psi_orb(f) + arg(hhat_22(x))
```

### Amplitude

Newtonian amplitude:

```
A_N(f) = sqrt(5/24) * Mc_sec^(5/6) / (dL_sec * pi^(2/3)) * f^(-7/6)
```

Balance-law SPA amplitude correction:

```
A(f) = A_N(f) * sqrt(-2 * d(E_real/M)/dx / nu)
```

### Final Waveform

```
h(f) = A(f) * exp(i * phase(f)) * W(f)
```

If `phase_only=True`, use `A(f) = A_N(f)` (Newtonian amplitude only).

## Scoring

Your implementation will be scored on 144 test cases (4 chirp masses x 4 eta x 3 distances x 3 lambda_RG values) using frequency-domain mismatch:

```
mismatch = 1 - max_{t,phi} <h_cand, h_ref> / sqrt(<h_cand, h_cand> <h_ref, h_ref>)
```

with PyCBC `aLIGOZeroDetHighPower` PSD, f_low=15 Hz, f_high=990 Hz.

## Deliverables

- `candidate_waveform.py` — standalone implementation of `h_of_f()`

When complete, print "NEW_PHYSICS_BENCH_COMPLETE" on its own line.
