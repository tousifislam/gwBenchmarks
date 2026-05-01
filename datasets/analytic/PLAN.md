# Analytic Bench — Agent Plan

## Objective

Derive a **complete analytic closed-form expression** for the (2,2) gravitational waveform mode h22(t) from non-spinning quasi-circular binary black hole mergers as a function of mass ratio q.

**Critical constraint**: the final model must be expressible as a closed-form mathematical formula — no lookup tables, no trained neural networks, no numerical interpolation. The expression must be writable in a single equation (or a small set of equations) using standard mathematical functions.

## Data

- **Training**: `analytic_training.h5` — non-spinning SXS simulations (q = 1–20)
- **Validation**: `analytic_validation.h5` — held-out mass ratios
- **Input**: q (mass ratio, 1D), time grid
- **Output**: complex h22(t) waveform

### Loading data

```python
import h5py, numpy as np

with h5py.File("datasets/analytic/analytic_training.h5", "r") as f:
    n = f.attrs["n_simulations"]
    for i in range(n):
        g = f[f"sim_{i:04d}"]
        q = g.attrs["q"]
        t = g["t"][:]
        h22_real = g["h22_real"][:]
        h22_imag = g["h22_imag"][:]
        h22 = h22_real + 1j * h22_imag
```

## Loss Function

```
L = mismatch + lambda * RMSE(coefficients)
```

- Time-domain mismatch between predicted and true h22
- Optional coefficient regularization

**Scoring**: t0 = 0.001s, alpha = 0.10

## Key Properties

1. **1D parameter space**: only mass ratio q — the simplest possible BBH parameter space
2. **Must be analytic**: the entire point is to find a closed-form expression
3. **Three regimes**: inspiral (quasi-circular orbit), merger (transition), ringdown (damped oscillations)
4. **Known physics**: extensive PN theory for inspiral, QNM frequencies for ringdown
5. **The challenge**: bridging inspiral -> merger -> ringdown smoothly in closed form

## Physical Structure

The waveform naturally decomposes into amplitude A(t) and phase phi(t):

```
h22(t) = A(t; q) * exp(-i * phi(t; q))
```

### Inspiral phase
- Well-described by post-Newtonian (PN) theory
- Phase: phi(t) ~ phi_0 + integral of orbital frequency
- Amplitude: A(t) ~ eta * x(t) where x = (M*omega)^(2/3)
- Key parameter: eta = q/(1+q)^2 (symmetric mass ratio)

### Merger
- Sharp amplitude peak at t = 0
- Rapid frequency evolution
- No clean analytical description exists — this is the creative challenge

### Ringdown
- Damped sinusoid: h ~ A_rd * exp(-t/tau) * exp(-i * omega_QNM * t)
- QNM frequencies known from Kerr BH perturbation theory
- Final spin chi_f(q) needed (itself an analytic fit)

## Parameter and Time Reparameterization

### Mass ratio
The input is q in [1, 20]. The agent should explore which parameterization gives the cleanest analytic expressions:
- **Symmetric mass ratio**: eta = q / (1+q)^2 in [0, 0.25] — appears naturally in PN theory
- **Mass difference**: delta_m = (q-1) / (q+1) in [0, ~0.9]
- **nu = eta**: standard PN convention
- **sqrt(eta)**, **eta^(1/5)**: power transforms that may linearize coefficient dependence

### Time variable
- **t = 0 at peak** (as stored): natural for merger-centric modeling
- **Retarded time**: tau = (t_c - t) / M where t_c is coalescence time — standard PN variable
- **Frequency-domain time**: t parameterized via orbital frequency x = (M*omega)^(2/3)
- **Normalized time**: tau = t / t_total mapping to [0, 1]

### Phase decomposition
- Model phi(t) relative to a PN baseline: phi(t) = phi_PN(t) + delta_phi(t; q), then find analytic delta_phi
- Use the TaylorT1/T2/T3/T4 phase as a starting point

## Approaches to Try (10–15)

### Physics-informed closed forms
1. **PN inspiral + Lorentzian merger + QNM ringdown**: Three-piece model with smooth stitching using tanh or sigmoid transitions.
2. **IMRPhenom-style**: Phenomenological amplitude/phase model with q-dependent coefficients fitted to NR.
3. **EOB-inspired**: Effective-one-body inspired closed-form with resummed PN expressions.
4. **Padé-resummed PN**: Padé approximants of the PN amplitude and phase, extended through merger.

### Symbolic regression
5. **PySR on amplitude A(t; q)**: Use symbolic regression to find a compact expression for the amplitude envelope.
6. **PySR on frequency omega(t; q)**: Find the instantaneous frequency evolution, then integrate for phase.
7. **PySR on full h22**: Direct symbolic regression on real/imaginary parts (ambitious).
8. **PySR on merger correction**: Use PN for inspiral, symbolic regression only for the merger patch.

### Matched asymptotic / composite
9. **Matched asymptotic expansion**: PN expansion for early times, QNM for late times, matched in an overlap region.
10. **Composite model**: A(t) = A_insp(t) * f_merge(t) * A_rd(t) with f_merge a simple transition function.

### Functional form optimization
11. **Amplitude as sum of Gaussians/Lorentzians**: Parameterize A(t; q) with a few basis functions.
12. **Phase as polynomial + log terms**: phi(t; q) = sum of polynomial and log(t_c - t) terms (PN-like).
13. **Damped sinusoid series**: Sum of damped sinusoids with q-dependent frequencies and damping times.
14. **Hypergeometric/Bessel ansatz**: Explore special functions that capture inspiral-to-ringdown transition.
15. **Dimensional analysis + fitting**: Use dimensional analysis to constrain functional forms, then fit free coefficients.

## Analytic Expression Guidelines

The final expression should:
- Be writable as a small set of equations (< 20 lines)
- Use only standard mathematical functions: +, -, *, /, ^, exp, log, sin, cos, tanh, sqrt, etc.
- Have q-dependent coefficients that are themselves analytic functions of q (or eta = q/(1+q)^2)
- Be evaluable in O(N) time where N is the number of time samples
- **Not** involve any numerical integration, ODE solving, or iterative procedures at evaluation time

## Evaluation Checklist

For each approach:
- [ ] Write down the closed-form expression explicitly
- [ ] Fit free coefficients to training data
- [ ] Evaluate on validation mass ratios
- [ ] Compute per-waveform: mismatch, phase RMSE, amplitude RMSE
- [ ] Verify the expression is truly closed-form (no hidden numerics)
- [ ] Time evaluation
- [ ] Save the expression in human-readable form alongside scorecard.json

## Final Deliverables

1. **Progress plot** (updated after every approach): mismatch and expression complexity vs. approach number
2. **Violin plot**: per-approach mismatch distribution on validation set
3. **Pareto plot**: mismatch vs. expression complexity (number of free parameters or terms)
4. **Summary table**: ranked approaches with mismatch, complexity, and combined score
5. **Best expression**: the winning closed-form formula, written out explicitly
6. **Overlay plots**: predicted vs. NR waveforms for several q values showing the best model

All outputs go in `results/<agent>/analytic/`. See `BENCHMARK_PLAN.md` for full directory structure and changelog format.
