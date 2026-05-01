# Analytic Bench — Haiku Agent

You are an autonomous agent deriving closed-form waveform expressions for the Analytic Bench.
Your work directory is: `results/haiku/analytic/`

## Task

Derive a **complete analytic closed-form expression** for the (2,2) gravitational waveform mode h22(t) from non-spinning quasi-circular BBH mergers as a function of mass ratio q. The final model must be expressible as a closed-form formula — no lookup tables, no trained neural networks, no numerical interpolation, **no SVD/PCA basis functions**, **no precomputed data-dependent bases**. You must try **at least 20 different modeling approaches** spanning **all four categories** below, and test **at least 3 different parameter reparameterizations**.

**CRITICAL: What "closed-form" means for this benchmark:**
- The expression for h22(t; q) must be writable as explicit mathematical formulas using standard functions (exp, log, sin, cos, tanh, sqrt, polynomials, rational functions, etc.)
- Coefficients in the expression may be fitted to training data, but the **functional form itself** must be specified analytically — not derived from data decomposition
- **FORBIDDEN**: SVD/PCA bases (these are data-dependent eigenvectors, not closed-form), empirical interpolation, stored lookup tables, trained neural network weights, any approach where the model structure comes from the data rather than from physics or symbolic regression
- **ALLOWED**: Physics-inspired ansatze with fitted coefficients, PySR-discovered expressions, Padé approximants, PN-inspired formulas, sums of Gaussians/Lorentzians with fitted parameters, polynomial/rational functions with fitted coefficients
- Each `expression.txt` must contain the full formula that a human could type into Mathematica/Python and evaluate without any external data files

## Completion Criteria (ALL must be met)

Before declaring DONE, verify:

- [ ] **>=20 approaches** implemented, trained, and evaluated (check `models/` directory)
- [ ] **>=3 reparameterizations** tested (e.g., q, eta=q/(1+q)^2, delta_m=(q-1)/(q+1))
- [ ] **All 4 approach categories** represented:
  - [ ] Physics-informed closed forms (PN+merger+QNM, IMRPhenom-style, EOB-inspired, Pade)
  - [ ] Symbolic regression — **must use PySR and gplearn** (install if needed: `pip install pysr gplearn`). PySR is the **primary tool** for this benchmark. Run PySR extensively with different reparameterizations (q, eta, delta_m), multiple complexity levels, and for different parts of the waveform (amplitude, phase, frequency, merger correction). Run gplearn's SymbolicRegressor for comparison. Report the best symbolic expressions found. (PySR on amplitude, frequency, merger correction, full h22)
  - [ ] Matched asymptotic / composite (overlap matching, composite transitions)
  - [ ] Functional form optimization (Gaussian/Lorentzian sums, polynomial+log phase, damped sinusoids)
- [ ] Each expression is truly closed-form (no hidden numerics, no ODE solving at eval time, **no SVD/PCA bases**, no data-dependent decompositions — the formula must be writable as explicit math that a human can evaluate without any precomputed data files)
- [ ] **PySR** was actually run (not a hand-crafted symbolic form) and the best expressions are saved
- [ ] **gplearn** was actually run and the best expressions are saved
- [ ] Each model directory `models/NN_name/` contains:
  - [ ] `train.py` — self-contained training script (reproducible: loads data, trains, saves model)
  - [ ] `predict.py` — importable prediction function
  - [ ] `saved_model/` — serialized model artifacts (pickle, joblib, numpy arrays, or PySR equation files)
  - [ ] `scorecard.json` — structured results
- [ ] `CHANGELOG.md` updated with an entry for every approach
- [ ] `comparison/progress.{png,pdf}` updated after every approach
- [ ] `comparison/error_histograms.{png,pdf}` — histograms of per-sample errors for each approach: separate training vs validation distributions (use transparency/hatching to distinguish). Must be true histograms, NOT bar charts or simplified violin plots
- [ ] `comparison/error_data.json` — raw per-sample error arrays for every approach (train + validation), saved for future reference
- [ ] `comparison/pareto_accuracy_speed.{png,pdf}` — final Pareto plot (loss vs eval time)
- [ ] `comparison/loss_only_comparison.{png,pdf}` — bar chart or scatter of raw loss (without runtime penalty) across approaches for interpretability
- [ ] `comparison/summary_table.json` — ranked list of all approaches with raw loss
- [ ] Best model identified in `comparison/best_model.json`
- [ ] `comparison/all_expressions.json` — collected closed-form expressions from ALL approaches (PySR Pareto fronts, gplearn results, hand-crafted formulas). Each entry: approach name, expression string, complexity, loss
- [ ] **All plots use descriptive labels** — approaches must be labeled by short descriptive name (e.g., "SVD+GPR (eta)", "PySR (log)", "Poly-15 (raw)"), NOT by bare numbers (1, 2, 3...). Use the approach name from the directory (strip the NN_ prefix). Legends must identify category colors. Pareto plot must label each point with its short name.
- [ ] Best expression written out explicitly in human-readable form

When ALL criteria are met, print "ANALYTIC_BENCH_COMPLETE" on its own line.

## Self-Check Loop

After each approach, run this self-check:

```
1. Count completed approaches in models/ (need >=20)
2. List unique reparameterizations used (need >=3)
3. Check category coverage (need all 4)
4. Verify PySR was actually run (not just a hand-crafted formula)
5. Verify gplearn was actually run
6. Verify each model dir has train.py, predict.py, saved_model/
7. Verify each expression is truly closed-form
8. If any criterion unmet → pick the next approach that fills the gap
9. If all met → generate final comparison plots and declare DONE
```

## Package Installation

Install any missing packages into the conda env before use:

```bash
pip install pysr gplearn
```

PySR may require Julia — if installation fails, try `pip install pysr` which auto-installs Julia. Allocate enough time for PySR runs (they can take minutes).

## Symbolic Regression Details

PySR is the **primary tool** for this benchmark. Use it extensively to discover closed-form expressions for different parts of the waveform.

### PySR (mandatory — primary tool)

```python
from pysr import PySRRegressor

# Example: fit amplitude A(t; q)
model = PySRRegressor(
    niterations=300,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "log", "exp", "sin", "cos", "tanh"],
    maxsize=35,
    populations=30,
    procs=4,
    loss="loss(prediction, target) = abs(prediction - target) / (abs(target) + 1e-10)",
)
# Fit amplitude as function of (t, q) or (t, eta) or (t, delta_m)
model.fit(X_train, y_amplitude)
print(model)  # shows Pareto front of expressions
best_expr = model.sympy()
```

Run PySR extensively:
- **Amplitude A(t; q)**: try with q, eta, delta_m as the mass ratio variable
- **Phase phi(t; q)**: try with different reparameterizations
- **Instantaneous frequency omega(t; q)**: derivative of phase
- **Merger correction**: fit residuals after subtracting PN baseline
- **Different complexity levels**: maxsize from 15 to 40
- **Different time variables**: raw t, retarded time tau = (t_peak - t)/M, normalized tau

Save the full Pareto front of expressions for each target. Pick the best accuracy-complexity tradeoff and optimize coefficients further with scipy.optimize.

### gplearn (mandatory)

```python
from gplearn.genetic import SymbolicRegressor

est = SymbolicRegressor(
    population_size=5000,
    generations=50,
    tournament_size=20,
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
    metric='mse',
    parsimony_coefficient=0.001,
    max_samples=1.0,
    verbose=1,
    random_state=42,
)
est.fit(X_train, y_amplitude)  # fit amplitude, phase, frequency separately
print(est._program)
```

Run gplearn for amplitude, phase, and frequency. Compare the expressions found by PySR vs gplearn.

## Data

Training: `datasets/analytic/analytic_training.h5`
Validation: `datasets/analytic/analytic_validation.h5`

```python
import h5py, numpy as np
with h5py.File("datasets/analytic/analytic_training.h5", "r") as f:
    n = f.attrs["n_simulations"]
    for i in range(n):
        g = f[f"sim_{i:04d}"]
        q = g.attrs["q"]
        t = g["t"][:]
        h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
```

1D parameter space (mass ratio q only). Decompose h22 = A(t) * exp(-i * phi(t)).

## Loss Function

```
L = mean frequency-domain mismatch over total masses [40, 80, 120, 160, 200] M☉
     (f_low=15 Hz, f_high=990 Hz, aLIGO ZeroDetHighPower PSD, maximized over time and phase)
```

## Reparameterizations to test

Must try at least 3 from:
1. **Mass ratio q** (direct)
2. **Symmetric mass ratio**: eta = q / (1+q)^2
3. **Mass difference**: delta_m = (q-1) / (q+1)
4. **Power transforms**: sqrt(eta), eta^(1/5)
5. **Time variables**: retarded time tau = (t_c - t) / M, normalized tau in [0,1]

## Approach ordering (suggested)

Start simple, increase complexity:
1. PN inspiral + Lorentzian merger + QNM ringdown (with tanh stitching)
2. IMRPhenom-style amplitude/phase model
3. Pade-resummed PN expressions
4. PySR on amplitude A(t; q)
5. PySR on instantaneous frequency omega(t; q)
6. PySR on merger correction (PN baseline + symbolic patch)
7. Matched asymptotic expansion (PN early + QNM late)
8. Composite model: A = A_insp * f_merge * A_rd
9. Sum of Gaussians/Lorentzians for amplitude
10. Damped sinusoid series with q-dependent parameters
11+ Hypergeometric ansatz, dimensional analysis, optimized versions

## Analytic Expression Guidelines

The final expression must:
- Be writable as < 20 lines of equations
- Use only standard functions: +, -, *, /, ^, exp, log, sin, cos, tanh, sqrt
- Have q-dependent coefficients that are themselves analytic functions of q (or eta)
- Be evaluable in O(N) time where N is number of time samples
- NOT involve numerical integration, ODE solving, or iterative procedures

## Scorecard format

Each `models/NN_name/scorecard.json`:
```json
{
    "approach": "pn_lorentzian_qnm",
    "approach_number": 1,
    "benchmark": "analytic",
    "agent": "haiku",
    "parameterization": "eta",
    "loss": 0.05,
    "loss_components": {"mismatch_40Msun": 0.04, "mismatch_80Msun": 0.035, "mismatch_120Msun": 0.03, "mismatch_160Msun": 0.028, "mismatch_200Msun": 0.025},
    "runtime_ms": 0.1,
    "n_train": 20,
    "n_val": 20,
    "n_params": 30,
    "n_terms": 15,
    "expression_file": "expression.txt",
    "notes": "..."
}
```

## LLM-Reasoned Hyperparameter Optimization

Do NOT treat hyperparameter tuning as a black-box numerical optimization (no blind grid search or random search alone). Use **LLM reasoning** to analyze results and make targeted adjustments:

1. **Diagnose**: after each approach, analyze the error patterns. Is the mismatch dominated by inspiral phase drift, merger amplitude, or ringdown damping? Which mass ratios are worst?
2. **Hypothesize**: form a physics-informed hypothesis. E.g., "the merger peak amplitude is off by 10% at q=8 — the transition function is too smooth, need a sharper sigmoid with q-dependent width" or "inspiral phase drifts at late times — adding a (t_c - t)^(-1/4) term should help"
3. **Prescribe**: choose specific changes based on the hypothesis. For PySR, adjust complexity limits, operator sets, or input features.
4. **Iterate**: evaluate, check if the hypothesis was correct, update understanding, try again.

For each approach, try at least one round of reasoned optimization beyond the initial fit. Record the reasoning chain in the CHANGELOG:
- What was observed → what was hypothesized → what was changed → what happened

## Rules

- **datasets/ is READ-ONLY** — never modify
- **gwbenchmarks/ is READ-ONLY** — import only
- Work only in `results/haiku/analytic/`
- All plots: Nature-style via `gwbenchmarks.plot_settings`, PNG + PDF, no titles
- Update CHANGELOG.md and progress plot after EVERY approach
- Write each closed-form expression explicitly in `expression.txt` alongside the scorecard
- Every model directory must have train.py, predict.py, saved_model/
