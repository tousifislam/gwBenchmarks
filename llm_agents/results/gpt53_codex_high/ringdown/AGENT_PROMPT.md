# Ringdown Bench — GPT-5.3 Codex High Agent

You are an autonomous agent building QNM frequency models for the Ringdown Bench.
Your work directory is: `llm_agents/results/gpt53_codex_high/ringdown/`

## Task

Build models to predict quasi-normal mode (QNM) frequencies (omega_R, omega_I) of Kerr black holes as functions of dimensionless spin a/M. You must try **at least 20 different modeling approaches** spanning **all four categories** below, and test **at least 3 different spin reparameterizations**.

## Completion Criteria (ALL must be met)

Before declaring DONE, verify:

- [ ] **>=20 approaches** implemented, trained, and evaluated (check `models/` directory)
- [ ] **>=3 reparameterizations** tested (e.g., raw a, -log(1-a), sqrt(1-a^2))
- [ ] **All 4 approach categories** represented:
  - [ ] Analytical/classical (polynomial, Chebyshev, Padé, rational)
  - [ ] Symbolic regression — **must use PySR and gplearn** (install if needed: `pip install pysr gplearn`). Run PySR with multiple complexity levels and optimize the discovered expressions. Run gplearn's SymbolicRegressor. Report the best symbolic expressions found.
  - [ ] Interpolation (spline, RBF, Chebyshev nodes)
  - [ ] Machine learning (GPR, neural network, random forest, or gradient boosting)
- [ ] **PySR** was actually run (not a hand-crafted symbolic form) and the best expressions are saved
- [ ] **gplearn** was actually run and the best expressions are saved
- [ ] Each model directory `models/NN_name/` contains:
  - [ ] `train.py` — self-contained training script (reproducible: loads data, trains, saves model)
  - [ ] `predict.py` — importable prediction function: `def predict(spin_array) -> (omega_r, omega_i)`
  - [ ] `saved_model/` — serialized model artifacts (pickle, joblib, numpy arrays, or PySR equation files)
  - [ ] `scorecard.json` — structured results
- [ ] `CHANGELOG.md` updated with an entry for every approach (include timestamp, method details, key observations, what to try next)
- [ ] `comparison/progress.{png,pdf}` updated after every approach
- [ ] `comparison/error_histograms.{png,pdf}` — histograms of per-sample errors for each approach: separate training vs validation distributions (use transparency/hatching to distinguish). Must be true histograms, NOT bar charts or simplified violin plots
- [ ] `comparison/error_data.json` — raw per-sample error arrays for every approach (train + validation), saved for future reference
- [ ] `comparison/pareto_accuracy_speed.{png,pdf}` — final Pareto plot (loss vs eval time)
- [ ] `comparison/loss_only_comparison.{png,pdf}` — bar chart or scatter of raw loss (without runtime penalty) across approaches for interpretability
- [ ] `comparison/summary_table.json` — ranked list of all approaches with raw loss
- [ ] Best model identified in `comparison/best_model.json`
- [ ] **All plots use descriptive labels** — approaches must be labeled by short descriptive name (e.g., "Poly-10 (raw)", "Rational [7,7] (log)", "PySR (log)", "GPR-RBF"), NOT by bare numbers (1, 2, 3...). Use the approach name from the directory (strip the NN_ prefix). Legends must identify category colors. Pareto plot must label each point with its short name.

When ALL criteria are met, print "RINGDOWN_BENCH_COMPLETE" on its own line.

## Self-Check Loop

After each approach, run this self-check:

```
1. Count completed approaches in models/ (need >=20)
2. List unique reparameterizations used (need >=3)
3. Check category coverage (need all 4)
4. Verify PySR was actually run (not just a hand-crafted formula)
5. Verify gplearn was actually run
6. Verify each model dir has train.py, predict.py, saved_model/
7. If any criterion unmet → pick the next approach that fills the gap
8. If all met → generate final comparison plots and declare DONE
```

## Package Installation

Install any missing packages into the conda env before use:

```bash
pip install pysr gplearn
```

PySR may require Julia — if installation fails, try `pip install pysr` which auto-installs Julia, or use `julia -e 'using Pkg; Pkg.add("SymbolicRegression")'` first. Allocate enough time for PySR runs (they can take minutes).

## Data

Training: `datasets/ringdown/ringdown_training.h5`
Validation: `datasets/ringdown/ringdown_validation.h5`

```python
import h5py, numpy as np
with h5py.File("datasets/ringdown/ringdown_training.h5", "r") as f:
    g = f["l2/m+2/n0"]  # example: l=2, m=+2, n=0 fundamental mode
    spin = g["spin"][:]          # 531 points in [0.001, 0.999999]
    omega_r = g["omega_real"][:] 
    omega_i = g["omega_imag"][:]
```

For simplicity, focus primarily on the (l=2, m=2, n=0) fundamental mode for the main comparison, but verify the best model generalizes to other modes (l=2,m=2,n=1; l=3,m=3,n=0; l=4,m=4,n=0).

## Loss Function

```
L = (mean(|pred - true| / |true|) for omega_R  +  same for omega_I) / 2
```

Mean of the two relative errors. Lower is better.

## Reparameterizations to test

Must try at least 3 from:
1. **Raw spin**: x = a (direct)
2. **Log-compactified**: x = -log(1 - a)
3. **Sqrt irreducible**: x = sqrt(1 - a^2)
4. **Compactified**: x = a / (1 - a)
5. **Chebyshev mapped**: x = 2*a - 1 (maps [0,1] to [-1,1])

## Symbolic Regression Details

### PySR (mandatory)

```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "log", "exp", "sin", "cos"],
    maxsize=30,
    populations=30,
    procs=4,
    loss="loss(prediction, target) = abs(prediction - target) / abs(target)",
)
model.fit(X_train, y_train)
print(model)  # shows Pareto front of expressions
best_expr = model.sympy()
```

Run PySR separately for omega_R and omega_I. Try at least 2 different reparameterizations as input. Save the full Pareto front of expressions. For each PySR/gplearn run, save all discovered expressions to `saved_model/expressions.json` (list of {expression, complexity, loss} dicts). Pick the best accuracy-complexity tradeoff and optimize coefficients further with scipy.optimize if possible.

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
est.fit(X_train, y_train)
print(est._program)
```

Run gplearn separately for omega_R and omega_I. Compare the expressions found by PySR vs gplearn.

## Approach ordering (suggested)

Start simple, increase complexity:
1. Polynomial fit (degree 10, raw a) — baseline
2. Polynomial fit (degree 15, -log(1-a)) — test reparameterization
3. Chebyshev polynomial expansion
4. Cubic spline interpolation
5. Rational approximation (Padé)
6. GPR with RBF kernel
7. RBF interpolation
8. Neural network (small MLP)
9. PySR symbolic regression (run with multiple reparameterizations, optimize best expressions)
10. gplearn symbolic regression (compare with PySR)
11. Random forest / gradient boosting
12+ Additional variations or optimized versions of best approaches

## Scorecard format

Each `models/NN_name/scorecard.json`:
```json
{
    "approach": "polynomial_deg10",
    "approach_number": 1,
    "benchmark": "ringdown",
    "agent": "gpt53_codex_high",
    "parameterization": "raw_a",
    "mode": "l2_m2_n0",
    "loss": 0.00123,
    "loss_components": {"rel_error_omega_real": 0.0006, "rel_error_omega_imag": 0.0007},
    "runtime_ms": 0.005,
    "n_train": 531,
    "n_val": 531,
    "n_params": 11,
    "notes": "..."
}
```

For symbolic regression approaches, also include:
```json
{
    "expression_omega_r": "0.373 - 0.148*(1-a)^0.89 + ...",
    "expression_omega_i": "-0.0890 + 0.0578*(1-a)^0.45 + ...",
    "expression_complexity": 15
}
```

## LLM-Reasoned Hyperparameter Optimization

Do NOT treat hyperparameter tuning as a black-box numerical optimization (no blind grid search or random search alone). Use **LLM reasoning** to analyze results and make targeted adjustments:

1. **Diagnose**: after each approach, analyze the error patterns. Where in spin space are errors largest? Is omega_I consistently harder than omega_R? Are errors systematic or random? Does the residual have structure?
2. **Hypothesize**: form a physics-informed hypothesis. E.g., "errors peak near a→1 because omega_I→0 causing large relative errors — a log-transform of the target should help" or "the polynomial oscillates near a=0.99 (Runge phenomenon) — Chebyshev nodes or a reparameterization will fix this"
3. **Prescribe**: choose specific hyperparameter changes based on the hypothesis. Explain the reasoning.
4. **Iterate**: evaluate, check if the hypothesis was correct, update understanding, try again.

For each approach, try at least one round of reasoned optimization beyond the initial fit. Record the reasoning chain in the CHANGELOG:
- What was observed → what was hypothesized → what was changed → what happened

Examples:
- "PySR found `c0 + c1*sqrt(1-a)` but residual shows `(1-a)^1.5` structure → adding that term and re-optimizing coefficients with scipy.optimize"
- "GPR lengthscale=0.3 but function varies on scale ~0.01 near extremal spin → reducing lengthscale or using non-stationary kernel"
- "Random forest errors biased for omega_I < 0.01 — can't extrapolate below training minimum → log-transforming target"
- "RBF interpolation with thin_plate works well at loss=1.7e-6 → try multiquadric and cubic kernels to see if we can do better"

## Rules

- **datasets/ is READ-ONLY** — never modify
- **gwbenchmarks/ is READ-ONLY** — import only
- Work only in `llm_agents/results/gpt53_codex_high/ringdown/`
- All plots: Nature-style via `gwbenchmarks.plot_settings`, PNG + PDF, no titles
- Update CHANGELOG.md and progress plot after EVERY approach
- Every model directory must have train.py, predict.py, saved_model/
