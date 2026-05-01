# Dynamics Bench — GPT-5.2 Agent

You are an autonomous agent building surrogate models for the Dynamics Bench.
Your work directory is: `llm_agents/results/gpt52/dynamics/`

## Task

Build surrogate models for eccentric BBH orbital dynamics. The target output is the PN frequency parameter x(t) only. You must try **at least 20 different modeling approaches** spanning **all four categories** below, and test **at least 3 different parameter reparameterizations**.

## Completion Criteria (ALL must be met)

Before declaring DONE, verify:

- [ ] **>=20 approaches** implemented, trained, and evaluated (check `models/` directory)
- [ ] **>=3 reparameterizations** tested (e.g., raw params, eta+chi_eff+log_e0, trig anomaly)
- [ ] **All 4 approach categories** represented:
  - [ ] SVD/decomposition-based (SVD+GPR, SVD+NN, SVD+polynomial, EIM)
  - [ ] Symbolic/physics-informed — **must use PySR and gplearn** (install if needed: `pip install pysr gplearn`). Run PySR with multiple complexity levels and optimize the discovered expressions. Run gplearn's SymbolicRegressor. Report the best symbolic expressions found. (PySR for ODEs, PN+corrections, symbolic ODE)
  - [ ] Interpolation/kernel (RBF, kernel methods, nearest-neighbor)
  - [ ] Machine learning (Neural ODE, LSTM/GRU, CNN, random forest, gradient boosting)
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
- [ ] **All plots use descriptive labels** — approaches must be labeled by short descriptive name (e.g., "SVD+GPR (eta)", "PySR (log)", "Poly-15 (raw)"), NOT by bare numbers (1, 2, 3...). Use the approach name from the directory (strip the NN_ prefix). Legends must identify category colors. Pareto plot must label each point with its short name.

When ALL criteria are met, print "DYNAMICS_BENCH_COMPLETE" on its own line.

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

PySR may require Julia — if installation fails, try `pip install pysr` which auto-installs Julia. Allocate enough time for PySR runs (they can take minutes).

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
    loss="loss(prediction, target) = abs(prediction - target)",
)
model.fit(X_train, y_train)  # X_train = params, y_train = SVD coefficients for e(t), zeta(t), x(t)
print(model)  # shows Pareto front of expressions
best_expr = model.sympy()
```

Run PySR on SVD coefficients for each dynamical quantity (e, zeta, x). Try at least 2 different reparameterizations as input. Save the full Pareto front of expressions. For each PySR/gplearn run, save all discovered expressions to `saved_model/expressions.json` (list of {expression, complexity, loss} dicts). Pick the best accuracy-complexity tradeoff and optimize coefficients further with scipy.optimize if possible.

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
est.fit(X_train, y_train)  # X_train = params, y_train = SVD coefficients for e(t), zeta(t), x(t)
print(est._program)
```

Run gplearn on SVD coefficients for dynamics. Compare the expressions found by PySR vs gplearn.

## Data

Training: `datasets/dynamics/dynamics_training.h5`
Validation: `datasets/dynamics/dynamics_validation.h5`

```python
import h5py, numpy as np
with h5py.File("datasets/dynamics/dynamics_training.h5", "r") as f:
    n = f.attrs["n_simulations"]
    for i in range(n):
        g = f[f"sim_{i:04d}"]
        q = g.attrs["q"]
        chi1z, chi2z = g.attrs["chi1z"], g.attrs["chi2z"]
        e0, zeta0, omega0 = g.attrs["e0"], g.attrs["zeta0"], g.attrs["omega0"]
        t = g["t"][:]
        e = g["e"][:]
        zeta = g["zeta"][:]
        x = g["x"][:]
```

## Loss Function

```
L = sqrt(mean((x_pred - x_true)^2 / x_true^2))   [pointwise RMS relative error on x(t)]
```

Weights all time steps equally in fractional terms, avoiding late-time bias from x growing monotonically.

## Time Conventions

Experiment with at least 2 time conventions:
- **t = 0 at end** (last time point): all evolutions end at the same point
- **t = 0 at start**: all begin at t = 0 (durations vary)
- **Normalized time**: tau in [0, 1] by dividing by total duration

## Reparameterizations to test

Must try at least 3 from:
1. **Raw parameters**: (q, chi1z, chi2z, e0, zeta0, omega0)
2. **Effective spin + log eccentricity**: (eta, chi_eff, chi_a, log(e0), zeta0, omega0)
3. **Trigonometric anomaly**: (eta, chi_eff, chi_a, e0, cos(zeta0), sin(zeta0), omega0)
4. **Log frequency**: (eta, chi_eff, chi_a, e0, zeta0, log(omega0))
5. **Fully transformed**: (eta, chi_eff, chi_a, log(e0), cos(zeta0), sin(zeta0), log(omega0))

## Approach ordering (suggested)

Start simple, increase complexity:
1. SVD + GPR (raw params) — baseline
2. SVD + GPR (eta+chi_eff reparameterization)
3. SVD + polynomial regression
4. SVD + neural network (MLP)
5. SVD + random forest
6. Neural ODE
7. LSTM / GRU sequence model
8. PN evolution + learned corrections
9. SVD + symbolic regression (PySR)
10. EIM + GPR
11+ RBF interpolation, gradient boosting, ensemble, optimized versions

## Scorecard format

Each `models/NN_name/scorecard.json`:
```json
{
    "approach": "svd_gpr",
    "approach_number": 1,
    "benchmark": "dynamics",
    "agent": "gpt52",
    "parameterization": "raw_6d",
    "time_convention": "t0_at_end",
    "loss": 0.015,
    "loss_components": {"rms_relative_error_x": 0.015},
    "runtime_ms": 15.0,
    "n_train": 250,
    "n_val": 250,
    "n_params": 5000,
    "notes": "..."
}
```

## LLM-Reasoned Hyperparameter Optimization

Do NOT treat hyperparameter tuning as a black-box numerical optimization (no blind grid search or random search alone). Use **LLM reasoning** to analyze results and make targeted adjustments:

1. **Diagnose**: after each approach, analyze the error patterns. Which output (e, x, zeta) has the largest error? Are errors worse for high eccentricity or long evolutions? Does the circular error in zeta dominate?
2. **Hypothesize**: form a physics-informed hypothesis. E.g., "eccentricity errors are worst at late times when e→0 — modeling log(e) should help" or "zeta errors are large because the anomaly wraps — embedding as (cos(zeta), sin(zeta)) in the SVD should fix this"
3. **Prescribe**: choose specific hyperparameter changes based on the hypothesis. Explain the reasoning.
4. **Iterate**: evaluate, check if the hypothesis was correct, update understanding, try again.

For each approach, try at least one round of reasoned optimization beyond the initial fit. Record the reasoning chain in the CHANGELOG:
- What was observed → what was hypothesized → what was changed → what happened

## Rules

- **datasets/ is READ-ONLY** — never modify
- **gwbenchmarks/ is READ-ONLY** — import only
- Work only in `llm_agents/results/gpt52/dynamics/`
- All plots: Nature-style via `gwbenchmarks.plot_settings`, PNG + PDF, no titles
- Update CHANGELOG.md and progress plot after EVERY approach
- Every model directory must have train.py, predict.py, saved_model/
