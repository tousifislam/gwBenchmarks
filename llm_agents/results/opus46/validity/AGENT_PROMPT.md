# Validity Bench — Opus 4.6 Agent

You are an autonomous agent building mismatch prediction models for the Validity Bench.
Your work directory is: `llm_agents/results/opus46/validity/`

## Task

Build models to predict the time-domain mismatch between SXS NR waveforms and NRHybSur3dq8 surrogate model for aligned-spin BBH systems (4D input → scalar output). You must try **at least 20 different modeling approaches** spanning **all four categories** below, and test **at least 3 different parameter reparameterizations**.

## Completion Criteria (ALL must be met)

Before declaring DONE, verify:

- [ ] **>=20 approaches** implemented, trained, and evaluated (check `models/` directory)
- [ ] **>=3 reparameterizations** tested (e.g., raw params, eta+chi_eff+chi_a, log(q)+interaction terms)
- [ ] **All 4 approach categories** represented:
  - [ ] Kernel/GP methods (GPR with RBF, GPR with Matern, KRR, SVR)
  - [ ] Symbolic/analytical — **must use PySR and gplearn** (install if needed: `pip install pysr gplearn`). Run PySR with multiple complexity levels and optimize the discovered expressions. Run gplearn's SymbolicRegressor. Report the best symbolic expressions found. (PySR, polynomial regression, physics-informed distance)
  - [ ] Interpolation (RBF interpolation, nearest-neighbor)
  - [ ] Machine learning (MLP, Bayesian NN, deep ensemble, random forest, XGBoost, LightGBM)
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

When ALL criteria are met, print "VALIDITY_BENCH_COMPLETE" on its own line.

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
    loss="loss(prediction, target) = (prediction - target)^2",
)
model.fit(X_train, y_log_mm)  # X_train = (q, chi1z, chi2z, omega0), y = log10(mm_td)
print(model)  # shows Pareto front of expressions
best_expr = model.sympy()
```

Run PySR to find a symbolic expression for log10(mismatch) as a function of binary parameters. Try at least 2 different reparameterizations as input. Save the full Pareto front of expressions. For each PySR/gplearn run, save all discovered expressions to `saved_model/expressions.json` (list of {expression, complexity, loss} dicts). Pick the best accuracy-complexity tradeoff and optimize coefficients further with scipy.optimize if possible.

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
est.fit(X_train, y_log_mm)  # X_train = (q, chi1z, chi2z, omega0), y = log10(mm_td)
print(est._program)
```

Run gplearn to find symbolic expressions for log10(mismatch). Compare the expressions found by PySR vs gplearn.

## Data

Training: `datasets/validity/validity_training.h5`
Validation: `datasets/validity/validity_validation.h5`

```python
import h5py, numpy as np
with h5py.File("datasets/validity/validity_training.h5", "r") as f:
    q = f["q"][:]
    chi1z = f["chi1z"][:]
    chi2z = f["chi2z"][:]
    omega0 = f["omega0"][:]
    mm_td = f["mm_td"][:]
```

Model in log10 space — mismatches span orders of magnitude (~1e-07 to ~1e-01).

## Loss Function

```
L = RMSE(log10(mm_pred), log10(mm_true))
```

RMSE in log10 space.

## Reparameterizations to test

Must try at least 3 from:
1. **Raw parameters**: (q, chi1z, chi2z, omega0)
2. **Effective spins**: (eta, chi_eff, chi_a, omega0)
3. **Log mass ratio**: (log(q), chi_eff, chi_a, log(omega0))
4. **Interaction terms**: (eta, chi_eff, chi_a, omega0, q*chi_eff, eta*chi_a)
5. **Boundary distance**: features measuring proximity to NRHybSur3dq8 valid region (q<=8, |chi|<=0.8)

## Approach ordering (suggested)

Start simple, increase complexity:
1. GPR (RBF kernel, raw params) — baseline
2. GPR (Matern kernel, eta+chi_eff reparameterization)
3. Random forest on log10(mm)
4. XGBoost on log10(mm)
5. MLP (feedforward network)
6. Kernel ridge regression
7. Polynomial regression in log space
8. Symbolic regression (PySR)
9. Deep ensemble (multiple MLPs)
10. Physics-informed (boundary distance features)
11+ LightGBM, Bayesian NN, ensemble stacking, optimized versions

## Scorecard format

Each `models/NN_name/scorecard.json`:
```json
{
    "approach": "gpr_rbf",
    "approach_number": 1,
    "benchmark": "validity",
    "agent": "opus46",
    "parameterization": "raw_4d",
    "loss": 0.35,
    "loss_components": {"log_rmse": 0.30},
    "runtime_ms": 0.1,
    "n_train": 393,
    "n_val": 393,
    "n_params": 200,
    "notes": "..."
}
```

## LLM-Reasoned Hyperparameter Optimization

Do NOT treat hyperparameter tuning as a black-box numerical optimization (no blind grid search or random search alone). Use **LLM reasoning** to analyze results and make targeted adjustments:

1. **Diagnose**: after each approach, analyze the error patterns. Are predictions worse in the extrapolation region (q>8 or |chi|>0.8)? Is the ECE dominated by over- or under-confidence? Are log-space residuals structured?
2. **Hypothesize**: form a physics-informed hypothesis. E.g., "mismatch predictions are biased high for q<4 — need a boundary-distance feature" or "ECE is poor because GPR variance is miscalibrated — post-hoc Platt scaling should fix this"
3. **Prescribe**: choose specific hyperparameter changes based on the hypothesis. Explain the reasoning.
4. **Iterate**: evaluate, check if the hypothesis was correct, update understanding, try again.

For each approach, try at least one round of reasoned optimization beyond the initial fit. Record the reasoning chain in the CHANGELOG:
- What was observed → what was hypothesized → what was changed → what happened

## Rules

- **datasets/ is READ-ONLY** — never modify
- **gwbenchmarks/ is READ-ONLY** — import only
- Work only in `llm_agents/results/opus46/validity/`
- All plots: Nature-style via `gwbenchmarks.plot_settings`, PNG + PDF, no titles
- Update CHANGELOG.md and progress plot after EVERY approach
- Every model directory must have train.py, predict.py, saved_model/
