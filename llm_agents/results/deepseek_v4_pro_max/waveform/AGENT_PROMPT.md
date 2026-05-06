# Waveform Bench — DeepSeek V4 Pro Max Agent

You are an autonomous agent building surrogate models for the Waveform Bench.
Your work directory is: `llm_agents/results/deepseek_v4_pro_max/waveform/`

## Task

Build surrogate models for the coprecessing-frame (2,2) gravitational waveform mode h22_copr(t) from precessing quasi-circular BBH mergers. The surrogate takes 7D binary parameters as input and outputs a complex time series. You must try **at least 20 different modeling approaches** spanning **all four categories** below, and test **at least 3 different parameter reparameterizations**.

## Completion Criteria (ALL must be met)

Before declaring DONE, verify:

- [ ] **>=20 approaches** implemented, trained, and evaluated (check `models/` directory)
- [ ] **>=3 reparameterizations** tested (e.g., raw params, eta+chi_eff+chi_p, spherical spins)
- [ ] **All 4 approach categories** represented:
  - [ ] SVD/decomposition-based (SVD+GPR, SVD+polynomial, SVD+NN, EIM)
  - [ ] Symbolic/analytical — **must use PySR and gplearn** (install if needed: `pip install pysr gplearn`). Run PySR with multiple complexity levels and optimize the discovered expressions. Run gplearn's SymbolicRegressor. Report the best symbolic expressions found. (PySR on SVD coefficients, PN+corrections, physics-informed)
  - [ ] Interpolation/kernel (RBF, kernel ridge, nearest-neighbor+correction)
  - [ ] Machine learning (neural network, random forest, gradient boosting)
- [ ] **PySR** was actually run (not a hand-crafted symbolic form) and the best expressions are saved
- [ ] **gplearn** was actually run and the best expressions are saved
- [ ] Each model directory `models/NN_name/` contains:
  - [ ] `train.py` — self-contained training script (reproducible: loads data, trains, saves model)
  - [ ] `predict.py` — importable prediction function
  - [ ] `saved_model/` — serialized model artifacts (pickle, joblib, numpy arrays, or PySR equation files)
  - [ ] `scorecard.json` — structured results
- [ ] `CHANGELOG.md` updated with an entry for every approach
- [ ] `comparison/progress.{png,pdf}` updated after every approach
- [ ] `comparison/error_histograms.{png,pdf}` — histograms of per-sample errors for each approach: separate training vs validation distributions (use transparency/hatching to distinguish). Overlay NR error floor as a vertical reference line. Must be true histograms, NOT bar charts or simplified violin plots
- [ ] `comparison/error_data.json` — raw per-sample error arrays for every approach (train + validation), saved for future reference
- [ ] `comparison/pareto_accuracy_speed.{png,pdf}` — final Pareto plot (loss vs eval time)
- [ ] `comparison/loss_only_comparison.{png,pdf}` — bar chart or scatter of raw loss (without runtime penalty) across approaches for interpretability
- [ ] `comparison/summary_table.json` — ranked list of all approaches with raw loss
- [ ] Best model identified in `comparison/best_model.json`
- [ ] **All plots use descriptive labels** — approaches must be labeled by short descriptive name (e.g., "SVD+GPR (eta)", "PySR (log)", "Poly-15 (raw)"), NOT by bare numbers (1, 2, 3...). Use the approach name from the directory (strip the NN_ prefix). Legends must identify category colors. Pareto plot must label each point with its short name.

When ALL criteria are met, print "WAVEFORM_BENCH_COMPLETE" on its own line.

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
model.fit(X_train, y_train)  # X_train = params, y_train = SVD coefficients
print(model)  # shows Pareto front of expressions
best_expr = model.sympy()
```

Run PySR on SVD coefficients (fit each coefficient as a function of binary parameters). Try at least 2 different reparameterizations as input. Save the full Pareto front of expressions. For each PySR/gplearn run, save all discovered expressions to `saved_model/expressions.json` (list of {expression, complexity, loss} dicts). Pick the best accuracy-complexity tradeoff and optimize coefficients further with scipy.optimize if possible.

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
est.fit(X_train, y_train)  # X_train = params, y_train = SVD coefficients
print(est._program)
```

Run gplearn on SVD coefficients. Compare the expressions found by PySR vs gplearn.

## Data

Training: `datasets/waveform/waveform_training.h5`
Validation: `datasets/waveform/waveform_validation.h5`

```python
import h5py, numpy as np
with h5py.File("datasets/waveform/waveform_training.h5", "r") as f:
    n = f.attrs["n_simulations"]
    for i in range(n):
        g = f[f"sim_{i:04d}"]
        q = g.attrs["q"]
        chi1 = [g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]]
        chi2 = [g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]]
        omega0 = g.attrs["omega0"]
        t = g["t"][:]
        h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
```

NR error floor: median FD mismatch ~ 1.4e-03.

## Loss Function

```
L = mean frequency-domain mismatch over total masses [40, 80, 120, 160, 200] M☉
     (f_low=15 Hz, f_high=990 Hz, aLIGO ZeroDetHighPower PSD, maximized over time and phase)
```

## Time and Phase Conventions

Experiment with at least 2 time conventions:
- **t = 0 at peak** (as stored)
- **t = 0 at start**: shift so all waveforms start at t = 0
- **Reversed time**: t = 0 at the last point

Phase alignment options:
- **phase = 0 at peak** (as stored)
- **Initial phase = 0**
- **Phase relative to PN baseline**

## Reparameterizations to test

Must try at least 3 from:
1. **Raw parameters**: (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
2. **Effective spins**: (eta, chi_eff, chi_p, |chi1|, |chi2|, theta1, theta2)
3. **Mass difference + spins**: (delta_m, chi_eff, chi_p, |chi1|, |chi2|, phi1, phi2)
4. **Spherical spins**: (eta, |chi1|, theta1, phi1, |chi2|, theta2, phi2)
5. **Include omega0**: add reference orbital frequency as an input parameter

## Approach ordering (suggested)

Start simple, increase complexity:
1. SVD + GPR (RBF kernel, raw params) — baseline
2. SVD + GPR (Matern kernel, eta+chi_eff reparameterization)
3. SVD + polynomial regression
4. SVD + neural network (MLP)
5. SVD + random forest
6. EIM + GPR
7. SVD + symbolic regression (PySR)
8. Autoencoder + MLP
9. Amplitude/phase decomposition + separate fits
10. RBF interpolation on SVD coefficients
11+ Additional variations or optimized versions

## Scorecard format

Each `models/NN_name/scorecard.json`:
```json
{
    "approach": "svd_gpr",
    "approach_number": 1,
    "benchmark": "waveform",
    "agent": "deepseek_v4_pro_max",
    "parameterization": "raw_7d",
    "time_convention": "t0_at_peak",
    "loss": 0.023,
    "loss_components": {"mismatch_40Msun": 0.021, "mismatch_80Msun": 0.019, "mismatch_120Msun": 0.018, "mismatch_160Msun": 0.017, "mismatch_200Msun": 0.016},
    "runtime_ms": 12.3,
    "n_train": 250,
    "n_val": 250,
    "n_params": 15000,
    "notes": "..."
}
```

## LLM-Reasoned Hyperparameter Optimization

Do NOT treat hyperparameter tuning as a black-box numerical optimization (no blind grid search or random search alone). Use **LLM reasoning** to analyze results and make targeted adjustments:

1. **Diagnose**: after each approach, analyze the error patterns. Which waveforms have the worst mismatch? Is it phase or amplitude that dominates? Are errors concentrated in a region of parameter space (high q, high spin, high precession)?
2. **Hypothesize**: form a physics-informed hypothesis. E.g., "phase errors dominate at high chi_p because precession modulations aren't captured by the SVD basis — need more basis vectors for precessing cases" or "mismatch is worst for long waveforms because phase error accumulates — modeling phase residual relative to PN baseline should help"
3. **Prescribe**: choose specific hyperparameter changes based on the hypothesis. Explain the reasoning.
4. **Iterate**: evaluate, check if the hypothesis was correct, update understanding, try again.

For each approach, try at least one round of reasoned optimization beyond the initial fit. Record the reasoning chain in the CHANGELOG:
- What was observed → what was hypothesized → what was changed → what happened

## Rules

- **datasets/ is READ-ONLY** — never modify
- **gwbenchmarks/ is READ-ONLY** — import only
- Work only in `llm_agents/results/deepseek_v4_pro_max/waveform/`
- All plots: Nature-style via `gwbenchmarks.plot_settings`, PNG + PDF, no titles
- Update CHANGELOG.md and progress plot after EVERY approach
- Every model directory must have train.py, predict.py, saved_model/
