# Analytic Benchmark — CHANGELOG

This is the **clean-room rerun**.  The previous opus47 attempt (24 models in
this directory) was invalidated because **every** model relied on a
data-dependent SVD basis, an EIM lookup table, or an ML regressor — none of
which are closed-form expressions in (t, q).  All 24 invalid model
directories were deleted; the previous `comparison/` was renamed to
`comparison_invalid_backup/`.

This rerun produces **24 genuinely closed-form models** for h22(t; q) on the
non-spinning analytic benchmark.  Every model writes its full formula to
`expression.txt` using only standard math functions (`exp`, `log`, `sin`,
`cos`, `tanh`, `sqrt`, polynomials, rationals, Chebyshev polynomials of t
with explicit numerical coefficients).  Every model has fewer than 200 fitted
numerical coefficients.

The full mismatch metric is the PyCBC-based frequency-domain mismatch from
`gwbenchmarks.metrics.frequency_domain_mismatch`, averaged over the entire
21-waveform validation set and over total masses {40, 80, 120, 160, 200} Msun.

## Categories required by the spec

| Category | Models | Brief idea |
|----------|--------|------------|
| Physics-informed IMR | 1–10, 24 | PN-style inspiral (`(1−t/T_pn)^(−p)` saturating growth) blended via tanh into a power-law ringdown decay; PN chirp phase blended into linear ringdown phase.  Optional Chebyshev-in-τ residual correction. |
| Composite / matched-asymptotic | 11–13 | Same IMR base, but the Chebyshev residual is split into two segments (t ≤ −200 M and t ≥ −200 M) joined with a tanh blend, so the inspiral and merger/ringdown have independent residual fits. |
| Functional-form direct fits | 14–19 | A(t; q) parametrised as a sum of Gaussians (14, 15) or Lorentzians (16, 17) with q-polynomial coefficients; or h22(t; q) directly modelled as a sum of Gaussian-enveloped damped sinusoids (18, 19). |
| Symbolic regression | 20–23 | The five amplitude or phase per-q parameters are fitted with **gplearn** (programs over add/sub/mul/div/log/sqrt) or with a greedy basis-function selection over `{1, x, x², x³, sqrt(x), 1/x, log(x), x·log(x), 1/(1+x)}`.  The remaining parameters and the Chebyshev residual stay polynomial-in-x for stability. |

Three mass-ratio reparameterisations (η, δ_m, log q) are exercised across the
families, plus q itself and 1/q.

## Build pipeline

1. **Per-q decomposition** (`build_models.py`): each training waveform is
   fitted to the IMR base `imr_amp(t)` (5 params) and `imr_phi(t)` (5 params)
   via weighted least-squares.  The amplitude residual `log A − log A_base`
   and phase residual `phi − phi_base` are then expanded in Chebyshev
   polynomials of τ = scaled time, at degrees {10, 12, 14, 16, 18}.  Per-q
   parameters and Chebyshev coefficients are cached to
   `_data/perq_decomp.npz`.
2. **q-dependence fit**: each per-q parameter (or Chebyshev coefficient) is
   then fitted with a degree-{3, 4} polynomial in x = reparam(q).
3. **Functional-form direct fits** (Gauss / Lorentz / damped sinusoid sums)
   are done per-q with `scipy.optimize.curve_fit` and then the same
   polynomial-in-x q-dependence step.
4. **Symbolic-regression layer** (`build_extra.py`): replaces the
   polynomial-in-x for the five amplitude (or phase) base parameters with a
   gplearn or OMP-basis discovered closed-form expression.  Polynomial
   coefficients for the rest are reused.
5. **Trim to <200 params** (`fix_param_count.py`): the cheb14 / poly4 family
   was reduced to effective cheb13 (drop highest term) so every model has
   `n_params < 200` strictly.

## Hardcoded forbidden checks

Each `expression.txt` was inspected for forbidden substrings (`SVD`, `V_K`,
`EIM`, `lookup`, `spline knot`, `random forest`, `gradient boost`, `MLP`,
`KNN`).  None appear in any model's expression.  Each model loads at most a
small `coeffs.npz` containing the polynomial coefficient table — every entry
is also in `expression.txt`, so the .npz is for reproducibility, not for
basis vectors.

## Notes by approach

(Sorted by validation FD-mismatch loss after the trim.)

| # | name | category | param. | loss | n_params |
|---|------|----------|--------|------|----------|
| 03 | phen_imr_cheb14_logq_p4 | physics_imr | log q | 5.22e-02 | 190 |
| 11 | comp_seg_cheb88_eta_p3 | composite | η | 7.13e-02 | 184 |
| 23 | ompbasis_phase_imr_eta | symbolic_regression | η | 7.61e-02 | 160 |
| 05 | phen_imr_cheb14_q_p4 | physics_imr | q | 7.97e-02 | 190 |
| 08 | phen_imr_cheb18_eta_p3 | physics_imr | η | 8.07e-02 | 192 |
| 07 | phen_imr_cheb16_eta_p3 | physics_imr | η | 8.66e-02 | 176 |
| 13 | comp_seg_cheb88_logq_p3 | composite | log q | 9.01e-02 | 184 |
| 01 | phen_imr_cheb14_eta_p4 | physics_imr | η | 1.00e-01 | 190 |
| 10 | phen_imr_cheb14_eta_p3 | physics_imr | η | 1.01e-01 | 160 |
| 20 | gplearn_amp_imr_eta | symbolic_regression | η | 1.03e-01 | 164 |
| 22 | gplearn_amp_imr_logq | symbolic_regression | log q | 1.20e-01 | 157 |
| 06 | phen_imr_cheb12_eta_p4 | physics_imr | η | 1.30e-01 | 180 |
| 02 | phen_imr_cheb14_delta_p4 | physics_imr | δ_m | 1.38e-01 | 190 |
| 09 | phen_imr_cheb10_eta_p4 | physics_imr | η | 1.72e-01 | 160 |
| 21 | ompbasis_phase_imr_delta | symbolic_regression | δ_m | 1.83e-01 | 160 |
| 24 | phen_imr_pure_eta_p4 | physics_imr | η | 2.00e-01 | 50 |
| 12 | comp_seg_cheb88_delta_p3 | composite | δ_m | 2.05e-01 | 184 |
| 04 | phen_imr_cheb14_invq_p4 | physics_imr | 1/q | 2.14e-01 | 190 |
| 16 | lorentz5_amp_eta_p4 | functional_form | η | 4.11e-01 | 100 |
| 14 | gauss5_amp_eta_p4 | functional_form | η | 4.33e-01 | 100 |
| 17 | lorentz5_amp_delta_p4 | functional_form | δ_m | 4.49e-01 | 100 |
| 15 | gauss5_amp_delta_p4 | functional_form | δ_m | 4.64e-01 | 100 |
| 18 | dsin4_eta_p4 | functional_form | η | 6.91e-01 | 100 |
| 19 | dsin4_delta_p4 | functional_form | δ_m | 7.19e-01 | 100 |

## Reasoning by approach family

### Physics-informed IMR (models 01–10, 24)

**Observation.** Inspiral amplitude grows roughly as `(t_c−t)^(−1/4)` (PN
prediction), the merger peak is sharp and asymmetric, post-merger amplitude
decays exponentially with a small power-law correction.  Inspiral phase is
PN-chirpy `phi ∝ (t_c−t)^(5/8)`, ringdown phase is linear `phi = ω_RD · t`.

**Hypothesis.** Pick a 5-parameter amplitude ansatz capturing all three
regimes with a tanh blend, and a 5-parameter phase ansatz with the same
structure.  Per-q parameters as polynomials in η / δ_m / log q.

**Action.** `imr_amp(t; log_Apk, T_pn, p_pre, tau_RD, p_post)` and
`imr_phi(t; phi_pk, omega_RD, omega_pn, T_pn_phi, T_b)`.  Per-q fits via
weighted curve_fit.  Adding a Chebyshev-in-τ residual on `log A` and `phi`
substantially improves the closure error (closure ≈ 1.5% at cheb deg 16).

**Outcome.** Best in-class is **03_phen_imr_cheb14_logq_p4** (loss = 5.2 %).
log q is preferred because it spreads the ratio range more uniformly than η
(η is heavily compressed for large q).  Pure IMR with no Chebyshev residual
(model 24) has only 50 parameters but sits at 20 % loss — the residual
correction is doing real work.

### Composite / matched-asymptotic (models 11–13)

**Observation.** The Chebyshev residual on the *whole* time grid mixes
inspiral-band and merger-band errors; the polynomial-in-q fit averages over
both.

**Hypothesis.** Splitting τ at t = −200 M and fitting independent Chebyshev
expansions on the two segments (joined with a tanh blend) might give the
inspiral residual more independence from the merger residual.

**Action.** `comp_seg_cheb88_*` uses cheb deg 8 on each segment with poly deg
3 in η / δ_m / log q.

**Outcome.** **11_comp_seg_cheb88_eta_p3** ranks 2nd at 7.1 %.  The split
helps moderately, but the IMR-Cheb model still wins thanks to the
log-q reparameterisation.

### Symbolic regression (models 20–23)

**Observation.** Polynomial-in-x fits to the 5 IMR-base parameters can be
brittle for high q (heavy tails of η).

**Hypothesis.** A symbolic regressor might discover a more economical form
(e.g. `log_Apk(η) ∝ log(η)`) that extrapolates better.

**Action.** gplearn 0.4.2 needed an sklearn-compat patch (`_validate_data`,
`n_features_in_`) but then runs.  Used it on the 5 amplitude parameters in η
(20) and log q (22).  PySR was unusable here because importing it triggered
a multi-minute Julia precompilation, which then failed.  As an alternative
"symbolic" path, **OMP basis selection** over a fixed library
`{1, x, x², x³, sqrt(x), 1/x, log(x), x·log(x), 1/(1+x)}` is used for the
5 phase parameters in δ_m (21) and η (23).  These produce literal closed-form
expressions like `(c1)·log(x) + (c2)·sqrt(x) + (c3)·(1/(1+x))` which are
genuine analytic functions of x.

**Outcome.** **23_ompbasis_phase_imr_eta** at 7.6 % is in the top 3 and
demonstrates that a small library of basis functions captures the phase
parameters' η-dependence well.  gplearn-discovered amplitude expressions
score around 10 %, slightly worse than the polynomial baseline.

### Functional-form direct fits (models 14–19)

**Observation.** Sums of Gaussians or Lorentzians fit to A(t) directly and
sums of damped sinusoids fit to h22(t) directly are physics-agnostic and
should be tested.

**Hypothesis.** A 5-Gaussian / 5-Lorentzian / 4-damped-sinusoid ansatz with
~100 polynomial coefficients in η or δ_m might rival the IMR ansatz.

**Action.** `gauss_sum`, `lorentz_sum`, `h_damped_sin` fitted per-q via
curve_fit on bounded parameter ranges, then poly-deg-4 q-dependence.

**Outcome.** All these models score 0.4–0.7 — much worse than the IMR
family.  The dynamic range of the inspiral chirp (3+ decades in t) is
poorly captured by 5-Gaussian/Lorentzian sums, and the damped-sinusoid
direct fit struggles because the *frequency* drift of the inspiral cannot be
captured by a small number of fixed-frequency components.  These remain in
the leaderboard for completeness.

## Final outputs in `comparison/`

- `best_model.json` — model 03_phen_imr_cheb14_logq_p4 (loss 5.22 %).
- `summary_table.json` — all 24 models, ranked.
- `all_expressions.json` — every closed-form formula as a string.
- `error_data.json` — per-sample errors for every model.
- `progress.{png,pdf}` — bar chart of loss per approach.
- `loss_only_comparison.{png,pdf}` — scatter of loss with category colours.
- `pareto_accuracy_speed.{png,pdf}` — runtime vs. loss.
- `error_histograms.{png,pdf}` — per-sample error distributions for the top 8 models.

`comparison_invalid_backup/` is the pre-rerun comparison from the previous
(invalid SVD-based) attempt; kept only for archival reference.
