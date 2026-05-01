# Waveform Bench — Agent Plan

## Objective

Build a surrogate model for the coprecessing-frame (2,2) gravitational waveform mode from precessing quasi-circular binary black hole mergers. The surrogate takes binary parameters as input and outputs a complex time series h22_copr(t).

## Data

- **Training**: `waveform_training.h5` — 250 SXS simulations
- **Validation**: `waveform_validation.h5` — 250 SXS simulations
- **Input parameters** (7D): q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z
- **Additional**: omega0 (reference orbital frequency, varies per sim)
- **Output**: complex h22_copr(t) stored as h22_real(t), h22_imag(t)
- **Time grid**: uniform dt = 0.1M, peak at t = 0, lengths vary (2947–10000 M)

### Loading data

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

## Loss Function

```
L = w1 * mismatch + w2 * RMSE(phase) + w3 * RMSE(log_amplitude)
```

- `mismatch`: time-domain mismatch 1 - <h_pred|h_true> / sqrt(<h_pred|h_pred><h_true|h_true>)
- `RMSE(phase)`: RMSE of unwrapped phase difference
- `RMSE(log_amplitude)`: RMSE of log-amplitude difference

**Scoring**: t0 = 0.01s, alpha = 0.10

## NR Error Floor

The NR resolution error (highest vs. second-highest Lev) sets the accuracy floor:
- Combined FD mismatch: **median = 1.4e-03**, 95th = 5.9e-02
- Stored per-sim in the HDF5 as `nr_fd_mm_M{40,80,120,160,200}` and `nr_fd_mm_combined`

**Target**: match the NR error floor (mismatch ~ 1e-03) while keeping evaluation time < 50 ms.

## Time and Phase Conventions

The data is stored with t = 0 at the coprecessing |h22| amplitude peak. The agent is free to **redefine the time axis** if it helps modeling — common choices in surrogate modeling include:

- **t = 0 at peak** (as stored): natural for merger-ringdown, but inspiral length varies
- **t = 0 at the start**: shift so t_start = 0 for all waveforms; inspiral starts at a common origin, merger happens at different times
- **Reversed time**: t = 0 at the end of the waveform (last point), counting backward

For **phase alignment**, the data has phase(h22) = 0 at t = 0 (peak). The agent may also explore:
- Initial phase = 0 (phase at the start of the waveform is zero)
- Phase referenced to a specific orbital frequency
- Decomposing into amplitude A(t) and phase phi(t) with phi modeled relative to a PN baseline

**The agent should experiment with time/phase conventions** and report which works best for each approach.

## Parameter Reparameterization

The raw parameters are (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z). The agent should explore whether reparameterizations improve modeling accuracy:

- **Symmetric mass ratio**: eta = q / (1 + q)^2 instead of q
- **Mass difference**: delta_m = (q - 1) / (q + 1)
- **Effective spin**: chi_eff = (q * chi1z + chi2z) / (1 + q)
- **Precessing spin**: chi_p (single-spin precession parameter)
- **Spin magnitudes + angles**: (|chi1|, theta1, phi1, |chi2|, theta2, phi2) in spherical coordinates
- **Approximately conserved quantities**: chi_eff and spin magnitudes are approximately conserved during inspiral, unlike chi1x/chi1y which precess rapidly
- **Log or power transforms**: log(q), q^(1/2), etc.
- **Include omega0**: the reference orbital frequency may improve accuracy since spin vectors are defined at this frequency

**Systematically test** at least 2–3 reparameterizations and report which gives the best results.

## Key Challenges

1. **Variable-length time series**: waveforms range from ~30k to 100k samples
2. **7D parameter space**: precessing spins create complex morphology
3. **Phase accuracy**: small phase errors accumulate over long inspirals
4. **Merger-ringdown**: sharp features near t = 0 require special treatment
5. **Spin reference frequency**: omega0 varies per sim; spins precess

## Approaches to Try (10–15)

### SVD-based surrogates (the gold standard for NR waveforms)
1. **SVD + GPR**: Decompose waveforms via SVD (on a common time grid), fit each coefficient with GPR. This is the NRSur approach.
2. **SVD + polynomial regression**: Same decomposition, fit with multivariate polynomials.
3. **SVD + neural network**: SVD basis coefficients predicted by an MLP.
4. **SVD + random forest / gradient boosting**: Tree-based regression on SVD coefficients.
5. **SVD + symbolic regression**: Use PySR to find analytic expressions for key SVD coefficients.

### Empirical interpolation
6. **EIM + GPR**: Greedy-select a reduced set of time nodes, fit the waveform values at those nodes with GPR, reconstruct via interpolation.
7. **EIM + polynomial fit**: Same but with polynomial fits at each node.

### Neural network approaches
8. **Autoencoder + MLP**: Train an autoencoder on waveforms to learn a latent space, then map parameters -> latent codes with an MLP.
9. **1D CNN / WaveNet**: Directly predict the waveform as a 1D signal conditioned on parameters.
10. **Neural ODE**: Model the waveform evolution as a learned dynamical system.

### Physics-informed
11. **PN inspiral + data-driven merger**: Use post-Newtonian expressions for the inspiral, train a model only for the merger-ringdown transition.
12. **Amplitude/phase decomposition**: Fit amplitude A(t) and phase phi(t) separately with different models.

### Hybrid / other
13. **RBF interpolation**: Radial basis function interpolation in 7D parameter space on SVD coefficients.
14. **Kernel ridge regression**: On SVD coefficients with optimized kernel.
15. **Nearest-neighbor + correction**: Start from nearest training waveform, learn a correction model.

## Evaluation Checklist

For each approach:
- [ ] Train on training set
- [ ] Predict on all 250 validation waveforms
- [ ] Compute per-waveform: mismatch, phase RMSE, log-amplitude RMSE
- [ ] Compute combined loss L
- [ ] Time 100 evaluations, report median wall-clock time per prediction
- [ ] Save scorecard.json with all metrics
- [ ] Generate diagnostic plots (worst cases, error distributions)

## Final Deliverables

1. **Progress plot** (updated after every approach): loss and eval time vs. approach number, NR error floor marked
2. **Violin plot**: per-approach distribution of log10(mismatch) on validation set
3. **Pareto plot**: combined loss vs. evaluation time (ms), with NR error floor marked
4. **Summary table**: all approaches ranked by combined score
5. **Best model**: the Pareto-optimal approach with lowest score

All outputs go in `results/<agent>/waveform/`. See `BENCHMARK_PLAN.md` for full directory structure and changelog format.
