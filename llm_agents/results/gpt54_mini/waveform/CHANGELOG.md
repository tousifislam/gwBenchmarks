# Waveform Bench Changelog

### 01. svd_gpr_raw
- Observed: train loss 2.4430e-01, val loss 2.5187e-01.
- Hypothesis: Baseline complex SVD coefficients with RBF GPR.
- Change: trained svd model with feature mode `raw7` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/01_svd_gpr_raw`.

### 02. svd_gpr_eff
- Observed: train loss 2.4425e-01, val loss 2.4152e-01.
- Hypothesis: Effective-spin reparameterization with Matern GPR.
- Change: trained svd model with feature mode `eff` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/02_svd_gpr_eff`.

### 03. svd_poly_delta
- Observed: train loss 2.5753e-01, val loss 2.4923e-01.
- Hypothesis: Low-order polynomial fit on mass-difference features.
- Change: trained svd model with feature mode `delta` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/03_svd_poly_delta`.

### 04. svd_mlp_spherical
- Observed: train loss 2.6549e-01, val loss 2.5252e-01.
- Hypothesis: MLP on spherical-spin features.
- Change: trained svd model with feature mode `spherical` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/04_svd_mlp_spherical`.

### 05. svd_rf_raw
- Observed: train loss 2.4131e-01, val loss 2.4315e-01.
- Hypothesis: Random forest on raw parameters.
- Change: trained svd model with feature mode `raw7_noomega` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/05_svd_rf_raw`.

### 06. svd_gb_eff
- Observed: train loss 2.3757e-01, val loss 2.3187e-01.
- Hypothesis: Gradient boosting on effective-spin features.
- Change: trained svd model with feature mode `eff` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/06_svd_gb_eff`.

### 07. svd_huber_omega
- Observed: train loss 2.8602e-01, val loss 2.7642e-01.
- Hypothesis: Robust linear model with omega0 included.
- Change: trained svd model with feature mode `omega` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/07_svd_huber_omega`.

### 08. svd_bayes_delta
- Observed: train loss 2.7178e-01, val loss 2.5944e-01.
- Hypothesis: Bayesian ridge regression on delta-mass features.
- Change: trained svd model with feature mode `delta` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/08_svd_bayes_delta`.

### 09. svd_rbf_raw
- Observed: train loss 2.4430e-01, val loss 2.5653e-01.
- Hypothesis: RBF interpolation of SVD coefficients.
- Change: trained svd model with feature mode `raw7` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/09_svd_rbf_raw`.

### 10. svd_krr_eff
- Observed: train loss 2.4375e-01, val loss 2.6140e-01.
- Hypothesis: Kernel ridge on effective-spin features.
- Change: trained svd model with feature mode `eff` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/10_svd_krr_eff`.

### 11. knn_correction_spherical
- Observed: train loss 2.4430e-01, val loss 2.5854e-01.
- Hypothesis: Nearest-neighbor interpolation with residual correction.
- Change: trained svd model with feature mode `spherical` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/11_knn_correction_spherical`.

### 12. eim_gpr_raw
- Observed: train loss 2.6324e-01, val loss 2.6475e-01.
- Hypothesis: EIM node values with GPR on raw parameters.
- Change: trained eim model with feature mode `raw7` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/12_eim_gpr_raw`.

### 13. eim_rf_eff
- Observed: train loss 2.5533e-01, val loss 2.5340e-01.
- Hypothesis: EIM node values with random forest on effective-spin features.
- Change: trained eim model with feature mode `eff` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/13_eim_rf_eff`.

### 14. ap_svd_gpr_peak
- Observed: train loss 1.1847e-02, val loss 5.2586e-01.
- Hypothesis: Amplitude/phase SVD with peak-aligned time ordering.
- Change: trained ap_svd model with feature mode `raw7` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/14_ap_svd_gpr_peak`.

### 15. ap_svd_mlp_reverse
- Observed: train loss 6.4087e-01, val loss 6.4126e-01.
- Hypothesis: Amplitude/phase SVD with reversed time ordering.
- Change: trained ap_svd model with feature mode `spherical` and time mode `reverse`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/15_ap_svd_mlp_reverse`.

### 16. ap_svd_poly_peak
- Observed: train loss 5.2435e-01, val loss 5.1512e-01.
- Hypothesis: Amplitude/phase SVD with polynomial coefficient fits.
- Change: trained ap_svd model with feature mode `delta` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/16_ap_svd_poly_peak`.

### 17. svd_svr_omega
- Observed: train loss 2.7633e-01, val loss 2.7092e-01.
- Hypothesis: Support-vector regression on omega0-augmented features.
- Change: trained svd model with feature mode `omega` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/17_svd_svr_omega`.

### 18. svd_extra_trees_delta
- Observed: train loss 2.4430e-01, val loss 2.3291e-01.
- Hypothesis: Extra-trees ensemble on delta-mass features.
- Change: trained svd model with feature mode `delta` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/18_svd_extra_trees_delta`.

### 19. pysr_svd_eff
- Observed: train loss 2.6335e-01, val loss 2.5363e-01.
- Hypothesis: PySR-discovered expressions calibrated on SVD coefficients.
- Change: trained symbolic model with feature mode `eff` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/19_pysr_svd_eff`.

### 20. gplearn_svd_spherical
- Observed: train loss 2.6574e-01, val loss 2.5490e-01.
- Hypothesis: gplearn expressions calibrated on SVD coefficients.
- Change: trained symbolic model with feature mode `spherical` and time mode `peak`.
- Result: saved to `/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/gpt54_mini/waveform/models/20_gplearn_svd_spherical`.

