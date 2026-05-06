# Validity Metric Recomputation (2026-05-02)

The original evaluation for GPT-5.3 Codex High used MAE on
log10(mismatch) as the loss metric.
The canonical metric is RMSE on log10(mismatch).

- Old loss (MAE): 0.53605082
- New loss (RMSE): 0.73990994

RMSE >= MAE for any non-degenerate distribution. The recomputation
loads the saved model, re-predicts on the 393-sample validation set,
and computes sqrt(mean((pred - true)^2)) in log10-space.

Script: llm_agents/recompute_remnant_validity_ringdown.py
