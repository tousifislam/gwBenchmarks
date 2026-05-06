# Remnant Metric Recomputation (2026-05-02)

The original evaluation for Gemini 3.1 Pro Preview used NMAE (mean absolute error / range) as the loss metric.
The canonical metric is NRMSE (root mean squared error / range).

- Old loss (NMAE): 0.05736311
- New loss (NRMSE): 0.09093549

NRMSE >= NMAE by Jensen's inequality. The recomputation loads the
saved model, re-predicts on the 1000-sample validation set, and
computes sqrt(mean((pred - true)^2)) / ptp(true) for kick velocity v_k.

Script: llm_agents/recompute_remnant_validity_ringdown.py
