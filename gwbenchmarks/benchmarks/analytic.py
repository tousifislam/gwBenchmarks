"""Benchmark 5: Analytic Bench (Non-spinning BBH, q in [1, 20])."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import mismatch as compute_mismatch
from gwbenchmarks.metrics import rmse


class AnalyticBench(Benchmark):
    """Evaluate analytic surrogate waveform predictions for non-spinning BBH.

    Inputs: q, time grid t_i
    Outputs: analytic surrogate waveform

    Loss: L = mismatch + lambda * RMSE(coefficients)

    Requirements:
    - Correct equal-mass limit
    - Smooth behavior in q
    - Correct test-mass trend near q=20
    - No unphysical oscillations
    """

    name = "analytic"
    t0 = 0.001  # 1 ms per waveform
    alpha = 0.10

    def __init__(self, config_path: str | Path | None = None):
        super().__init__(config_path)
        self.lam = self.config.get("lambda", 1.0)

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        dt = targets.get("dt", np.array(1.0)).item()
        h_pred = predictions["waveform"]
        h_true = targets["waveform"]

        m = compute_mismatch(h_pred, h_true, dt=dt)

        coeff_err = 0.0
        if "coefficients" in predictions and "coefficients" in targets:
            coeff_err = rmse(predictions["coefficients"], targets["coefficients"])

        loss = m + self.lam * coeff_err
        components = {
            "mismatch": m,
            "coefficient_rmse": coeff_err,
        }
        return loss, components
