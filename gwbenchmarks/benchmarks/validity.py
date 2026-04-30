"""Benchmark 6: Validity Bench (NRHybSur3dq8 extrapolation awareness)."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import expected_calibration_error, rmse


class ValidityBench(Benchmark):
    """Evaluate extrapolation awareness and reliability prediction.

    Inputs: q, chi1, chi2
    Outputs: predicted mismatch M_hat

    Loss: L = RMSE(log M_hat, log M*) + ECE

    Goal: assess whether the model knows when its predictions are unreliable.
    """

    name = "validity"
    t0 = 0.001  # 1 ms per point
    alpha = 0.05

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        m_hat = np.asarray(predictions["mismatch"])
        m_true = np.asarray(targets["mismatch"])

        eps = 1e-30
        log_rmse = rmse(np.log(m_hat + eps), np.log(m_true + eps))

        m_max = np.max(m_true) if np.max(m_true) > 0 else 1.0
        ece = expected_calibration_error(m_hat / m_max, m_true / m_max)

        loss = log_rmse + ece
        components = {
            "log_rmse": log_rmse,
            "ece": ece,
        }
        return loss, components
