"""Benchmark 6: Validity Bench (NRHybSur3dq8 extrapolation awareness)."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import rmse


class ValidityBench(Benchmark):
    """Evaluate extrapolation awareness and reliability prediction.

    Inputs:  q, chi1z, chi2z
    Outputs: predicted mismatch M_hat

    Loss: log-space RMSE  E = RMSE(log M_hat, log M*)
    """

    name = "validity"

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        m_hat = np.asarray(predictions["mismatch"])
        m_true = np.asarray(targets["mismatch"])

        eps = 1e-30
        log_rmse = rmse(np.log(m_hat + eps), np.log(m_true + eps))
        return log_rmse, {"log_rmse": log_rmse}
