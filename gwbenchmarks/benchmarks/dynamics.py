"""Benchmark 3: Dynamics Bench (Eccentric Spinning)."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import circular_error, rmse


class DynamicsBench(Benchmark):
    """Evaluate eccentric spinning orbital dynamics predictions.

    Inputs: q, chi1, chi2, e0, x0, zeta0, time grid t_i
    Outputs: e(t_i), x(t_i), zeta(t_i)

    Loss: L = RMSE(e) + RMSE(x) + mean(1 - cos(zeta - zeta*))
    """

    name = "dynamics"
    t0 = 0.01  # 10 ms per trajectory
    alpha = 0.10

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        e_err = rmse(predictions["e"], targets["e"])
        x_err = rmse(predictions["x"], targets["x"])
        zeta_err = circular_error(predictions["zeta"], targets["zeta"])

        loss = e_err + x_err + zeta_err
        components = {
            "rmse_e": e_err,
            "rmse_x": x_err,
            "circular_error_zeta": zeta_err,
        }
        return loss, components
