"""Benchmark 3: Dynamics Bench (Eccentric Spinning)."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import rms_relative_error


class DynamicsBench(Benchmark):
    """Evaluate eccentric spinning orbital dynamics predictions for x(t).

    Inputs:  q, chi1z, chi2z, e0, x0, zeta0, time grid t_i
    Outputs: x(t_i)  (PN frequency parameter)

    Loss: pointwise RMS relative error
          E = sqrt( mean( ((x_pred - x_true) / x_true)^2 ) )

    This weights every time step equally in fractional terms, avoiding the
    late-time bias of a global L2 norm when x(t) grows toward merger.
    """

    name = "dynamics"

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        err = rms_relative_error(predictions["x"], targets["x"])
        return err, {"rms_relative_error_x": err}
