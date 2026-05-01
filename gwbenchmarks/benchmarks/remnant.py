"""Benchmark 2: Remnant Bench."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import nrmse


class RemnantBench(Benchmark):
    """Evaluate remnant kick-velocity magnitude predictions.

    Inputs:  q, chi1, chi2
    Outputs: |v_k| (kick magnitude)

    Loss: NRMSE(|v_k|)
    """

    name = "remnant"

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        vk_err = nrmse(predictions["v_k"], targets["v_k"])
        return vk_err, {"nrmse_v_k": vk_err}
