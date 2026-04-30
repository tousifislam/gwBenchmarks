"""Benchmark 2: Remnant Bench."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import nrmse


class RemnantBench(Benchmark):
    """Evaluate remnant property predictions (final mass, spin, kick).

    Inputs: q, chi1, chi2
    Outputs: Mf/M, chi_f (3-vector), v_k (3-vector)

    Loss: L = NRMSE(Mf) + NRMSE(chi_f) + NRMSE(v_k)
    """

    name = "remnant"
    t0 = 0.0001  # 0.1 ms per point
    alpha = 0.05

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        mf_err = nrmse(predictions["Mf"], targets["Mf"])
        chif_err = nrmse(predictions["chi_f"], targets["chi_f"])
        vk_err = nrmse(predictions["v_k"], targets["v_k"])

        loss = mf_err + chif_err + vk_err
        components = {
            "nrmse_Mf": mf_err,
            "nrmse_chi_f": chif_err,
            "nrmse_v_k": vk_err,
        }
        return loss, components
