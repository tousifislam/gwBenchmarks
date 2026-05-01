"""Benchmark 5: Analytic Bench (Non-spinning BBH, q in [1, 20])."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import FD_MASSES_MSUN, frequency_domain_mismatch, mean_fd_mismatch


class AnalyticBench(Benchmark):
    """Evaluate analytic surrogate waveform predictions for non-spinning BBH.

    Inputs:  q, time grid t_i
    Outputs: waveform h22(t_i)

    Loss: mean frequency-domain mismatch (aLIGO PSD) over
          M_tot in {40, 80, 120, 160, 200} M_sun.
    """

    name = "analytic"

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        dt = float(targets.get("dt", np.array(1.0)))
        h_pred = np.asarray(predictions["waveform"])
        h_true = np.asarray(targets["waveform"])

        per_mass = {
            f"mismatch_{int(m)}Msun": frequency_domain_mismatch(
                h_pred, h_true, dt_geometric=dt, mtot_msun=m
            )
            for m in FD_MASSES_MSUN
        }
        loss = float(np.mean(list(per_mass.values())))
        components = {"mean_fd_mismatch": loss, **per_mass}
        return loss, components
