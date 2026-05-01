"""Benchmark 1: Waveform Bench (Co-precessing h22)."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import FD_MASSES_MSUN, frequency_domain_mismatch, mean_fd_mismatch


class WaveformBench(Benchmark):
    """Evaluate co-precessing (2,2) mode waveform predictions.

    Inputs:  q, chi1, chi2, time grid t_i
    Outputs: Re(h22_copr(t_i)), Im(h22_copr(t_i))

    Loss: mean frequency-domain mismatch (aLIGO PSD) over
          M_tot in {40, 80, 120, 160, 200} M_sun.
    """

    name = "waveform"

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        dt = float(targets.get("dt", np.array(1.0)))
        h_pred = predictions["h22_real"] + 1j * predictions["h22_imag"]
        h_true = targets["h22_real"] + 1j * targets["h22_imag"]

        per_mass = {
            f"mismatch_{int(m)}Msun": frequency_domain_mismatch(
                h_pred, h_true, dt_geometric=dt, mtot_msun=m
            )
            for m in FD_MASSES_MSUN
        }
        loss = float(np.mean(list(per_mass.values())))
        components = {"mean_fd_mismatch": loss, **per_mass}
        return loss, components
