"""Benchmark 1: Waveform Bench (Co-precessing h22)."""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import log_amplitude_rmse, mismatch, phase_rmse


class WaveformBench(Benchmark):
    """Evaluate co-precessing (2,2) mode waveform predictions.

    Inputs: q, chi1, chi2, time grid t_i
    Outputs: Re(h22_copr(t_i)), Im(h22_copr(t_i))

    Loss: L = w1*M + w2*RMSE(phi) + w3*RMSE(log A)
    """

    name = "waveform"
    t0 = 0.01  # 10 ms per waveform
    alpha = 0.10

    def __init__(self, config_path: str | Path | None = None):
        super().__init__(config_path)
        weights = self.config.get("weights", {})
        self.w_mismatch = weights.get("mismatch", 1.0)
        self.w_phase = weights.get("phase_rmse", 1.0)
        self.w_logamp = weights.get("logamp_rmse", 1.0)

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        dt = targets.get("dt", np.array(1.0)).item()
        h_pred = predictions["h22_real"] + 1j * predictions["h22_imag"]
        h_true = targets["h22_real"] + 1j * targets["h22_imag"]

        m = mismatch(h_pred, h_true, dt=dt)
        phi_err = phase_rmse(h_pred, h_true)
        logamp_err = log_amplitude_rmse(h_pred, h_true)

        loss = self.w_mismatch * m + self.w_phase * phi_err + self.w_logamp * logamp_err
        components = {
            "mismatch": m,
            "phase_rmse": phi_err,
            "logamp_rmse": logamp_err,
        }
        return loss, components
