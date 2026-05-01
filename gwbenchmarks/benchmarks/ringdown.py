"""Benchmark 4: Ringdown Bench (QNM frequencies)."""

from pathlib import Path
from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.metrics import relative_error


class RingdownBench(Benchmark):
    """Evaluate quasi-normal mode frequency predictions.

    Inputs:  Mf, chi_f, mode indices (l, m, n)
    Outputs: omega_real, omega_imag

    Loss: mean relative error  E = (|dw_R/w_R*| + |dw_I/w_I*|) / 2
    """

    name = "ringdown"

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        omega_r_pred = np.asarray(predictions["omega_real"])
        omega_r_true = np.asarray(targets["omega_real"])
        omega_i_pred = np.asarray(predictions["omega_imag"])
        omega_i_true = np.asarray(targets["omega_imag"])

        rel_err_real = float(np.mean(
            [relative_error(p, t) for p, t in zip(omega_r_pred.flat, omega_r_true.flat)]
        ))
        rel_err_imag = float(np.mean(
            [relative_error(p, t) for p, t in zip(omega_i_pred.flat, omega_i_true.flat)]
        ))

        loss = 0.5 * (rel_err_real + rel_err_imag)
        components = {
            "rel_error_omega_real": rel_err_real,
            "rel_error_omega_imag": rel_err_imag,
        }
        return loss, components
