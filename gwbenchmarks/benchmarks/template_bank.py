"""Template Bank Bench: a filter bank and detection statistic for signals in Gaussian noise.

The agent submits up to 32 filter waveforms ``b_i`` and a function
``detection_statistic(z)`` of the complex filter outputs ``z_i = <d, b_i>``.
The evaluator simulates noise-only data and signal-plus-noise data
``d = rho * h + n`` (stationary Gaussian noise, ``p(rho) ~ rho^-4`` for
``rho >= 5``), sets the threshold at false-alarm probability 1e-4, and reports
the fraction of hidden signals detected.

The loss is the fractional loss of sensitive volume relative to an exactly
known single template (the "oracle"):

    loss = 1 - detection_fraction / oracle_detection_fraction

Because SNR scales as 1/distance and sources are uniform in volume, the
detection fraction is proportional to the sensitive volume, so this ratio is
VT / VT_oracle. Lower is better; 0 is unattainable for a finite bank.

All numerics live in ``template_bank_core`` (a self-contained module that is
also copied into isolated agent workspaces). This file only adapts it to the
suite's ``Benchmark`` interface.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark
from gwbenchmarks.benchmarks.template_bank_core import *  # noqa: F401,F403 (public helper API)
from gwbenchmarks.benchmarks.template_bank_core import (
    HIDDEN_NOISE_TRIALS,
    HIDDEN_SIGNAL_REPEATS,
    score_detection,
)


class TemplateBankBench(Benchmark):
    """Detection-in-noise template-bank benchmark (loss = 1 - VT / VT_oracle)."""

    name = "template_bank"

    def evaluate_bank(
        self,
        bank_waveforms: np.ndarray,
        target_waveforms: np.ndarray,
        weights: np.ndarray,
        detection_statistic: Callable,
        waveform_indices: np.ndarray | None = None,
        seed: int = 0,
        n_noise_trials: int = HIDDEN_NOISE_TRIALS,
        n_signal_repeats: int = HIDDEN_SIGNAL_REPEATS,
    ) -> dict:
        """Return the full ``score_detection`` metrics dictionary."""

        return score_detection(
            bank_waveforms, target_waveforms, weights, detection_statistic,
            waveform_indices, seed=seed, n_noise_trials=n_noise_trials,
            n_signal_repeats=n_signal_repeats,
        )

    def compute_loss(
        self, predictions: Dict[str, object], targets: Dict[str, object]
    ) -> tuple[float, Dict[str, float]]:
        """Score a submission.

        predictions: ``bank_waveforms`` (n_rows, n_freq), ``detection_statistic``
            (callable) and optionally ``waveform_indices``.
        targets: ``waveforms`` (unit-norm hidden targets), ``weights``, and
            optionally ``seed``, ``n_noise_trials``, ``n_signal_repeats``.
        """

        metrics = self.evaluate_bank(
            predictions["bank_waveforms"],
            targets["waveforms"],
            targets["weights"],
            predictions["detection_statistic"],
            predictions.get("waveform_indices"),
            seed=int(targets.get("seed", 0)),
            n_noise_trials=int(targets.get("n_noise_trials", HIDDEN_NOISE_TRIALS)),
            n_signal_repeats=int(targets.get("n_signal_repeats", HIDDEN_SIGNAL_REPEATS)),
        )
        loss = 1.0 - metrics["relative_to_oracle"]
        components = {
            "volume_loss_vs_oracle": loss,
            "detection_fraction": metrics["detection_fraction"],
            "detection_fraction_stderr": metrics["detection_fraction_stderr"],
            "oracle_detection_fraction": metrics["oracle_detection_fraction"],
            "n_rows": float(metrics["n_rows"]),
            "threshold": metrics["threshold"],
        }
        return loss, components
