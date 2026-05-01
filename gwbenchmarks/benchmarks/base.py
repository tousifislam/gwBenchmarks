"""Abstract base class for all GW benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results."""

    loss: float
    runtime: float
    loss_components: Dict[str, float]


class Benchmark(ABC):
    """Base class for gravitational wave benchmarks.

    Subclasses must implement ``compute_loss``.
    """

    name: str = "base"

    def __init__(self, config_path: str | Path | None = None):
        self.config: Dict[str, Any] = {}
        if config_path is not None:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

    @abstractmethod
    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        """Compute benchmark-specific accuracy loss.

        Parameters
        ----------
        predictions : dict
            Model predictions keyed by output name.
        targets : dict
            Ground-truth values keyed by output name.

        Returns
        -------
        loss : float
            Accuracy loss.
        components : dict
            Individual loss components for diagnostics.
        """

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        runtime: float,
    ) -> BenchmarkResult:
        """Compute loss and wrap result with runtime.

        Parameters
        ----------
        predictions : dict
            Model predictions.
        targets : dict
            Ground-truth values.
        runtime : float
            Evaluation runtime in seconds (recorded, not penalised).

        Returns
        -------
        BenchmarkResult
        """
        loss, components = self.compute_loss(predictions, targets)
        return BenchmarkResult(loss=loss, runtime=runtime, loss_components=components)
