"""Abstract base class for all GW benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from gwbenchmarks.scoring import compute_score


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results."""

    loss: float
    score: float
    runtime: float
    loss_components: Dict[str, float]


class Benchmark(ABC):
    """Base class for gravitational wave benchmarks.

    Subclasses must implement ``compute_loss`` and define their
    cost parameters (t0, alpha).
    """

    name: str = "base"
    t0: float = 1.0
    alpha: float = 0.1

    def __init__(self, config_path: str | Path | None = None):
        self.config: Dict[str, Any] = {}
        if config_path is not None:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
            self.t0 = self.config.get("t0", self.t0)
            self.alpha = self.config.get("alpha", self.alpha)

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
            Combined accuracy loss L_b.
        components : dict
            Individual loss components for diagnostics.
        """

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        runtime: float,
    ) -> BenchmarkResult:
        """Run full evaluation: loss + cost-penalized score.

        Parameters
        ----------
        predictions : dict
            Model predictions.
        targets : dict
            Ground-truth values.
        runtime : float
            Total evaluation runtime in seconds.

        Returns
        -------
        BenchmarkResult
        """
        loss, components = self.compute_loss(predictions, targets)
        score = compute_score(loss, runtime, self.t0, self.alpha)
        return BenchmarkResult(
            loss=loss, score=score, runtime=runtime, loss_components=components
        )
