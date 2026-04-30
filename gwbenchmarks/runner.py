"""Benchmark runner: evaluate one or more models across benchmarks."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark, BenchmarkResult
from gwbenchmarks.models.base import LLMModel


@dataclass
class RunResult:
    """Result from running a single model on a single benchmark."""

    model_name: str
    benchmark_name: str
    result: BenchmarkResult
    n_samples: int


class BenchmarkRunner:
    """Run benchmarks across multiple models and collect results."""

    def __init__(self, benchmarks: List[Benchmark], models: List[LLMModel]):
        self.benchmarks = {b.name: b for b in benchmarks}
        self.models = {m.name: m for m in models}

    def run(
        self,
        benchmark_name: str,
        model_name: str,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        runtime: float,
    ) -> RunResult:
        """Evaluate a model's predictions on a benchmark."""
        bench = self.benchmarks[benchmark_name]
        result = bench.evaluate(predictions, targets, runtime)
        n = len(next(iter(targets.values())))
        return RunResult(
            model_name=model_name,
            benchmark_name=benchmark_name,
            result=result,
            n_samples=n,
        )

    def run_all(
        self,
        results_map: Dict[str, Dict[str, tuple]],
    ) -> List[RunResult]:
        """Evaluate all model-benchmark pairs.

        Parameters
        ----------
        results_map : dict
            Nested dict: results_map[benchmark_name][model_name] = (predictions, targets, runtime)

        Returns
        -------
        list of RunResult
        """
        all_results = []
        for bench_name, model_results in results_map.items():
            for model_name, (preds, targs, runtime) in model_results.items():
                r = self.run(bench_name, model_name, preds, targs, runtime)
                all_results.append(r)
        return all_results

    @staticmethod
    def summary_table(results: List[RunResult]) -> str:
        """Format results as a text table."""
        header = f"{'Model':<25} {'Benchmark':<15} {'Loss':>10} {'Score':>10} {'Runtime (s)':>12}"
        lines = [header, "-" * len(header)]
        for r in sorted(results, key=lambda x: (x.benchmark_name, x.model_name)):
            lines.append(
                f"{r.model_name:<25} {r.benchmark_name:<15} "
                f"{r.result.loss:>10.6f} {r.result.score:>10.6f} "
                f"{r.result.runtime:>12.4f}"
            )
        return "\n".join(lines)

    @staticmethod
    def to_json(results: List[RunResult], path: str | Path) -> None:
        """Save results to JSON."""
        records = []
        for r in results:
            records.append({
                "model": r.model_name,
                "benchmark": r.benchmark_name,
                "loss": r.result.loss,
                "score": r.result.score,
                "runtime": r.result.runtime,
                "loss_components": r.result.loss_components,
                "n_samples": r.n_samples,
            })
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
