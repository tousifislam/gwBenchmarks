"""gwbenchmarks: Benchmark suite for LLM-based gravitational wave modelling."""

__version__ = "0.1.0"

from gwbenchmarks.scoring import compute_score
from gwbenchmarks.benchmarks import (
    WaveformBench,
    RemnantBench,
    DynamicsBench,
    RingdownBench,
    AnalyticBench,
    ValidityBench,
)
