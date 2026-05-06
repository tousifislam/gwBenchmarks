"""gwbenchmarks: Benchmark suite for LLM-based gravitational wave modelling."""

__version__ = "0.1.0"

from gwbenchmarks import plot_settings
from gwbenchmarks.benchmarks import (
    WaveformBench,
    RemnantBench,
    DynamicsBench,
    RingdownBench,
    AnalyticBench,
    ValidityBench,
    TemplateBankBench,
    NewPhysicsBench,
)
from gwbenchmarks.models.registry import MODELS, get_model
from gwbenchmarks.runner import BenchmarkRunner
