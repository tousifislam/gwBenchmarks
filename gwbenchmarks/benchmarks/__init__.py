"""Gravitational wave LLM benchmarks."""

from gwbenchmarks.benchmarks.waveform import WaveformBench
from gwbenchmarks.benchmarks.remnant import RemnantBench
from gwbenchmarks.benchmarks.dynamics import DynamicsBench
from gwbenchmarks.benchmarks.ringdown import RingdownBench
from gwbenchmarks.benchmarks.analytic import AnalyticBench
from gwbenchmarks.benchmarks.validity import ValidityBench
from gwbenchmarks.benchmarks.template_bank import TemplateBankBench

__all__ = [
    "WaveformBench",
    "RemnantBench",
    "DynamicsBench",
    "RingdownBench",
    "AnalyticBench",
    "ValidityBench",
    "TemplateBankBench",
]
