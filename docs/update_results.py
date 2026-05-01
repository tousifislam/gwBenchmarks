#!/usr/bin/env python3
"""
Read best_model.json from every agent/benchmark result directory and write
docs/results.json.  Run this from the gwBenchmarks/ root after benchmark runs.

Usage:
    python docs/update_results.py
"""
import json
from pathlib import Path

AGENTS     = [
    "haiku",
    "opus46",
    "opus47",
    "sonnet46",
    "gpt55_high",
    "gpt54_mini",
    "gpt53_codex_high",
    "gpt52",
]
BENCHMARKS = ["waveform", "remnant", "dynamics", "ringdown", "validity", "analytic"]
ROOT       = Path(__file__).parent.parent   # gwBenchmarks/

results = {}
for agent in AGENTS:
    results[agent] = {}
    for bench in BENCHMARKS:
        path = ROOT / "llm_agents" / "results" / agent / bench / "comparison" / "best_model.json"
        if path.exists():
            data = json.loads(path.read_text())
            results[agent][bench] = {
                "loss":  data.get("loss"),
                "model": data.get("name"),
            }
        else:
            results[agent][bench] = None

out = Path(__file__).parent / "results.json"
out.write_text(json.dumps(results, indent=2))
print(f"Written {out}")

# summary to stdout
print(f"\n{'Agent':<20}", end="")
for b in BENCHMARKS:
    print(f"  {b[:10]:<12}", end="")
print()
for agent in AGENTS:
    print(f"{agent:<20}", end="")
    for bench in BENCHMARKS:
        v = results[agent][bench]
        val = f"{v['loss']:.3e}" if v and v["loss"] is not None else "—"
        print(f"  {val:<12}", end="")
    print()
