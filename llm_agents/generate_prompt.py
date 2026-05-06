#!/usr/bin/env python3
"""
Generate a benchmark agent prompt for a specific agent.

Usage:
    python generate_prompt.py <agent> <benchmark>
    python generate_prompt.py opus47 waveform
    python generate_prompt.py gemini31_pro_preview waveform

    # Write directly to the agent's result directory:
    python generate_prompt.py opus47 waveform --write

Supported agents : haiku, opus46, opus47, sonnet46,
                   gpt55_high, gpt54_mini, gpt53_codex_high, gpt52
Supported benchmarks: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank
                   gpt55_high, gpt54_mini, gpt53_codex_high, gpt52,
                   gemini31_pro_preview, gemini25_pro, gemini3_flash_preview, gemini25_flash,
                    hy3_preview_free, kimi_k26, deepseek_v4_pro_max
Supported benchmarks: waveform, remnant, dynamics, ringdown, validity, analytic, new_physics
"""

import sys
from pathlib import Path

AGENT_LABELS = {
    "haiku":            "Haiku",
    "opus46":           "Opus 4.6",
    "opus47":           "Opus 4.7",
    "sonnet46":         "Sonnet 4.6",
    "gpt55_high":       "GPT-5.5 High",
    "gpt54_mini":       "GPT-5.4 Mini",
    "gpt53_codex_high": "GPT-5.3 Codex High",
    "gpt52":            "GPT-5.2",
    "gemini31_pro_preview": "Gemini 3.1 Pro Preview",
    "gemini25_pro":         "Gemini 2.5 Pro",
    "gemini3_flash_preview": "Gemini 3 Flash Preview",
    "gemini25_flash":        "Gemini 2.5 Flash",
    "hy3_preview_free":  "Hy3 Preview Free",
    "kimi_k26":          "Kimi K2.6",
    "deepseek_v4_pro_max": "DeepSeek V4 Pro Max",
}

BENCHMARKS = [
    "waveform",
    "remnant",
    "dynamics",
    "ringdown",
    "validity",
    "analytic",
    "template_bank",
]
BENCHMARKS = ["waveform", "remnant", "dynamics", "ringdown", "validity", "analytic", "new_physics"]


def generate(agent: str, benchmark: str) -> str:
    if agent not in AGENT_LABELS:
        raise ValueError(f"Unknown agent '{agent}'. Choose from: {list(AGENT_LABELS)}")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Choose from: {BENCHMARKS}")

    template = (Path(__file__).parent / "agent_prompts" / f"{benchmark}.md").read_text()

    prompt = (
        template
        .replace("{AGENT_LABEL}", AGENT_LABELS[agent])
        .replace("{AGENT}", agent)
    )
    return prompt


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv

    if len(args) != 2:
        print(__doc__)
        sys.exit(1)

    agent, benchmark = args
    prompt = generate(agent, benchmark)

    if write:
        out = Path(__file__).parent / "results" / agent / benchmark / "AGENT_PROMPT.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(prompt)
        print(f"Written to {out}")
    else:
        print(prompt)
