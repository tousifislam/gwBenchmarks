# LLM Agent Benchmark Runner

## Step 1 — Start Claude Code

Open a terminal in the `gwBenchmarks/` root and launch Claude Code with
`--dangerously-skip-permissions` and the model you want to evaluate.

### Haiku (`claude-haiku-4-5-20251001`)
```bash
claude --dangerously-skip-permissions --model claude-haiku-4-5-20251001
```

### Sonnet 4.6 (`claude-sonnet-4-6`)
```bash
claude --dangerously-skip-permissions --model claude-sonnet-4-6
```

### Opus 4.6 (`claude-opus-4-6`)
```bash
claude --dangerously-skip-permissions --model claude-opus-4-6
```

### Opus 4.7 (`claude-opus-4-7`)
```bash
claude --dangerously-skip-permissions --model claude-opus-4-7
```

---

## Step 2 — Launch the benchmark loop

Once inside the Claude Code session, paste the loop command for your model.
The agent will run all six benchmarks one by one, in order.

### Haiku
```
/ralph-loop:ralph-loop "You are the Haiku agent for the gwBenchmarks suite. Your agent ID is 'haiku'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py haiku <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/haiku/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it — do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE." --max-iterations 3
```

### Sonnet 4.6
```
/ralph-loop:ralph-loop "You are the Sonnet 4.6 agent for the gwBenchmarks suite. Your agent ID is 'sonnet46'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py sonnet46 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/sonnet46/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it — do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE." --max-iterations 3
```

### Opus 4.6
```
/ralph-loop:ralph-loop "You are the Opus 4.6 agent for the gwBenchmarks suite. Your agent ID is 'opus46'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py opus46 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/opus46/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it — do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE." --max-iterations 3
```

### Opus 4.7
```
/ralph-loop:ralph-loop "You are the Opus 4.7 agent for the gwBenchmarks suite. Your agent ID is 'opus47'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py opus47 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/opus47/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it — do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE." --max-iterations 3
```

---

## What gets saved

Each agent writes results into `llm_agents/results/<agent>/<benchmark>/`:

```
llm_agents/results/opus47/waveform/
├── AGENT_PROMPT.md              # generated automatically in step (1)
├── CHANGELOG.md                 # updated after every approach
├── models/
│   ├── 01_svd_gpr_raw/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── saved_model/         # gitignored (pkl, joblib, etc.)
│   │   └── scorecard.json
│   └── ...
└── comparison/
    ├── error_data.json
    ├── summary_table.json
    ├── best_model.json
    ├── progress.{png,pdf}
    ├── loss_only_comparison.{png,pdf}
    └── error_histograms.{png,pdf}
```

Model binaries (`saved_model/`, `*.pkl`, `*.joblib`) are gitignored.
Scripts, scorecards, plots, and JSON summaries are committed.

---

## Updating a prompt

Edit `llm_agents/agent_prompts/<benchmark>.md` (one file, shared across all agents).
The agent regenerates its copy automatically at the start of each benchmark via
`generate_prompt.py`.

## Wiping a run

```bash
rm -rf llm_agents/results/opus47/waveform/
```

Then re-run the loop — the agent will regenerate the prompt and start fresh.
