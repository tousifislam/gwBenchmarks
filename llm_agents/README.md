# LLM Agent Benchmark Runner

Run each LLM agent through the six gwBenchmarks tasks and store its work under
`llm_agents/results/<agent>/<benchmark>/`.

## Supported Agents

| Agent ID | Label | Runner | Model | Reasoning |
|---|---|---|---|---|
| `haiku` | Haiku | Claude Code | `claude-haiku-4-5-20251001` | default |
| `sonnet46` | Sonnet 4.6 | Claude Code | `claude-sonnet-4-6` | default |
| `opus46` | Opus 4.6 | Claude Code | `claude-opus-4-6` | default |
| `opus47` | Opus 4.7 | Claude Code | `claude-opus-4-7` | default |
| `gpt55_high` | GPT-5.5 High | Codex | `gpt-5.5` | high |
| `gpt54_mini` | GPT-5.4 Mini | Codex | `gpt-5.4-mini` | default |
| `gpt53_codex_high` | GPT-5.3 Codex High | Codex | `gpt-5.3-codex` | high |
| `gpt52` | GPT-5.2 | Codex | `gpt-5.2` | default |
| `hy3_preview_free` | Hy3 Preview Free | OpenCode | `opencode/hy3-preview-free` | default |
| `kimi_k26` | Kimi K2.6 | OpenCode | `opencode-go/kimi-k2.6` | default |

## Step 1 - Start The Agent

Open a terminal in the `gwBenchmarks/` root.

### Gemini CLI

```bash
gemini --approval-mode=yolo --skip-trust --model gemini-3.1-pro-preview
gemini --approval-mode=yolo --skip-trust --model gemini-2.5-pro
gemini --approval-mode=yolo --skip-trust --model gemini-3-flash-preview
gemini --approval-mode=yolo --skip-trust --model gemini-2.5-flash
```

### Claude Code

```bash
claude --dangerously-skip-permissions --model claude-haiku-4-5-20251001
claude --dangerously-skip-permissions --model claude-sonnet-4-6
claude --dangerously-skip-permissions --model claude-opus-4-6
claude --dangerously-skip-permissions --model claude-opus-4-7
```

### Codex

```bash
codex -C . -m gpt-5.5 -c model_reasoning_effort="high" -s workspace-write -a never
codex -C . -m gpt-5.4-mini -s workspace-write -a never
codex -C . -m gpt-5.3-codex -c model_reasoning_effort="high" -s workspace-write -a never
codex -C . -m gpt-5.2 -s workspace-write -a never
```

### OpenCode CLI

```bash
opencode
opencode --model opencode-go/kimi-k2.6
```

## Step 2 - Launch The Benchmark Loop

Paste the command or prompt for the agent you launched. Each agent runs all six
benchmarks in this order: waveform, remnant, dynamics, ringdown, validity,
analytic.

### Claude Code Prompts

#### Haiku
```text
/ralph-loop:ralph-loop "You are the Haiku agent for the gwBenchmarks suite. Your agent ID is 'haiku'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py haiku <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/haiku/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it - do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE." --max-iterations 3
```

#### Sonnet 4.6
```text
/ralph-loop:ralph-loop "You are the Sonnet 4.6 agent for the gwBenchmarks suite. Your agent ID is 'sonnet46'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py sonnet46 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/sonnet46/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it - do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE." --max-iterations 3
```

#### Opus 4.6
```text
/ralph-loop:ralph-loop "You are the Opus 4.6 agent for the gwBenchmarks suite. Your agent ID is 'opus46'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py opus46 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/opus46/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it - do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE." --max-iterations 3
```

#### Opus 4.7
```text
/ralph-loop:ralph-loop "You are the Opus 4.7 agent for the gwBenchmarks suite. Your agent ID is 'opus47'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py opus47 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/opus47/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it - do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE." --max-iterations 3
```

### Codex Prompts

#### GPT-5.5 High
```text
You are the GPT-5.5 High Codex agent for the gwBenchmarks suite. Your agent ID is 'gpt55_high'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py gpt55_high <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gpt55_high/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

#### GPT-5.4 Mini
```text
You are the GPT-5.4 Mini Codex agent for the gwBenchmarks suite. Your agent ID is 'gpt54_mini'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py gpt54_mini <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gpt54_mini/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

#### GPT-5.3 Codex High
```text
You are the GPT-5.3 Codex High agent for the gwBenchmarks suite. Your agent ID is 'gpt53_codex_high'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py gpt53_codex_high <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gpt53_codex_high/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

#### GPT-5.2
```text
You are the GPT-5.2 Codex agent for the gwBenchmarks suite. Your agent ID is 'gpt52'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py gpt52 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gpt52/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

### OpenCode Prompts

#### Hy3 Preview Free
```text
You are the Hy3 Preview Free agent for the gwBenchmarks suite. Your agent ID is 'hy3_preview_free'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py hy3_preview_free <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/hy3_preview_free/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

#### Kimi K2.6
```text
You are the Kimi K2.6 agent for the gwBenchmarks suite. Your agent ID is 'kimi_k26'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py kimi_k26 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/kimi_k26/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

#### Gemini 3.1 Pro Preview
```text
You are the Gemini 3.1 Pro Preview agent for the gwBenchmarks suite. Your agent ID is 'gemini31_pro_preview'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py gemini31_pro_preview <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gemini31_pro_preview/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

#### Gemini 2.5 Pro
```text
You are the Gemini 2.5 Pro agent for the gwBenchmarks suite. Your agent ID is 'gemini25_pro'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py gemini25_pro <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gemini25_pro/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

#### Gemini 3 Flash Preview
```text
You are the Gemini 3 Flash Preview agent for the gwBenchmarks suite. Your agent ID is 'gemini3_flash_preview'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py gemini3_flash_preview <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gemini3_flash_preview/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

#### Gemini 2.5 Flash
```text
You are the Gemini 2.5 Flash agent for the gwBenchmarks suite. Your agent ID is 'gemini25_flash'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run `python llm_agents/generate_prompt.py gemini25_flash <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gemini25_flash/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE.
```

## Optional One-Shot Codex Commands

For non-interactive automated runs, pass the same prompt to `codex exec`.
The examples keep the sandbox at `workspace-write` and use `-a never` so the
benchmark does not stop for approvals.

```bash
codex exec -C . -m gpt-5.5 -c model_reasoning_effort="high" -s workspace-write -a never "You are the GPT-5.5 High Codex agent for the gwBenchmarks suite. Your agent ID is 'gpt55_high'. Run all six benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic. For each benchmark: (1) run \`python llm_agents/generate_prompt.py gpt55_high <benchmark> --write\` from the gwBenchmarks/ root to generate your task prompt, (2) read \`llm_agents/results/gpt55_high/<benchmark>/AGENT_PROMPT.md\` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE."
```

## What Gets Saved

Each agent writes results into `llm_agents/results/<agent>/<benchmark>/`:

```text
llm_agents/results/gpt55_high/waveform/
|-- AGENT_PROMPT.md              # generated automatically in step (1)
|-- CHANGELOG.md                 # updated after every approach
|-- models/
|   |-- 01_svd_gpr_raw/
|   |   |-- train.py
|   |   |-- predict.py
|   |   |-- saved_model/         # gitignored (pkl, joblib, etc.)
|   |   `-- scorecard.json
|   `-- ...
`-- comparison/
    |-- error_data.json
    |-- summary_table.json
    |-- best_model.json
    |-- progress.{png,pdf}
    |-- loss_only_comparison.{png,pdf}
    `-- error_histograms.{png,pdf}
```

Model binaries (`saved_model/`, `*.pkl`, `*.joblib`) are gitignored.
Scripts, scorecards, plots, and JSON summaries are committed.

## Updating A Prompt

Edit `llm_agents/agent_prompts/<benchmark>.md` (one file, shared across all agents).
The agent regenerates its copy automatically at the start of each benchmark via
`generate_prompt.py`.

## Wiping A Run

```bash
rm -rf llm_agents/results/gpt55_high/waveform/
```

Then re-run the loop. The agent will regenerate the prompt and start fresh.
GENT_PROMPT.md\` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE."
```

## What Gets Saved

Each agent writes results into `llm_agents/results/<agent>/<benchmark>/`:

```text
llm_agents/results/gpt55_high/waveform/
|-- AGENT_PROMPT.md              # generated automatically in step (1)
|-- CHANGELOG.md                 # updated after every approach
|-- models/
|   |-- 01_svd_gpr_raw/
|   |   |-- train.py
|   |   |-- predict.py
|   |   |-- saved_model/         # gitignored (pkl, joblib, etc.)
|   |   `-- scorecard.json
|   `-- ...
`-- comparison/
    |-- error_data.json
    |-- summary_table.json
    |-- best_model.json
    |-- progress.{png,pdf}
    |-- loss_only_comparison.{png,pdf}
    `-- error_histograms.{png,pdf}
```

Model binaries (`saved_model/`, `*.pkl`, `*.joblib`) are gitignored.
Scripts, scorecards, plots, and JSON summaries are committed.

## Updating A Prompt

Edit `llm_agents/agent_prompts/<benchmark>.md` (one file, shared across all agents).
The agent regenerates its copy automatically at the start of each benchmark via
`generate_prompt.py`.

## Wiping A Run

```bash
rm -rf llm_agents/results/gpt55_high/waveform/
```

Then re-run the loop. The agent will regenerate the prompt and start fresh.
file, shared across all agents).
The agent regenerates its copy automatically at the start of each benchmark via
`generate_prompt.py`.

## Wiping A Run

```bash
rm -rf llm_agents/results/gpt55_high/waveform/
```

Then re-run the loop. The agent will regenerate the prompt and start fresh.
