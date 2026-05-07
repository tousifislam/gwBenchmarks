# LLM Agent Benchmark Runner

Run each LLM agent through the gwBenchmarks tasks and store its work under
`gwBenchmarks/llm_agents/results/<agent>/<benchmark>/`.

## Supported Agents

| Agent ID | Label | Provider | API Model ID | CLI | Reasoning |
|---|---|---|---|---|---|
| `haiku` | Haiku 4.5 | Anthropic | `claude-haiku-4-5-20251001` | Claude Code | default |
| `sonnet46` | Sonnet 4.6 | Anthropic | `claude-sonnet-4-6-20250514` | Claude Code | default |
| `opus46` | Opus 4.6 | Anthropic | `claude-opus-4-6-20250529` | Claude Code | default |
| `opus47` | Opus 4.7 | Anthropic | `claude-opus-4-7-20250715` | Claude Code | default |
| `gpt55_high` | GPT-5.5 | OpenAI | `gpt-5.5` | Codex | high |
| `gpt54_mini` | GPT-5.4 Mini | OpenAI | `gpt-5.4-mini` | Codex | default |
| `gpt53_codex_high` | GPT-5.3 Codex | OpenAI | `gpt-5.3-codex` | Codex | high |
| `gpt52` | GPT-5.2 | OpenAI | `gpt-5.2` | Codex | default |
| `gemini31_pro_preview` | Gemini 3.1 Pro | Google | `gemini-3.1-pro-preview` | Gemini CLI | default |
| `gemini3_flash_preview` | Gemini 3 Flash | Google | `gemini-3-flash-preview` | Gemini CLI | default |
| `kimi_k26` | Kimi K2.6 | Moonshot AI | `kimi-k2.6` | OpenCode | default |
| `deepseek_v4_pro_max` | DeepSeek V4 Pro Max | DeepSeek | `deepseek-v4-pro-max` | OpenCode | default |

## Benchmarks

All agents run 8 benchmarks. The prompt for each is a Markdown template in
`gwBenchmarks/llm_agents/agent_prompts/` with `{AGENT}` and `{AGENT_LABEL}` placeholders filled by
`gwBenchmarks/llm_agents/generate_prompt.py`.

| Benchmark | Prompt template | Completion string | Lines |
|---|---|---|---|
| Waveform | `gwBenchmarks/llm_agents/agent_prompts/waveform.md` | `WAVEFORM_BENCH_COMPLETE` | 215 |
| Remnant | `gwBenchmarks/llm_agents/agent_prompts/remnant.md` | `REMNANT_BENCH_COMPLETE` | 200 |
| Dynamics | `gwBenchmarks/llm_agents/agent_prompts/dynamics.md` | `DYNAMICS_BENCH_COMPLETE` | 209 |
| Ringdown | `gwBenchmarks/llm_agents/agent_prompts/ringdown.md` | `RINGDOWN_BENCH_COMPLETE` | 215 |
| Validity | `gwBenchmarks/llm_agents/agent_prompts/validity.md` | `VALIDITY_BENCH_COMPLETE` | 198 |
| Analytic | `gwBenchmarks/llm_agents/agent_prompts/analytic.md` | `ANALYTIC_BENCH_COMPLETE` | 233 |
| Template Bank | `gwBenchmarks/llm_agents/agent_prompts/template_bank.md` | `TEMPLATE_BANK_BENCH_COMPLETE` | 87 |
| New Physics | `gwBenchmarks/llm_agents/agent_prompts/new_physics.md` | `NEW_PHYSICS_BENCH_COMPLETE` | 97 |

Each prompt specifies: task definition, data format, loss function, required
approach categories (>=20 approaches, >=3 reparameterizations for surrogate
benchmarks), output format, and rules. The New Physics prompt additionally
references `gwBenchmarks/llm_agents/agent_prompts/new_physics_formulas.md` (216 lines of PN formulas).

## Step 1 -- Start the Agent

Open a terminal in the `gwBenchmarks/` root.

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

### Gemini CLI

```bash
gemini --approval-mode=yolo --skip-trust --model gemini-3.1-pro-preview
gemini --approval-mode=yolo --skip-trust --model gemini-3-flash-preview
```

### OpenCode CLI

```bash
opencode --model opencode-go/kimi-k2.6
opencode --model opencode-go/deepseek-v4-pro
```

## Step 2 -- Launch the Benchmark Loop

Paste the appropriate prompt into the agent's CLI. Each agent runs all eight
benchmarks sequentially: waveform, remnant, dynamics, ringdown, validity,
analytic, template_bank, new_physics.

### Claude Code Prompts

#### Haiku 4.5
```text
/ralph-loop:ralph-loop "You are the Haiku agent for the gwBenchmarks suite. Your agent ID is 'haiku'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py haiku <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/haiku/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it - do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE." --max-iterations 3
```

#### Sonnet 4.6
```text
/ralph-loop:ralph-loop "You are the Sonnet 4.6 agent for the gwBenchmarks suite. Your agent ID is 'sonnet46'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py sonnet46 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/sonnet46/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it - do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE." --max-iterations 3
```

#### Opus 4.6
```text
/ralph-loop:ralph-loop "You are the Opus 4.6 agent for the gwBenchmarks suite. Your agent ID is 'opus46'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py opus46 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/opus46/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it - do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE." --max-iterations 3
```

#### Opus 4.7
```text
/ralph-loop:ralph-loop "You are the Opus 4.7 agent for the gwBenchmarks suite. Your agent ID is 'opus47'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py opus47 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read llm_agents/results/opus47/<benchmark>/AGENT_PROMPT.md carefully, (3) execute every task described in it - do not stop until the completion string is printed, (4) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE." --max-iterations 3
```

### Codex Prompts

#### GPT-5.5
```text
You are the GPT-5.5 High Codex agent for the gwBenchmarks suite. Your agent ID is 'gpt55_high'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py gpt55_high <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gpt55_high/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE.
```

#### GPT-5.4 Mini
```text
You are the GPT-5.4 Mini Codex agent for the gwBenchmarks suite. Your agent ID is 'gpt54_mini'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py gpt54_mini <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gpt54_mini/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE.
```

#### GPT-5.3 Codex
```text
You are the GPT-5.3 Codex High agent for the gwBenchmarks suite. Your agent ID is 'gpt53_codex_high'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py gpt53_codex_high <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gpt53_codex_high/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE.
```

#### GPT-5.2
```text
You are the GPT-5.2 Codex agent for the gwBenchmarks suite. Your agent ID is 'gpt52'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py gpt52 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gpt52/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE.
```

### Gemini CLI Prompts

#### Gemini 3.1 Pro
```text
You are the Gemini 3.1 Pro Preview agent for the gwBenchmarks suite. Your agent ID is 'gemini31_pro_preview'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py gemini31_pro_preview <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gemini31_pro_preview/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE.
```

#### Gemini 3 Flash
```text
You are the Gemini 3 Flash Preview agent for the gwBenchmarks suite. Your agent ID is 'gemini3_flash_preview'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py gemini3_flash_preview <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/gemini3_flash_preview/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE.
```

### OpenCode CLI Prompts

#### Kimi K2.6
```text
You are the Kimi K2.6 agent for the gwBenchmarks suite. Your agent ID is 'kimi_k26'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py kimi_k26 <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/kimi_k26/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE.
```

#### DeepSeek V4 Pro Max
```text
You are the DeepSeek V4 Pro Max agent for the gwBenchmarks suite. Your agent ID is 'deepseek_v4_pro_max'. Run all eight benchmarks sequentially in this order: waveform, remnant, dynamics, ringdown, validity, analytic, template_bank, new_physics. For each benchmark: (1) run `python llm_agents/generate_prompt.py deepseek_v4_pro_max <benchmark> --write` from the gwBenchmarks/ root to generate your task prompt, (2) read `llm_agents/results/deepseek_v4_pro_max/<benchmark>/AGENT_PROMPT.md` carefully, (3) execute every task described in it, (4) do not stop until the benchmark completion string is printed, and (5) only then move on to the next benchmark. Completion strings: WAVEFORM_BENCH_COMPLETE, REMNANT_BENCH_COMPLETE, DYNAMICS_BENCH_COMPLETE, RINGDOWN_BENCH_COMPLETE, VALIDITY_BENCH_COMPLETE, ANALYTIC_BENCH_COMPLETE, TEMPLATE_BANK_BENCH_COMPLETE, NEW_PHYSICS_BENCH_COMPLETE.
```

## One-Shot Codex Exec

For non-interactive automated runs, pass the same prompt to `codex exec`:

```bash
codex exec -C . -m gpt-5.5 -c model_reasoning_effort="high" \
  -s workspace-write -a never \
  "You are the GPT-5.5 High Codex agent for the gwBenchmarks suite. ..."
```

## Tool Access Policy

All agents share the same access policy:

- **Code execution**: Full shell access (bash/Python). Agents can run arbitrary code.
- **Package installation**: Agents can `pip install` any package.
- **File system**: Read access to `gwBenchmarks/datasets/` and `gwBenchmarks/gwbenchmarks/` (read-only). Write access only to `gwBenchmarks/llm_agents/results/<agent>/<benchmark>/`.
- **Validation data**: Agents can read and evaluate on validation sets during development. There is no held-out hidden test set for surrogate benchmarks (the validation set IS the test set). Template Bank has a separate hidden test evaluated post-hoc.
- **Internet**: No internet access during runs. All data and dependencies are local.
- **Other agents**: No access to other agents' results.

## What Gets Saved

Each agent writes results into `gwBenchmarks/llm_agents/results/<agent>/<benchmark>/`:

```text
llm_agents/results/opus46/waveform/
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

## Updating a Prompt

Edit `gwBenchmarks/llm_agents/agent_prompts/<benchmark>.md` (one file, shared across all agents).
The agent regenerates its copy automatically at the start of each benchmark via
`gwBenchmarks/llm_agents/generate_prompt.py`.

## Evaluating Template Bank Submissions

```bash
python gwBenchmarks/llm_agents/evaluate_template_bank.py --agent opus46
```

This requires public data in `gwBenchmarks/datasets/template_bank/`. Hidden-test evaluation
additionally requires `bank_wf_params_test.npy`.

## Wiping a Run

```bash
rm -rf llm_agents/results/opus46/waveform/
```

Then re-run the loop. The agent will regenerate the prompt and start fresh.
