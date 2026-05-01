# LLM Agent Benchmark Runner

Instructions for launching Claude Code agents to run the gwBenchmarks suite.

## Directory structure

```
llm_agents/
├── agent_prompts/          # canonical benchmark prompts (edit here to update all agents)
│   ├── waveform.md
│   ├── remnant.md
│   ├── dynamics.md
│   ├── ringdown.md
│   ├── validity.md
│   └── analytic.md
├── generate_prompt.py      # fills {AGENT}/{AGENT_LABEL} placeholders
├── results/
│   ├── comparison/         # cross-agent plotting scripts
│   ├── haiku/{bench}/      # run artifacts land here
│   ├── opus46/{bench}/
│   ├── opus47/{bench}/
│   └── sonnet46/{bench}/
└── README.md               # this file
```

## Step 1 — Generate the agent prompt

Before launching, write the agent-specific prompt into the result directory:

```bash
# from the gwBenchmarks/ root
python llm_agents/generate_prompt.py <agent> <benchmark> --write
```

Examples:

```bash
python llm_agents/generate_prompt.py haiku    waveform --write
python llm_agents/generate_prompt.py opus46   remnant  --write
python llm_agents/generate_prompt.py opus47   ringdown --write
python llm_agents/generate_prompt.py sonnet46 dynamics --write
```

Supported agents: `haiku`, `opus46`, `opus47`, `sonnet46`  
Supported benchmarks: `waveform`, `remnant`, `dynamics`, `ringdown`, `validity`, `analytic`

To regenerate all 24 prompts at once:

```bash
for agent in haiku opus46 opus47 sonnet46; do
  for bench in waveform remnant dynamics ringdown validity analytic; do
    python llm_agents/generate_prompt.py $agent $bench --write
  done
done
```

## Step 2 — Start Claude Code with the right model

Open a terminal in the `gwBenchmarks/` root and launch Claude Code with
`--dangerously-skip-permissions` so the agent can read/write files and run
code without manual approval.

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

## Step 3 — Launch the benchmark loop

Once inside the Claude Code session, launch the loop with:

```
/ralph-loop:ralph-loop "Read llm_agents/results/<agent>/<benchmark>/AGENT_PROMPT.md carefully and execute every task described in it. Work from the gwBenchmarks/ root directory. Do not stop until the completion string (e.g. WAVEFORM_BENCH_COMPLETE) is printed." --max-iterations 3
```

### Full examples

**Haiku — waveform:**
```
/ralph-loop:ralph-loop "Read llm_agents/results/haiku/waveform/AGENT_PROMPT.md carefully and execute every task described in it. Work from the gwBenchmarks/ root directory. Do not stop until WAVEFORM_BENCH_COMPLETE is printed." --max-iterations 3
```

**Opus 4.7 — ringdown:**
```
/ralph-loop:ralph-loop "Read llm_agents/results/opus47/ringdown/AGENT_PROMPT.md carefully and execute every task described in it. Work from the gwBenchmarks/ root directory. Do not stop until RINGDOWN_BENCH_COMPLETE is printed." --max-iterations 3
```

**Sonnet 4.6 — dynamics:**
```
/ralph-loop:ralph-loop "Read llm_agents/results/sonnet46/dynamics/AGENT_PROMPT.md carefully and execute every task described in it. Work from the gwBenchmarks/ root directory. Do not stop until DYNAMICS_BENCH_COMPLETE is printed." --max-iterations 3
```

Completion strings per benchmark:

| Benchmark | Completion string |
|-----------|------------------|
| waveform  | `WAVEFORM_BENCH_COMPLETE` |
| remnant   | `REMNANT_BENCH_COMPLETE` |
| dynamics  | `DYNAMICS_BENCH_COMPLETE` |
| ringdown  | `RINGDOWN_BENCH_COMPLETE` |
| validity  | `VALIDITY_BENCH_COMPLETE` |
| analytic  | `ANALYTIC_BENCH_COMPLETE` |

## What gets saved

Each agent writes its results into `llm_agents/results/<agent>/<benchmark>/`:

```
llm_agents/results/opus47/waveform/
├── AGENT_PROMPT.md              # generated in Step 1
├── CHANGELOG.md                 # updated after every approach
├── models/
│   ├── 01_svd_gpr_raw/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── saved_model/         # gitignored (pkl, joblib, etc.)
│   │   └── scorecard.json
│   └── ...
└── comparison/
    ├── error_data.json          # raw per-sample validation errors
    ├── summary_table.json       # ranked model comparison
    ├── best_model.json
    ├── progress.{png,pdf}
    ├── loss_only_comparison.{png,pdf}
    └── error_histograms.{png,pdf}
```

Model binaries (`saved_model/`, `*.pkl`, `*.joblib`) are gitignored. Everything
else — scripts, scorecards, plots, JSON summaries — is committed.

## Updating a prompt

Edit `llm_agents/agent_prompts/<benchmark>.md` (one file), then regenerate:

```bash
python llm_agents/generate_prompt.py opus47 waveform --write
```

## Wiping a run

To delete one agent's results for a benchmark and start fresh:

```bash
rm -rf llm_agents/results/opus47/waveform/
mkdir  llm_agents/results/opus47/waveform/
python llm_agents/generate_prompt.py opus47 waveform --write
```
