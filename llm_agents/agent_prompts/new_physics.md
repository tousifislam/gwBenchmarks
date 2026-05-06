# New Physics Bench — {AGENT_LABEL} Agent

You are joining a new gravitational-wave forecasting project for the first
time. Your workspace contains only this prompt and a compact source packet.
Your work directory is: `llm_agents/results/{AGENT}/new_physics/`

## Allowed Files

Only inspect:

- This prompt (your AGENT_PROMPT.md)
- `llm_agents/agent_prompts/new_physics_formulas.md`

You may create `candidate_waveform.py`, `notes.md`, and temporary sanity-check
scripts. Do not inspect project source code (`gwbenchmarks/`), previous
benchmark runs, hidden scorers, hidden test cases, other agents' results,
internet sources, or external code.

## Task

Write a standalone Python file named `candidate_waveform.py` defining:

```python
def h_of_f(
    f,
    Mc,
    eta,
    dL,
    tc=0.0,
    phic=0.0,
    lambda_RG=1.0,
    f_low=20.0,
    fmax_over_fisco=1.3,
    sigma_taper_over_fisco=0.01,
    phase_only=False,
):
    ...
```

The function must accept a NumPy frequency array in Hz and return a complex
frequency-domain strain array with the same shape.

Use only ordinary scientific Python packages such as `numpy` and `scipy`.

## Inputs and Cutoffs

Use detector-frame masses and geometric seconds internally:

```text
MSUN_SEC = 4.925491025543576e-6
MPC_SEC  = 1.0292712503e14
M_sec    = Mc * MSUN_SEC / eta^(3/5)
dL_sec   = dL * MPC_SEC
x_f      = (pi M_sec f)^(2/3)
```

Use

```text
f_isco = 1 / (pi 6^(3/2) M_sec)
f_cut  = fmax_over_fisco * f_isco
sigma  = sigma_taper_over_fisco * f_isco
W(f)   = 1 / [1 + exp((f - f_isco) / sigma)]
```

Set the waveform exactly to zero for `f < f_low` and `f >= f_cut`.
Apply the Fermi taper W(f) within the valid band.

## Source Formula

Use `llm_agents/agent_prompts/new_physics_formulas.md` as the local source
for the dominant nonspinning `(2,2)` RG-tail ingredients. The hidden evaluator
will score the generated waveform numerically; it will not grade prose.

## Scoring

Your implementation will be scored on 144 test cases (4 chirp masses x 4 eta
x 3 distances x 3 lambda_RG values) using frequency-domain mismatch:

```
mismatch = 1 - max_{t,phi} <h_cand, h_ref> / sqrt(<h_cand, h_cand> <h_ref, h_ref>)
```

with PyCBC `aLIGOZeroDetHighPower` PSD, f_low=15 Hz, f_high=990 Hz.

## Deliverables

Required:

- `candidate_waveform.py`

Recommended:

- `notes.md`, briefly documenting the waveform convention and any equation
  checks you performed.

When complete, print "NEW_PHYSICS_BENCH_COMPLETE" on its own line.
