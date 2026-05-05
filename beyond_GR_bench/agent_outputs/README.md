# Agent Outputs

This folder stores the generated waveform code and score files for the
no-skill agent runs used in `results_summary.md`.

Each model has the same structure:

```text
agent_outputs/<model>/no_skills/RG/
|-- candidate_waveform.py
`-- score_level13.json

agent_outputs/<model>/no_skills/finite_size/
|-- candidate_waveform.py
`-- score_finite_size.json
```

The score files use the single-detector convention described in the benchmark
README files: PyCBC `aLIGOZeroDetHighPower`, `f_low = 15 Hz`,
`f_high = 990 Hz`, and `df = 0.125 Hz`.

Only generated code and machine-readable scores are kept here. Longer
diagnostic notes, exploratory plots, and intermediate artifacts are omitted to
keep the branch reviewable.
