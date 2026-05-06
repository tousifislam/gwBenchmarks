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

Some folders may also include `error_diagnosis.md` when a manual failure
diagnosis was written.

When an RG candidate has a documented pure Fourier-convention mismatch, the
folder may also include `candidate_waveform_fourier_fixed.py`,
`score_level13_fourier_fixed.json`, and `fourier_convention_rescore.md`. These
files preserve the raw agent output while recording the convention-normalized
diagnostic score used in the summary plot.

The score files use the single-detector convention described in the benchmark
README files: PyCBC `aLIGOZeroDetHighPower`, `f_low = 15 Hz`,
`f_high = 990 Hz`, and `df = 0.125 Hz`.

Only generated code, machine-readable scores, and selected concise diagnoses
are kept here. Exploratory plots and intermediate artifacts are omitted to keep
the branch reviewable.
