# Template Bank Detection Agent Rules

The full task description is in `AGENT_PROMPT.md`; read it completely.

## Python environment

Use `{PYTHON}` for every Python command. Required dependencies (numpy, scipy,
numba, lal, lalsimulation) are already installed. Do not create another
environment or install packages.

## Constraints

- Work only inside this folder; treat `data/` as read-only.
- The final submission in `output/` must not depend on files outside `output/`.
- Use five or fewer CPU cores for any parallel work.
- The hidden evaluator uses the same noise model and procedure as
  `template_bank_core.score_detection`, with 2,000,000 noise trials,
  10,000 signal trials, false-alarm probability 1e-4 and SNR >= 5 drawn from
  p(rho) ~ rho^-4.
