# Source-Packet No-Skill Finite-Size Waveform Benchmark Prompt

You are joining a new gravitational-wave forecasting project for the first
time. Your workspace contains only this prompt and a compact source packet.

## Allowed Files

Only inspect:

- `prompt_source_packet_no_skills.md`
- `2410.00294_relevant_formulas.md`
- `README.md`

You may create `candidate_waveform.py`, `notes.md`, and temporary sanity-check
scripts. Do not inspect project source code, previous benchmark runs, hidden
scorers, hidden test cases, skill files, internet sources, or external code.

## Task

Write a standalone Python file named `candidate_waveform.py` defining:

```python
def h_of_f(
    f,
    Mc,
    eta,
    dL,
    chi1=0.0,
    chi2=0.0,
    tc=0.0,
    phic=0.0,
    kappa_s=1.0,
    kappa_a=0.0,
    H0=0.0,
    Lambda_tilde=0.0,
    H1E=0.0,
    H1B=0.0,
    H1E_bar=0.0,
    H1B_bar=0.0,
    f_low=20.0,
    fmax_over_fisco=1.0,
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
MPC_SEC  = 3.0856775814913673e22 / 299792458
M_sec    = Mc * MSUN_SEC / eta^(3/5)
dL_sec   = dL * MPC_SEC
v_f      = (pi M_sec f)^(1/3)
```

Use

```text
f_isco = 1 / (pi 6^(3/2) M_sec)
f_cut  = fmax_over_fisco f_isco
sigma  = sigma_taper_over_fisco f_isco
W(f)   = 1 / [1 + exp((f - f_cut)/sigma)].
```

Set the waveform exactly to zero for `f < f_low` and `f >= f_cut`.

## Source Formula

Use `2410.00294_relevant_formulas.md` as the local source for the binding
energy, flux at infinity, absorbed flux, and compact frequency-domain
construction. The hidden evaluator will score the generated waveform
numerically; it will not grade prose.

The important physical requirement is that the waveform should be derived from
the balance law, not from copying a closed-form finite-size phase:

```text
-F_infty - dotM = dE/dt.
```

## Deliverables

Required:

- `candidate_waveform.py`

Recommended:

- `notes.md`, briefly documenting the waveform convention and any equation
  checks you performed.
