# Template Bank Detection Workspace

## Files

- `template_bank_core.py`: waveform generation, inner products, noise model, and `score_detection`
- `detection_statistic.py`: starter statistic, `max_i |z_i|^2`
- `create_submission.py`: starter script that writes a valid submission
- `data/`: public inputs (read-only)
- `output/`: submission directory

## Quick start

```bash
{PYTHON} create_submission.py
```

This writes a 16-row farthest-first physical-template bank with the starter
statistic and reports its public detection fraction at false-alarm probability
1e-4.

## Public data rules

`data/bank_wf_params.npy` is the only public parameter pool. Split it into a
public training subset and a public evaluation subset. The hidden evaluator
uses a separate hidden set of target waveforms and private noise seeds.
