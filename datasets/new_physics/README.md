# New Physics Bench Dataset

Frequency-domain RG-tail inspiral waveform benchmark based on arXiv:2602.08833.

## Benchmark Type

Unlike other benchmarks in this suite, the New Physics Bench is **formula-driven**:
- The agent receives physics formulas (a "source packet") describing the dominant nonspinning (2,2) mode with RG-tail corrections.
- The agent must implement `h_of_f()` from first principles.
- Scoring compares the agent's waveform against a reference implementation using PyCBC FD mismatch.

There is **no HDF5 training/validation dataset**.

## Reference Implementation

The ground truth lives in `gwbenchmarks/rg_tail_reference.py`. It implements:

```
hhat_22 = H_eff * T_22 * rho_22^2 * exp(i * delta_22)
```

with:
- H_eff: effective source from conservative circular-orbit energy
- T_22: radiative external tail factor with RG running
- rho_22: residual amplitude (4PN)
- delta_22: residual phase

The beyond-GR parameter `lambda_RG` scales the running anomalous dimension inside T_22. Setting `lambda_RG = 1` recovers GR.

## Parameter Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| Mc | [12, 20, 28.3, 40] M_sun | Detector-frame chirp mass |
| eta | [0.12, 0.16, 0.22, 0.247] | Symmetric mass ratio |
| dL | [200, 410, 1000] Mpc | Luminosity distance |
| lambda_RG | [0.8, 1.0, 1.2] | RG deformation parameter |

Total: 4 x 4 x 3 x 3 = **144 test cases**

## Scoring

- **PSD**: PyCBC `aLIGOZeroDetHighPower`
- **f_low**: 15 Hz
- **f_high**: 990 Hz
- **delta_f**: 0.125 Hz
- **Metric**: PyCBC match maximized over time and phase shifts
- **Loss**: mean mismatch = mean(1 - match) across 144 cases

## Agent Source Packet

The agent receives `llm_agents/agent_prompts/new_physics.md` which contains:
- Function signature for `h_of_f()`
- Compact formula sheet from arXiv:2602.08833 Section IV
- Frequency array conventions and cutoff prescriptions
- No access to the reference implementation
