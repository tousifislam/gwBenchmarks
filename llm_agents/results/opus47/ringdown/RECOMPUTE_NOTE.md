# Ringdown Scope Recomputation (2026-05-02)

The original evaluation for Opus 4.7 evaluated all QNM modes
(51 modes, ~1.2M validation samples). For fair comparison with
other agents, the loss was recomputed on the (2,2,0) mode only
(531 validation samples).

- Old loss (all modes, n=1,204,578): 5.634280e-06
- New loss ((2,2,0) only, n=531):    1.129938e-05
  - Re(omega) relative error: 8.349459e-08
  - Im(omega) relative error: 2.251527e-05

Script: llm_agents/recompute_remnant_validity_ringdown.py
