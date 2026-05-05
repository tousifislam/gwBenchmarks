# RG Waveform Source Packet

This is a lean source packet for the RG waveform benchmark.  It is meant to
avoid slow PDF reading while still leaving the agent responsible for translating
the formulas into code.

## Files

- `prompt_source_packet_no_skills.md`: the benchmark prompt.
- `2602.08833_relevant_formulas.md`: compact formula sheet extracted and
  cleaned from the relevant parts of arXiv:2602.08833.

## Boundary

For a no-skill benchmark, give the agent only this folder in a fresh workspace.
Do not include project source code, old candidate waveforms, old notes, score
files, hidden cases, hidden scorer code, or skill files.

This packet intentionally omits workflow advice, known failure diagnoses, and
model-specific hints.  Those belong in the skill condition, not the source
packet condition.
