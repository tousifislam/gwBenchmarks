# Finite-Size Balance-Law SPA Waveform Benchmark Source Packet

This packet supports a benchmark where agents implement a frequency-domain
inspiral waveform with finite-size corrections from arXiv:2410.00294 by
starting from the binding energy, flux at infinity, and absorbed flux.

Use this packet when the benchmark condition allows a compact source packet
instead of requiring the agent to read the full paper.

## Files

- `prompt_source_packet_no_skills.md`: task prompt for an agent with no project
  skills and no access to the repository reference implementation.
- `2410.00294_relevant_formulas.md`: extracted formulas and conventions needed
  to build the benchmark waveform.

## Hidden Reference

The benchmark reference implementation lives outside this packet at
`src/finite_size/waveform.py`. Candidate agents should not inspect that file.
