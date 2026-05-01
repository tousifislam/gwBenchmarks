# Results

Agent work products organized by agent name and benchmark.

## Structure

```
results/
    <agent_name>/
        CHANGELOG.md                    # running log (append after every approach)
        waveform/
            models/01_svd_gpr/          # numbered chronologically
            models/02_svd_nn/
            ...
            comparison/                 # cross-approach comparison
                error_histograms.{png,pdf}
                loss_only_comparison.{png,pdf}
                pareto_accuracy_speed.{png,pdf}
                progress.{png,pdf}
                summary_table.json
                error_data.json
        remnant/
            models/...
            comparison/...
        dynamics/...
        ringdown/...
        validity/...
        analytic/...
```

## Agents

| Agent | Model | Directory | Status |
|-------|-------|-----------|--------|
| Opus 4.7 | claude-opus-4-7 | `opus47/` | — |
| Opus 4.6 | claude-opus-4-6 | `opus46/` | — |
| Sonnet 4.6 | claude-sonnet-4-6 | `sonnet46/` | — |
| Haiku 4.5 | claude-haiku-4-5 | `haiku/` | — |
| GPT-5.5 High | gpt-5.5, reasoning high | `gpt55_high/` | — |
| GPT-5.4 Mini | gpt-5.4-mini | `gpt54_mini/` | — |
| GPT-5.3 Codex High | gpt-5.3-codex, reasoning high | `gpt53_codex_high/` | — |
| GPT-5.2 | gpt-5.2 | `gpt52/` | — |

## Per-approach outputs

Each `models/<NN>_<name>/` directory must contain:
- `train.py` — training script (reproducible)
- `predict.py` — prediction function (must be importable)
- `saved_model/` — serialized model artifacts
- `saved_model/expressions.json` — for PySR/gplearn approaches: all discovered expressions (list of {expression, complexity, loss})
- `scorecard.json` — structured results with both `loss` (raw accuracy) and `score` (loss + runtime penalty)
- `plots/` — approach-specific diagnostic plots

## Comparison outputs

Each `comparison/` directory must contain:
- `error_histograms.{png,pdf}` — histograms of per-sample errors (separate train vs validation distributions). For waveform/remnant, overlay NR error floor as reference. Must be true histograms, NOT bar charts.
- `error_data.json` — raw per-sample error arrays for all approaches (train + validation), saved for future reference
- `loss_only_comparison.{png,pdf}` — raw loss (without runtime penalty) across approaches for interpretability
- `pareto_accuracy_speed.{png,pdf}` — Pareto plot: combined score vs eval time
- `progress.{png,pdf}` — loss and eval time vs approach number (updated incrementally)
- `summary_table.json` — all approaches with both raw loss AND combined score (ranked by loss)
- `best_model.json` — pointer to the winning approach

### Analytic bench additional outputs
- `all_expressions.json` — collected closed-form expressions from all approaches (PySR Pareto fronts, gplearn results, hand-crafted formulas)

## Loss vs Score

- **Loss** = raw accuracy metric from the benchmark-specific loss function. Lower is better. This is the primary metric for comparing model quality.
- **Score** = `loss * [1 + alpha * log(1 + runtime / t0)]`. Penalizes slow models. Lower is better.
- Both must be reported in scorecards and summary tables.
- Comparison plots must show raw loss prominently — the score is secondary.

## Rules

1. **`datasets/` is READ-ONLY**: agents may load data but must never modify anything in `datasets/`
2. **No cross-agent access**: each agent works only in `results/<own_name>/` — never read or reference another agent's directory
3. **`gwbenchmarks/` is READ-ONLY**: agents may import shared code (metrics, plot_settings, benchmarks) but must not modify it
4. **Append-only changelog**: never edit previous entries in CHANGELOG.md
5. **Number approaches chronologically**: `01_`, `02_`, etc.
6. **Update progress plot after every approach**: the plot must reflect the current state
7. **All plots in PNG + PDF**: Nature-style, using `gwbenchmarks.plot_settings`
