"""Starter submission: farthest-first physical templates + max-|z|^2 statistic.

Run it from your work directory; it writes a complete, valid submission to
``output/`` and prints its public detection fraction. It is a baseline to beat,
not a recommended design.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from detection_statistic import detection_statistic

# The helper module is ``template_bank_core.py`` next to this file in an isolated
# workspace, or ``gwbenchmarks.benchmarks.template_bank_core`` inside the repository.
try:
    from template_bank_core import (
        MAX_BANK_ROWS,
        PUBLIC_EVAL_PHASE_SEED,
        append_phase_column,
        generate_normalized_waveforms,
        greedy_farthest_first_order,
        load_public_data,
        pairwise_overlap_matrix,
        score_detection,
        write_submission,
    )
except ImportError:
    from gwbenchmarks.benchmarks.template_bank_core import (
        MAX_BANK_ROWS,
        PUBLIC_EVAL_PHASE_SEED,
        append_phase_column,
        generate_normalized_waveforms,
        greedy_farthest_first_order,
        load_public_data,
        pairwise_overlap_matrix,
        score_detection,
        write_submission,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-size", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=128)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--noise-trials", type=int, default=200_000)
    parser.add_argument("--signal-repeats", type=int, default=4)
    parser.add_argument("--output-dir", default="output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.bank_size <= MAX_BANK_ROWS:
        raise ValueError(f"--bank-size must be in [1, {MAX_BANK_ROWS}]")
    data = load_public_data()
    n = len(data.benchmark_params)
    n_train = int(round(args.train_fraction * n))
    train_indices = np.arange(n_train, dtype=np.int64)
    eval_indices = np.arange(n_train, n, dtype=np.int64)

    candidates = append_phase_column(data.benchmark_params[train_indices[: args.candidate_limit]], seed=0)
    candidate_waveforms = generate_normalized_waveforms(candidates, data)
    order = greedy_farthest_first_order(pairwise_overlap_matrix(candidate_waveforms, data.weights))
    chosen = np.asarray(order[: args.bank_size], dtype=np.int64)
    bank = candidate_waveforms[chosen]
    waveform_indices = train_indices[: args.candidate_limit][chosen]

    eval_waveforms = generate_normalized_waveforms(
        append_phase_column(data.benchmark_params[eval_indices], seed=PUBLIC_EVAL_PHASE_SEED), data
    )
    metrics = score_detection(
        bank,
        eval_waveforms,
        data.weights,
        detection_statistic,
        waveform_indices,
        seed=12345,
        n_noise_trials=args.noise_trials,
        n_signal_repeats=args.signal_repeats,
    )
    summary = {
        "method": "farthest_first_physical_templates_max_abs_overlap",
        "n_waveforms": int(len(bank)),
        "public_train_size": int(len(train_indices)),
        "public_eval_size": int(len(eval_indices)),
        "public_eval_noise_trials": metrics["n_noise_trials"],
        "public_eval_signal_trials": metrics["n_signal_trials"],
        "public_eval_false_alarm_probability": metrics["false_alarm_probability"],
        "public_eval_detection_fraction": metrics["detection_fraction"],
        "public_eval_detection_fraction_stderr": metrics["detection_fraction_stderr"],
        "public_eval_oracle_detection_fraction": metrics["oracle_detection_fraction"],
        "notes": "Starter baseline; agents are expected to improve both the bank and the statistic.",
    }
    write_submission(
        args.output_dir, bank, summary, waveform_indices, train_indices, eval_indices
    )
    shutil.copyfile(
        Path(__file__).resolve().parent / "detection_statistic.py",
        Path(args.output_dir) / "detection_statistic.py",
    )
    print(f"Detection fraction {metrics['detection_fraction']:.4f} "
          f"(oracle {metrics['oracle_detection_fraction']:.4f}) at FAP "
          f"{metrics['false_alarm_probability']:g}")


if __name__ == "__main__":
    main()
