#!/usr/bin/env python3
"""Hidden evaluator for the Template Bank Bench (detection in Gaussian noise).

Usage (from the gwBenchmarks root):

    python llm_agents/evaluate_template_bank.py --agent opus48
    python llm_agents/evaluate_template_bank.py --submission-dir /path/to/output

A submission directory contains ``bank_waveforms.npy``,
``detection_statistic.py`` and, optionally, ``waveform_indices.npy``,
``eval_indices.npy`` and ``run_summary.json`` (see
``llm_agents/agent_prompts/template_bank.md``).

The evaluator draws 2,000,000 noise-only trials and 10 signal trials for each
of the 1,000 hidden targets in ``bank_wf_params_test.npy``, sets the threshold
at false-alarm probability 1e-4, and reports the detection fraction and the
headline loss ``volume_loss_vs_oracle = 1 - VT / VT_oracle``. It takes
roughly 1-30 min depending on the cost of the submitted statistic.

Keep this file and ``bank_wf_params_test.npy`` out of reach of a running
agent: the seeds below and the hidden targets are what make the score a
held-out measurement (see ``llm_agents/README.md``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gwbenchmarks.benchmarks.template_bank_core import (  # noqa: E402
    FALSE_ALARM_PROBABILITY,
    HIDDEN_NOISE_TRIALS,
    HIDDEN_SIGNAL_REPEATS,
    PUBLIC_EVAL_PHASE_SEED,
    SNR_MIN,
    append_phase_column,
    generate_normalized_waveforms,
    load_public_data,
    score_detection,
)

# Evaluator-private seeds: reference phases of the hidden targets, and the
# noise/SNR draws. They never appear in agent-facing files.
HIDDEN_TEST_PHASE_SEED = 2718
HIDDEN_NOISE_SEED = 314159

DATA_DIR = REPO_ROOT / "datasets" / "template_bank"
REFERENCE_SCORES = REPO_ROOT / "llm_agents" / "template_bank_reference_scores.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--agent", help="agent ID; scores llm_agents/results/<agent>/template_bank/output")
    where.add_argument("--submission-dir", help="directory containing bank_waveforms.npy and detection_statistic.py")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="public arrays (f_amp, Aref_weights, bank_wf_params)")
    parser.add_argument("--hidden-params", default=str(DATA_DIR / "bank_wf_params_test.npy"))
    parser.add_argument("--noise-trials", type=int, default=HIDDEN_NOISE_TRIALS)
    parser.add_argument("--signal-repeats", type=int, default=HIDDEN_SIGNAL_REPEATS)
    parser.add_argument("--skip-public-eval", action="store_true",
                        help="skip re-scoring the agent's own public evaluation split")
    parser.add_argument("--report-path", default=None, help="default: <submission-dir>/../evaluation_report.json")
    return parser.parse_args()


def load_detection_statistic(submission_dir: Path) -> Callable:
    """Import ``detection_statistic`` from the submission (it may load files next to it)."""

    path = submission_dir / "detection_statistic.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing submission file: {path}")
    spec = importlib.util.spec_from_file_location("submission_detection_statistic", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(submission_dir))
    function = getattr(module, "detection_statistic", None)
    if not callable(function):
        raise AttributeError(f"{path} must define detection_statistic(complex_overlaps, bank_waveforms, indices)")
    return function


def load_submission(submission_dir: Path):
    bank = np.load(submission_dir / "bank_waveforms.npy")
    indices_path = submission_dir / "waveform_indices.npy"
    indices = np.load(indices_path) if indices_path.exists() else np.arange(len(bank))
    summary_path = submission_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return bank, np.asarray(indices, dtype=np.int64), summary, load_detection_statistic(submission_dir)


def build_report(
    submission_dir: Path,
    data_dir: Path = DATA_DIR,
    hidden_params_path: Path = DATA_DIR / "bank_wf_params_test.npy",
    noise_trials: int = HIDDEN_NOISE_TRIALS,
    signal_repeats: int = HIDDEN_SIGNAL_REPEATS,
    public_eval: bool = True,
) -> dict:
    data = load_public_data(data_dir)
    bank, indices, summary, statistic = load_submission(submission_dir)
    hidden_params = append_phase_column(np.load(hidden_params_path), seed=HIDDEN_TEST_PHASE_SEED)
    hidden_waveforms = generate_normalized_waveforms(hidden_params, data)

    started = time.time()
    hidden = score_detection(
        bank, hidden_waveforms, data.weights, statistic, indices,
        seed=HIDDEN_NOISE_SEED, n_noise_trials=noise_trials, n_signal_repeats=signal_repeats,
    )
    hidden["statistic_wall_seconds"] = time.time() - started
    # Headline metric (the suite's loss): fractional sensitive-volume loss vs the exact-template oracle.
    hidden["volume_loss_vs_oracle"] = 1.0 - hidden["relative_to_oracle"]
    # Secondary, for appendices only (the task does not ask agents to optimize it): volume per filter row.
    hidden["detection_fraction_per_row"] = hidden["detection_fraction"] / hidden["n_rows"]

    # Re-score the agent's own public evaluation split as a sanity check on its run_summary.json.
    public = None
    eval_path = submission_dir / "eval_indices.npy"
    if public_eval and eval_path.exists():
        eval_waveforms = generate_normalized_waveforms(
            append_phase_column(data.benchmark_params[np.load(eval_path)], seed=PUBLIC_EVAL_PHASE_SEED), data
        )
        public = score_detection(
            bank, eval_waveforms, data.weights, statistic, indices,
            seed=HIDDEN_NOISE_SEED + 1, n_noise_trials=noise_trials // 4, n_signal_repeats=4,
        )

    references = json.loads(REFERENCE_SCORES.read_text()) if REFERENCE_SCORES.exists() else None
    return {
        "submission_dir": str(submission_dir),
        "false_alarm_probability": FALSE_ALARM_PROBABILITY,
        "snr_min": SNR_MIN,
        "hidden_test_size": int(len(hidden_waveforms)),
        "submission_metrics": hidden,
        "public_eval_metrics": public,
        "agent_summary": summary,
        "reference_scores": references,
    }


def main() -> None:
    args = parse_args()
    if args.agent:
        submission_dir = REPO_ROOT / "llm_agents" / "results" / args.agent / "template_bank" / "output"
    else:
        submission_dir = Path(args.submission_dir).resolve()
    report = build_report(
        submission_dir, Path(args.data_dir), Path(args.hidden_params),
        args.noise_trials, args.signal_repeats, public_eval=not args.skip_public_eval,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    report_path = Path(args.report_path) if args.report_path else submission_dir.parent / "evaluation_report.json"
    report_path.write_text(text + "\n")
    m = report["submission_metrics"]
    print(f"rows {m['n_rows']}  detection fraction {m['detection_fraction']:.4f} +- {m['detection_fraction_stderr']:.4f}  "
          f"(oracle {m['oracle_detection_fraction']:.4f})  loss 1 - VT/VT_oracle = {m['volume_loss_vs_oracle']:.4f}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
