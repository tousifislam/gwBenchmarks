#!/usr/bin/env python3
"""Evaluate a Template Bank Bench waveform-reconstruction submission."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Callable

import numpy as np

from gwbenchmarks.benchmarks.template_bank import (
    FULL_HM_MODES,
    ORTHOGONAL_MODE_GROUPS,
    PUBLIC_EVAL_PHASE_SEED,
    TemplateBankBench,
    append_phase_column,
    complex_overlap,
    compute_hplus_hcross,
    load_template_bank_data,
    normalize_waveform,
    parameter_dict_from_row,
)


# Evaluator-private deterministic seed. This value is deliberately absent from
# the public benchmark helper and the agent prompt.
HIDDEN_TEST_PHASE_SEED = 2718


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", help="Agent ID under llm_agents/results/<agent>/template_bank"
    )
    parser.add_argument(
        "--submission-dir", help="Directory containing bank_waveforms.npy"
    )
    parser.add_argument("--data-dir", default="datasets/template_bank")
    parser.add_argument(
        "--hidden-data", default="datasets/template_bank/bank_wf_params_test.npy"
    )
    parser.add_argument(
        "--literature-params",
        default="datasets/template_bank/calpha_grid_params.txt",
    )
    parser.add_argument("--config", default="configs/template_bank.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-coverage", type=float, default=None)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def orthogonalize_waveforms(
    waveforms: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return a weighted-orthonormal basis spanning the supplied waveforms."""

    n_waveforms = len(waveforms)
    covariance = np.empty((n_waveforms, n_waveforms), dtype=np.complex128)
    for row in range(n_waveforms):
        for column in range(n_waveforms):
            covariance[row, column] = complex_overlap(
                waveforms[row], waveforms[column], weights
            )
    reverse_inverse = np.linalg.inv(covariance[::-1, ::-1])
    cholesky = np.linalg.cholesky(reverse_inverse)[::-1, ::-1]
    return np.dot(cholesky.conj().T, waveforms)


def make_orthogonal_mode_templates(params: np.ndarray, data) -> np.ndarray:
    """Build the five literature mode-pair filters for one intrinsic point."""

    par_dict = parameter_dict_from_row(params)
    modes = np.empty(
        (len(ORTHOGONAL_MODE_GROUPS), len(data.frequencies_hz)),
        dtype=np.complex128,
    )
    for index, mode_group in enumerate(ORTHOGONAL_MODE_GROUPS):
        waveform = compute_hplus_hcross(
            data.frequencies_hz,
            par_dict,
            approximant="IMRPhenomXHM",
            harmonic_modes=mode_group,
        )[0] / data.amplitude_reference
        modes[index] = normalize_waveform(waveform, data.weights)
    return orthogonalize_waveforms(modes, data.weights)


def evaluate_literature_reference(
    literature_params_path: Path,
    evaluation_params: np.ndarray,
    data,
    threshold: float,
    target_coverage: float,
) -> dict | None:
    """Evaluate the mode-decomposed literature comparison on hidden targets."""

    if not literature_params_path.exists():
        return None

    literature_params = np.loadtxt(literature_params_path)
    template_groups = np.asarray(
        [make_orthogonal_mode_templates(params, data) for params in literature_params]
    )
    best_matches = []
    for params in evaluation_params:
        target = normalize_waveform(
            compute_hplus_hcross(
                data.frequencies_hz,
                parameter_dict_from_row(params),
                approximant="IMRPhenomXHM",
                harmonic_modes=FULL_HM_MODES,
            )[0]
            / data.amplitude_reference,
            data.weights,
        )
        group_matches = []
        for group in template_groups:
            overlaps = np.asarray(
                [complex_overlap(target, waveform, data.weights) for waveform in group]
            )
            group_matches.append(float(np.sqrt(np.sum(np.abs(overlaps) ** 2))))
        best_matches.append(max(group_matches))

    best_matches = np.clip(np.asarray(best_matches, dtype=np.float64), 0.0, 1.0)
    coverage_count = int(np.count_nonzero(best_matches >= threshold))
    coverage_fraction = float(coverage_count / len(best_matches))
    return {
        "n_intrinsic_points": int(len(literature_params)),
        "n_waveforms": int(
            len(literature_params) * len(ORTHOGONAL_MODE_GROUPS)
        ),
        "threshold": float(threshold),
        "target_coverage": float(target_coverage),
        "coverage_operator": ">",
        "match_operator": ">=",
        "coverage_count": coverage_count,
        "required_coverage_count": int(
            np.floor(target_coverage * len(best_matches))
        )
        + 1,
        "coverage_fraction": coverage_fraction,
        "median_match": float(np.median(best_matches)),
        "min_match": float(np.min(best_matches)),
        "passed_target": coverage_fraction > target_coverage,
    }


def load_reconstruction_function(submission_dir: Path) -> Callable:
    function_path = submission_dir / "reconstruction_function.py"
    if not function_path.exists():
        raise FileNotFoundError(f"Missing submission file: {function_path}")
    spec = importlib.util.spec_from_file_location(
        "template_bank_submission_reconstruction", function_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reconstruction module: {function_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reconstruct = getattr(module, "reconstruct_from_overlaps", None)
    if not callable(reconstruct):
        raise AttributeError(
            f"{function_path} must define callable reconstruct_from_overlaps("
            "complex_overlaps, bank_waveforms, indices)."
        )
    return reconstruct


def load_submission(
    submission_dir: Path, expected_frequency_count: int
) -> tuple[np.ndarray, np.ndarray, dict, Callable]:
    bank_path = submission_dir / "bank_waveforms.npy"
    indices_path = submission_dir / "waveform_indices.npy"
    summary_path = submission_dir / "run_summary.json"
    for path in (bank_path, indices_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing submission file: {path}")

    bank_waveforms = np.asarray(np.load(bank_path), dtype=np.complex128)
    if bank_waveforms.ndim != 2 or bank_waveforms.shape[1] != expected_frequency_count:
        raise ValueError(
            "bank_waveforms.npy must have shape (n_waveforms, n_frequencies); "
            f"expected second dimension {expected_frequency_count}, got "
            f"{bank_waveforms.shape}."
        )
    if len(bank_waveforms) == 0:
        raise ValueError("Submitted waveform bank is empty.")

    waveform_indices = np.asarray(np.load(indices_path), dtype=np.int64)
    if waveform_indices.shape != (len(bank_waveforms),):
        raise ValueError(
            "waveform_indices.npy must have shape (n_waveforms,), got "
            f"{waveform_indices.shape}."
        )
    with open(summary_path, encoding="utf-8") as stream:
        summary = json.load(stream)
    return (
        bank_waveforms,
        waveform_indices,
        summary,
        load_reconstruction_function(submission_dir),
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.submission_dir:
        submission_dir = (repo_root / args.submission_dir).resolve()
    elif args.agent:
        submission_dir = (
            repo_root / "llm_agents" / "results" / args.agent / "template_bank"
        )
    else:
        raise SystemExit("Provide --agent or --submission-dir.")

    data_dir = (repo_root / args.data_dir).resolve()
    hidden_data_path = (repo_root / args.hidden_data).resolve()
    literature_params_path = (repo_root / args.literature_params).resolve()
    bench = TemplateBankBench(config_path=repo_root / args.config)
    if args.threshold is not None:
        bench.config["threshold"] = args.threshold
    if args.target_coverage is not None:
        bench.config["target_coverage"] = args.target_coverage

    data = load_template_bank_data(data_dir)
    hidden_params = append_phase_column(
        np.load(hidden_data_path), seed=HIDDEN_TEST_PHASE_SEED
    )
    bank_waveforms, waveform_indices, agent_summary, reconstruct = load_submission(
        submission_dir, len(data.frequencies_hz)
    )
    submission_metrics = bench.evaluate_bank(
        bank_waveforms,
        hidden_params,
        data,
        reconstruct_from_overlaps=reconstruct,
        waveform_indices=waveform_indices,
    )

    public_eval_metrics = None
    eval_indices_path = submission_dir / "eval_indices.npy"
    if eval_indices_path.exists():
        eval_indices = np.asarray(np.load(eval_indices_path), dtype=np.int64)
        public_eval_params = append_phase_column(
            data.public_params[eval_indices], seed=PUBLIC_EVAL_PHASE_SEED
        )
        public_eval_metrics = bench.evaluate_bank(
            bank_waveforms,
            public_eval_params,
            data,
            reconstruct_from_overlaps=reconstruct,
            waveform_indices=waveform_indices,
        )

    threshold = float(bench.config.get("threshold", 0.97))
    target_coverage = float(bench.config.get("target_coverage", 0.9))
    literature_metrics = evaluate_literature_reference(
        literature_params_path,
        hidden_params,
        data,
        threshold,
        target_coverage,
    )
    report = {
        "submission_dir": str(submission_dir),
        "threshold": threshold,
        "target_coverage": target_coverage,
        "coverage_operator": ">",
        "match_operator": ">=",
        "agent_summary": agent_summary,
        "public_eval_metrics": public_eval_metrics,
        "hidden_test_size": int(len(hidden_params)),
        "submission_waveform_shape": list(bank_waveforms.shape),
        "submission_metrics": submission_metrics,
        "hidden_literature_reference": literature_metrics,
    }

    if args.report_path is not None:
        report_path = (repo_root / args.report_path).resolve()
    else:
        report_path = submission_dir / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
