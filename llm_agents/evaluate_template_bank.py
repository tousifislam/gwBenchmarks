#!/usr/bin/env python3
"""Evaluate a Template Bank Bench submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gwbenchmarks.benchmarks.template_bank import (
    HIDDEN_TEST_PHASE_SEED,
    PUBLIC_EVAL_PHASE_SEED,
    TemplateBankBench,
    append_phase_column,
    compute_hplus_hcross,
    load_template_bank_data,
    normalize_waveform,
    overlap,
    parameter_dict_from_row,
)


FULL_HM_MODES = [
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (2, -2),
    (2, -1),
    (3, -3),
    (3, -2),
    (4, -4),
]

ORTHOGONAL_MODE_GROUPS = [
    [(2, 2), (2, -2)],
    [(3, 3), (3, -3)],
    [(4, 4), (4, -4)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", help="Agent ID under llm_agents/results/<agent>/template_bank")
    parser.add_argument("--submission-dir", help="Directory containing bank_params.npy")
    parser.add_argument("--data-dir", default="datasets/template_bank")
    parser.add_argument("--hidden-data", default="datasets/template_bank/bank_wf_params_test.npy")
    parser.add_argument("--literature-params", default="datasets/template_bank/calpha_grid_params.npy")
    parser.add_argument("--config", default="configs/template_bank.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-coverage", type=float, default=None)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def orthogonalize_wfs(waveforms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cov = np.zeros((3, 3), dtype=np.complex128)
    for j in range(3):
        for k in range(j, 3):
            cov[j, k] = np.sum(weights * waveforms[j] * np.conj(waveforms[k]), axis=-1)
    cov[np.tril_indices(3)] = cov.T.conj()[np.tril_indices(3)]
    chol = np.linalg.cholesky(np.linalg.inv(cov[::-1, ::-1]))[::-1, ::-1]
    return np.dot(chol.conj().T, waveforms)


def make_orthogonal_mode_templates(params: np.ndarray, data) -> np.ndarray:
    par_dict = parameter_dict_from_row(params)
    modes = np.zeros((3, len(data.frequencies_hz)), dtype=np.complex128)
    for idx, mode_group in enumerate(ORTHOGONAL_MODE_GROUPS):
        modes[idx] = compute_hplus_hcross(
            data.frequencies_hz,
            par_dict,
            approximant="IMRPhenomXHM",
            harmonic_modes=mode_group,
        )[0] / data.amplitude_reference
        modes[idx] /= np.sqrt(overlap(modes[idx], modes[idx], data.weights))
    return orthogonalize_wfs(modes, data.weights)


def evaluate_literature_reference(literature_params_path: Path, evaluation_params: np.ndarray, data, threshold: float) -> dict | None:
    if not literature_params_path.exists():
        return None

    literature_params = np.load(literature_params_path)
    template_wfs = np.asarray(
        [make_orthogonal_mode_templates(params, data) for params in literature_params]
    )

    best_scores = []
    for params in evaluation_params:
        href = normalize_waveform(
            compute_hplus_hcross(
                data.frequencies_hz,
                parameter_dict_from_row(params),
                approximant="IMRPhenomXHM",
                harmonic_modes=FULL_HM_MODES,
            )[0]
            / data.amplitude_reference,
            data.weights,
        )
        z = np.empty((template_wfs.shape[0], 3), dtype=np.float64)
        for i in range(template_wfs.shape[0]):
            for j in range(3):
                z[i, j] = overlap(href, template_wfs[i, j], data.weights)
        best_scores.append(float(np.sqrt(np.max(np.sum(z**2, axis=1)))))

    best_scores = np.asarray(best_scores, dtype=np.float64)
    return {
        "n_templates": int(len(literature_params)),
        "coverage_fraction_ge_threshold": float(np.mean(best_scores >= threshold)),
        "median_best_score": float(np.median(best_scores)),
        "min_best_score": float(np.min(best_scores)),
    }


def load_submission(submission_dir: Path) -> tuple[np.ndarray, dict]:
    bank_path = submission_dir / "bank_params.npy"
    summary_path = submission_dir / "run_summary.json"
    if not bank_path.exists():
        raise FileNotFoundError(f"Missing submission file: {bank_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing submission file: {summary_path}")
    with open(summary_path, encoding="utf-8") as stream:
        summary = json.load(stream)
    return np.load(bank_path), summary


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.submission_dir:
        submission_dir = (repo_root / args.submission_dir).resolve()
    elif args.agent:
        submission_dir = repo_root / "llm_agents" / "results" / args.agent / "template_bank"
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
    bank_params, agent_summary = load_submission(submission_dir)
    submission_metrics = bench.evaluate_bank(bank_params, hidden_params, data)

    public_eval_metrics = None
    eval_indices_path = submission_dir / "eval_indices.npy"
    if eval_indices_path.exists():
        eval_indices = np.load(eval_indices_path)
        public_eval_params = append_phase_column(
            data.public_params[eval_indices], seed=PUBLIC_EVAL_PHASE_SEED
        )
        public_eval_metrics = bench.evaluate_bank(bank_params, public_eval_params, data)

    literature_metrics = evaluate_literature_reference(
        literature_params_path,
        hidden_params,
        data,
        float(bench.config.get("threshold", 0.97)),
    )

    report = {
        "submission_dir": str(submission_dir),
        "threshold": float(bench.config.get("threshold", 0.97)),
        "target_coverage": float(bench.config.get("target_coverage", 0.5)),
        "agent_summary": agent_summary,
        "public_eval_metrics": public_eval_metrics,
        "hidden_test_size": int(len(hidden_params)),
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
