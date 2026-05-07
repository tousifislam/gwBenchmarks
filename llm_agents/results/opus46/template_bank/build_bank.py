"""Build an ordered template bank with multi-phase templates for HM coverage."""

import sys
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.stdout.reconfigure(line_buffering=True)

from gwbenchmarks.benchmarks.template_bank import (
    load_template_bank_data,
    make_full_reference_waveform,
    normalize_waveform,
    overlap,
    append_phase_column,
    PUBLIC_EVAL_PHASE_SEED,
)

DATA_DIR = Path(__file__).resolve().parents[4] / "datasets" / "template_bank"
OUTPUT_DIR = Path(__file__).resolve().parent

TRAIN_FRACTION = 0.7
RNG_SEED = 42
N_PHASES = 16
N_REPRESENTATIVE = 600


def farthest_point_sampling(points, n_select, seed=42):
    """Select n_select points using Farthest Point Sampling in normalized space."""
    rng = np.random.default_rng(seed)
    # Normalize each dimension to [0, 1]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (points - mins) / ranges

    n = len(points)
    selected = [rng.integers(n)]
    min_dists = np.full(n, np.inf)

    for _ in range(n_select - 1):
        last = normed[selected[-1]]
        dists = np.sum((normed - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1  # exclude already selected
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return np.array(selected)


def generate_single_waveform(args):
    """Worker function for parallel waveform generation."""
    params_5d, freq_hz, amp_ref, weights = args
    from gwbenchmarks.benchmarks.template_bank import (
        make_full_reference_waveform,
        normalize_waveform,
        TemplateBankData,
    )
    data = TemplateBankData(
        frequencies_hz=freq_hz,
        amplitude_reference=amp_ref,
        weights=weights,
        public_params=np.empty((0, 4)),
    )
    wf = make_full_reference_waveform(params_5d, data)
    return normalize_waveform(wf, weights)


def generate_waveforms_parallel(params_array, data, n_workers=None):
    """Generate normalized waveforms in parallel."""
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    args_list = [
        (params_array[i], data.frequencies_hz, data.amplitude_reference, data.weights)
        for i in range(len(params_array))
    ]

    results = []
    with Pool(n_workers) as pool:
        for i, wf in enumerate(pool.imap(generate_single_waveform, args_list, chunksize=50)):
            results.append(wf)
            if (i + 1) % 500 == 0:
                print(f"    Generated {i+1}/{len(params_array)} waveforms...")

    return np.array(results, dtype=np.complex128)


def compute_overlap_matrix(h1_matrix, h2_matrix, weights):
    """Compute all pairwise overlaps using matrix multiplication."""
    h1_weighted = h1_matrix * weights[None, :]
    return np.abs(h1_weighted @ h2_matrix.conj().T)


def main():
    print("Loading data...")
    data = load_template_bank_data(DATA_DIR)
    n_total = data.public_params.shape[0]
    print(f"Total public parameters: {n_total}")

    # Split into train/eval
    rng = np.random.default_rng(RNG_SEED)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_train = int(n_total * TRAIN_FRACTION)
    train_indices = np.sort(indices[:n_train])
    eval_indices = np.sort(indices[n_train:])
    print(f"Train: {len(train_indices)}, Eval: {len(eval_indices)}")

    # Save splits
    np.save(OUTPUT_DIR / "train_indices.npy", train_indices)
    np.save(OUTPUT_DIR / "eval_indices.npy", eval_indices)

    # Generate eval waveforms with deterministic phase
    eval_params_4d = data.public_params[eval_indices]
    eval_params_5d = append_phase_column(eval_params_4d, seed=PUBLIC_EVAL_PHASE_SEED)
    print(f"Generating {len(eval_params_5d)} eval waveforms...")
    eval_waveforms = generate_waveforms_parallel(eval_params_5d, data)
    print(f"Eval waveforms generated. Shape: {eval_waveforms.shape}")

    # Select representative subset using FPS
    train_params_4d = data.public_params[train_indices]
    print(f"Selecting {N_REPRESENTATIVE} representative parameters via FPS...")
    fps_local_indices = farthest_point_sampling(train_params_4d, N_REPRESENTATIVE, seed=RNG_SEED)
    representative_params_4d = train_params_4d[fps_local_indices]
    print(f"Selected {len(representative_params_4d)} representative parameters.")

    # Create multi-phase candidates: each representative × N_PHASES phases
    phases = np.linspace(0, 2 * np.pi, N_PHASES, endpoint=False)
    candidate_params_5d = []
    for p4d in representative_params_4d:
        for phi in phases:
            candidate_params_5d.append(np.append(p4d, phi))
    candidate_params_5d = np.array(candidate_params_5d)
    print(f"Total candidates: {len(candidate_params_5d)} ({N_REPRESENTATIVE} params × {N_PHASES} phases)")

    # Generate candidate waveforms
    print(f"Generating {len(candidate_params_5d)} candidate waveforms...")
    candidate_waveforms = generate_waveforms_parallel(candidate_params_5d, data)
    print(f"Candidate waveforms generated. Shape: {candidate_waveforms.shape}")

    # Compute full overlap matrix: (n_eval, n_candidates)
    print("Computing overlap matrix...")
    overlap_matrix = compute_overlap_matrix(eval_waveforms, candidate_waveforms, data.weights)
    print(f"Overlap matrix shape: {overlap_matrix.shape}")
    print(f"Max overlap per eval signal: min={overlap_matrix.max(axis=1).min():.4f}, "
          f"median={np.median(overlap_matrix.max(axis=1)):.4f}, "
          f"max={overlap_matrix.max(axis=1).max():.4f}")

    n_eval = len(eval_waveforms)
    n_candidates = len(candidate_waveforms)
    overlap_evaluations = n_eval * n_candidates

    # Check potential coverage
    best_possible = overlap_matrix.max(axis=1)
    potential_coverage = float(np.mean(best_possible >= 0.97))
    print(f"Potential coverage (if all candidates used): {potential_coverage:.4f}")
    print(f"Potential median best overlap: {np.median(best_possible):.4f}")

    # Greedy selection
    running_best = np.zeros(n_eval, dtype=np.float64)
    bank_order = []
    used = np.zeros(n_candidates, dtype=bool)
    threshold = 0.97
    target_coverage = 0.5

    print("Running greedy bank construction...")
    while True:
        coverage = float(np.mean(running_best >= threshold))
        if coverage >= target_coverage and len(bank_order) > 0:
            print(f"  Target coverage {target_coverage} reached at bank size {len(bank_order)}")
            break

        if np.sum(~used) == 0:
            print("  All candidates exhausted.")
            break

        uncovered_mask = running_best < threshold
        n_uncovered = int(np.sum(uncovered_mask))
        if n_uncovered == 0:
            break

        # Greedy: pick candidate that covers most uncovered evals
        candidate_overlaps = overlap_matrix[uncovered_mask, :]
        covers_count = np.sum(candidate_overlaps >= threshold, axis=0)
        covers_count[used] = -1

        best_idx = int(np.argmax(covers_count))
        if covers_count[best_idx] <= 0:
            # No candidate covers any new eval - pick one that maximizes sum of improvements
            improvements = np.maximum(overlap_matrix - running_best[:, None], 0)
            sum_improvements = np.sum(improvements, axis=0)
            sum_improvements[used] = -1
            best_idx = int(np.argmax(sum_improvements))

        used[best_idx] = True
        bank_order.append(best_idx)
        running_best = np.maximum(running_best, overlap_matrix[:, best_idx])

        if len(bank_order) % 20 == 0:
            coverage = float(np.mean(running_best >= threshold))
            median_bo = float(np.median(running_best))
            print(f"  Bank size: {len(bank_order)}, Coverage: {coverage:.4f}, "
                  f"Median overlap: {median_bo:.4f}")

    # Continue adding for robustness
    target_bank_size = min(len(bank_order) + 200, n_candidates)
    extra = target_bank_size - len(bank_order)
    if extra > 0:
        print(f"Adding {extra} more templates for robustness...")
        for i in range(extra):
            available = np.where(~used)[0]
            if len(available) == 0:
                break
            uncovered_mask = running_best < threshold
            if np.any(uncovered_mask):
                candidate_overlaps = overlap_matrix[uncovered_mask, :]
                covers_count = np.sum(candidate_overlaps >= threshold, axis=0)
                covers_count[used] = -1
                best_idx = int(np.argmax(covers_count))
                if covers_count[best_idx] <= 0:
                    improvements = np.maximum(overlap_matrix - running_best[:, None], 0)
                    sum_improvements = np.sum(improvements, axis=0)
                    sum_improvements[used] = -1
                    best_idx = int(np.argmax(sum_improvements))
            else:
                improvements = np.maximum(overlap_matrix - running_best[:, None], 0)
                sum_improvements = np.sum(improvements, axis=0)
                sum_improvements[used] = -1
                best_idx = int(np.argmax(sum_improvements))

            used[best_idx] = True
            bank_order.append(best_idx)
            running_best = np.maximum(running_best, overlap_matrix[:, best_idx])

    # Build final bank_params
    bank_params = candidate_params_5d[bank_order]
    print(f"\nFinal bank size: {len(bank_params)}")
    print(f"Bank params shape: {bank_params.shape}")

    # Final metrics
    final_coverage = float(np.mean(running_best >= threshold))
    median_best_overlap = float(np.median(running_best))
    print(f"Final coverage: {final_coverage:.4f}")
    print(f"Final median best overlap: {median_best_overlap:.4f}")

    # Replay to get prefix_length_to_50pct
    running_best_replay = np.zeros(n_eval, dtype=np.float64)
    prefix_length_to_50pct = None
    for idx, bidx in enumerate(bank_order):
        running_best_replay = np.maximum(running_best_replay, overlap_matrix[:, bidx])
        cov = float(np.mean(running_best_replay >= threshold))
        if cov >= target_coverage:
            prefix_length_to_50pct = idx + 1
            break

    print(f"Prefix length to 50% coverage: {prefix_length_to_50pct}")

    # Save outputs
    np.save(OUTPUT_DIR / "bank_params.npy", bank_params)

    # Per-sample results
    running_best_final = np.zeros(n_eval, dtype=np.float64)
    for bidx in bank_order:
        running_best_final = np.maximum(running_best_final, overlap_matrix[:, bidx])
    per_sample = {
        "eval_indices": eval_indices.tolist(),
        "best_overlaps": running_best_final.tolist(),
        "covered_at_097": (running_best_final >= threshold).tolist(),
    }
    with open(OUTPUT_DIR / "per_sample_results.json", "w") as f:
        json.dump(per_sample, f)

    run_summary = {
        "method": "greedy_multiphase_fps_coverage",
        "n_bank": int(len(bank_params)),
        "overlap_evaluations": int(overlap_evaluations),
        "public_train_size": int(len(train_indices)),
        "public_eval_size": int(len(eval_indices)),
        "public_eval_threshold": float(threshold),
        "public_eval_target_coverage": float(target_coverage),
        "public_eval_coverage_fraction": float(final_coverage),
        "public_eval_median_best_overlap": float(median_best_overlap),
        "public_eval_prefix_length_to_50pct": prefix_length_to_50pct,
    }

    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("\nrun_summary.json:")
    print(json.dumps(run_summary, indent=2))
    print("\nDone!")


if __name__ == "__main__":
    main()
