"""Ordered template-bank construction for the Template Bank Bench (opus48).

Strategy (greedy max-coverage / set cover):
  1. Generate normalised IMRPhenomXHM frequency-domain waveforms for the public
     parameter pool (cached).
  2. Split the pool into a public training subset (candidate pool + coverage
     objective) and a disjoint public evaluation subset.
  3. Greedily order templates: repeatedly add the candidate that newly covers
     the most still-uncovered training waveforms at overlap >= 0.97. This is the
     classic greedy set-cover ordering and minimises templates for a coverage
     target.
  4. Stop once the public-eval coverage comfortably exceeds 50% (margin for
     hidden-test generalisation), then report prefix-coverage metrics.

All overlaps use the benchmark's own weighted complex overlap (phase-insensitive
via abs), so phi_ref is a weak degree of freedom; bank rows use phi_ref = 0.
"""
import sys, json, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from gwbenchmarks.benchmarks import template_bank as TB

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"; CACHE.mkdir(exist_ok=True)
THRESH = 0.97
TARGET = 0.50
EVAL_FRAC = 0.30
COVER_GOAL = 0.72          # build train coverage past 50% for hidden-test margin
SEED = 20259


def get_waveforms(data):
    f = CACHE / "wf_all.npy"
    if f.exists():
        return np.load(f)
    t = time.time()
    W = TB.generate_normalized_waveforms(data.public_params, data)
    np.save(f, W)
    print(f"generated {len(W)} waveforms in {time.time()-t:.0f}s")
    return W


def overlap_matrix(V, W, weights):
    """|<V_i, W_j>| for normalised rows. Returns (nV, nW) real."""
    Vw = V * weights                      # (nV, nf)
    G = Vw @ np.conj(W).T                 # (nV, nW) complex
    return np.abs(G)


def main():
    data = TB.load_template_bank_data(str(REPO / "datasets" / "template_bank"))
    weights = data.weights
    W = get_waveforms(data)
    n = len(W)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_eval = int(EVAL_FRAC * n)
    eval_idx = np.sort(perm[:n_eval])
    train_idx = np.sort(perm[n_eval:])
    Wtr = W[train_idx]; Wev = W[eval_idx]

    # candidate-vs-train coverage matrix (greedy objective)
    t = time.time()
    O = overlap_matrix(Wtr, Wtr, weights)          # (ntr, ntr)
    print(f"overlap matrix {O.shape} in {time.time()-t:.0f}s")
    cover = O >= THRESH

    # greedy set cover on the training set
    ntr = len(Wtr)
    covered = np.zeros(ntr, bool)
    order = []
    overlap_evals = int(O.size)                    # matrix build cost
    while covered.mean() < COVER_GOAL:
        gains = (cover & ~covered).sum(axis=1)
        b = int(np.argmax(gains))
        if gains[b] == 0:
            break
        order.append(b)
        covered |= cover[b]
        if len(order) % 25 == 0:
            print(f"  bank={len(order)} train_cov={covered.mean():.3f}")
    print(f"greedy bank size {len(order)}, train coverage {covered.mean():.3f}")

    bank_local = np.array(order)
    bank_global = train_idx[bank_local]
    bank_params4 = data.public_params[bank_global]
    bank_params = np.column_stack([bank_params4, np.zeros(len(bank_params4))])  # phi_ref=0

    # evaluate ordered prefix coverage on the held-out public eval set
    Obank_eval = overlap_matrix(Wev, W[bank_global], weights)   # (n_eval, n_bank)
    overlap_evals += int(Obank_eval.size)
    running = np.zeros(len(Wev))
    prefix_cov, prefix_med = [], []
    cov50 = None
    for j in range(Obank_eval.shape[1]):
        running = np.maximum(running, Obank_eval[:, j])
        c = float(np.mean(running >= THRESH))
        prefix_cov.append(c); prefix_med.append(float(np.median(running)))
        if cov50 is None and c >= TARGET:
            cov50 = j + 1

    summary = {
        "method": "greedy_max_coverage_set_cover_IMRPhenomXHM",
        "n_bank": int(len(bank_params)),
        "overlap_evaluations": int(overlap_evals),
        "public_train_size": int(len(train_idx)),
        "public_eval_size": int(len(eval_idx)),
        "public_eval_threshold": THRESH,
        "public_eval_target_coverage": TARGET,
        "public_eval_coverage_fraction": float(prefix_cov[-1]),
        "public_eval_median_best_overlap": float(prefix_med[-1]),
        "public_eval_prefix_length_to_50pct": (int(cov50) if cov50 else None),
        "build_coverage_goal": COVER_GOAL,
        "seed": SEED,
    }
    np.save(HERE / "bank_params.npy", bank_params)
    np.save(HERE / "train_indices.npy", train_idx)
    np.save(HERE / "eval_indices.npy", eval_idx)
    (HERE / "run_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(CACHE / "prefix_coverage.npy", np.array(prefix_cov))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
