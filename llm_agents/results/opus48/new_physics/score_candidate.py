#!/usr/bin/env python3
"""Post-hoc official scorer for opus48's new_physics candidate_waveform.h_of_f.

Uses the canonical gwbenchmarks scorer (NewPhysicsBench.score_h_of_f), which
computes PyCBC frequency-domain mismatch against the reference RG-tail waveform
over the 144-case grid (aLIGOZeroDetHighPower PSD, f_low=15, f_high=990).

Writes the same output files the other agents have:
  - mismatch_summary.json   (mean/median/max/min/std + argmax/argmin + n_failed)
  - mismatch_per_case.json   (per-case mismatch with Mc/eta/dL/lambda_RG)
  - per_sample_results.json  (per-case waveform diagnostics: norm, f_isco, ...)
  - sanity_summary.json      (structural checks)

Run from the gwBenchmarks root with the gwbench env:
    envs/gwbench/bin/python llm_agents/results/opus48/new_physics/score_candidate.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def load_candidate():
    spec = importlib.util.spec_from_file_location(
        "opus48_candidate", HERE / "candidate_waveform.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.h_of_f


def main() -> None:
    from gwbenchmarks.benchmarks.new_physics import NewPhysicsBench

    bench = NewPhysicsBench(config_path=REPO / "configs" / "new_physics.yaml")
    h_of_f = load_candidate()

    loss, components = bench.score_h_of_f(h_of_f)

    cases = bench.test_cases
    mismatches = [
        components[
            f"mismatch_mc{c['Mc']:g}_eta{c['eta']:g}_dl{c['dL']:g}_rg{c['lambda_RG']:g}"
        ]
        for c in cases
    ]
    mismatches = np.asarray(mismatches, dtype=np.float64)

    # ---- mismatch_per_case.json ----
    per_case = [
        {
            "Mc": c["Mc"],
            "eta": c["eta"],
            "dL": c["dL"],
            "lambda_RG": c["lambda_RG"],
            "mismatch": float(m),
        }
        for c, m in zip(cases, mismatches)
    ]

    # ---- mismatch_summary.json ----
    imax = int(np.argmax(mismatches))
    imin = int(np.argmin(mismatches))
    summary = {
        "mean_fd_mismatch": float(np.mean(mismatches)),
        "median_fd_mismatch": float(np.median(mismatches)),
        "max_fd_mismatch": float(np.max(mismatches)),
        "min_fd_mismatch": float(np.min(mismatches)),
        "std_fd_mismatch": float(np.std(mismatches)),
        "n_cases": int(len(mismatches)),
        "n_failed": int(np.sum(mismatches >= 1.0)),
        "argmax": {**{k: cases[imax][k] for k in ("Mc", "eta", "dL", "lambda_RG")},
                   "mismatch": float(mismatches[imax])},
        "argmin": {**{k: cases[imin][k] for k in ("Mc", "eta", "dL", "lambda_RG")},
                   "mismatch": float(mismatches[imin])},
        "scorer": "gwbenchmarks.benchmarks.new_physics.NewPhysicsBench.score_h_of_f",
        "psd": "aLIGOZeroDetHighPower",
        "f_low_hz": bench._f_low,
        "f_high_hz": bench._f_high,
        "delta_f": bench._delta_f,
    }

    # ---- per_sample_results.json (waveform diagnostics) ----
    f = bench.frequency_array()
    MSUN_SEC = 4.925491025543576e-6
    per_sample = []
    all_finite = True
    all_cutoffs_ok = True
    for c in cases:
        h = np.asarray(
            h_of_f(
                f, Mc=c["Mc"], eta=c["eta"], dL=c["dL"],
                tc=c["tc"], phic=c["phic"], lambda_RG=c["lambda_RG"],
                f_low=bench._f_low,
                fmax_over_fisco=c["fmax_over_fisco"],
                sigma_taper_over_fisco=c["sigma_taper_over_fisco"],
            )
        )
        M_sec = c["Mc"] * MSUN_SEC / c["eta"] ** 0.6
        f_isco = 1.0 / (np.pi * 6 ** 1.5 * M_sec)
        f_cut = c["fmax_over_fisco"] * f_isco
        nonzero = h != 0
        finite = bool(np.all(np.isfinite(h)))
        all_finite = all_finite and finite
        below = bool(np.all(h[f < bench._f_low] == 0))
        all_cutoffs_ok = all_cutoffs_ok and below
        per_sample.append({
            "Mc": c["Mc"], "eta": c["eta"], "dL": c["dL"],
            "lambda_RG": c["lambda_RG"],
            "f_isco": float(f_isco), "f_cut": float(f_cut),
            "nonzero_bins": int(np.sum(nonzero)),
            "max_abs": float(np.max(np.abs(h))) if h.size else 0.0,
            "norm": float(np.sum(np.abs(h) ** 2)),
        })

    sanity = {
        "all_finite": all_finite,
        "all_cutoffs_ok": all_cutoffs_ok,
        "frequency_bins": int(len(f)),
        "n_cases": len(cases),
    }

    (HERE / "mismatch_per_case.json").write_text(json.dumps(per_case, indent=2, sort_keys=True) + "\n")
    (HERE / "mismatch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (HERE / "per_sample_results.json").write_text(json.dumps(per_sample, indent=2, sort_keys=True) + "\n")
    (HERE / "sanity_summary.json").write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nLoss (mean FD mismatch) = {loss:.6e}")


if __name__ == "__main__":
    main()
