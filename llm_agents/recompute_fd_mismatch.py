#!/usr/bin/env python3
"""Recompute waveform & analytic losses using proper PyCBC FD mismatch.

GPT-5.5 High and GPT-5.3 Codex High used a time-domain L2 proxy instead
of the real PyCBC frequency-domain mismatch. This script loads each saved
model, re-predicts on the validation set, computes the correct FD mismatch,
and updates the scorecards + best_model.json.

Usage (from gwBenchmarks root, with gwbench conda env):
    envs/gwbench/bin/python llm_agents/recompute_fd_mismatch.py
"""

import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np

from gwbenchmarks.metrics import frequency_domain_mismatch, FD_MASSES_MSUN

ROOT = Path(__file__).resolve().parents[1]

AGENTS = ["gpt55_high", "gpt53_codex_high"]
BENCHMARKS = ["waveform", "analytic"]


def load_waveform_grid(split, n_grid=128):
    path = ROOT / f"datasets/waveform/waveform_{split}.h5"
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        names = [f"sim_{i:04d}" for i in range(n)]
        tmin = max(float(np.min(f[k]["t"][:])) for k in names)
        tmax = min(float(np.max(f[k]["t"][:])) for k in names)
        grid = np.linspace(max(tmin, -1500), min(tmax, 120), n_grid)
        dt = grid[1] - grid[0]
        X, Y = [], []
        for k in names:
            g = f[k]
            X.append([g.attrs["q"], g.attrs["chi1x"], g.attrs["chi1y"],
                       g.attrs["chi1z"], g.attrs["chi2x"], g.attrs["chi2y"],
                       g.attrs["chi2z"], g.attrs["omega0"]])
            re = np.interp(grid, g["t"][:], g["h22_real"][:])
            im = np.interp(grid, g["t"][:], g["h22_imag"][:])
            Y.append(np.r_[re, im])
    return np.asarray(X), np.asarray(Y), dt


def load_analytic_grid(split, n_grid=160):
    path = ROOT / f"datasets/analytic/analytic_{split}.h5"
    with h5py.File(path, "r") as f:
        groups = list(f["sims"].keys())
        tmin = max(float(np.min(f["sims"][k]["t"][:])) for k in groups)
        tmax = min(float(np.max(f["sims"][k]["t"][:])) for k in groups)
        grid = np.linspace(max(tmin, -1200), min(tmax, 120), n_grid)
        dt = grid[1] - grid[0]
        X, Y = [], []
        for k in groups:
            g = f["sims"][k]
            X.append([g.attrs["q"]])
            re = np.interp(grid, g["t"][:], g["h22_real"][:])
            im = np.interp(grid, g["t"][:], g["h22_imag"][:])
            Y.append(np.r_[re, im])
    return np.asarray(X), np.asarray(Y), dt


def raw_features(arr, bench, param):
    """Reproduce the parameterization transforms from suite_runner.py."""
    X = np.asarray(arr, dtype=np.float64)
    if bench == "analytic":
        return X
    if param == "raw_7d":
        return X[:, :7]
    if param == "raw_6d":
        return X[:, :6]
    if param == "effective_spins":
        q = X[:, 0]; chi1z = X[:, 3]; chi2z = X[:, 6]
        eta = q / (1 + q)**2
        chi_eff = (q * chi1z + chi2z) / (1 + q)
        chi_p = np.sqrt(X[:, 1]**2 + X[:, 2]**2)
        return np.column_stack([eta, chi_eff, chi_p])
    if param == "massdiff_spins":
        q = X[:, 0]
        dm = (q - 1) / (q + 1)
        chi1_mag = np.sqrt(X[:, 1]**2 + X[:, 2]**2 + X[:, 3]**2)
        chi2_mag = np.sqrt(X[:, 4]**2 + X[:, 5]**2 + X[:, 6]**2)
        return np.column_stack([dm, chi1_mag, X[:, 3], chi2_mag, X[:, 6]])
    if param == "spherical_spins":
        q = X[:, 0]; eta = q / (1 + q)**2
        chi1_mag = np.sqrt(X[:, 1]**2 + X[:, 2]**2 + X[:, 3]**2)
        chi1_theta = np.arccos(X[:, 3] / np.maximum(chi1_mag, 1e-12))
        chi2_mag = np.sqrt(X[:, 4]**2 + X[:, 5]**2 + X[:, 6]**2)
        chi2_theta = np.arccos(X[:, 6] / np.maximum(chi2_mag, 1e-12))
        return np.column_stack([eta, chi1_mag, chi1_theta, chi2_mag, chi2_theta])
    if param == "with_omega0":
        return X[:, :8]
    return X[:, :7]


def y_to_complex(y):
    """Convert [re, im] concatenated array to complex array."""
    n = y.shape[1] // 2
    return y[:, :n] + 1j * y[:, n:]


def compute_fd_loss(h_pred, h_true, dt):
    """Compute mean FD mismatch over masses, for all validation cases."""
    n = h_pred.shape[0]
    per_mass_sums = {m: 0.0 for m in FD_MASSES_MSUN}
    per_sample = np.zeros(n)
    for i in range(n):
        sample_vals = []
        for m in FD_MASSES_MSUN:
            try:
                mm = frequency_domain_mismatch(h_pred[i], h_true[i],
                                                dt_geometric=dt, mtot_msun=m)
            except Exception:
                mm = 1.0
            per_mass_sums[m] += mm
            sample_vals.append(mm)
        per_sample[i] = np.mean(sample_vals)

    per_mass_means = {f"mismatch_{int(m)}Msun": per_mass_sums[m] / n
                      for m in FD_MASSES_MSUN}
    loss = float(np.mean(list(per_mass_means.values())))
    per_mass_means["mean_fd_mismatch"] = loss
    return loss, per_mass_means, per_sample


def recompute_agent_bench(agent, bench, X_val_raw, y_val, dt):
    """Recompute all model scorecards for one agent/benchmark."""
    work = ROOT / "llm_agents" / "results" / agent / bench
    models_dir = work / "models"
    if not models_dir.exists():
        print(f"  No models dir for {agent}/{bench}, skipping")
        return

    h_true = y_to_complex(y_val)
    summaries = []

    for mdir in sorted(models_dir.iterdir()):
        if not mdir.is_dir():
            continue
        model_path = mdir / "saved_model" / "model.joblib"
        scorecard_path = mdir / "scorecard.json"
        if not model_path.exists() or not scorecard_path.exists():
            print(f"    {mdir.name}: skipping (no model or scorecard)")
            continue

        sc = json.loads(scorecard_path.read_text())
        old_loss = sc.get("loss", None)

        try:
            model = joblib.load(model_path)
            param = sc.get("parameterization", "raw_7d")
            Xv = raw_features(X_val_raw, bench, param)
            y_pred_raw = model.predict(Xv)
            h_pred = y_to_complex(y_pred_raw)

            t0 = time.time()
            loss, components, per_sample = compute_fd_loss(h_pred, h_true, dt)
            elapsed = time.time() - t0

            sc["loss"] = loss
            sc["loss_components"] = components
            sc["loss_old_proxy"] = old_loss
            scorecard_path.write_text(json.dumps(sc, indent=2))

            summaries.append(sc)
            print(f"    {mdir.name}: {old_loss:.4e} -> {loss:.4e}  ({elapsed:.1f}s)")

        except Exception as e:
            print(f"    {mdir.name}: FAILED ({e})")
            sc["loss"] = old_loss
            summaries.append(sc)

    if summaries:
        ranked = sorted(summaries, key=lambda s: s.get("loss", float("inf")))
        comp_dir = work / "comparison"
        comp_dir.mkdir(exist_ok=True)
        (comp_dir / "best_model.json").write_text(json.dumps(ranked[0], indent=2))
        (comp_dir / "summary_table.json").write_text(json.dumps(ranked, indent=2))
        print(f"  Best: {ranked[0].get('approach', '?')} = {ranked[0]['loss']:.4e}")


def main():
    print("Loading waveform validation data...")
    X_wf, Y_wf, dt_wf = load_waveform_grid("validation")
    print(f"  {len(X_wf)} cases, dt={dt_wf:.4f} M")

    print("Loading analytic validation data...")
    X_an, Y_an, dt_an = load_analytic_grid("validation")
    print(f"  {len(X_an)} cases, dt={dt_an:.4f} M")

    for agent in AGENTS:
        print(f"\n{'='*60}")
        print(f"Agent: {agent}")
        print(f"{'='*60}")

        print(f"\n  Waveform:")
        recompute_agent_bench(agent, "waveform", X_wf, Y_wf, dt_wf)

        print(f"\n  Analytic:")
        recompute_agent_bench(agent, "analytic", X_an, Y_an, dt_an)

    print("\n\nDone. Run 'python docs/update_results.py' to refresh the website.")


if __name__ == "__main__":
    main()
