#!/usr/bin/env python3
"""Recompute waveform losses using proper PyCBC FD mismatch on ALL validation cases.

Targets:
  - Opus 4.7:    correct metric but only 20/250 → rerun all 250
  - Sonnet 4.6:  naive FFT proxy → PyCBC on all 250
  - Haiku 4.5:   naive FFT proxy → PyCBC on all 250
  - GPT-5.4 Mini: correct metric but only 64/250 → rerun all 250

Usage (from gwBenchmarks root, with gwbench conda env):
    envs/gwbench/bin/python llm_agents/recompute_fd_mismatch_v2.py
"""

import importlib.util
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwbenchmarks.metrics import frequency_domain_mismatch, FD_MASSES_MSUN


def compute_fd_loss_single(h_pred, h_true, dt):
    """Compute mean FD mismatch for a single waveform pair."""
    vals = []
    for m in FD_MASSES_MSUN:
        try:
            mm = frequency_domain_mismatch(h_pred, h_true, dt_geometric=dt, mtot_msun=m)
        except Exception:
            mm = 1.0
        vals.append(mm)
    return float(np.mean(vals)), {f"mismatch_{int(m)}Msun": v for m, v in zip(FD_MASSES_MSUN, vals)}


def compute_fd_loss_batch(h_preds, h_trues, dt):
    """Compute mean FD mismatch over all cases."""
    n = len(h_preds)
    per_mass_accum = {f"mismatch_{int(m)}Msun": [] for m in FD_MASSES_MSUN}
    per_sample = []
    for i in range(n):
        loss_i, comp_i = compute_fd_loss_single(h_preds[i], h_trues[i], dt)
        per_sample.append(loss_i)
        for k, v in comp_i.items():
            per_mass_accum[k].append(v)
    per_mass_means = {k: float(np.mean(v)) for k, v in per_mass_accum.items()}
    loss = float(np.mean(list(per_mass_means.values())))
    per_mass_means["mean_fd_mismatch"] = loss
    return loss, per_mass_means, np.array(per_sample)


# ── Opus 4.7 ──────────────────────────────────────────────

def recompute_opus47():
    print("\n" + "=" * 60)
    print("Opus 4.7 — waveform (20 → all 250)")
    print("=" * 60)
    agent_dir = ROOT / "llm_agents" / "results" / "opus47" / "waveform"
    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location("opus47_lib", agent_dir / "_lib.py")
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)

    _, h_val, _, omega0_val = lib.load_split("validation")
    params_val, _, _, _ = lib.load_split("validation")
    dt = lib.T_GRID_DT
    n_val = len(h_val)
    print(f"  Loaded {n_val} validation cases, dt={dt}")

    models_dir = agent_dir / "models"
    summaries = []
    for mdir in sorted(models_dir.iterdir()):
        if not mdir.is_dir():
            continue
        sc_path = mdir / "scorecard.json"
        if not sc_path.exists():
            continue
        sc = json.loads(sc_path.read_text())
        old_loss = sc.get("loss")

        try:
            pred_spec = importlib.util.spec_from_file_location(
                f"opus47_pred_{mdir.name}", mdir / "predict.py")
            pred_mod = importlib.util.module_from_spec(pred_spec)
            pred_spec.loader.exec_module(pred_mod)

            mode = sc.get("parameterization", "raw7")
            X = lib.reparam(params_val, mode, omega0=omega0_val)
            h_pred = pred_mod.predict(X)
            if not np.iscomplexobj(h_pred):
                n_t = h_pred.shape[1] // 2
                h_pred = h_pred[:, :n_t] + 1j * h_pred[:, n_t:]

            t0 = time.time()
            loss, components, _ = compute_fd_loss_batch(h_pred, h_val, dt)
            elapsed = time.time() - t0

            sc["loss"] = loss
            sc["loss_components"] = components
            sc["loss_old_subset20"] = old_loss
            sc_path.write_text(json.dumps(sc, indent=2))
            summaries.append(sc)
            print(f"  {mdir.name}: {old_loss:.4e} -> {loss:.4e}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {mdir.name}: FAILED ({e})")
            traceback.print_exc()
            summaries.append(sc)

    _write_best(agent_dir, summaries)
    sys.path.remove(str(agent_dir))


# ── Sonnet 4.6 ────────────────────────────────────────────

def recompute_sonnet46():
    print("\n" + "=" * 60)
    print("Sonnet 4.6 — waveform (naive FFT → PyCBC, all 250)")
    print("=" * 60)
    agent_dir = ROOT / "llm_agents" / "results" / "sonnet46" / "waveform"

    # Load validation data in Sonnet's format
    with h5py.File(ROOT / "datasets" / "waveform" / "waveform_validation.h5", "r") as f:
        n = f.attrs["n_simulations"]
        val_data = []
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            val_data.append({
                'q': g.attrs["q"],
                'chi1': [g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]],
                'chi2': [g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]],
                'omega0': g.attrs["omega0"],
                't': g["t"][:],
                'h22': g["h22_real"][:] + 1j * g["h22_imag"][:],
            })
    print(f"  Loaded {len(val_data)} validation cases")

    models_dir = agent_dir / "models"
    summaries = []
    for mdir in sorted(models_dir.iterdir()):
        if not mdir.is_dir():
            continue
        sc_path = mdir / "scorecard.json"
        model_path = mdir / "saved_model" / "model.pkl"
        if not sc_path.exists() or not model_path.exists():
            continue
        sc = json.loads(sc_path.read_text())
        old_loss = sc.get("loss")

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            fn = model.get("fn") if isinstance(model, dict) else None
            if fn is None:
                print(f"  {mdir.name}: skipping (no 'fn' in model dict)")
                summaries.append(sc)
                continue

            # Get max_len and delta_t from model or compute from data
            max_len = model.get("max_len", max(len(d['h22']) for d in val_data))
            delta_t = model.get("delta_t", val_data[0]['t'][1] - val_data[0]['t'][0])

            # Predict and compute mismatch per case
            per_sample = []
            per_mass_accum = {f"mismatch_{int(m)}Msun": [] for m in FD_MASSES_MSUN}
            t0 = time.time()
            for d in val_data:
                try:
                    x = np.array([d['q']] + d['chi1'] + d['chi2']).reshape(1, -1)
                    h_pred_raw = fn(x)
                    if h_pred_raw.ndim == 2:
                        h_pred_raw = h_pred_raw[0]
                    # Pad/trim to match true waveform length
                    h_true = np.real(d['h22'])
                    if len(h_pred_raw) < len(h_true):
                        h_pred = np.zeros(len(h_true))
                        h_pred[:len(h_pred_raw)] = np.real(h_pred_raw)
                    else:
                        h_pred = np.real(h_pred_raw[:len(h_true)])

                    dt_phys = d['t'][1] - d['t'][0]
                    loss_i, comp_i = compute_fd_loss_single(h_pred, h_true, dt_phys)
                    per_sample.append(loss_i)
                    for k, v in comp_i.items():
                        per_mass_accum[k].append(v)
                except Exception:
                    per_sample.append(1.0)
                    for k in per_mass_accum:
                        per_mass_accum[k].append(1.0)

            elapsed = time.time() - t0
            per_mass_means = {k: float(np.mean(v)) for k, v in per_mass_accum.items()}
            loss = float(np.mean(list(per_mass_means.values())))
            per_mass_means["mean_fd_mismatch"] = loss

            sc["loss"] = loss
            sc["loss_components"] = per_mass_means
            sc["loss_old_naive_fft"] = old_loss
            sc_path.write_text(json.dumps(sc, indent=2))
            summaries.append(sc)
            print(f"  {mdir.name}: {old_loss:.4e} -> {loss:.4e}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {mdir.name}: FAILED ({e})")
            traceback.print_exc()
            summaries.append(sc)

    _write_best(agent_dir, summaries)


# ── Haiku 4.5 ─────────────────────────────────────────────

def recompute_haiku():
    print("\n" + "=" * 60)
    print("Haiku 4.5 — waveform (naive FFT → PyCBC, all 250)")
    print("=" * 60)
    agent_dir = ROOT / "llm_agents" / "results" / "haiku" / "waveform"

    # Load validation data
    with h5py.File(ROOT / "datasets" / "waveform" / "waveform_validation.h5", "r") as f:
        n = f.attrs["n_simulations"]
        val_data = []
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            val_data.append({
                'q': g.attrs["q"],
                'chi1': [g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]],
                'chi2': [g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]],
                'omega0': g.attrs["omega0"],
                't': g["t"][:],
                'h22': g["h22_real"][:] + 1j * g["h22_imag"][:],
            })
    print(f"  Loaded {len(val_data)} validation cases")

    models_dir = agent_dir / "models"
    summaries = []
    for mdir in sorted(models_dir.iterdir()):
        if not mdir.is_dir():
            continue
        sc_path = mdir / "scorecard.json"
        if not sc_path.exists():
            continue
        sc = json.loads(sc_path.read_text())
        old_loss = sc.get("loss")

        try:
            # Haiku saves multiple pkl files - load them
            saved = mdir / "saved_model"
            with open(saved / "svd_basis.pkl", "rb") as f:
                U = pickle.load(f)
            with open(saved / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open(saved / "gpr_models.pkl", "rb") as f:
                gpr_models = pickle.load(f)
            max_len_path = saved / "max_len.pkl"
            if max_len_path.exists():
                with open(max_len_path, "rb") as f:
                    max_len = pickle.load(f)
            else:
                max_len = U.shape[0]

            per_sample = []
            per_mass_accum = {f"mismatch_{int(m)}Msun": [] for m in FD_MASSES_MSUN}
            t0 = time.time()
            for d in val_data:
                try:
                    x = np.array([d['q']] + d['chi1'] + d['chi2']).reshape(1, -1)
                    x_scaled = scaler.transform(x)
                    coeffs = np.array([gpr.predict(x_scaled)[0] for gpr in gpr_models])
                    h_recon = np.dot(U, coeffs)

                    h_true = np.real(d['h22'])
                    if len(h_recon) < len(h_true):
                        h_pred = np.zeros(len(h_true))
                        h_pred[:len(h_recon)] = np.real(h_recon)
                    else:
                        h_pred = np.real(h_recon[:len(h_true)])

                    dt_phys = d['t'][1] - d['t'][0]
                    loss_i, comp_i = compute_fd_loss_single(h_pred, h_true, dt_phys)
                    per_sample.append(loss_i)
                    for k, v in comp_i.items():
                        per_mass_accum[k].append(v)
                except Exception:
                    per_sample.append(1.0)
                    for k in per_mass_accum:
                        per_mass_accum[k].append(1.0)

            elapsed = time.time() - t0
            per_mass_means = {k: float(np.mean(v)) for k, v in per_mass_accum.items()}
            loss = float(np.mean(list(per_mass_means.values())))
            per_mass_means["mean_fd_mismatch"] = loss

            sc["loss"] = loss
            sc["loss_components"] = per_mass_means
            sc["loss_old_naive_fft"] = old_loss
            sc_path.write_text(json.dumps(sc, indent=2))
            summaries.append(sc)
            print(f"  {mdir.name}: {old_loss:.4e} -> {loss:.4e}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {mdir.name}: FAILED ({e})")
            traceback.print_exc()
            summaries.append(sc)

    _write_best(agent_dir, summaries)


# ── GPT-5.4 Mini ──────────────────────────────────────────

def recompute_gpt54_mini():
    print("\n" + "=" * 60)
    print("GPT-5.4 Mini — waveform (64 → all 250)")
    print("=" * 60)
    agent_dir = ROOT / "llm_agents" / "results" / "gpt54_mini" / "waveform"
    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location("gpt54_lib", agent_dir / "_lib.py")
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)

    val_sims = lib.load_split("validation")
    n_val = len(val_sims)
    print(f"  Loaded {n_val} validation cases")

    models_dir = agent_dir / "models"
    summaries = []
    for mdir in sorted(models_dir.iterdir()):
        if not mdir.is_dir():
            continue
        sc_path = mdir / "scorecard.json"
        model_path = mdir / "saved_model" / "model.pkl"
        if not sc_path.exists() or not model_path.exists():
            continue
        sc = json.loads(sc_path.read_text())
        old_loss = sc.get("loss")

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            per_sample = []
            per_mass_accum = {f"mismatch_{int(m)}Msun": [] for m in FD_MASSES_MSUN}
            t0 = time.time()
            for sim in val_sims:
                try:
                    h_pred = lib.predict_from_model(model, sim)
                    h_true = sim["h"]
                    dt = sim.get("dt", lib.DT_GEOMETRIC)
                    loss_i, comp_i = compute_fd_loss_single(h_pred, h_true, dt)
                    per_sample.append(loss_i)
                    for k, v in comp_i.items():
                        per_mass_accum[k].append(v)
                except Exception:
                    per_sample.append(1.0)
                    for k in per_mass_accum:
                        per_mass_accum[k].append(1.0)

            elapsed = time.time() - t0
            per_mass_means = {k: float(np.mean(v)) for k, v in per_mass_accum.items()}
            loss = float(np.mean(list(per_mass_means.values())))
            per_mass_means["mean_fd_mismatch"] = loss

            sc["loss"] = loss
            sc["loss_components"] = per_mass_means
            sc["loss_old_subset64"] = old_loss
            sc_path.write_text(json.dumps(sc, indent=2))
            summaries.append(sc)
            print(f"  {mdir.name}: {old_loss:.4e} -> {loss:.4e}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {mdir.name}: FAILED ({e})")
            traceback.print_exc()
            summaries.append(sc)

    _write_best(agent_dir, summaries)
    sys.path.remove(str(agent_dir))


# ── Shared ─────────────────────────────────────────────────

def _write_best(agent_dir, summaries):
    valid = [s for s in summaries if s.get("loss") is not None and np.isfinite(s["loss"])]
    if valid:
        ranked = sorted(valid, key=lambda s: s["loss"])
        comp_dir = agent_dir / "comparison"
        comp_dir.mkdir(exist_ok=True)
        (comp_dir / "best_model.json").write_text(json.dumps(ranked[0], indent=2))
        (comp_dir / "summary_table.json").write_text(json.dumps(ranked, indent=2))
        print(f"  Best: {ranked[0].get('approach', ranked[0].get('name', '?'))} = {ranked[0]['loss']:.4e}")
    else:
        print("  No valid results!")


def main():
    recompute_opus47()
    recompute_sonnet46()
    recompute_haiku()
    recompute_gpt54_mini()
    print("\n\nDone. Run 'python docs/update_results.py' to refresh the website.")


if __name__ == "__main__":
    main()
