#!/usr/bin/env python3
"""Recompute waveform losses using proper PyCBC FD mismatch on ALL 250 validation cases.

Targets:
  - Opus 4.7:     correct metric but only 20/250 → rerun all 250
  - Sonnet 4.6:   naive FFT/RMSE proxy → PyCBC on all 250
  - Haiku 4.5:    naive FFT/RMSE proxy → PyCBC on all 250
  - GPT-5.4 Mini: correct metric but only 64/250 → rerun all 250

Usage (from gwBenchmarks root, with gwbench conda env):
    envs/gwbench/bin/python llm_agents/recompute_fd_mismatch_v3.py
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
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwbenchmarks.metrics import frequency_domain_mismatch, FD_MASSES_MSUN


def compute_fd_loss_single(h_pred, h_true, dt):
    vals = []
    for m in FD_MASSES_MSUN:
        try:
            mm = frequency_domain_mismatch(h_pred, h_true, dt_geometric=dt, mtot_msun=m)
        except Exception:
            mm = 1.0
        vals.append(mm)
    return float(np.mean(vals))


def compute_fd_loss_batch(h_preds, h_trues, dt):
    per_mass_accum = {f"mismatch_{int(m)}Msun": [] for m in FD_MASSES_MSUN}
    per_sample = []
    for i in range(len(h_preds)):
        for m in FD_MASSES_MSUN:
            try:
                mm = frequency_domain_mismatch(h_preds[i], h_trues[i],
                                                dt_geometric=dt, mtot_msun=m)
            except Exception:
                mm = 1.0
            per_mass_accum[f"mismatch_{int(m)}Msun"].append(mm)
        per_sample.append(np.mean([per_mass_accum[k][-1] for k in per_mass_accum]))
    per_mass_means = {k: float(np.mean(v)) for k, v in per_mass_accum.items()}
    loss = float(np.mean(list(per_mass_means.values())))
    per_mass_means["mean_fd_mismatch"] = loss
    return loss, per_mass_means, np.array(per_sample)


def load_waveform_validation():
    path = ROOT / "datasets" / "waveform" / "waveform_validation.h5"
    sims = []
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            sims.append({
                'q': float(g.attrs["q"]),
                'chi1x': float(g.attrs["chi1x"]),
                'chi1y': float(g.attrs["chi1y"]),
                'chi1z': float(g.attrs["chi1z"]),
                'chi2x': float(g.attrs["chi2x"]),
                'chi2y': float(g.attrs["chi2y"]),
                'chi2z': float(g.attrs["chi2z"]),
                'omega0': float(g.attrs["omega0"]),
                't': g["t"][:].astype(np.float64),
                'h22': g["h22_real"][:] + 1j * g["h22_imag"][:],
            })
    return sims


def load_waveform_training():
    path = ROOT / "datasets" / "waveform" / "waveform_training.h5"
    sims = []
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            sims.append({
                't': g["t"][:].astype(np.float64),
                'h22': g["h22_real"][:] + 1j * g["h22_imag"][:],
            })
    return sims


def _write_best(agent_dir, summaries, recomputed_names):
    """Write best_model.json from only successfully recomputed models."""
    valid = [s for s in summaries
             if s.get("_recomputed") and s.get("loss") is not None
             and np.isfinite(s["loss"])]
    if valid:
        ranked = sorted(valid, key=lambda s: s["loss"])
        for s in ranked:
            s.pop("_recomputed", None)
        comp_dir = agent_dir / "comparison"
        comp_dir.mkdir(exist_ok=True)
        (comp_dir / "best_model.json").write_text(json.dumps(ranked[0], indent=2))
        (comp_dir / "summary_table.json").write_text(json.dumps(ranked, indent=2))
        name = ranked[0].get('approach', ranked[0].get('name', '?'))
        print(f"  Best ({len(valid)} successful): {name} = {ranked[0]['loss']:.4e}")
    else:
        print("  No valid recomputed results!")


# ── Opus 4.7 ──────────────────────────────────────────────

def recompute_opus47(val_sims):
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
    print(f"  {len(h_val)} validation cases, dt={dt}")

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
            sc["_recomputed"] = True
            sc_path.write_text(json.dumps({k: v for k, v in sc.items()
                                            if k != "_recomputed"}, indent=2))
            summaries.append(sc)
            print(f"  {mdir.name}: {old_loss:.4e} -> {loss:.4e}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {mdir.name}: FAILED ({e})")
            sc["_recomputed"] = False
            summaries.append(sc)

    _write_best(agent_dir, summaries, set())
    sys.path.remove(str(agent_dir))


# ── Sonnet 4.6 ────────────────────────────────────────────

def _sonnet_reparam(p, name):
    """Reparameterize Sonnet-style: p = [q, chi1x..chi2z] (7 values)."""
    q, c1x, c1y, c1z, c2x, c2y, c2z = p
    if name == 'raw':
        return np.array(p, dtype=float)
    elif name == 'eff':
        m1, m2 = q/(1+q), 1/(1+q)
        eta = q/(1+q)**2
        chi_eff = m1*c1z + m2*c2z
        chi1m = np.sqrt(c1x**2+c1y**2+c1z**2)
        chi2m = np.sqrt(c2x**2+c2y**2+c2z**2)
        chi1p = np.sqrt(c1x**2+c1y**2)
        chi2p = np.sqrt(c2x**2+c2y**2)
        chi_p = max(chi1p, (4*m2+3*m1)/(4*m1+3*m2)*(m2/m1)*chi2p)
        th1 = np.arctan2(chi1p, c1z) if chi1m > 1e-10 else 0.
        th2 = np.arctan2(chi2p, c2z) if chi2m > 1e-10 else 0.
        return np.array([eta, chi_eff, chi_p, chi1m, chi2m, th1, th2])
    elif name == 'sph':
        eta = q/(1+q)**2
        chi1m = np.sqrt(c1x**2+c1y**2+c1z**2)
        chi2m = np.sqrt(c2x**2+c2y**2+c2z**2)
        th1 = np.arccos(np.clip(c1z/max(chi1m,1e-10),-1,1))
        th2 = np.arccos(np.clip(c2z/max(chi2m,1e-10),-1,1))
        ph1 = np.arctan2(c1y, c1x)
        ph2 = np.arctan2(c2y, c2x)
        return np.array([eta, chi1m, th1, ph1, chi2m, th2, ph2])
    elif name == 'md':
        m1, m2 = q/(1+q), 1/(1+q)
        dm = m1-m2
        chi_eff = m1*c1z + m2*c2z
        chi1m = np.sqrt(c1x**2+c1y**2+c1z**2)
        chi2m = np.sqrt(c2x**2+c2y**2+c2z**2)
        chi1p = np.sqrt(c1x**2+c1y**2)
        chi2p = np.sqrt(c2x**2+c2y**2)
        chi_p = max(chi1p, (4*m2+3*m1)/(4*m1+3*m2)*(m2/m1)*chi2p)
        ph1 = np.arctan2(c1y, c1x)
        ph2 = np.arctan2(c2y, c2x)
        return np.array([dm, chi_eff, chi_p, chi1m, chi2m, ph1, ph2])
    raise ValueError(f"Unknown reparam: {name}")


def _sonnet_apply_reparam(P, name):
    return np.array([_sonnet_reparam(p, name) for p in P])


def _sonnet_decode(c_ri, basis):
    """Decode complex SVD coefficients from real [re, im] stacked format."""
    n = len(c_ri) // 2
    c_cplx = c_ri[:n] + 1j * c_ri[n:]
    return c_cplx @ basis[:n]


REPARAM_MAP = {
    'raw': 'raw', 'eff': 'eff', 'sph': 'sph', 'md': 'md',
    't0_start': 'raw',
}


def _sonnet_predict_one(model, x_1d, basis, sc_name):
    """Predict a single waveform from a Sonnet model dict.
    x_1d: 1D array of input features (already reparameterized).
    Returns complex waveform array.
    """
    x_2d = x_1d.reshape(1, -1)
    keys = set(model.keys())

    # GPR-based SVD
    if 'gpr' in keys and 'basis' in keys and 'nodes' not in keys:
        sc = model.get('sc')
        n_c = model.get('n_c', basis.shape[0])
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        c_ri = model['gpr'].predict(x_sc).flatten()
        return _sonnet_decode(c_ri, basis)

    # RF-based SVD
    if 'rf' in keys:
        c_ri = model['rf'].predict(x_2d).flatten()
        return _sonnet_decode(c_ri, basis)

    # MLP-based SVD
    if 'mlp' in keys and 'basis' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        c_ri = model['mlp'].predict(x_sc).flatten()
        return _sonnet_decode(c_ri, basis)

    # GBR-based SVD
    if 'gbr' in keys:
        c_ri = model['gbr'].predict(x_2d).flatten()
        return _sonnet_decode(c_ri, basis)

    # KRR-based SVD
    if 'krr' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        c_ri = model['krr'].predict(x_sc).flatten()
        return _sonnet_decode(c_ri, basis)

    # KNN-based SVD
    if 'knn' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        c_ri = model['knn'].predict(x_sc).flatten()
        return _sonnet_decode(c_ri, basis)

    # ExtraTrees SVD
    if 'et' in keys:
        c_ri = model['et'].predict(x_2d).flatten()
        return _sonnet_decode(c_ri, basis)

    # Poly + Ridge SVD
    if 'poly' in keys and ('r' in keys or 'ridge' in keys):
        sc = model.get('sc', model.get('scaler'))
        ridge = model.get('r', model.get('ridge'))
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        x_poly = model['poly'].transform(x_sc)
        c_ri = ridge.predict(x_poly).flatten()
        return _sonnet_decode(c_ri, basis)

    # RBF interp
    if 'rbf' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        c_ri = model['rbf'].predict(x_sc).flatten()
        return _sonnet_decode(c_ri, basis)

    # Amplitude/Phase GPR
    if ('g_a' in keys or 'ga' in keys) and ('g_p' in keys or 'gp' in keys):
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        g_a = model.get('g_a', model.get('ga'))
        g_p = model.get('g_p', model.get('gp'))
        B_a = model.get('Ba')
        B_p = model.get('Bp')
        ca = g_a.predict(x_sc).flatten()
        cp = g_p.predict(x_sc).flatten()
        amp = ca @ B_a[:len(ca)]
        phase = cp @ B_p[:len(cp)]
        return np.abs(amp) * np.exp(1j * phase)

    # EIM GPR
    if 'gpr' in keys and 'nodes' in keys and 'V' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        c_ri = model['gpr'].predict(x_sc).flatten()
        return _sonnet_decode(c_ri, basis)

    # gplearn
    if 'gpl_r' in keys and 'gpl_i' in keys:
        cr = model['gpl_r'].predict(x_2d).flatten()
        ci = model['gpl_i'].predict(x_2d).flatten()
        c_ri = np.concatenate([cr, ci])
        return _sonnet_decode(c_ri, basis)

    raise ValueError(f"Unknown model format: {list(keys)}")


def recompute_sonnet46(val_sims, train_sims_for_grid):
    print("\n" + "=" * 60)
    print("Sonnet 4.6 — waveform (naive RMSE → PyCBC, all 250)")
    print("=" * 60)
    agent_dir = ROOT / "llm_agents" / "results" / "sonnet46" / "waveform"
    models_dir = agent_dir / "models"

    # Derive the same T_GRID Sonnet used
    t_max_start_tr = max(s['t'][0] for s in train_sims_for_grid)
    t_max_start_va = max(s['t'][0] for s in val_sims)
    t_common_start = max(t_max_start_tr, t_max_start_va)
    T_GRID = np.linspace(t_common_start, 100.0, 2048)
    dt = float(T_GRID[1] - T_GRID[0])
    print(f"  T_GRID: {T_GRID[0]:.1f}..{T_GRID[-1]:.1f}, n=2048, dt={dt:.4f}")

    # Interpolate validation waveforms to T_GRID
    from scipy.interpolate import interp1d
    H_val = []
    for s in val_sims:
        fr = interp1d(s['t'], s['h22'].real, bounds_error=False, fill_value=0.0)
        fi = interp1d(s['t'], s['h22'].imag, bounds_error=False, fill_value=0.0)
        H_val.append(fr(T_GRID) + 1j * fi(T_GRID))
    H_val = np.array(H_val)

    # Raw params for reparameterization
    P_val = np.array([[s['q'], s['chi1x'], s['chi1y'], s['chi1z'],
                        s['chi2x'], s['chi2y'], s['chi2z']] for s in val_sims])

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

        # Skip NN-prefix models that are from Haiku's format
        is_nn = mdir.name.startswith("NN")

        try:
            if is_nn:
                # Haiku-style model in Sonnet directory
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                loss = _recompute_haiku_style_model(
                    model, sc, val_sims, mdir.name)
            else:
                model = joblib.load(model_path)
                basis = model.get('basis')
                if basis is None:
                    print(f"  {mdir.name}: skipping (no basis)")
                    summaries.append(sc)
                    continue

                reparam_name = sc.get('parameterization', 'raw')
                if reparam_name in ('t0_start', 't0_at_peak'):
                    reparam_name = 'raw'
                Xv = _sonnet_apply_reparam(P_val, reparam_name)

                h_preds = []
                for i in range(len(val_sims)):
                    h_pred = _sonnet_predict_one(model, Xv[i], basis, reparam_name)
                    h_preds.append(h_pred)
                h_preds = np.array(h_preds)

                t0 = time.time()
                loss, components, _ = compute_fd_loss_batch(h_preds, H_val, dt)
                elapsed = time.time() - t0

            sc["loss"] = loss
            sc["loss_old_naive"] = old_loss
            sc["_recomputed"] = True
            sc_path.write_text(json.dumps({k: v for k, v in sc.items()
                                            if k != "_recomputed"}, indent=2))
            summaries.append(sc)
            print(f"  {mdir.name}: {old_loss:.4e} -> {loss:.4e}")
        except Exception as e:
            print(f"  {mdir.name}: FAILED ({e})")
            traceback.print_exc()
            sc["_recomputed"] = False
            summaries.append(sc)

    _write_best(agent_dir, summaries, set())


def _haiku_reparam(q, chi1, chi2, func_name):
    """Haiku reparameterization from build_comprehensive.py."""
    if func_name == 'reparameterize_raw' or func_name == 'raw':
        return np.array([q] + list(chi1) + list(chi2))
    elif func_name == 'reparameterize_eta_chi' or func_name == 'eta_chi':
        eta = q / (1 + q)**2
        chi1_mag = np.linalg.norm(chi1)
        chi2_mag = np.linalg.norm(chi2)
        chi_eff = (chi1[2] + q * chi2[2]) / (1 + q)
        chi_p = max(
            np.sqrt(chi1[0]**2 + chi1[1]**2),
            (4*q)/(3*(1+q)**2) * np.sqrt(chi2[0]**2 + chi2[1]**2)
        )
        theta1 = np.arccos(chi1[2] / (chi1_mag + 1e-10)) if chi1_mag > 0 else 0
        theta2 = np.arccos(chi2[2] / (chi2_mag + 1e-10)) if chi2_mag > 0 else 0
        return np.array([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])
    elif func_name == 'reparameterize_spherical' or func_name == 'spherical':
        eta = q / (1 + q)**2
        chi1_mag = np.linalg.norm(chi1)
        chi2_mag = np.linalg.norm(chi2)
        theta1 = np.arccos(np.clip(chi1[2] / max(chi1_mag, 1e-10), -1, 1))
        theta2 = np.arccos(np.clip(chi2[2] / max(chi2_mag, 1e-10), -1, 1))
        phi1 = np.arctan2(chi1[1], chi1[0])
        phi2 = np.arctan2(chi2[1], chi2[0])
        return np.array([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])
    raise ValueError(f"Unknown param_func: {func_name}")


def _recompute_haiku_style_model(model, sc, val_sims, model_name):
    """Recompute a Haiku-format model (models, U, scaler, max_len).
    These models operate on real parts only and use their own SVD basis.
    """
    U = model['U']
    scaler = model['scaler']
    models_list = model['models']
    max_len = model['max_len']
    param_func = model.get('param_func', 'reparameterize_raw')

    per_sample = []
    for sim in val_sims:
        q = sim['q']
        chi1 = [sim['chi1x'], sim['chi1y'], sim['chi1z']]
        chi2 = [sim['chi2x'], sim['chi2y'], sim['chi2z']]

        x = _haiku_reparam(q, chi1, chi2, param_func).reshape(1, -1)
        x_sc = scaler.transform(x)
        coeffs = np.array([m.predict(x_sc)[0] for m in models_list])
        h_recon = np.dot(U, coeffs)

        # h_recon is real, length max_len. Compare to real part of true waveform.
        h_true_real = np.real(sim['h22'])
        dt = sim['t'][1] - sim['t'][0]

        # Pad/trim
        if len(h_recon) < len(h_true_real):
            h_pred = np.zeros(len(h_true_real))
            h_pred[:len(h_recon)] = h_recon
        else:
            h_pred = h_recon[:len(h_true_real)]

        loss_i = compute_fd_loss_single(h_pred, h_true_real, dt)
        per_sample.append(loss_i)

    return float(np.mean(per_sample))


# ── Haiku 4.5 ─────────────────────────────────────────────

def recompute_haiku(val_sims):
    print("\n" + "=" * 60)
    print("Haiku 4.5 — waveform (naive FFT → PyCBC, all 250)")
    print("=" * 60)
    agent_dir = ROOT / "llm_agents" / "results" / "haiku" / "waveform"
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
        saved = mdir / "saved_model"

        try:
            # Format 1: NN1 style (separate pkl files)
            if (saved / "svd_basis.pkl").exists():
                with open(saved / "svd_basis.pkl", "rb") as f:
                    U = pickle.load(f)
                with open(saved / "scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                with open(saved / "gpr_models.pkl", "rb") as f:
                    gpr_models = pickle.load(f)

                per_sample = []
                for sim in val_sims:
                    q = sim['q']
                    chi1 = [sim['chi1x'], sim['chi1y'], sim['chi1z']]
                    chi2 = [sim['chi2x'], sim['chi2y'], sim['chi2z']]
                    x = np.array([q] + chi1 + chi2).reshape(1, -1)
                    x_sc = scaler.transform(x)
                    coeffs = np.array([gpr.predict(x_sc)[0] for gpr in gpr_models])
                    h_recon = np.dot(U, coeffs)

                    h_true_real = np.real(sim['h22'])
                    dt = sim['t'][1] - sim['t'][0]
                    if len(h_recon) < len(h_true_real):
                        h_pred = np.zeros(len(h_true_real))
                        h_pred[:len(h_recon)] = h_recon
                    else:
                        h_pred = h_recon[:len(h_true_real)]
                    per_sample.append(compute_fd_loss_single(h_pred, h_true_real, dt))

                loss = float(np.mean(per_sample))

            # Format 2: model.pkl with models, U, scaler
            elif (saved / "model.pkl").exists():
                with open(saved / "model.pkl", "rb") as f:
                    model = pickle.load(f)
                if isinstance(model, dict) and 'U' in model and 'models' in model:
                    loss = _recompute_haiku_style_model(model, sc, val_sims, mdir.name)
                else:
                    raise ValueError(f"Unrecognized model.pkl format: {list(model.keys()) if isinstance(model, dict) else type(model)}")
            else:
                print(f"  {mdir.name}: skipping (no loadable model)")
                summaries.append(sc)
                continue

            sc["loss"] = loss
            sc["loss_old_naive_fft"] = old_loss
            sc["_recomputed"] = True
            sc_path.write_text(json.dumps({k: v for k, v in sc.items()
                                            if k != "_recomputed"}, indent=2))
            summaries.append(sc)
            print(f"  {mdir.name}: {old_loss:.4e} -> {loss:.4e}")
        except Exception as e:
            print(f"  {mdir.name}: FAILED ({e})")
            sc["_recomputed"] = False
            summaries.append(sc)

    _write_best(agent_dir, summaries, set())


# ── GPT-5.4 Mini ──────────────────────────────────────────

def recompute_gpt54_mini(val_sims):
    print("\n" + "=" * 60)
    print("GPT-5.4 Mini — waveform (64 → all 250)")
    print("=" * 60)
    agent_dir = ROOT / "llm_agents" / "results" / "gpt54_mini" / "waveform"
    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location("gpt54_lib", agent_dir / "_lib.py")
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)

    # Use _lib's own loader which returns sim dicts
    val_sims_lib = lib._load_split("validation")
    n_val = len(val_sims_lib)
    dt = lib.DT_GEOMETRIC
    print(f"  {n_val} validation cases, dt={dt}")

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

            # Predict on all validation sims (per-sample, varying lengths)
            t0 = time.time()
            per_mass_accum = {f"mismatch_{int(m)}Msun": [] for m in FD_MASSES_MSUN}
            per_sample = []
            for sim in val_sims_lib:
                h_pred = lib.predict_from_model(model, sim)
                h_true = sim["h"]
                loss_i = compute_fd_loss_single(h_pred, h_true, dt)
                per_sample.append(loss_i)
                for m in FD_MASSES_MSUN:
                    try:
                        mm = frequency_domain_mismatch(h_pred, h_true,
                                                        dt_geometric=dt, mtot_msun=m)
                    except Exception:
                        mm = 1.0
                    per_mass_accum[f"mismatch_{int(m)}Msun"].append(mm)
            per_mass_means = {k: float(np.mean(v)) for k, v in per_mass_accum.items()}
            loss = float(np.mean(list(per_mass_means.values())))
            per_mass_means["mean_fd_mismatch"] = loss
            components = per_mass_means
            elapsed = time.time() - t0

            sc["loss"] = loss
            sc["loss_components"] = components
            sc["loss_old_subset64"] = old_loss
            sc["_recomputed"] = True
            sc_path.write_text(json.dumps({k: v for k, v in sc.items()
                                            if k != "_recomputed"}, indent=2))
            summaries.append(sc)
            print(f"  {mdir.name}: {old_loss:.4e} -> {loss:.4e}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {mdir.name}: FAILED ({e})")
            traceback.print_exc()
            sc["_recomputed"] = False
            summaries.append(sc)

    _write_best(agent_dir, summaries, set())
    sys.path.remove(str(agent_dir))


def main():
    print("Loading validation data...")
    val_sims = load_waveform_validation()
    print(f"  {len(val_sims)} validation cases")

    print("Loading training data (for Sonnet grid)...")
    train_sims = load_waveform_training()
    print(f"  {len(train_sims)} training cases")

    recompute_opus47(val_sims)
    recompute_sonnet46(val_sims, train_sims)
    recompute_haiku(val_sims)
    recompute_gpt54_mini(val_sims)

    print("\n\nDone. Run 'python docs/update_results.py' to refresh the website.")


if __name__ == "__main__":
    main()
