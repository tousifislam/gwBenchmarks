#!/usr/bin/env python3
"""Recompute per-sample PyCBC FD mismatch for best models missing this data.

Waveform benchmark (6 agents): Opus 4.7, Sonnet 4.6, Haiku, GPT-5.5 High,
                                GPT-5.4 Mini, GPT-5.3 Codex High
Analytic benchmark (3 agents):  GPT-5.5 High, GPT-5.4 Mini, GPT-5.3 Codex High

Saves per_sample_loss (list of floats) into each best_model.json.

Usage:
    envs/gwbench/bin/python llm_agents/recompute_per_sample.py
"""

import importlib.util
import json
import pickle
import sys
import traceback
from pathlib import Path

import h5py
import joblib
import numpy as np
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwbenchmarks.metrics import frequency_domain_mismatch, FD_MASSES_MSUN


# ── Shared helpers ──────────────────────────────────────────

def per_sample_fd_mismatch(h_pred, h_true, dt):
    """Mean FD mismatch over 5 masses for a single waveform pair."""
    vals = []
    for m in FD_MASSES_MSUN:
        try:
            mm = frequency_domain_mismatch(h_pred, h_true, dt_geometric=dt, mtot_msun=m)
        except Exception:
            mm = 1.0
        vals.append(mm)
    return float(np.mean(vals))


def save_per_sample(agent, bench, per_sample_losses):
    """Write per_sample_loss into best_model.json."""
    bm_path = ROOT / "llm_agents" / "results" / agent / bench / "comparison" / "best_model.json"
    bm = json.loads(bm_path.read_text())
    bm["per_sample_loss"] = [float(v) for v in per_sample_losses]
    bm_path.write_text(json.dumps(bm, indent=2))
    arr = np.array(per_sample_losses)
    print(f"  Saved {len(arr)} per-sample losses: "
          f"mean={np.mean(arr):.4e}, median={np.median(arr):.4e}, "
          f"p5={np.percentile(arr, 5):.4e}, p95={np.percentile(arr, 95):.4e}")


# ── Waveform data loaders ──────────────────────────────────

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


def load_waveform_grid(split, n_grid=128):
    """Load waveform data on the 128-point grid used by GPT-5.5/GPT-5.3 suite_runner."""
    path = ROOT / f"datasets/waveform/waveform_{split}.h5"
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        names = [f"sim_{i:04d}" for i in range(n)]
        tmin = max(float(np.min(f[k]["t"][:])) for k in names)
        tmax = min(float(np.max(f[k]["t"][:])) for k in names)
        grid = np.linspace(max(tmin, -1500), min(tmax, 120), n_grid)
        dt = float(grid[1] - grid[0])
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


# ── Analytic data loader ───────────────────────────────────

def load_analytic_grid(split, n_grid=160):
    """Load analytic data on the 160-point grid used by GPT-5.5/GPT-5.3 suite_runner."""
    path = ROOT / f"datasets/analytic/analytic_{split}.h5"
    with h5py.File(path, "r") as f:
        groups = list(f["sims"].keys())
        tmin = max(float(np.min(f["sims"][k]["t"][:])) for k in groups)
        tmax = min(float(np.max(f["sims"][k]["t"][:])) for k in groups)
        grid = np.linspace(max(tmin, -1200), min(tmax, 120), n_grid)
        dt = float(grid[1] - grid[0])
        X, Y = [], []
        for k in groups:
            g = f["sims"][k]
            X.append([g.attrs["q"]])
            re = np.interp(grid, g["t"][:], g["h22_real"][:])
            im = np.interp(grid, g["t"][:], g["h22_imag"][:])
            Y.append(np.r_[re, im])
    return np.asarray(X), np.asarray(Y), dt


def load_analytic_validation_raw():
    """Load analytic validation data at original resolution (for native-metric agents)."""
    path = ROOT / "datasets" / "analytic" / "analytic_validation.h5"
    sims = []
    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt_geometric"])
        for k in sorted(f["sims"].keys()):
            g = f["sims"][k]
            sims.append({
                'q': float(g.attrs["q"]),
                'h22': g["h22_real"][:] + 1j * g["h22_imag"][:],
                't': g["t"][:].astype(np.float64),
            })
    return sims, dt


# ── Suite-runner parameterizations (GPT-5.5 / GPT-5.3) ────

def suite_runner_param(X, bench, param):
    X = np.asarray(X, dtype=np.float64)
    if bench == "analytic":
        q = X[:, 0]
        eta = lambda q_: q_ / (1 + q_) ** 2
        if param == "q": return q[:, None]
        if param == "eta": return eta(q)[:, None]
        if param == "delta_m": return ((q - 1) / (q + 1))[:, None]
        if param == "sqrt_eta": return np.sqrt(eta(q))[:, None]
        if param == "eta_power": return np.column_stack([eta(q), eta(q) ** 0.2])
        return X
    # waveform params
    q = X[:, 0]
    if param == "raw_7d": return X[:, :7]
    if param == "raw_6d": return X[:, :6]
    if param == "effective_spins":
        eta = q / (1 + q) ** 2
        chi_eff = (q * X[:, 3] + X[:, 6]) / (1 + q)
        chi_p = np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2)
        return np.column_stack([eta, chi_eff, chi_p])
    if param == "massdiff_spins":
        dm = (q - 1) / (q + 1)
        chi1_mag = np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2)
        chi2_mag = np.sqrt(X[:, 4] ** 2 + X[:, 5] ** 2 + X[:, 6] ** 2)
        return np.column_stack([dm, chi1_mag, X[:, 3], chi2_mag, X[:, 6]])
    if param == "spherical_spins":
        eta = q / (1 + q) ** 2
        chi1_mag = np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2)
        chi1_theta = np.arccos(X[:, 3] / np.maximum(chi1_mag, 1e-12))
        chi2_mag = np.sqrt(X[:, 4] ** 2 + X[:, 5] ** 2 + X[:, 6] ** 2)
        chi2_theta = np.arccos(X[:, 6] / np.maximum(chi2_mag, 1e-12))
        return np.column_stack([eta, chi1_mag, chi1_theta, chi2_mag, chi2_theta])
    if param == "with_omega0": return X[:, :8]
    return X[:, :7]


def y_to_complex(y):
    n = y.shape[1] // 2
    return y[:, :n] + 1j * y[:, n:]


# ── Sonnet 4.6 helpers (from v3 script) ───────────────────

def _sonnet_reparam(p, name):
    q, c1x, c1y, c1z, c2x, c2y, c2z = p
    if name == 'raw':
        return np.array(p, dtype=float)
    m1, m2 = q / (1 + q), 1 / (1 + q)
    eta = q / (1 + q) ** 2
    chi_eff = m1 * c1z + m2 * c2z
    chi1m = np.sqrt(c1x ** 2 + c1y ** 2 + c1z ** 2)
    chi2m = np.sqrt(c2x ** 2 + c2y ** 2 + c2z ** 2)
    chi1p = np.sqrt(c1x ** 2 + c1y ** 2)
    chi2p = np.sqrt(c2x ** 2 + c2y ** 2)
    if name == 'eff':
        chi_p = max(chi1p, (4 * m2 + 3 * m1) / (4 * m1 + 3 * m2) * (m2 / m1) * chi2p)
        th1 = np.arctan2(chi1p, c1z) if chi1m > 1e-10 else 0.
        th2 = np.arctan2(chi2p, c2z) if chi2m > 1e-10 else 0.
        return np.array([eta, chi_eff, chi_p, chi1m, chi2m, th1, th2])
    if name == 'sph':
        th1 = np.arccos(np.clip(c1z / max(chi1m, 1e-10), -1, 1))
        th2 = np.arccos(np.clip(c2z / max(chi2m, 1e-10), -1, 1))
        ph1 = np.arctan2(c1y, c1x)
        ph2 = np.arctan2(c2y, c2x)
        return np.array([eta, chi1m, th1, ph1, chi2m, th2, ph2])
    if name == 'md':
        dm = m1 - m2
        chi_p = max(chi1p, (4 * m2 + 3 * m1) / (4 * m1 + 3 * m2) * (m2 / m1) * chi2p)
        ph1 = np.arctan2(c1y, c1x)
        ph2 = np.arctan2(c2y, c2x)
        return np.array([dm, chi_eff, chi_p, chi1m, chi2m, ph1, ph2])
    raise ValueError(f"Unknown reparam: {name}")


def _sonnet_decode(c_ri, basis):
    n = len(c_ri) // 2
    c_cplx = c_ri[:n] + 1j * c_ri[n:]
    return c_cplx @ basis[:n]


def _sonnet_predict_one(model, x_1d, basis):
    x_2d = x_1d.reshape(1, -1)
    keys = set(model.keys())
    if 'gpr' in keys and 'basis' in keys and 'nodes' not in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        return _sonnet_decode(model['gpr'].predict(x_sc).flatten(), basis)
    if 'rf' in keys:
        return _sonnet_decode(model['rf'].predict(x_2d).flatten(), basis)
    if 'mlp' in keys and 'basis' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        return _sonnet_decode(model['mlp'].predict(x_sc).flatten(), basis)
    if 'gbr' in keys:
        return _sonnet_decode(model['gbr'].predict(x_2d).flatten(), basis)
    if 'krr' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        return _sonnet_decode(model['krr'].predict(x_sc).flatten(), basis)
    if 'knn' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        return _sonnet_decode(model['knn'].predict(x_sc).flatten(), basis)
    if 'et' in keys:
        return _sonnet_decode(model['et'].predict(x_2d).flatten(), basis)
    if 'poly' in keys and ('r' in keys or 'ridge' in keys):
        sc = model.get('sc', model.get('scaler'))
        ridge = model.get('r', model.get('ridge'))
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        x_poly = model['poly'].transform(x_sc)
        return _sonnet_decode(ridge.predict(x_poly).flatten(), basis)
    if 'rbf' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        return _sonnet_decode(model['rbf'].predict(x_sc).flatten(), basis)
    if ('g_a' in keys or 'ga' in keys) and ('g_p' in keys or 'gp' in keys):
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        g_a = model.get('g_a', model.get('ga'))
        g_p = model.get('g_p', model.get('gp'))
        ca = g_a.predict(x_sc).flatten()
        cp = g_p.predict(x_sc).flatten()
        amp = ca @ model['Ba'][:len(ca)]
        phase = cp @ model['Bp'][:len(cp)]
        return np.abs(amp) * np.exp(1j * phase)
    if 'gpr' in keys and 'nodes' in keys and 'V' in keys:
        sc = model.get('sc')
        x_sc = sc.transform(x_2d) if sc is not None else x_2d
        return _sonnet_decode(model['gpr'].predict(x_sc).flatten(), basis)
    if 'gpl_r' in keys and 'gpl_i' in keys:
        cr = model['gpl_r'].predict(x_2d).flatten()
        ci = model['gpl_i'].predict(x_2d).flatten()
        return _sonnet_decode(np.concatenate([cr, ci]), basis)
    raise ValueError(f"Unknown model format: {list(keys)}")


# ── Haiku helpers ──────────────────────────────────────────

def _haiku_reparam(q, chi1, chi2, func_name):
    if func_name in ('reparameterize_raw', 'raw'):
        return np.array([q] + list(chi1) + list(chi2))
    if func_name in ('reparameterize_eta_chi', 'eta_chi'):
        eta = q / (1 + q) ** 2
        chi1_mag = np.linalg.norm(chi1)
        chi2_mag = np.linalg.norm(chi2)
        chi_eff = (chi1[2] + q * chi2[2]) / (1 + q)
        chi_p = max(np.sqrt(chi1[0] ** 2 + chi1[1] ** 2),
                    (4 * q) / (3 * (1 + q) ** 2) * np.sqrt(chi2[0] ** 2 + chi2[1] ** 2))
        theta1 = np.arccos(chi1[2] / (chi1_mag + 1e-10)) if chi1_mag > 0 else 0
        theta2 = np.arccos(chi2[2] / (chi2_mag + 1e-10)) if chi2_mag > 0 else 0
        return np.array([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])
    if func_name in ('reparameterize_spherical', 'spherical'):
        eta = q / (1 + q) ** 2
        chi1_mag = np.linalg.norm(chi1)
        chi2_mag = np.linalg.norm(chi2)
        theta1 = np.arccos(np.clip(chi1[2] / max(chi1_mag, 1e-10), -1, 1))
        theta2 = np.arccos(np.clip(chi2[2] / max(chi2_mag, 1e-10), -1, 1))
        phi1 = np.arctan2(chi1[1], chi1[0])
        phi2 = np.arctan2(chi2[1], chi2[0])
        return np.array([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])
    raise ValueError(f"Unknown param_func: {func_name}")


def _haiku_predict_one(model_dict, sim):
    """Predict one waveform from a Haiku-style dict model."""
    U = model_dict['U']
    scaler = model_dict['scaler']
    models_list = model_dict['models']
    param_func = model_dict.get('param_func', 'reparameterize_raw')
    q = sim['q']
    chi1 = [sim['chi1x'], sim['chi1y'], sim['chi1z']]
    chi2 = [sim['chi2x'], sim['chi2y'], sim['chi2z']]
    x = _haiku_reparam(q, chi1, chi2, param_func).reshape(1, -1)
    x_sc = scaler.transform(x)
    coeffs = np.array([m.predict(x_sc)[0] for m in models_list])
    return np.dot(U, coeffs)


# ════════════════════════════════════════════════════════════
#  WAVEFORM BENCHMARK — per-sample recomputation
# ════════════════════════════════════════════════════════════

def waveform_opus47(val_sims):
    print("\n--- Opus 4.7 waveform ---")
    agent_dir = ROOT / "llm_agents" / "results" / "opus47" / "waveform"
    bm = json.loads((agent_dir / "comparison" / "best_model.json").read_text())
    model_name = bm["approach"]
    mdir = None
    for d in (agent_dir / "models").iterdir():
        if d.is_dir() and model_name in d.name:
            mdir = d
            break
    if mdir is None:
        print(f"  ERROR: cannot find model dir for {model_name}")
        return

    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location("opus47_lib", agent_dir / "_lib.py")
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)

    params_val, h_val, _, omega0_val = lib.load_split("validation")
    dt = lib.T_GRID_DT

    pred_spec = importlib.util.spec_from_file_location(
        f"opus47_pred_{mdir.name}", mdir / "predict.py")
    pred_mod = importlib.util.module_from_spec(pred_spec)
    pred_spec.loader.exec_module(pred_mod)

    param_mode = bm.get("parameterization", "raw7")
    X = lib.reparam(params_val, param_mode, omega0=omega0_val)
    h_pred = pred_mod.predict(X)
    if not np.iscomplexobj(h_pred):
        n_t = h_pred.shape[1] // 2
        h_pred = h_pred[:, :n_t] + 1j * h_pred[:, n_t:]

    per_sample = []
    for i in range(len(h_val)):
        per_sample.append(per_sample_fd_mismatch(h_pred[i], h_val[i], dt))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(h_val)} done")
    save_per_sample("opus47", "waveform", per_sample)
    sys.path.remove(str(agent_dir))


def waveform_sonnet46(val_sims, train_sims):
    print("\n--- Sonnet 4.6 waveform ---")
    agent_dir = ROOT / "llm_agents" / "results" / "sonnet46" / "waveform"
    bm = json.loads((agent_dir / "comparison" / "best_model.json").read_text())
    model_name = bm.get("approach", bm.get("name", ""))
    print(f"  Best model: {model_name}")

    # Find model dir
    mdir = None
    for d in sorted((agent_dir / "models").iterdir()):
        if not d.is_dir():
            continue
        sc_p = d / "scorecard.json"
        if sc_p.exists():
            sc = json.loads(sc_p.read_text())
            if sc.get("approach") == model_name or sc.get("name") == model_name:
                mdir = d
                break
    if mdir is None:
        print(f"  ERROR: cannot find model dir for {model_name}")
        return

    # Derive Sonnet's T_GRID
    t_max_start_tr = max(s['t'][0] for s in train_sims)
    t_max_start_va = max(s['t'][0] for s in val_sims)
    t_common_start = max(t_max_start_tr, t_max_start_va)
    T_GRID = np.linspace(t_common_start, 100.0, 2048)
    dt = float(T_GRID[1] - T_GRID[0])

    # Interpolate validation to T_GRID
    H_val = []
    for s in val_sims:
        fr = interp1d(s['t'], s['h22'].real, bounds_error=False, fill_value=0.0)
        fi = interp1d(s['t'], s['h22'].imag, bounds_error=False, fill_value=0.0)
        H_val.append(fr(T_GRID) + 1j * fi(T_GRID))
    H_val = np.array(H_val)

    P_val = np.array([[s['q'], s['chi1x'], s['chi1y'], s['chi1z'],
                        s['chi2x'], s['chi2y'], s['chi2z']] for s in val_sims])

    model = joblib.load(mdir / "saved_model" / "model.pkl")
    basis = model.get('basis')
    reparam_name = bm.get('parameterization', 'raw')
    if reparam_name in ('t0_start', 't0_at_peak'):
        reparam_name = 'raw'
    Xv = np.array([_sonnet_reparam(p, reparam_name) for p in P_val])

    per_sample = []
    for i in range(len(H_val)):
        h_pred = _sonnet_predict_one(model, Xv[i], basis)
        per_sample.append(per_sample_fd_mismatch(h_pred, H_val[i], dt))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(H_val)} done")
    save_per_sample("sonnet46", "waveform", per_sample)


def waveform_haiku(val_sims):
    print("\n--- Haiku waveform ---")
    agent_dir = ROOT / "llm_agents" / "results" / "haiku" / "waveform"
    bm = json.loads((agent_dir / "comparison" / "best_model.json").read_text())
    model_name = bm.get("approach", bm.get("name", ""))
    print(f"  Best model: {model_name}")

    # Find model dir
    mdir = None
    for d in sorted((agent_dir / "models").iterdir()):
        if not d.is_dir():
            continue
        sc_p = d / "scorecard.json"
        if sc_p.exists():
            sc = json.loads(sc_p.read_text())
            if sc.get("approach") == model_name:
                mdir = d
                break
    if mdir is None:
        print(f"  ERROR: cannot find model dir for {model_name}")
        return

    saved = mdir / "saved_model"
    if (saved / "svd_basis.pkl").exists():
        with open(saved / "svd_basis.pkl", "rb") as f:
            U = pickle.load(f)
        with open(saved / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(saved / "gpr_models.pkl", "rb") as f:
            gpr_models = pickle.load(f)
        model_dict = {'U': U, 'scaler': scaler, 'models': gpr_models,
                       'param_func': 'reparameterize_raw'}
    elif (saved / "model.pkl").exists():
        with open(saved / "model.pkl", "rb") as f:
            model_dict = pickle.load(f)
    else:
        print("  ERROR: no loadable model")
        return

    per_sample = []
    for sim in val_sims:
        h_recon = _haiku_predict_one(model_dict, sim)
        h_true_real = np.real(sim['h22'])
        dt = sim['t'][1] - sim['t'][0]
        if len(h_recon) < len(h_true_real):
            h_pred = np.zeros(len(h_true_real))
            h_pred[:len(h_recon)] = h_recon
        else:
            h_pred = h_recon[:len(h_true_real)]
        per_sample.append(per_sample_fd_mismatch(h_pred, h_true_real, dt))
        if (len(per_sample)) % 50 == 0:
            print(f"    {len(per_sample)}/{len(val_sims)} done")
    save_per_sample("haiku", "waveform", per_sample)


def waveform_suite_runner(agent, X_val_raw, Y_val, dt):
    """GPT-5.5 High or GPT-5.3 Codex High waveform (suite_runner format)."""
    print(f"\n--- {agent} waveform ---")
    agent_dir = ROOT / "llm_agents" / "results" / agent / "waveform"
    bm = json.loads((agent_dir / "comparison" / "best_model.json").read_text())
    model_name = bm["approach"]
    print(f"  Best model: {model_name}")

    # Find model dir by approach number or name
    mdir = None
    for d in sorted((agent_dir / "models").iterdir()):
        if not d.is_dir():
            continue
        sc_p = d / "scorecard.json"
        if sc_p.exists():
            sc = json.loads(sc_p.read_text())
            if sc.get("approach") == model_name:
                mdir = d
                break
    if mdir is None:
        print(f"  ERROR: cannot find model dir for {model_name}")
        return

    model = joblib.load(mdir / "saved_model" / "model.joblib")
    param = bm.get("parameterization", "raw_7d")
    Xv = suite_runner_param(X_val_raw, "waveform", param)
    y_pred = model.predict(Xv)
    h_pred = y_to_complex(y_pred)
    h_true = y_to_complex(Y_val)

    per_sample = []
    for i in range(len(h_pred)):
        per_sample.append(per_sample_fd_mismatch(h_pred[i], h_true[i], dt))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(h_pred)} done")
    save_per_sample(agent, "waveform", per_sample)


def waveform_gpt54_mini(val_sims):
    print("\n--- GPT-5.4 Mini waveform ---")
    agent_dir = ROOT / "llm_agents" / "results" / "gpt54_mini" / "waveform"
    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location("gpt54_lib", agent_dir / "_lib.py")
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)

    val_sims_lib = lib._load_split("validation")
    dt = lib.DT_GEOMETRIC

    bm = json.loads((agent_dir / "comparison" / "best_model.json").read_text())
    model_name = bm["approach"]
    print(f"  Best model: {model_name}")

    # Find model dir
    mdir = None
    for d in sorted((agent_dir / "models").iterdir()):
        if not d.is_dir():
            continue
        sc_p = d / "scorecard.json"
        if sc_p.exists():
            sc = json.loads(sc_p.read_text())
            if sc.get("approach") == model_name:
                mdir = d
                break
    if mdir is None:
        print(f"  ERROR: cannot find model dir for {model_name}")
        return

    with open(mdir / "saved_model" / "model.pkl", "rb") as f:
        model = pickle.load(f)

    per_sample = []
    for sim in val_sims_lib:
        h_pred = lib.predict_from_model(model, sim)
        h_true = sim["h"]
        per_sample.append(per_sample_fd_mismatch(h_pred, h_true, dt))
        if len(per_sample) % 50 == 0:
            print(f"    {len(per_sample)}/{len(val_sims_lib)} done")
    save_per_sample("gpt54_mini", "waveform", per_sample)
    sys.path.remove(str(agent_dir))


# ════════════════════════════════════════════════════════════
#  ANALYTIC BENCHMARK — per-sample recomputation
# ════════════════════════════════════════════════════════════

def analytic_suite_runner(agent, X_val_raw, Y_val, dt):
    """GPT-5.5 High or GPT-5.3 Codex High analytic (suite_runner format)."""
    print(f"\n--- {agent} analytic ---")
    agent_dir = ROOT / "llm_agents" / "results" / agent / "analytic"
    bm = json.loads((agent_dir / "comparison" / "best_model.json").read_text())
    model_name = bm["approach"]
    print(f"  Best model: {model_name}")

    mdir = None
    for d in sorted((agent_dir / "models").iterdir()):
        if not d.is_dir():
            continue
        sc_p = d / "scorecard.json"
        if sc_p.exists():
            sc = json.loads(sc_p.read_text())
            if sc.get("approach") == model_name:
                mdir = d
                break
    if mdir is None:
        print(f"  ERROR: cannot find model dir for {model_name}")
        return

    model = joblib.load(mdir / "saved_model" / "model.joblib")
    param = bm.get("parameterization", "q")
    Xv = suite_runner_param(X_val_raw, "analytic", param)
    y_pred = model.predict(Xv)
    h_pred = y_to_complex(y_pred)
    h_true = y_to_complex(Y_val)

    per_sample = []
    for i in range(len(h_pred)):
        per_sample.append(per_sample_fd_mismatch(h_pred[i], h_true[i], dt))
    save_per_sample(agent, "analytic", per_sample)


def analytic_gpt54_mini():
    """GPT-5.4 Mini analytic — predict(q, t) returns complex h22."""
    print("\n--- GPT-5.4 Mini analytic ---")
    agent_dir = ROOT / "llm_agents" / "results" / "gpt54_mini" / "analytic"
    bm = json.loads((agent_dir / "comparison" / "best_model.json").read_text())
    model_name = bm["approach"]
    print(f"  Best model: {model_name}")

    mdir = None
    for d in sorted((agent_dir / "models").iterdir()):
        if not d.is_dir():
            continue
        sc_p = d / "scorecard.json"
        if sc_p.exists():
            sc = json.loads(sc_p.read_text())
            if sc.get("approach") == model_name:
                mdir = d
                break
    if mdir is None:
        print(f"  ERROR: cannot find model dir for {model_name}")
        return

    # Import _lib from the agent's analytic directory
    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location(
        "gpt54_analytic_lib", agent_dir / "_lib.py")
    lib_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib_mod)

    # Load the model
    with open(mdir / "saved_model" / "model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load validation data at original resolution
    val_sims, dt = load_analytic_validation_raw()
    print(f"  {len(val_sims)} validation sims, dt={dt}")

    eval_fn = (lib_mod.evaluate_linear_model if model.get("type") == "linear_basis"
               else lib_mod.evaluate_curve_basis_model)

    per_sample = []
    for sim in val_sims:
        q = sim['q']
        t = sim['t']
        h_true = sim['h22']
        try:
            h_pred = eval_fn(model, q, t)
            per_sample.append(per_sample_fd_mismatch(h_pred, h_true, dt))
        except Exception as e:
            print(f"  Warning: sample q={q:.2f} failed ({e}), using 1.0")
            per_sample.append(1.0)
    save_per_sample("gpt54_mini", "analytic", per_sample)
    sys.path.remove(str(agent_dir))


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Per-sample PyCBC FD mismatch recomputation")
    print("=" * 60)

    # ── Load data ──
    print("\nLoading waveform validation data...")
    val_sims = load_waveform_validation()
    print(f"  {len(val_sims)} waveform validation cases")

    print("Loading waveform training data (for Sonnet grid)...")
    train_sims = load_waveform_training()
    print(f"  {len(train_sims)} waveform training cases")

    print("Loading waveform grid data (for GPT-5.5/5.3)...")
    X_wf, Y_wf, dt_wf = load_waveform_grid("validation")
    print(f"  {len(X_wf)} cases on 128-pt grid, dt={dt_wf:.4f}")

    print("Loading analytic grid data (for GPT-5.5/5.3)...")
    X_an, Y_an, dt_an = load_analytic_grid("validation")
    print(f"  {len(X_an)} cases on 160-pt grid, dt={dt_an:.4f}")

    # ── Waveform benchmark ──
    print("\n" + "=" * 60)
    print("WAVEFORM BENCHMARK — 6 agents")
    print("=" * 60)

    try:
        waveform_opus47(val_sims)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    try:
        waveform_sonnet46(val_sims, train_sims)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    try:
        waveform_haiku(val_sims)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    try:
        waveform_suite_runner("gpt55_high", X_wf, Y_wf, dt_wf)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    try:
        waveform_gpt54_mini(val_sims)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    try:
        waveform_suite_runner("gpt53_codex_high", X_wf, Y_wf, dt_wf)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    # ── Analytic benchmark ──
    print("\n" + "=" * 60)
    print("ANALYTIC BENCHMARK — 3 agents")
    print("=" * 60)

    try:
        analytic_suite_runner("gpt55_high", X_an, Y_an, dt_an)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    try:
        analytic_suite_runner("gpt53_codex_high", X_an, Y_an, dt_an)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    try:
        analytic_gpt54_mini()
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
