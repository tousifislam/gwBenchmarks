import os
import h5py
import numpy as np
import json
import time
import shutil
import matplotlib.pyplot as plt
from gwbenchmarks.metrics import mean_fd_mismatch, FD_MASSES_MSUN
from gwbenchmarks.plot_settings import setup_plots

setup_plots()

DATA_DIR = "datasets/waveform"
MODELS_DIR = "llm_agents/results/gemini31_pro_preview/waveform/models"
PLOT_DIR = "llm_agents/results/gemini31_pro_preview/waveform/comparison"
AGENT_NAME = "gemini31_pro_preview"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data(split="training"):
    path = os.path.join(DATA_DIR, f"waveform_{split}.h5")
    X = []
    y = []
    times = None
    with h5py.File(path, "r") as f:
        n = f.attrs["n_simulations"]
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            params = [
                g.attrs["q"], g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"],
                g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]
            ]
            X.append(params)
            h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
            y.append(h22)
            if times is None:
                times = g["t"][:]
    return np.array(X), np.array(y), times

def dt_geometric(t):
    return t[1] - t[0]

def reparametrize(X, kind="raw_7d"):
    """
    kind: 
    - raw_7d: q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z
    - eta_chi_eff_chi_p: eta, chi_eff, chi_p, |chi1|, |chi2|, theta1, theta2
    - spherical: eta, |chi1|, theta1, phi1, |chi2|, theta2, phi2
    """
    X_new = []
    for params in X:
        q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = params
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        eta = q / (1 + q)**2
        chi1 = np.linalg.norm([chi1x, chi1y, chi1z])
        chi2 = np.linalg.norm([chi2x, chi2y, chi2z])
        theta1 = np.arccos(chi1z / chi1) if chi1 > 0 else 0
        theta2 = np.arccos(chi2z / chi2) if chi2 > 0 else 0
        phi1 = np.arctan2(chi1y, chi1x)
        phi2 = np.arctan2(chi2y, chi2x)
        chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
        chi1_perp = np.linalg.norm([chi1x, chi1y])
        chi2_perp = np.linalg.norm([chi2x, chi2y])
        S1_perp = m1**2 * chi1_perp
        S2_perp = m2**2 * chi2_perp
        Sp = max(S1_perp, (4*m2 + 3*m1)/(4*m1 + 3*m2)*S2_perp)
        chi_p = Sp / (m1**2 * max(1, q * (4*m2 + 3*m1)/(4*m1 + 3*m2)))
        
        if kind == "raw_7d":
            X_new.append(params)
        elif kind == "eta_chi_eff_chi_p":
            X_new.append([eta, chi_eff, chi_p, chi1, chi2, theta1, theta2])
        elif kind == "spherical":
            X_new.append([eta, chi1, theta1, phi1, chi2, theta2, phi2])
        else:
            raise ValueError(f"Unknown reparameterization: {kind}")
    return np.array(X_new)

def apply_time_convention(y, t, convention="t0_at_peak"):
    if convention == "t0_at_peak":
        return y, t # Assuming it's stored this way
    elif convention == "t0_at_start":
        return y, t - t[0]
    elif convention == "reversed":
        return y, t[-1] - t
    else:
        raise ValueError(f"Unknown time convention: {convention}")

def evaluate_model(model_predict, X_val, y_val, dt):
    start = time.time()
    y_pred = model_predict(X_val)
    runtime_ms = (time.time() - start) / len(X_val) * 1000
    
    losses = []
    loss_components = []
    for i in range(len(X_val)):
        from gwbenchmarks.metrics import frequency_domain_mismatch, FD_MASSES_MSUN
        comp = {}
        mismatch_sum = 0
        for m in FD_MASSES_MSUN:
            mm = frequency_domain_mismatch(y_pred[i], y_val[i], dt, m)
            comp[f"mismatch_{int(m)}Msun"] = mm
            mismatch_sum += mm
        mean_mm = mismatch_sum / len(FD_MASSES_MSUN)
        losses.append(mean_mm)
        loss_components.append(comp)
        
    return np.mean(losses), np.array(losses), np.mean([c["mismatch_80Msun"] for c in loss_components]), runtime_ms

def save_scorecard(approach_dir, approach_name, approach_number, param, time_conv, loss, losses_val, losses_train, loss_comp_means, runtime_ms, n_train, n_val, n_params, notes):
    scorecard = {
        "approach": approach_name,
        "approach_number": approach_number,
        "benchmark": "waveform",
        "agent": AGENT_NAME,
        "parameterization": param,
        "time_convention": time_conv,
        "loss": loss,
        "loss_components": loss_comp_means,
        "runtime_ms": runtime_ms,
        "n_train": n_train,
        "n_val": n_val,
        "n_params": n_params,
        "notes": notes
    }
    with open(os.path.join(approach_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
        
    # Also update global progress
    update_plots()

def update_plots():
    pass # To be implemented in another script for aggregation

if __name__ == "__main__":
    X_tr, y_tr, t = load_data("training")
    X_val, y_val, t = load_data("validation")
    print(f"Data loaded: {X_tr.shape}, {y_tr.shape}, {X_val.shape}, {y_val.shape}")
