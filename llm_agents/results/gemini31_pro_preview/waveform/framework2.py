import os
import sys
import h5py
import numpy as np
import json
import time
import shutil
import pickle
from pathlib import Path
from typing import Dict, Any

from gwbenchmarks.metrics import mean_fd_mismatch, FD_MASSES_MSUN
from gwbenchmarks.plot_settings import setup_plots
import matplotlib.pyplot as plt

# Try importing pycbc, if not mock frequency_domain_mismatch for speed / test?
# Actually, metric.py has frequency_domain_mismatch. Let's just use it.
from gwbenchmarks.metrics import frequency_domain_mismatch

# Setup Paths
BASE_DIR = Path("llm_agents/results/gemini31_pro_preview/waveform")
MODELS_DIR = BASE_DIR / "models"
COMP_DIR = BASE_DIR / "comparison"
DATA_DIR = Path("datasets/waveform")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
COMP_DIR.mkdir(parents=True, exist_ok=True)
setup_plots()

# Data Loaders
def load_data(split="training"):
    path = DATA_DIR / f"waveform_{split}.h5"
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

# Helpers
def reparametrize(X, kind="raw_7d"):
    X_new = []
    for params in X:
        q, c1x, c1y, c1z, c2x, c2y, c2z = params
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        eta = q / (1 + q)**2
        c1 = np.linalg.norm([c1x, c1y, c1z])
        c2 = np.linalg.norm([c2x, c2y, c2z])
        th1 = np.arccos(c1z / c1) if c1 > 1e-8 else 0
        th2 = np.arccos(c2z / c2) if c2 > 1e-8 else 0
        ph1 = np.arctan2(c1y, c1x)
        ph2 = np.arctan2(c2y, c2x)
        chi_eff = (m1 * c1z + m2 * c2z) / (m1 + m2)
        c1p = np.linalg.norm([c1x, c1y])
        c2p = np.linalg.norm([c2x, c2y])
        S1p = m1**2 * c1p
        S2p = m2**2 * c2p
        Sp = max(S1p, (4*m2 + 3*m1)/(4*m1 + 3*m2)*S2p)
        chi_p = Sp / (m1**2 * max(1e-8, q * (4*m2 + 3*m1)/(4*m1 + 3*m2)))
        
        if kind == "raw_7d":
            X_new.append(params)
        elif kind == "eta_chi_eff_chi_p":
            X_new.append([eta, chi_eff, chi_p, c1, c2, th1, th2])
        elif kind == "spherical":
            X_new.append([eta, c1, th1, ph1, c2, th2, ph2])
        elif kind == "mass_diff":
            dm = m1 - m2
            X_new.append([dm, chi_eff, chi_p, c1, c2, ph1, ph2])
    return np.array(X_new)

def apply_time_conv(t, y, conv):
    if conv == "t0_at_peak":
        return t
    elif conv == "t0_at_start":
        return t - t[0]
    elif conv == "reversed":
        return t[-1] - t
    return t

# Create train.py and predict.py templates
TRAIN_TEMPLATE = """\
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
{extra_imports}

def train_model(X_train, y_train, savedir):
    # Perform PCA on real and imag parts separately
    # Or as complex SVD. Let's do PCA on concatenated real/imag
    y_stacked = np.hstack([np.real(y_train), np.imag(y_train)])
    pca = PCA(n_components={n_pca})
    y_pca = pca.fit_transform(y_stacked)
    
    # Model
    model = {model_init}
    model.fit(X_train, y_pca)
    
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(savedir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
        
    # Additional PySR logic if needed
    {pysr_logic}

if __name__ == "__main__":
    # In practice, called from master script
    pass
"""

PREDICT_TEMPLATE = """\
import os
import pickle
import numpy as np

def load_model(savedir):
    with open(os.path.join(savedir, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(savedir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return pca, model

def predict(X, savedir):
    pca, model = load_model(savedir)
    y_pca = model.predict(X)
    y_stacked = pca.inverse_transform(y_pca)
    n = y_stacked.shape[1] // 2
    return y_stacked[:, :n] + 1j * y_stacked[:, n:]
"""
