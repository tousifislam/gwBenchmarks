import h5py
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("datasets/remnant")
WORK_DIR = Path("llm_agents/results/haiku/remnant")

class RemnantData:
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.n_train = 0
        self.n_val = 0

    def load_training(self):
        if self.train_data is not None:
            return self.train_data
        data = []
        with h5py.File(DATA_DIR / "remnant_training.h5", "r") as f:
            n = len(f["q"])
            for i in range(n):
                data.append({
                    'q': f["q"][i],
                    'chi1': [f["chi1x"][i], f["chi1y"][i], f["chi1z"][i]],
                    'chi2': [f["chi2x"][i], f["chi2y"][i], f["chi2z"][i]],
                    'Mf': f["Mf"][i],
                    'chif': f["chif_mag"][i],
                    'vf': f["vf_mag"][i],
                })
        self.train_data = data
        self.n_train = len(data)
        return data

    def load_validation(self):
        if self.val_data is not None:
            return self.val_data
        data = []
        with h5py.File(DATA_DIR / "remnant_validation.h5", "r") as f:
            n = len(f["q"])
            for i in range(n):
                data.append({
                    'q': f["q"][i],
                    'chi1': [f["chi1x"][i], f["chi1y"][i], f["chi1z"][i]],
                    'chi2': [f["chi2x"][i], f["chi2y"][i], f["chi2z"][i]],
                    'Mf': f["Mf"][i],
                    'chif': f["chif_mag"][i],
                    'vf': f["vf_mag"][i],
                })
        self.val_data = data
        self.n_val = len(data)
        return data

def reparameterize_raw(q, chi1, chi2):
    """Raw parameterization: (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)"""
    return np.array([q] + chi1 + chi2)

def reparameterize_eta_chi(q, chi1, chi2):
    """Effective spin parameterization: (eta, chi_eff, chi_p, |chi1|, |chi2|, theta1, theta2)"""
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

def reparameterize_spherical(q, chi1, chi2):
    """Spherical spin parameterization: (eta, |chi1|, theta1, phi1, |chi2|, theta2, phi2)"""
    eta = q / (1 + q)**2
    chi1_mag = np.linalg.norm(chi1)
    chi2_mag = np.linalg.norm(chi2)

    if chi1_mag > 0:
        theta1 = np.arccos(chi1[2] / chi1_mag)
        phi1 = np.arctan2(chi1[1], chi1[0])
    else:
        theta1, phi1 = 0, 0

    if chi2_mag > 0:
        theta2 = np.arccos(chi2[2] / chi2_mag)
        phi2 = np.arctan2(chi2[1], chi2[0])
    else:
        theta2, phi2 = 0, 0

    return np.array([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])

def compute_nrmse(y_true, y_pred):
    """Compute NRMSE: RMSE / range."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    y_range = np.max(y_true) - np.min(y_true)
    return rmse / (y_range + 1e-10) if y_range > 0 else 0

def save_progress(model_name, scorecard):
    """Save model scorecard."""
    score_path = WORK_DIR / "models" / model_name / "scorecard.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with open(score_path, 'w') as f:
        json.dump(scorecard, f, indent=2)

def update_changelog(entry_text):
    """Append to CHANGELOG.md."""
    changelog_path = WORK_DIR / "CHANGELOG.md"
    with open(changelog_path, 'a') as f:
        f.write(entry_text + "\n")
