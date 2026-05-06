import h5py
import numpy as np
import json
from pathlib import Path
from scipy.signal import morlet2
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("datasets/waveform")
WORK_DIR = Path("llm_agents/results/haiku/waveform")

class WaveformData:
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.n_train = 0
        self.n_val = 0

    def load_training(self):
        if self.train_data is not None:
            return self.train_data
        data = []
        with h5py.File(DATA_DIR / "waveform_training.h5", "r") as f:
            n = f.attrs["n_simulations"]
            for i in range(n):
                g = f[f"sim_{i:04d}"]
                data.append({
                    'q': g.attrs["q"],
                    'chi1': [g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]],
                    'chi2': [g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]],
                    'omega0': g.attrs["omega0"],
                    't': g["t"][:],
                    'h22': g["h22_real"][:] + 1j * g["h22_imag"][:],
                })
        self.train_data = data
        self.n_train = len(data)
        return data

    def load_validation(self):
        if self.val_data is not None:
            return self.val_data
        data = []
        with h5py.File(DATA_DIR / "waveform_validation.h5", "r") as f:
            n = f.attrs["n_simulations"]
            for i in range(n):
                g = f[f"sim_{i:04d}"]
                data.append({
                    'q': g.attrs["q"],
                    'chi1': [g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]],
                    'chi2': [g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]],
                    'omega0': g.attrs["omega0"],
                    't': g["t"][:],
                    'h22': g["h22_real"][:] + 1j * g["h22_imag"][:],
                })
        self.val_data = data
        self.n_val = len(data)
        return data

def compute_svd_basis(data_list, n_components=15):
    """Compute SVD basis for waveforms."""
    # Find max length
    max_len = max(len(d['h22']) for d in data_list)

    # Pad all waveforms to same length
    wf_list = []
    for d in data_list:
        h22_padded = np.real(d['h22']).copy()
        if len(h22_padded) < max_len:
            padding = np.zeros(max_len - len(h22_padded))
            h22_padded = np.concatenate([h22_padded, padding])
        wf_list.append(h22_padded)

    # Stack all real parts
    wf_matrix = np.column_stack(wf_list)
    U, s, Vt = np.linalg.svd(wf_matrix, full_matrices=False)
    return U[:, :n_components], s[:n_components], Vt[:n_components], max_len

def pad_waveform(h22, target_len):
    """Pad waveform to target length."""
    h22_real = np.real(h22).copy()
    if len(h22_real) < target_len:
        padding = np.zeros(target_len - len(h22_real))
        h22_real = np.concatenate([h22_real, padding])
    return h22_real[:target_len]

def project_onto_basis(h22, basis, max_len):
    """Project a waveform onto the SVD basis."""
    h22_padded = pad_waveform(h22, max_len)
    return np.dot(basis.T, h22_padded)

def reconstruct_from_basis(coeffs, basis):
    """Reconstruct waveform from SVD coefficients."""
    return np.dot(basis, coeffs)

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

def compute_frequency_domain_mismatch(h_true, h_model, delta_t, masses=[40, 80, 120, 160, 200]):
    """Compute frequency-domain mismatch loss."""
    try:
        # FFT both
        fft_true = fft(h_true)
        fft_model = fft(h_model)
        freqs = fftfreq(len(h_true), delta_t)

        # Alignment and mismatch
        norm_true = np.sqrt(np.sum(np.abs(fft_true)**2))
        norm_model = np.sqrt(np.sum(np.abs(fft_model)**2))

        if norm_true == 0 or norm_model == 0:
            return 1.0

        # Compute overlap
        overlap = np.sum(fft_true * np.conj(fft_model)) / (norm_true * norm_model)
        mismatch = 1.0 - np.abs(overlap)
        return float(np.clip(mismatch, 0, 1))
    except:
        return 1.0

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
