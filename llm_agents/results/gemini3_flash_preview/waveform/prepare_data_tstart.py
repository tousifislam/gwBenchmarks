import h5py
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib
import os

def load_data_tstart(file_path, common_t=None):
    with h5py.File(file_path, "r") as f:
        n = f.attrs["n_simulations"]
        params = []
        waveforms = []
        
        lengths = []
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            lengths.append(len(g["t"]))
        max_len = max(lengths)
        dt = 0.1
        
        if common_t is None:
            common_t = np.arange(0, max_len * dt, dt)
        
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            q = g.attrs["q"]
            chi1 = [g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]]
            chi2 = [g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]]
            params.append([q, chi1[0], chi1[1], chi1[2], chi2[0], chi2[1], chi2[2]])
            
            h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
            # Pad at the end
            h_pad = np.zeros(len(common_t), dtype=complex)
            l = min(len(h22), len(common_t))
            h_pad[:l] = h22[:l]
            waveforms.append(h_pad)
            
    return np.array(params), np.array(waveforms), common_t, dt

if __name__ == "__main__":
    # T0 at start
    X_train, y_train, t, dt = load_data_tstart("datasets/waveform/waveform_training.h5")
    X_val, y_val, _, _ = load_data_tstart("datasets/waveform/waveform_validation.h5", common_t=t)
    
    y_train_flat = np.hstack([y_train.real, y_train.imag])
    svd = TruncatedSVD(n_components=50)
    y_train_reduced = svd.fit_transform(y_train_flat)
    
    os.makedirs("llm_agents/results/gemini3_flash_preview/waveform/data_cache", exist_ok=True)
    joblib.dump((X_train, y_train, X_val, y_val, t, dt, svd, y_train_reduced), "llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data_tstart.pkl")
    print("SVD (tstart) completed and data cached.")
