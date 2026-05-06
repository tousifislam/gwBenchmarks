import h5py
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib
import os

def load_data(file_path, common_t=None):
    with h5py.File(file_path, "r") as f:
        n = f.attrs["n_simulations"]
        params = []
        waveforms = []
        
        if common_t is None:
            t_min = 0
            t_max = -1e10
            dt = 0.1
            for i in range(n):
                g = f[f"sim_{i:04d}"]
                t_arr = g["t"][:]
                t_min = min(t_min, t_arr[0])
                t_max = max(t_max, t_arr[-1])
            common_t = np.arange(t_min, t_max + dt/2, dt)
        
        dt = common_t[1] - common_t[0]
        
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            q = g.attrs["q"]
            chi1 = [g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]]
            chi2 = [g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]]
            params.append([q, chi1[0], chi1[1], chi1[2], chi2[0], chi2[1], chi2[2]])
            
            t_arr = g["t"][:]
            h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
            
            # Interpolate to common_t, zero-pad before start
            h_interp = np.zeros(len(common_t), dtype=complex)
            mask = (common_t >= t_arr[0]) & (common_t <= t_arr[-1])
            h_interp[mask] = np.interp(common_t[mask], t_arr, h22)
            waveforms.append(h_interp)
            
    return np.array(params), np.array(waveforms), common_t, dt

def get_reparameterized_params(params, type='raw'):
    # params: [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]
    q = params[:, 0]
    chi1x = params[:, 1]
    chi1y = params[:, 2]
    chi1z = params[:, 3]
    chi2x = params[:, 4]
    chi2y = params[:, 5]
    chi2z = params[:, 6]
    
    eta = q / (1 + q)**2
    m1 = q / (1 + q)
    m2 = 1 / (1 + q)
    
    if type == 'raw':
        return params
    elif type == 'effective_spins':
        chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
        chi1p = np.sqrt(chi1x**2 + chi1y**2)
        chi2p = np.sqrt(chi2x**2 + chi2y**2)
        chi_p = np.maximum(chi1p, (4*q + 3)/(3*q + 4) * q * chi2p)
        abs_chi1 = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
        abs_chi2 = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
        theta1 = np.arccos(np.clip(chi1z / (abs_chi1 + 1e-10), -1, 1))
        theta2 = np.arccos(np.clip(chi2z / (abs_chi2 + 1e-10), -1, 1))
        return np.column_stack([eta, chi_eff, chi_p, abs_chi1, abs_chi2, theta1, theta2])
    elif type == 'spherical':
        abs_chi1 = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
        theta1 = np.arccos(np.clip(chi1z / (abs_chi1 + 1e-10), -1, 1))
        phi1 = np.arctan2(chi1y, chi1x)
        abs_chi2 = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
        theta2 = np.arccos(np.clip(chi2z / (abs_chi2 + 1e-10), -1, 1))
        phi2 = np.arctan2(chi2y, chi2x)
        return np.column_stack([eta, abs_chi1, theta1, phi1, abs_chi2, theta2, phi2])
    else:
        raise ValueError("Unknown reparameterization type")

if __name__ == "__main__":
    X_train, y_train, t, dt = load_data("datasets/waveform/waveform_training.h5")
    X_val, y_val, _, _ = load_data("datasets/waveform/waveform_validation.h5", common_t=t)
    
    # Flatten waveforms for SVD: (n_sim, 2 * n_times)
    y_train_flat = np.hstack([y_train.real, y_train.imag])
    
    n_components = 50
    svd = TruncatedSVD(n_components=n_components)
    y_train_reduced = svd.fit_transform(y_train_flat)
    
    os.makedirs("llm_agents/results/gemini3_flash_preview/waveform/data_cache", exist_ok=True)
    joblib.dump((X_train, y_train, X_val, y_val, t, dt, svd, y_train_reduced), "llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")
    print("SVD completed and data cached.")
