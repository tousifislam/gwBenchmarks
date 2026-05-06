import h5py
import numpy as np
import os
import joblib
import time
import json

def load_data(file_path):
    with h5py.File(file_path, "r") as f:
        q = f["q"][:]
        chi1x, chi1y, chi1z = f["chi1x"][:], f["chi1y"][:], f["chi1z"][:]
        chi2x, chi2y, chi2z = f["chi2x"][:], f["chi2y"][:], f["chi2z"][:]
        # Target: vf_mag (kick velocity)
        y = f["vf_mag"][:]
        params = np.column_stack([q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z])
    return params, y

def get_reparameterized_params(params, type='raw'):
    q = params[:, 0]
    chi1x, chi1y, chi1z = params[:, 1], params[:, 2], params[:, 3]
    chi2x, chi2y, chi2z = params[:, 4], params[:, 5], params[:, 6]
    
    eta = q / (1 + q)**2
    m1 = q / (1 + q)
    m2 = 1 / (1 + q)
    delta_m = (m1 - m2) / (m1 + m2) # which is (q-1)/(q+1)
    
    chi1_mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2_mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
    
    if type == 'raw':
        return params
    elif type == 'effective_spins':
        chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
        chi1p = np.sqrt(chi1x**2 + chi1y**2)
        chi2p = np.sqrt(chi2x**2 + chi2y**2)
        chi_p = np.maximum(chi1p, (4*q + 3)/(3*q + 4) * q * chi2p)
        theta1 = np.arccos(np.clip(chi1z / (chi1_mag + 1e-10), -1, 1))
        theta2 = np.arccos(np.clip(chi2z / (chi2_mag + 1e-10), -1, 1))
        return np.column_stack([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])
    elif type == 'delta_m_chi_a':
        chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
        chi_a = (chi1z - chi2z) / 2
        return np.column_stack([delta_m, chi_eff, chi_a, chi1_mag, chi2_mag])
    elif type == 'spherical':
        theta1 = np.arccos(np.clip(chi1z / (chi1_mag + 1e-10), -1, 1))
        phi1 = np.arctan2(chi1y, chi1x)
        theta2 = np.arccos(np.clip(chi2z / (chi2_mag + 1e-10), -1, 1))
        phi2 = np.arctan2(chi2y, chi2x)
        return np.column_stack([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])
    else:
        raise ValueError("Unknown reparameterization type")

def nrmse(pred, true):
    rmse = np.sqrt(np.mean((pred - true)**2))
    return rmse / (np.max(true) - np.min(true))

def evaluate_model(model_dir, predict_fn, X_val, y_val):
    start_time = time.time()
    y_pred = predict_fn(X_val)
    eval_time = (time.time() - start_time) / len(X_val) * 1000 # ms per sample
    
    loss = nrmse(y_pred, y_val)
    errors = np.abs(y_pred - y_val) # Absolute error for histograms
    # Or relative? Prompt says "per-sample errors". NRMSE uses RMSE.
    # I'll save relative errors too.
    rel_errors = np.abs(y_pred - y_val) / (np.max(y_val) - np.min(y_val))
    
    return loss, eval_time, rel_errors, y_pred

def save_approach(approach_name, approach_number, params_type, loss, runtime_ms, n_params, notes, model_dir, train_errors, val_errors):
    scorecard = {
        "approach": approach_name,
        "approach_number": approach_number,
        "benchmark": "remnant",
        "agent": "gemini3_flash_preview",
        "parameterization": params_type,
        "loss": float(loss),
        "loss_components": {"nrmse_v_k": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": 10000,
        "n_val": 2000,
        "n_params": int(n_params),
        "notes": notes
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    
    # Update error_data.json
    error_data_path = "llm_agents/results/gemini3_flash_preview/remnant/comparison/error_data.json"
    if os.path.exists(error_data_path):
        with open(error_data_path, "r") as f:
            error_data = json.load(f)
    else:
        error_data = {}
    
    error_data[approach_name] = {
        "train": [float(e) for e in train_errors],
        "validation": [float(e) for e in val_errors]
    }
    with open(error_data_path, "w") as f:
        json.dump(error_data, f, indent=4)

    # Update CHANGELOG.md
    with open("llm_agents/results/gemini3_flash_preview/remnant/CHANGELOG.md", "a") as f:
        f.write(f"## Approach {approach_number}: {approach_name}\n")
        f.write(f"- **Parameterization**: {params_type}\n")
        f.write(f"- **Loss**: {loss:.6f}\n")
        f.write(f"- **Notes**: {notes}\n\n")

if __name__ == "__main__":
    X_train, y_train = load_data("datasets/remnant/remnant_training.h5")
    X_val, y_val = load_data("datasets/remnant/remnant_validation.h5")
    os.makedirs("llm_agents/results/gemini3_flash_preview/remnant/data_cache", exist_ok=True)
    joblib.dump((X_train, y_train, X_val, y_val), "llm_agents/results/gemini3_flash_preview/remnant/data_cache/data.pkl")
    print("Data loaded and cached.")
