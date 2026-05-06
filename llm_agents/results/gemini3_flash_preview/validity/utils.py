import h5py
import numpy as np
import os
import joblib
import time
import json

def load_data(file_path):
    with h5py.File(file_path, "r") as f:
        q = f["q"][:]
        chi1z = f["chi1z"][:]
        chi2z = f["chi2z"][:]
        omega0 = f["omega0"][:]
        mm_td = f["mm_td"][:]
    params = np.column_stack([q, chi1z, chi2z, omega0])
    # Target is log10(mm_td)
    y = np.log10(np.maximum(mm_td, 1e-10))
    return params, y

def get_reparameterized_params(params, type='raw'):
    # params: [q, chi1z, chi2z, omega0]
    q = params[:, 0]
    chi1z, chi2z = params[:, 1], params[:, 2]
    omega0 = params[:, 3]
    
    eta = q / (1 + q)**2
    chi_eff = (q * chi1z + chi2z) / (1 + q)
    chi_a = (chi1z - chi2z) / 2
    
    if type == 'raw':
        return params
    elif type == 'effective_spins':
        return np.column_stack([eta, chi_eff, chi_a, omega0])
    elif type == 'log_q':
        return np.column_stack([np.log10(q), chi_eff, chi_a, np.log10(omega0)])
    elif type == 'interaction':
        return np.column_stack([eta, chi_eff, chi_a, omega0, q*chi_eff, eta*chi_a])
    elif type == 'boundary':
        dist_q = np.maximum(0, q - 8)
        dist_chi1 = np.maximum(0, np.abs(chi1z) - 0.8)
        dist_chi2 = np.maximum(0, np.abs(chi2z) - 0.8)
        return np.column_stack([q, chi1z, chi2z, omega0, dist_q, dist_chi1, dist_chi2])
    else:
        raise ValueError("Unknown reparameterization type")

def evaluate_model(model_dir, predict_fn, X_val, y_val):
    start_time = time.time()
    y_pred = predict_fn(X_val)
    eval_time = (time.time() - start_time) / len(X_val) * 1000 # ms
    
    # y is log10(mm)
    loss = np.sqrt(np.mean((y_pred - y_val)**2))
    per_sample_errors = np.abs(y_pred - y_val)
    
    return loss, eval_time, per_sample_errors, y_pred

def save_approach(approach_name, approach_number, params_type, loss, runtime_ms, n_params, notes, model_dir, train_errors, val_errors):
    scorecard = {
        "approach": approach_name,
        "approach_number": approach_number,
        "benchmark": "validity",
        "agent": "gemini3_flash_preview",
        "parameterization": params_type,
        "loss": float(loss),
        "loss_components": {"log10_rmse": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": 393,
        "n_val": 393,
        "n_params": int(n_params),
        "notes": notes
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    
    error_data_path = "llm_agents/results/gemini3_flash_preview/validity/comparison/error_data.json"
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

    with open("llm_agents/results/gemini3_flash_preview/validity/CHANGELOG.md", "a") as f:
        f.write(f"## Approach {approach_number}: {approach_name}\n")
        f.write(f"- **Parameterization**: {params_type}\n")
        f.write(f"- **Loss**: {loss:.6f}\n")
        f.write(f"- **Notes**: {notes}\n\n")

if __name__ == "__main__":
    X_train, y_train = load_data("datasets/validity/validity_training.h5")
    X_val, y_val = load_data("datasets/validity/validity_validation.h5")
    os.makedirs("llm_agents/results/gemini3_flash_preview/validity/data_cache", exist_ok=True)
    joblib.dump((X_train, y_train, X_val, y_val), "llm_agents/results/gemini3_flash_preview/validity/data_cache/data.pkl")
    print("Data loaded and cached.")
