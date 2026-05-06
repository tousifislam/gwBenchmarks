import h5py
import numpy as np
import os
import joblib
import time
import json
from sklearn.decomposition import TruncatedSVD

def load_data(file_path, n_grid=1000):
    with h5py.File(file_path, "r") as f:
        n_sim = f.attrs["n_simulations"]
        params = []
        x_data = []
        for i in range(n_sim):
            g = f[f"sim_{i:04d}"]
            q = g.attrs["q"]
            chi1z, chi2z = g.attrs["chi1z"], g.attrs["chi2z"]
            e0, zeta0, omega0 = g.attrs["e0"], g.attrs["zeta0"], g.attrs["omega0"]
            params.append([q, chi1z, chi2z, e0, zeta0, omega0])
            
            t = g["t"][:]
            x = g["x"][:]
            # Normalized time
            tau = (t - t[0]) / (t[-1] - t[0])
            common_tau = np.linspace(0, 1, n_grid)
            x_interp = np.interp(common_tau, tau, x)
            x_data.append(x_interp)
            
    return np.array(params), np.array(x_data)

def get_reparameterized_params(params, type='raw'):
    # params: [q, chi1z, chi2z, e0, zeta0, omega0]
    q = params[:, 0]
    chi1z, chi2z = params[:, 1], params[:, 2]
    e0, zeta0, omega0 = params[:, 3], params[:, 4], params[:, 5]
    
    eta = q / (1 + q)**2
    chi_eff = (q * chi1z + chi2z) / (1 + q)
    chi_a = (chi1z - chi2z) / 2
    
    if type == 'raw':
        return params
    elif type == 'eff_log_e':
        return np.column_stack([eta, chi_eff, chi_a, np.log(e0), zeta0, omega0])
    elif type == 'trig_anomaly':
        return np.column_stack([eta, chi_eff, chi_a, e0, np.cos(zeta0), np.sin(zeta0), omega0])
    elif type == 'fully_transformed':
        return np.column_stack([eta, chi_eff, chi_a, np.log(e0), np.cos(zeta0), np.sin(zeta0), np.log(omega0)])
    else:
        raise ValueError("Unknown reparameterization type")

def dynamics_loss(x_pred, x_true):
    # L = sqrt(mean((x_pred - x_true)^2 / x_true^2))
    rel_error_sq = ((x_pred - x_true) / x_true)**2
    return np.sqrt(np.mean(rel_error_sq))

def evaluate_model(model_dir, predict_fn, X_val_raw, y_val_true):
    start_time = time.time()
    y_pred = predict_fn(X_val_raw)
    eval_time = (time.time() - start_time) / len(X_val_raw) * 1000 # ms
    
    losses = []
    for i in range(len(y_val_true)):
        losses.append(dynamics_loss(y_pred[i], y_val_true[i]))
    
    mean_loss = np.mean(losses)
    return mean_loss, eval_time, losses, y_pred

def save_approach(approach_name, approach_number, params_type, loss, runtime_ms, n_params, notes, model_dir, train_losses, val_losses):
    scorecard = {
        "approach": approach_name,
        "approach_number": approach_number,
        "benchmark": "dynamics",
        "agent": "gemini3_flash_preview",
        "parameterization": params_type,
        "time_convention": "normalized_tau",
        "loss": float(loss),
        "loss_components": {"rms_relative_error_x": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": 250,
        "n_val": 250,
        "n_params": int(n_params),
        "notes": notes
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    
    error_data_path = "llm_agents/results/gemini3_flash_preview/dynamics/comparison/error_data.json"
    if os.path.exists(error_data_path):
        with open(error_data_path, "r") as f:
            error_data = json.load(f)
    else:
        error_data = {}
    
    error_data[approach_name] = {
        "train": [float(l) for l in train_losses],
        "validation": [float(l) for l in val_losses]
    }
    with open(error_data_path, "w") as f:
        json.dump(error_data, f, indent=4)

    with open("llm_agents/results/gemini3_flash_preview/dynamics/CHANGELOG.md", "a") as f:
        f.write(f"## Approach {approach_number}: {approach_name}\n")
        f.write(f"- **Parameterization**: {params_type}\n")
        f.write(f"- **Loss**: {loss:.6f}\n")
        f.write(f"- **Notes**: {notes}\n\n")

if __name__ == "__main__":
    X_train, y_train = load_data("datasets/dynamics/dynamics_training.h5")
    X_val, y_val = load_data("datasets/dynamics/dynamics_validation.h5")
    
    svd = TruncatedSVD(n_components=20)
    y_train_reduced = svd.fit_transform(y_train)
    
    os.makedirs("llm_agents/results/gemini3_flash_preview/dynamics/data_cache", exist_ok=True)
    joblib.dump((X_train, y_train, X_val, y_val, svd, y_train_reduced), "llm_agents/results/gemini3_flash_preview/dynamics/data_cache/data.pkl")
    print("Data prepared and cached.")
