import h5py
import numpy as np
import os
import joblib
import time
import json

def load_data(file_path, mode="l2/m+2/n0"):
    with h5py.File(file_path, "r") as f:
        g = f[mode]
        spin = g["spin"][:]
        omega_r = g["omega_real"][:]
        omega_i = g["omega_imag"][:]
    return spin, omega_r, omega_i

def get_reparameterized_spin(a, type='raw'):
    if type == 'raw':
        return a
    elif type == 'log_compact':
        return -np.log(np.maximum(1 - a, 1e-10))
    elif type == 'sqrt_irred':
        return np.sqrt(np.maximum(1 - a**2, 0))
    elif type == 'compact':
        return a / np.maximum(1 - a, 1e-10)
    elif type == 'chebyshev':
        return 2 * a - 1
    else:
        raise ValueError("Unknown reparameterization type")

def transform_target(y, type='none'):
    if type == 'none':
        return y
    elif type == 'log':
        return np.log(np.abs(y))
    else:
        raise ValueError("Unknown target transform")

def inverse_transform_target(y, type='none', sign=-1):
    if type == 'none':
        return y
    elif type == 'log':
        return sign * np.exp(y)
    else:
        raise ValueError("Unknown target transform")

def ringdown_loss(pred_r, pred_i, true_r, true_i):
    # L = (mean(|pred - true| / |true|) for omega_R + same for omega_I) / 2
    rel_err_r = np.abs(pred_r - true_r) / np.abs(true_r)
    rel_err_i = np.abs(pred_i - true_i) / np.abs(true_i)
    return (np.mean(rel_err_r) + np.mean(rel_err_i)) / 2

def evaluate_model(model_dir, predict_fn, spin_val, r_val, i_val):
    start_time = time.time()
    pred_r, pred_i = predict_fn(spin_val)
    eval_time = (time.time() - start_time) / len(spin_val) * 1000 # ms
    
    rel_err_r = np.abs(pred_r - r_val) / np.abs(r_val)
    rel_err_i = np.abs(pred_i - i_val) / np.abs(i_val)
    
    loss = (np.mean(rel_err_r) + np.mean(rel_err_i)) / 2
    per_sample_losses = (rel_err_r + rel_err_i) / 2
    
    return loss, eval_time, per_sample_losses, (pred_r, pred_i)

def save_approach(approach_name, approach_number, params_type, loss, runtime_ms, n_params, notes, model_dir, train_losses, val_losses, mode="l2_m2_n0"):
    scorecard = {
        "approach": approach_name,
        "approach_number": approach_number,
        "benchmark": "ringdown",
        "agent": "gemini3_flash_preview",
        "parameterization": params_type,
        "mode": mode,
        "loss": float(loss),
        "loss_components": {"rel_error_combined": float(loss)},
        "runtime_ms": float(runtime_ms),
        "n_train": 531,
        "n_val": 531,
        "n_params": int(n_params),
        "notes": notes
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    
    error_data_path = "llm_agents/results/gemini3_flash_preview/ringdown/comparison/error_data.json"
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

    with open("llm_agents/results/gemini3_flash_preview/ringdown/CHANGELOG.md", "a") as f:
        f.write(f"## Approach {approach_number}: {approach_name}\n")
        f.write(f"- **Parameterization**: {params_type}\n")
        f.write(f"- **Loss**: {loss:.6e}\n")
        f.write(f"- **Notes**: {notes}\n\n")

if __name__ == "__main__":
    spin_train, r_train, i_train = load_data("datasets/ringdown/ringdown_training.h5")
    spin_val, r_val, i_val = load_data("datasets/ringdown/ringdown_validation.h5")
    
    os.makedirs("llm_agents/results/gemini3_flash_preview/ringdown/data_cache", exist_ok=True)
    joblib.dump((spin_train, r_train, i_train, spin_val, r_val, i_val), "llm_agents/results/gemini3_flash_preview/ringdown/data_cache/data.pkl")
    print("Data prepared and cached.")
