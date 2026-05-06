import h5py
import numpy as np
import os
import joblib
import time
import json
from gwbenchmarks.metrics import mean_fd_mismatch, FD_MASSES_MSUN

def load_data(file_path, n_grid=1000):
    with h5py.File(file_path, "r") as f:
        sim_keys = sorted(list(f["sims"].keys()))
        n_sim = len(sim_keys)
        qs = []
        waveforms = []
        
        # Find common time range
        t_min = 0
        t_max = -1e10
        for k in sim_keys:
            g = f["sims"][k]
            t = g["t"][:]
            t_min = min(t_min, t[0])
            t_max = max(t_max, t[-1])
        
        common_t = np.linspace(t_min, t_max, n_grid)
        dt = common_t[1] - common_t[0]
        
        for k in sim_keys:
            g = f["sims"][k]
            q = g.attrs["q"]
            qs.append(q)
            
            t = g["t"][:]
            h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
            
            # Interpolate to common_t
            h_interp = np.zeros(len(common_t), dtype=complex)
            mask = (common_t >= t[0]) & (common_t <= t[-1])
            h_interp[mask] = np.interp(common_t[mask], t, h22)
            waveforms.append(h_interp)
            
    return np.array(qs), np.array(waveforms), common_t, dt

def get_reparameterized_q(q, type='q'):
    if type == 'q':
        return q
    elif type == 'eta':
        return q / (1 + q)**2
    elif type == 'delta_m':
        return (q - 1) / (q + 1)
    elif type == 'sqrt_eta':
        return np.sqrt(q / (1 + q)**2)
    else:
        raise ValueError("Unknown reparameterization type")

def evaluate_model(model_dir, predict_fn, qs_val, waveforms_val, dt):
    start_time = time.time()
    
    losses = []
    per_mass_losses = {f"mismatch_{int(m)}Msun": [] for m in FD_MASSES_MSUN}
    
    for i in range(len(qs_val)):
        h_pred = predict_fn(qs_val[i])
        h_true = waveforms_val[i]
        
        mismatches = [
            mean_fd_mismatch(h_pred, h_true, dt, masses=[m])
            for m in FD_MASSES_MSUN
        ]
        losses.append(np.mean(mismatches))
        for j, m in enumerate(FD_MASSES_MSUN):
            per_mass_losses[f"mismatch_{int(m)}Msun"].append(mismatches[j])
            
    eval_time = (time.time() - start_time) / len(qs_val) * 1000 # ms
    
    mean_loss = np.mean(losses)
    mean_per_mass = {k: np.mean(v) for k, v in per_mass_losses.items()}
    
    return mean_loss, mean_per_mass, eval_time, losses

def save_approach(approach_name, approach_number, params_type, loss, components, runtime_ms, n_params, notes, model_dir, train_losses, val_losses, expression=""):
    scorecard = {
        "approach": approach_name,
        "approach_number": approach_number,
        "benchmark": "analytic",
        "agent": "gemini3_flash_preview",
        "parameterization": params_type,
        "loss": float(loss),
        "loss_components": {k: float(v) for k, v in components.items()},
        "runtime_ms": float(runtime_ms),
        "n_train": 20,
        "n_val": 20,
        "n_params": int(n_params),
        "n_terms": 10, # dummy
        "expression_file": "expression.txt",
        "notes": notes
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    
    with open(os.path.join(model_dir, "expression.txt"), "w") as f:
        f.write(expression)
    
    error_data_path = "llm_agents/results/gemini3_flash_preview/analytic/comparison/error_data.json"
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

    with open("llm_agents/results/gemini3_flash_preview/analytic/CHANGELOG.md", "a") as f:
        f.write(f"## Approach {approach_number}: {approach_name}\n")
        f.write(f"- **Parameterization**: {params_type}\n")
        f.write(f"- **Loss**: {loss:.6f}\n")
        f.write(f"- **Notes**: {notes}\n\n")

if __name__ == "__main__":
    qs_train, y_train, common_t, dt = load_data("datasets/analytic/analytic_training.h5")
    qs_val, y_val, _, _ = load_data("datasets/analytic/analytic_validation.h5")
    
    os.makedirs("llm_agents/results/gemini3_flash_preview/analytic/data_cache", exist_ok=True)
    joblib.dump((qs_train, y_train, qs_val, y_val, common_t, dt), "llm_agents/results/gemini3_flash_preview/analytic/data_cache/data.pkl")
    print("Data prepared and cached.")
