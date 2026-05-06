import numpy as np
import joblib
import json
import os
import time
from gwbenchmarks.metrics import mean_fd_mismatch, FD_MASSES_MSUN
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params

def evaluate_model(model_dir, predict_fn, X_val, y_val, dt):
    losses = []
    per_mass_losses = {f"mismatch_{int(m)}Msun": [] for m in FD_MASSES_MSUN}
    
    start_time = time.time()
    for i in range(len(X_val)):
        h_pred = predict_fn(X_val[i])
        h_true = y_val[i]
        
        mismatches = [
            mean_fd_mismatch(h_pred, h_true, dt, masses=[m])
            for m in FD_MASSES_MSUN
        ]
        losses.append(np.mean(mismatches))
        for j, m in enumerate(FD_MASSES_MSUN):
            per_mass_losses[f"mismatch_{int(m)}Msun"].append(mismatches[j])
            
    eval_time = (time.time() - start_time) / len(X_val) * 1000 # ms
    
    mean_loss = np.mean(losses)
    mean_per_mass = {k: np.mean(v) for k, v in per_mass_losses.items()}
    
    return mean_loss, mean_per_mass, eval_time, losses

def save_approach(approach_name, approach_number, params_type, time_conv, loss, components, runtime_ms, n_params, notes, model_dir, train_losses, val_losses):
    scorecard = {
        "approach": approach_name,
        "approach_number": approach_number,
        "benchmark": "waveform",
        "agent": "gemini3_flash_preview",
        "parameterization": params_type,
        "time_convention": time_conv,
        "loss": float(loss),
        "loss_components": {k: float(v) for k, v in components.items()},
        "runtime_ms": float(runtime_ms),
        "n_train": 250,
        "n_val": 250,
        "n_params": int(n_params),
        "notes": notes
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    
    # Update error_data.json
    error_data_path = "llm_agents/results/gemini3_flash_preview/waveform/comparison/error_data.json"
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

    # Update CHANGELOG.md
    with open("llm_agents/results/gemini3_flash_preview/waveform/CHANGELOG.md", "a") as f:
        f.write(f"## Approach {approach_number}: {approach_name}\n")
        f.write(f"- **Parameterization**: {params_type}\n")
        f.write(f"- **Loss**: {loss:.6f}\n")
        f.write(f"- **Notes**: {notes}\n\n")
