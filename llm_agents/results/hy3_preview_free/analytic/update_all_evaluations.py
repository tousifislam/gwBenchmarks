"""Update all evaluations with per-sample errors."""

import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import h5py
import json
from pathlib import Path
from gwbenchmarks.metrics import mean_fd_mismatch

PROJECT_ROOT = Path("/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks")
MODELS_DIR = PROJECT_ROOT / "llm_agents/results/hy3_preview_free/analytic/models"

def load_val_data():
    q_list, h22_list, t_list = [], [], []
    with h5py.File(PROJECT_ROOT / "datasets/analytic/analytic_validation.h5", 'r') as f:
        for sim_id in f["sims"].keys():
            g = f["sims"][sim_id]
            q_list.append(g.attrs["q"])
            t = g["t"][:]
            h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
            h22_list.append(h22)
            t_list.append(t)
    return np.array(q_list), h22_list, t_list

def evaluate_predict_func(val_q, val_h22, val_t, predict_func):
    """Evaluate a prediction function and return per-sample mismatches."""
    mismatches = []
    for q, t, h22_ref in zip(val_q, val_t, val_h22):
        try:
            h22_pred = predict_func(t, q)
            dt = float(t[1] - t[0])
            mm = mean_fd_mismatch(h22_pred, h22_ref, dt, masses=[40, 80, 120, 160, 200])
            mismatches.append(mm)
        except Exception as e:
            mismatches.append(1.0)
    return mismatches

def main():
    print("Loading validation data...")
    val_q, val_h22, val_t = load_val_data()
    print(f"Loaded {len(val_q)} samples\n")
    
    # Import all predict functions from batch scripts
    sys.path.insert(0, str(PROJECT_ROOT / "llm_agents/results/hy3_preview_free/analytic"))
    
    # Map approach name to predict function
    predict_funcs = {}
    
    # From batch_approaches_1.py
    from batch_approaches_1 import approach_5, approach_6
    predict_funcs["05_lorentzian_q"] = approach_5
    predict_funcs["06_gaussian_sum_eta"] = approach_6
    
    # From batch_approaches_2.py  
    from batch_approaches_2 import (approach_7, approach_8, approach_9, approach_10,
                                      approach_11, approach_12, approach_13, approach_14, approach_15)
    predict_funcs["07_tanh_transition_eta"] = approach_7
    predict_funcs["08_damped_sinusoid_q"] = approach_8
    predict_funcs["09_powerlaw_exp_delta"] = approach_9
    predict_funcs["10_rational_eta"] = approach_10
    predict_funcs["11_sigmoid_eta"] = approach_11
    predict_funcs["12_composite_bump_q"] = approach_12
    predict_funcs["13_freq_based_eta"] = approach_13
    predict_funcs["14_delta_m_param"] = approach_14
    predict_funcs["15_sqrt_eta_param"] = approach_15
    
    # From batch_approaches_3.py
    from batch_approaches_3 import approach_16, approach_17, approach_18, approach_19, approach_20
    predict_funcs["16_pn_qnm_eta"] = approach_16
    predict_funcs["17_modified_lorentzian_q"] = approach_17
    predict_funcs["18_exp_inspiral_eta"] = approach_18
    predict_funcs["19_imrphenom_style_eta"] = approach_19
    predict_funcs["20_pade_eta"] = approach_20
    
    # Evaluate each approach
    updated = 0
    for approach_dir in sorted(MODELS_DIR.iterdir()):
        if not approach_dir.is_dir():
            continue
        
        approach_name = approach_dir.name
        
        if approach_name not in predict_funcs:
            continue
        
        print(f"Evaluating {approach_name}...")
        mismatches = evaluate_predict_func(val_q, val_h22, val_t, predict_funcs[approach_name])
        
        # Save evaluation.json
        eval_data = {
            "mean_mismatch": float(np.mean(mismatches)),
            "std_mismatch": float(np.std(mismatches)),
            "per_sample": [float(m) for m in mismatches]
        }
        
        with open(approach_dir / "evaluation.json", "w") as f:
            json.dump(eval_data, f, indent=2)
        
        # Update scorecard.json
        sc_path = approach_dir / "scorecard.json"
        if sc_path.exists():
            with open(sc_path) as f:
                sc = json.load(f)
            sc["loss"] = float(np.mean(mismatches))
            sc["loss_components"] = {"mean": float(np.mean(mismatches)), "std": float(np.std(mismatches))}
            with open(sc_path, "w") as f:
                json.dump(sc, f, indent=2)
        
        updated += 1
        print(f"  Mean mismatch: {eval_data['mean_mismatch']:.6f}")
    
    print(f"\nUpdated {updated} approaches with per-sample errors")

if __name__ == "__main__":
    main()
