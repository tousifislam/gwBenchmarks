"""Evaluate a single approach by name."""

import sys
import numpy as np
import h5py
from gwbenchmarks.metrics import mean_fd_mismatch

def load_data(file_path):
    q_list = []
    h22_list = []
    t_list = []
    with h5py.File(file_path, 'r') as f:
        for sim_id in f['sims'].keys():
            group = f['sims'][sim_id]
            q_list.append(group.attrs['q'])
            t = group['t'][:]
            h22 = group['h22_real'][:] + 1j * group['h22_imag'][:]
            h22_list.append(h22)
            t_list.append(t)
    return np.array(q_list), h22_list, t_list

def predict_approach_3(t, q):
    """Approach 3: Polynomial eta model."""
    eta = q / (1 + q)**2
    tau = np.maximum(-t, 1e-6)
    
    # Amplitude
    A0 = np.sqrt(32/45 * eta**2)
    A = A0 * tau**(-1/6) * (1 + 0.05 * (eta - 0.25) * np.log(tau))
    
    # Phase
    phi = - (5/8) * eta**(-3/5) * tau**(5/8) * (1 + 0.1 * eta)
    
    return A * np.exp(1j * phi)

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_approach.py <approach_number>")
        print("Approaches: 1, 2, 3")
        sys.exit(1)
    
    approach = sys.argv[1]
    
    print(f"Evaluating Approach {approach}...")
    
    # Load validation data
    val_q, val_h22, val_t = load_data(
        '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/analytic/analytic_validation.h5'
    )
    
    # Select predict function
    if approach == "1":
        from llm_agents.results.hy3_preview_free.analytic.models.01_pn_lorentzian_qnm.predict import predict as predict_func
        # Need to load model params
        import json
        from pathlib import Path
        params = json.load(open(Path(__file__).parent / 'models/01_pn_lorentzian_qnm/saved_model/model_params.json'))
        predict_func = lambda t, q: predict_func(t, q, params)
    elif approach == "2":
        from llm_agents.results.hy3_preview_free.analytic.models.02_pn_full_eta.predict import predict as predict_func
    elif approach == "3":
        predict_func = predict_approach_3
    else:
        print(f"Unknown approach: {approach}")
        sys.exit(1)
    
    # Evaluate
    mismatches = []
    for i, (q, t, h22_ref) in enumerate(zip(val_q, val_t, val_h22)):
        try:
            h22_pred = predict_func(t, q)
            dt = float(t[1] - t[0])
            mm = mean_fd_mismatch(h22_pred, h22_ref, dt, masses=[40, 80, 120, 160, 200])
            mismatches.append(mm)
        except Exception as e:
            print(f"Error for q={q}: {e}")
            mismatches.append(1.0)
    
    print(f"\nResults for Approach {approach}:")
    print(f"Mean mismatch: {np.mean(mismatches):.6f}")
    print(f"Std mismatch: {np.std(mismatches):.6f}")
    print(f"Min mismatch: {np.min(mismatches):.6f}")
    print(f"Max mismatch: {np.max(mismatches):.6f}")

if __name__ == "__main__":
    main()
