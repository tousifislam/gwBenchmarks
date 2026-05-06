"""Fix per-sample error consistency with loss metrics."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import pickle
import h5py
from pathlib import Path

RESULTS_BASE = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26')

def fix_remnant_consistency():
    """Remnant uses NRMSE = RMSE / range. Per-sample errors should be |pred-true|/range."""
    print('=== REMNANT ===')
    best_name = '02_gpr_matern_effective'
    sc_path = RESULTS_BASE / 'remnant' / 'models' / best_name / 'scorecard.json'
    
    with open(sc_path) as f:
        sc = json.load(f)
    
    val_losses = sc.get('val_losses', [])
    if val_losses:
        # NRMSE = sqrt(mean((pred-true)^2)) / range
        # The per-sample errors should be (pred-true)^2 / range^2, then sqrt(mean) = NRMSE
        # OR we can store |pred-true|/range and note that NRMSE = sqrt(mean of squares)
        # Let's store the squared normalized errors so sqrt(mean) gives NRMSE
        squared_errors = [e**2 for e in val_losses]
        nrmse_from_errors = np.sqrt(np.mean(squared_errors))
        print(f'  Current mean of per-sample errors: {np.mean(val_losses):.6f}')
        print(f'  NRMSE from squared errors: {nrmse_from_errors:.6f}')
        print(f'  Scorecard loss: {sc["loss"]:.6f}')
        print(f'  Match: {abs(nrmse_from_errors - sc["loss"]) < 0.001}')
        
        # Update to store squared normalized errors for consistency
        sc['val_losses'] = squared_errors
        with open(sc_path, 'w') as f:
            json.dump(sc, f, indent=2)
        print('  Fixed: per-sample errors now squared normalized errors')

def fix_validity_consistency():
    """Validity uses RMSE = sqrt(mean((pred-true)^2)). Per-sample errors should be (pred-true)^2."""
    print('\n=== VALIDITY ===')
    best_name = '05_rf_raw'
    sc_path = RESULTS_BASE / 'validity' / 'models' / best_name / 'scorecard.json'
    
    with open(sc_path) as f:
        sc = json.load(f)
    
    val_losses = sc.get('val_losses', [])
    if val_losses:
        # RMSE = sqrt(mean((pred-true)^2))
        # Store squared errors so sqrt(mean) = RMSE
        squared_errors = [e**2 for e in val_losses]
        rmse_from_errors = np.sqrt(np.mean(squared_errors))
        print(f'  Current mean of per-sample errors: {np.mean(val_losses):.6f}')
        print(f'  RMSE from squared errors: {rmse_from_errors:.6f}')
        print(f'  Scorecard loss: {sc["loss"]:.6f}')
        print(f'  Match: {abs(rmse_from_errors - sc["loss"]) < 0.001}')
        
        sc['val_losses'] = squared_errors
        with open(sc_path, 'w') as f:
            json.dump(sc, f, indent=2)
        print('  Fixed: per-sample errors now squared errors')

def update_error_data():
    print('\n=== UPDATING error_data.json ===')
    for bench in ['remnant', 'validity']:
        comparison_dir = RESULTS_BASE / bench / 'comparison'
        
        with open(comparison_dir / 'summary_table.json') as f:
            summary = json.load(f)
        
        best_name = summary[0]['model_name']
        sc_path = RESULTS_BASE / bench / 'models' / best_name / 'scorecard.json'
        
        with open(sc_path) as f:
            sc = json.load(f)
        
        err_path = comparison_dir / 'error_data.json'
        with open(err_path) as f:
            err_data = json.load(f)
        
        for i, entry in enumerate(err_data):
            if entry['model_name'] == best_name:
                err_data[i]['val_losses'] = sc.get('val_losses', [])
                break
        
        with open(err_path, 'w') as f:
            json.dump(err_data, f, indent=2)
        
        print(f'  {bench}: updated')

if __name__ == '__main__':
    fix_remnant_consistency()
    fix_validity_consistency()
    update_error_data()
    print('\nDone!')
