"""Run all waveform benchmark approaches systematically."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import time
from pathlib import Path
from train_evaluate import (
    load_basis, load_ref_sims, train_and_save_approach, 
    evaluate_and_save_approach, get_param_reparameterization, normalize_features,
    fit_coefficient_models, predict_coefficients
)

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/waveform')

# Define all approaches
# Format: (name, model_type, ptype, n_basis_use, category)
APPROACHES = [
    # SVD/Decomposition-based
    ('01_svd_gpr_rbf_raw', 'gpr_rbf', 'raw', 15, 'svd_decomposition'),
    ('02_svd_gpr_matern_effective', 'gpr_matern', 'effective', 15, 'svd_decomposition'),
    ('03_svd_polynomial_raw', 'poly', 'raw', 15, 'svd_decomposition'),
    ('04_svd_mlp_raw', 'mlp', 'raw', 15, 'svd_decomposition'),
    ('05_svd_rf_raw', 'rf', 'raw', 15, 'svd_decomposition'),
    ('06_eim_gpr_rbf_raw', 'gpr_rbf', 'raw', 25, 'svd_decomposition'),
    ('07_svd_gpr_rbf_mass_diff', 'gpr_rbf', 'mass_diff', 15, 'svd_decomposition'),
    ('08_svd_mlp_effective', 'mlp', 'effective', 15, 'svd_decomposition'),
    
    # Interpolation/Kernel
    ('09_svd_kr_rbf_raw', 'kr', 'raw', 15, 'interpolation_kernel'),
    ('10_svd_knn_raw', 'knn', 'raw', 15, 'interpolation_kernel'),
    ('11_svd_svr_raw', 'svr', 'raw', 15, 'interpolation_kernel'),
    ('12_svd_kr_rbf_effective', 'kr', 'effective', 15, 'interpolation_kernel'),
    
    # Machine Learning
    ('13_svd_gbr_raw', 'gbr', 'raw', 15, 'machine_learning'),
    ('14_svd_extra_trees_raw', 'extra_trees', 'raw', 15, 'machine_learning'),
    ('15_svd_mlp_large_raw', 'mlp_large', 'raw', 20, 'machine_learning'),
    ('16_svd_ridge_raw', 'ridge', 'raw', 15, 'machine_learning'),
    ('17_svd_elastic_net_raw', 'elastic_net', 'raw', 15, 'machine_learning'),
    ('18_svd_lasso_raw', 'lasso', 'raw', 15, 'machine_learning'),
    ('19_svd_huber_raw', 'huber', 'raw', 15, 'machine_learning'),
    ('20_svd_linear_raw', 'linear', 'raw', 15, 'machine_learning'),
    ('21_svd_bayesian_ridge_raw', 'bayesian_ridge', 'raw', 15, 'machine_learning'),
    ('22_svd_mlp_small_raw', 'mlp_small', 'raw', 15, 'machine_learning'),
]

def run_all_approaches():
    print('Loading data...')
    basis_data = load_basis()
    train_sims, val_sims = load_ref_sims()
    train_params = basis_data['train_params']
    val_params = basis_data['val_params']
    
    results = []
    
    for i, (name, model_type, ptype, n_basis_use, category) in enumerate(APPROACHES):
        print(f'\n{"="*60}')
        print(f'Approach {i+1}/{len(APPROACHES)}: {name}')
        print(f'{"="*60}')
        
        try:
            models_real, models_imag, mean_x, std_x = train_and_save_approach(
                name, model_type, ptype, n_basis_use, basis_data,
                train_sims, train_params, val_sims, val_params
            )
            
            scorecard = evaluate_and_save_approach(
                name, models_real, models_imag, mean_x, std_x,
                basis_data, val_sims, val_params, ptype, n_basis_use, model_type
            )
            
            results.append({
                'name': name,
                'loss': scorecard['loss'],
                'category': category,
                'ptype': ptype,
            })
            
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()
    
    # Print summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    results_sorted = sorted(results, key=lambda x: x['loss'])
    for r in results_sorted:
        print(f'{r["loss"]:.4f}  {r["name"]:30s}  ({r["category"]}, {r["ptype"]})')
    
    return results

if __name__ == '__main__':
    results = run_all_approaches()
    
    # Save summary
    with open(RESULTS_DIR / 'approach_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
