"""Run gplearn symbolic regression."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import time
import pickle
import warnings
from pathlib import Path
from train_evaluate import (
    load_basis, load_ref_sims, get_param_reparameterization, normalize_features,
    evaluate_approach, RESULTS_DIR
)
from sklearn.linear_model import Ridge
from gplearn.genetic import SymbolicRegressor

warnings.filterwarnings('ignore')

def run_gplearn_approach():
    print('=== gplearn Approach ===')
    
    basis_data = load_basis()
    train_sims, val_sims = load_ref_sims()
    train_params = basis_data['train_params']
    val_params = basis_data['val_params']
    
    ptype = 'raw'
    n_basis_use = 8
    n_gp_coeffs = 3
    
    X_train = get_param_reparameterization(train_params, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize_features(X_train, get_param_reparameterization(val_params, ptype))
    
    coeffs_real_train = basis_data['coeffs_real_train'][:, :n_basis_use]
    coeffs_imag_train = basis_data['coeffs_imag_train'][:, :n_basis_use]
    
    models_real = []
    models_imag = []
    expressions = []
    
    for i in range(n_basis_use):
        if i < n_gp_coeffs:
            print(f'  Fitting gplearn on coefficient {i}...')
            t0 = time.time()
            
            est = SymbolicRegressor(
                population_size=1000,
                generations=20,
                tournament_size=20,
                function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
                metric='mse',
                parsimony_coefficient=0.001,
                max_samples=1.0,
                verbose=0,
                random_state=42,
                n_jobs=2,
            )
            
            y_train = coeffs_real_train[:, i]
            est.fit(Xn_train, y_train)
            expr = str(est._program)
            print(f'    Expression: {expr}')
            print(f'    Time: {time.time()-t0:.1f}s')
            
            expressions.append({'coeff': i, 'expr': expr, 'type': 'real'})
            models_real.append(est)
            
            est_imag = SymbolicRegressor(
                population_size=1000,
                generations=20,
                tournament_size=20,
                function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
                metric='mse',
                parsimony_coefficient=0.001,
                max_samples=1.0,
                verbose=0,
                random_state=43,
                n_jobs=2,
            )
            y_imag = coeffs_imag_train[:, i]
            est_imag.fit(Xn_train, y_imag)
            expr_imag = str(est_imag._program)
            expressions.append({'coeff': i, 'expr': expr_imag, 'type': 'imag'})
            models_imag.append(est_imag)
        else:
            model = Ridge(alpha=1.0)
            model.fit(Xn_train, coeffs_real_train[:, i])
            models_real.append(model)
            
            model_imag = Ridge(alpha=1.0)
            model_imag.fit(Xn_train, coeffs_imag_train[:, i])
            models_imag.append(model_imag)
    
    model_dir = RESULTS_DIR / 'models' / '24_svd_gplearn_raw'
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({'models_real': models_real, 'models_imag': models_imag,
                     'mean_x': mean_x, 'std_x': std_x, 'ptype': ptype,
                     'n_basis_use': n_basis_use}, f)
    
    with open(save_dir / 'expressions.json', 'w') as f:
        json.dump(expressions, f, indent=2)
    
    print('  Evaluating gplearn...')
    losses = evaluate_approach(models_real, models_imag, mean_x, std_x, basis_data,
                               val_sims, val_params, ptype, n_basis_use)
    mean_loss = float(np.mean(losses))
    print(f'  Mean loss: {mean_loss:.4f}')
    
    scorecard = {
        'approach': '24_svd_gplearn_raw',
        'approach_number': 24,
        'benchmark': 'waveform',
        'agent': 'kimi_k26',
        'parameterization': ptype,
        'time_convention': 't0_at_peak',
        'loss': mean_loss,
        'loss_components': {'mean_fd_mismatch': mean_loss},
        'runtime_ms': 0,
        'n_train': len(train_sims),
        'n_val': len(val_sims),
        'n_params': 0,
        'notes': 'gplearn on top 3 SVD coefficients, Ridge for rest',
        'val_losses': losses,
    }
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    return mean_loss

if __name__ == '__main__':
    gplearn_loss = run_gplearn_approach()
    print(f'\ngplearn loss: {gplearn_loss:.4f}')
