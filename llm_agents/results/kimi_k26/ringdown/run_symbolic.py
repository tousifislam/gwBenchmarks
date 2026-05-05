"""Add symbolic regression to ringdown."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import pickle
from pathlib import Path
from run_all import load_data, get_X, relative_error, RESULTS_DIR

def run_pysr():
    print('=== PySR ===')
    from pysr import PySRRegressor
    
    train, val = load_data()
    spin_train, omega_r_train, omega_i_train = train
    spin_val, omega_r_val, omega_i_val = val
    
    ptype = 'log'
    X_train = get_X(spin_train, ptype)
    X_val = get_X(spin_val, ptype)
    
    model_r = PySRRegressor(
        niterations=50, binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "exp"],
        maxsize=15, populations=10, procs=4,
        loss="loss(prediction, target) = abs(prediction - target) / abs(target)",
        verbosity=0, random_state=42,
    )
    model_r.fit(X_train, omega_r_train)
    
    model_i = PySRRegressor(
        niterations=50, binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "exp"],
        maxsize=15, populations=10, procs=4,
        loss="loss(prediction, target) = abs(prediction - target) / abs(target)",
        verbosity=0, random_state=43,
    )
    model_i.fit(X_train, omega_i_train)
    
    pred_r = model_r.predict(X_val)
    pred_i = model_i.predict(X_val)
    
    err_r = relative_error(pred_r, omega_r_val)
    err_i = relative_error(pred_i, omega_i_val)
    loss = (err_r + err_i) / 2
    print(f'  Mean: {loss:.6f}')
    
    name = '23_pysr_log'
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({'model_r': model_r, 'model_i': model_i}, f)
    
    scorecard = {
        'approach': name, 'approach_number': 23, 'benchmark': 'ringdown', 'agent': 'kimi_k26',
        'parameterization': ptype, 'loss': loss,
        'loss_components': {'rel_error_omega_real': err_r, 'rel_error_omega_imag': err_i},
        'notes': 'PySR symbolic regression',
    }
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    with open(model_dir / 'train.py', 'w') as f:
        f.write('# train.py\n')
    with open(model_dir / 'predict.py', 'w') as f:
        f.write('# predict.py\n')

def run_gplearn():
    print('=== gplearn ===')
    from gplearn.genetic import SymbolicRegressor
    
    train, val = load_data()
    spin_train, omega_r_train, omega_i_train = train
    spin_val, omega_r_val, omega_i_val = val
    
    ptype = 'log'
    X_train = get_X(spin_train, ptype)
    X_val = get_X(spin_val, ptype)
    
    model_r = SymbolicRegressor(
        population_size=500, generations=15,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log'],
        metric='mse', parsimony_coefficient=0.001,
        verbose=0, random_state=42, n_jobs=2,
    )
    model_r.fit(X_train, omega_r_train)
    
    model_i = SymbolicRegressor(
        population_size=500, generations=15,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log'],
        metric='mse', parsimony_coefficient=0.001,
        verbose=0, random_state=43, n_jobs=2,
    )
    model_i.fit(X_train, omega_i_train)
    
    pred_r = model_r.predict(X_val)
    pred_i = model_i.predict(X_val)
    
    err_r = relative_error(pred_r, omega_r_val)
    err_i = relative_error(pred_i, omega_i_val)
    loss = (err_r + err_i) / 2
    print(f'  Mean: {loss:.6f}')
    
    name = '24_gplearn_log'
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({'model_r': model_r, 'model_i': model_i}, f)
    
    scorecard = {
        'approach': name, 'approach_number': 24, 'benchmark': 'ringdown', 'agent': 'kimi_k26',
        'parameterization': ptype, 'loss': loss,
        'loss_components': {'rel_error_omega_real': err_r, 'rel_error_omega_imag': err_i},
        'notes': 'gplearn symbolic regression',
    }
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    with open(model_dir / 'train.py', 'w') as f:
        f.write('# train.py\n')
    with open(model_dir / 'predict.py', 'w') as f:
        f.write('# predict.py\n')

if __name__ == '__main__':
    run_pysr()
    run_gplearn()
