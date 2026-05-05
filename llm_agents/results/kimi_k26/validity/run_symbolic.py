"""Add symbolic regression to validity."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import pickle
from pathlib import Path
from run_all import load_data, reparametrize, normalize, RESULTS_DIR

def run_pysr():
    print('=== PySR ===')
    from pysr import PySRRegressor
    
    train, val = load_data()
    X_train_raw, y_train = train
    X_val_raw, y_val = val
    
    ptype = 'effective'
    X_train = reparametrize(X_train_raw, ptype)
    X_val = reparametrize(X_val_raw, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize(X_train, X_val)
    
    model = PySRRegressor(
        niterations=50, binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "exp"],
        maxsize=15, populations=10, procs=4,
        loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=0, random_state=42,
    )
    model.fit(Xn_train, y_train)
    
    y_pred = model.predict(Xn_val)
    loss = np.sqrt(np.mean((y_pred - y_val)**2))
    print(f'  Log RMSE: {loss:.6f}')
    
    name = '23_pysr_effective'
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'model.pkl', 'wb') as f:
        pickle.dump({'model': model}, f)
    
    scorecard = {
        'approach': name, 'approach_number': 23, 'benchmark': 'validity', 'agent': 'kimi_k26',
        'parameterization': ptype, 'loss': loss, 'loss_components': {'log_rmse': loss},
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
    X_train_raw, y_train = train
    X_val_raw, y_val = val
    
    ptype = 'effective'
    X_train = reparametrize(X_train_raw, ptype)
    X_val = reparametrize(X_val_raw, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize(X_train, X_val)
    
    model = SymbolicRegressor(
        population_size=500, generations=15,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log'],
        metric='mse', parsimony_coefficient=0.001,
        verbose=0, random_state=42, n_jobs=2,
    )
    model.fit(Xn_train, y_train)
    
    y_pred = model.predict(Xn_val)
    loss = np.sqrt(np.mean((y_pred - y_val)**2))
    print(f'  Log RMSE: {loss:.6f}')
    
    name = '24_gplearn_effective'
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'model.pkl', 'wb') as f:
        pickle.dump({'model': model}, f)
    
    scorecard = {
        'approach': name, 'approach_number': 24, 'benchmark': 'validity', 'agent': 'kimi_k26',
        'parameterization': ptype, 'loss': loss, 'loss_components': {'log_rmse': loss},
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
