"""Add symbolic regression to dynamics benchmark."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import pickle
from pathlib import Path
from run_all import load_data, build_basis, get_params, normalize, rms_relative_error, RESULTS_DIR

def run_pysr():
    print('=== PySR ===')
    from pysr import PySRRegressor
    
    train, val = load_data()
    basis_data = build_basis(train, val, n_basis=10, target_npts=500)
    
    ptype = 'raw'
    X_train = get_params(train, ptype)
    X_val = get_params(val, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize(X_train, X_val)
    
    models = []
    for i in range(basis_data['coeffs_train'].shape[1]):
        model = PySRRegressor(
            niterations=50, binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "exp"],
            maxsize=15, populations=10, procs=4,
            loss="loss(prediction, target) = abs(prediction - target)",
            verbosity=0, random_state=42,
        )
        model.fit(Xn_train, basis_data['coeffs_train'][:, i])
        models.append(model)
    
    losses = []
    for i, s in enumerate(val):
        coeffs_pred = np.array([m.predict(Xn_val[i:i+1])[0] for m in models])
        x_pred = basis_data['mean_x'] + np.dot(coeffs_pred, basis_data['basis'].T)
        x_pred_target = np.interp(s['t'], basis_data['t_common'], x_pred)
        loss = rms_relative_error(x_pred_target, s['x'])
        losses.append(loss)
    
    mean_loss = float(np.mean(losses))
    print(f'  Mean loss: {mean_loss:.6f}')
    
    name = '23_svd_pysr_raw'
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({'models': models, 'mean_x': mean_x, 'std_x': std_x}, f)
    
    scorecard = {
        'approach': name, 'approach_number': 23, 'benchmark': 'dynamics', 'agent': 'kimi_k26',
        'parameterization': ptype, 'loss': mean_loss,
        'loss_components': {'rms_relative_error_x': mean_loss},
        'notes': 'PySR on SVD coefficients',
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
    basis_data = build_basis(train, val, n_basis=8, target_npts=500)
    
    ptype = 'raw'
    X_train = get_params(train, ptype)
    X_val = get_params(val, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize(X_train, X_val)
    
    models = []
    for i in range(basis_data['coeffs_train'].shape[1]):
        est = SymbolicRegressor(
            population_size=500, generations=15,
            function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log'],
            metric='mse', parsimony_coefficient=0.001,
            verbose=0, random_state=42, n_jobs=2,
        )
        est.fit(Xn_train, basis_data['coeffs_train'][:, i])
        models.append(est)
    
    losses = []
    for i, s in enumerate(val):
        coeffs_pred = np.array([m.predict(Xn_val[i:i+1])[0] for m in models])
        x_pred = basis_data['mean_x'] + np.dot(coeffs_pred, basis_data['basis'].T)
        x_pred_target = np.interp(s['t'], basis_data['t_common'], x_pred)
        loss = rms_relative_error(x_pred_target, s['x'])
        losses.append(loss)
    
    mean_loss = float(np.mean(losses))
    print(f'  Mean loss: {mean_loss:.6f}')
    
    name = '24_svd_gplearn_raw'
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({'models': models, 'mean_x': mean_x, 'std_x': std_x}, f)
    
    scorecard = {
        'approach': name, 'approach_number': 24, 'benchmark': 'dynamics', 'agent': 'kimi_k26',
        'parameterization': ptype, 'loss': mean_loss,
        'loss_components': {'rms_relative_error_x': mean_loss},
        'notes': 'gplearn on SVD coefficients',
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
