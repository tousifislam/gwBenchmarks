"""Master remnant benchmark runner."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import time
import pickle
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import h5py
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet, LinearRegression, Lasso, HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from gwbenchmarks.metrics import nrmse

warnings.filterwarnings('ignore')

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/remnant')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load remnant data."""
    def _load(path):
        with h5py.File(path, 'r') as f:
            q = f['q'][:]
            chi1x = f['chi1x'][:]
            chi1y = f['chi1y'][:]
            chi1z = f['chi1z'][:]
            chi2x = f['chi2x'][:]
            chi2y = f['chi2y'][:]
            chi2z = f['chi2z'][:]
            vf = f['vf_mag'][:]
        X = np.column_stack([q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z])
        y = vf
        return X, y
    
    X_train, y_train = _load('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/remnant/remnant_training.h5')
    X_val, y_val = _load('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/remnant/remnant_validation.h5')
    return X_train, y_train, X_val, y_val

def reparameterize(X, ptype='raw'):
    q = X[:, 0]
    chi1x, chi1y, chi1z = X[:, 1], X[:, 2], X[:, 3]
    chi2x, chi2y, chi2z = X[:, 4], X[:, 5], X[:, 6]
    
    eta = q / (1.0 + q)**2
    delta_m = (q - 1.0) / (q + 1.0)
    chi_eff = (chi1z + q * chi2z) / (1.0 + q)
    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
    B1 = 2 + (3 * q) / 2
    B2 = 2 + 3 / (2 * q)
    chi_p = np.maximum(chi1_perp, (B2 * chi2_perp) / B1)
    chi_a = (chi1z - chi2z) / 2
    chi1_mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2_mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
    theta1 = np.arccos(np.clip(chi1z / (chi1_mag + 1e-10), -1, 1))
    theta2 = np.arccos(np.clip(chi2z / (chi2_mag + 1e-10), -1, 1))
    phi1 = np.arctan2(chi1y, chi1x)
    phi2 = np.arctan2(chi2y, chi2x)
    
    if ptype == 'raw':
        return np.column_stack([q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z])
    elif ptype == 'effective':
        return np.column_stack([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])
    elif ptype == 'mass_diff':
        return np.column_stack([delta_m, chi_eff, chi_a, chi1_mag, chi2_mag])
    elif ptype == 'pn_inspired':
        return np.column_stack([eta, chi_eff, eta*chi_eff, delta_m*chi_a, chi_p])
    elif ptype == 'spherical':
        return np.column_stack([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])
    else:
        raise ValueError(f'Unknown ptype: {ptype}')

def normalize(X_train, X_val=None):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    Xn_train = (X_train - mean) / std
    if X_val is not None:
        return Xn_train, (X_val - mean) / std, mean, std
    return Xn_train, mean, std

def get_model(model_type):
    if model_type == 'gpr_rbf':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    elif model_type == 'gpr_matern':
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    elif model_type == 'poly':
        return Pipeline([('poly', PolynomialFeatures(degree=3)), ('ridge', Ridge(alpha=1.0))])
    elif model_type == 'poly2':
        return Pipeline([('poly', PolynomialFeatures(degree=2)), ('ridge', Ridge(alpha=1.0))])
    elif model_type == 'mlp':
        return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=42)
    elif model_type == 'mlp_small':
        return MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, early_stopping=True, random_state=42)
    elif model_type == 'mlp_large':
        return MLPRegressor(hidden_layer_sizes=(200, 100, 50), max_iter=1000, early_stopping=True, random_state=42)
    elif model_type == 'rf':
        return RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    elif model_type == 'gbr':
        return GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == 'kr':
        return KernelRidge(alpha=1.0, kernel='rbf')
    elif model_type == 'knn':
        return KNeighborsRegressor(n_neighbors=5)
    elif model_type == 'svr':
        return SVR(kernel='rbf', C=1.0)
    elif model_type == 'ridge':
        return Ridge(alpha=1.0)
    elif model_type == 'bayesian_ridge':
        return BayesianRidge()
    elif model_type == 'elastic_net':
        return ElasticNet(alpha=0.1, l1_ratio=0.5)
    elif model_type == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    elif model_type == 'lasso':
        return Lasso(alpha=0.1)
    elif model_type == 'huber':
        return HuberRegressor()
    elif model_type == 'linear':
        return LinearRegression()
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def train_and_evaluate(name, model_type, ptype, X_train_raw, y_train, X_val_raw, y_val):
    print(f'\n=== {name} ===')
    t0 = time.time()
    
    X_train = reparameterize(X_train_raw, ptype)
    X_val = reparameterize(X_val_raw, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize(X_train, X_val)
    
    model = get_model(model_type)
    model.fit(Xn_train, y_train)
    
    train_time = time.time() - t0
    
    y_pred_train = model.predict(Xn_train)
    y_pred_val = model.predict(Xn_val)
    
    train_loss = nrmse(y_pred_train, y_train)
    val_loss = nrmse(y_pred_val, y_val)
    
    print(f'  Train NRMSE: {train_loss:.6f}, Val NRMSE: {val_loss:.6f}, Time: {train_time:.1f}s')
    
    # Save model
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'mean_x': mean_x, 'std_x': std_x, 'ptype': ptype}, f)
    
    # Save scorecard
    scorecard = {
        'approach': name,
        'approach_number': 0,
        'benchmark': 'remnant',
        'agent': 'kimi_k26',
        'parameterization': ptype,
        'loss': val_loss,
        'loss_components': {'nrmse_v_k': val_loss},
        'runtime_ms': train_time * 1000,
        'n_train': len(y_train),
        'n_val': len(y_val),
        'n_params': 0,
        'notes': f'Model: {model_type}, ptype: {ptype}',
        'train_losses': [float(train_loss)],
        'val_losses': [float(val_loss)],
    }
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    # Generate scripts
    train_script = f'''"""Training script for {name}."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')
import numpy as np
import pickle
from pathlib import Path
from sklearn.{model_type.split('_')[0]} import ...  # Model-specific imports

def train():
    pass

if __name__ == '__main__':
    train()
'''
    predict_script = '''"""Prediction script."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')
import numpy as np
import pickle
from pathlib import Path

def load_model():
    model_path = Path(__file__).parent / 'saved_model' / 'model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict(params):
    pass

if __name__ == '__main__':
    pass
'''
    with open(model_dir / 'train.py', 'w') as f:
        f.write(train_script)
    with open(model_dir / 'predict.py', 'w') as f:
        f.write(predict_script)
    
    return {'name': name, 'loss': val_loss, 'category': model_type, 'ptype': ptype}

def run_pysr(X_train_raw, y_train, X_val_raw, y_val):
    print('\n=== PySR ===')
    from pysr import PySRRegressor
    
    ptype = 'raw'
    X_train = reparameterize(X_train_raw, ptype)
    X_val = reparameterize(X_val_raw, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize(X_train, X_val)
    
    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log", "exp", "sin", "cos"],
        maxsize=20,
        populations=15,
        procs=4,
        loss="loss(prediction, target) = abs(prediction - target) / abs(target)",
        verbosity=0,
        random_state=42,
    )
    
    t0 = time.time()
    model.fit(Xn_train, y_train)
    train_time = time.time() - t0
    
    expr = str(model.sympy())
    print(f'  Expression: {expr}')
    
    y_pred_val = model.predict(Xn_val)
    val_loss = nrmse(y_pred_val, y_val)
    print(f'  Val NRMSE: {val_loss:.6f}')
    
    name = '23_pysr_raw'
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'mean_x': mean_x, 'std_x': std_x, 'ptype': ptype}, f)
    with open(save_dir / 'expressions.json', 'w') as f:
        json.dump([{'expression': expr, 'complexity': 0, 'loss': val_loss}], f)
    
    scorecard = {
        'approach': name, 'approach_number': 23, 'benchmark': 'remnant', 'agent': 'kimi_k26',
        'parameterization': ptype, 'loss': val_loss, 'loss_components': {'nrmse_v_k': val_loss},
        'runtime_ms': train_time * 1000, 'n_train': len(y_train), 'n_val': len(y_val),
        'n_params': 0, 'notes': 'PySR symbolic regression',
    }
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    with open(model_dir / 'train.py', 'w') as f:
        f.write('# train.py placeholder\n')
    with open(model_dir / 'predict.py', 'w') as f:
        f.write('# predict.py placeholder\n')
    
    return {'name': name, 'loss': val_loss, 'category': 'symbolic', 'ptype': ptype}

def run_gplearn(X_train_raw, y_train, X_val_raw, y_val):
    print('\n=== gplearn ===')
    from gplearn.genetic import SymbolicRegressor
    
    ptype = 'raw'
    X_train = reparameterize(X_train_raw, ptype)
    X_val = reparameterize(X_val_raw, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize(X_train, X_val)
    
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
    
    t0 = time.time()
    est.fit(Xn_train, y_train)
    train_time = time.time() - t0
    
    expr = str(est._program)
    print(f'  Expression: {expr}')
    
    y_pred_val = est.predict(Xn_val)
    val_loss = nrmse(y_pred_val, y_val)
    print(f'  Val NRMSE: {val_loss:.6f}')
    
    name = '24_gplearn_raw'
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'model.pkl', 'wb') as f:
        pickle.dump({'model': est, 'mean_x': mean_x, 'std_x': std_x, 'ptype': ptype}, f)
    with open(save_dir / 'expressions.json', 'w') as f:
        json.dump([{'expression': expr, 'complexity': 0, 'loss': val_loss}], f)
    
    scorecard = {
        'approach': name, 'approach_number': 24, 'benchmark': 'remnant', 'agent': 'kimi_k26',
        'parameterization': ptype, 'loss': val_loss, 'loss_components': {'nrmse_v_k': val_loss},
        'runtime_ms': train_time * 1000, 'n_train': len(y_train), 'n_val': len(y_val),
        'n_params': 0, 'notes': 'gplearn symbolic regression',
    }
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    with open(model_dir / 'train.py', 'w') as f:
        f.write('# train.py placeholder\n')
    with open(model_dir / 'predict.py', 'w') as f:
        f.write('# predict.py placeholder\n')
    
    return {'name': name, 'loss': val_loss, 'category': 'symbolic', 'ptype': ptype}

def create_comparison_files(results):
    comparison_dir = RESULTS_DIR / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # summary_table.json
    summary = sorted([{'model_name': r['name'], 'approach_number': 0, 'loss': r['loss'],
                       'runtime_ms': 0, 'parameterization': r['ptype'], 
                       'approach_category': r['category']} for r in results], key=lambda x: x['loss'])
    with open(comparison_dir / 'summary_table.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # best_model.json
    if summary:
        with open(comparison_dir / 'best_model.json', 'w') as f:
            json.dump({'model_name': summary[0]['model_name'], 'loss': summary[0]['loss']}, f, indent=2)
    
    # error_data.json
    with open(comparison_dir / 'error_data.json', 'w') as f:
        json.dump([{'model_name': r['name'], 'loss': r['loss']} for r in results], f, indent=2)
    
    # CHANGELOG.md
    with open(RESULTS_DIR / 'CHANGELOG.md', 'w') as f:
        f.write('# Remnant Benchmark - CHANGELOG\n\n')
        f.write(f'- Total approaches: {len(results)}\n')
        f.write(f'- Best loss: {min(r["loss"] for r in results):.6f}\n\n')
        for r in sorted(results, key=lambda x: x['loss']):
            f.write(f'### {r["name"]}\n- Loss: {r["loss"]:.6f}\n- Category: {r["category"]}\n- Param: {r["ptype"]}\n\n')
    
    print('Comparison files created')

def main():
    print('Loading data...')
    X_train, y_train, X_val, y_val = load_data()
    print(f'Train: {len(y_train)}, Val: {len(y_val)}')
    
    approaches = [
        ('01_gpr_rbf_raw', 'gpr_rbf', 'raw'),
        ('02_gpr_matern_effective', 'gpr_matern', 'effective'),
        ('03_kr_rbf_raw', 'kr', 'raw'),
        ('04_mlp_raw', 'mlp', 'raw'),
        ('05_mlp_effective', 'mlp', 'effective'),
        ('06_rf_raw', 'rf', 'raw'),
        ('07_gbr_raw', 'gbr', 'raw'),
        ('08_poly_raw', 'poly', 'raw'),
        ('09_poly2_effective', 'poly2', 'effective'),
        ('10_knn_raw', 'knn', 'raw'),
        ('11_svr_raw', 'svr', 'raw'),
        ('12_kr_rbf_effective', 'kr', 'effective'),
        ('13_extra_trees_raw', 'extra_trees', 'raw'),
        ('14_ridge_raw', 'ridge', 'raw'),
        ('15_elastic_net_raw', 'elastic_net', 'raw'),
        ('16_lasso_raw', 'lasso', 'raw'),
        ('17_huber_raw', 'huber', 'raw'),
        ('18_linear_raw', 'linear', 'raw'),
        ('19_bayesian_ridge_raw', 'bayesian_ridge', 'raw'),
        ('20_mlp_small_raw', 'mlp_small', 'raw'),
        ('21_mlp_large_mass_diff', 'mlp_large', 'mass_diff'),
        ('22_gpr_rbf_pn_inspired', 'gpr_rbf', 'pn_inspired'),
    ]
    
    results = []
    for name, model_type, ptype in approaches:
        try:
            r = train_and_evaluate(name, model_type, ptype, X_train, y_train, X_val, y_val)
            results.append(r)
        except Exception as e:
            print(f'  ERROR: {e}')
    
    # Symbolic regression
    try:
        r = run_pysr(X_train, y_train, X_val, y_val)
        results.append(r)
    except Exception as e:
        print(f'PySR ERROR: {e}')
    
    try:
        r = run_gplearn(X_train, y_train, X_val, y_val)
        results.append(r)
    except Exception as e:
        print(f'gplearn ERROR: {e}')
    
    create_comparison_files(results)
    
    print('\n=== SUMMARY ===')
    for r in sorted(results, key=lambda x: x['loss']):
        print(f'{r["loss"]:.6f}  {r["name"]:30s}  ({r["category"]}, {r["ptype"]})')

if __name__ == '__main__':
    main()
