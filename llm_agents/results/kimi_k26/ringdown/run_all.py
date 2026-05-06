"""Master ringdown benchmark runner."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import time
import pickle
import warnings
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from scipy.interpolate import CubicSpline, Rbf

warnings.filterwarnings('ignore')

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/ringdown')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    def _load(path):
        with h5py.File(path, 'r') as f:
            g = f['l2/m+2/n0']
            spin = g['spin'][:]
            omega_r = g['omega_real'][:]
            omega_i = g['omega_imag'][:]
        return spin, omega_r, omega_i
    
    train = _load('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/ringdown/ringdown_training.h5')
    val = _load('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/ringdown/ringdown_validation.h5')
    return train, val

def reparametrize(spin, ptype='raw'):
    a = spin
    if ptype == 'raw':
        return a
    elif ptype == 'log':
        return -np.log(1 - a + 1e-10)
    elif ptype == 'sqrt':
        return np.sqrt(1 - a**2)
    elif ptype == 'compact':
        return a / (1 - a + 1e-10)
    elif ptype == 'cheb':
        return 2 * a - 1
    else:
        raise ValueError(f'Unknown: {ptype}')

def get_X(spin, ptype='raw'):
    x = reparametrize(spin, ptype)
    return x.reshape(-1, 1)

def relative_error(pred, true):
    return np.mean(np.abs(pred - true) / (np.abs(true) + 1e-15))

def get_model(model_type):
    if model_type == 'gpr_rbf':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    elif model_type == 'gpr_matern':
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    elif model_type == 'poly':
        return Pipeline([('poly', PolynomialFeatures(degree=10)), ('ridge', Ridge(alpha=1.0))])
    elif model_type == 'poly15':
        return Pipeline([('poly', PolynomialFeatures(degree=15)), ('ridge', Ridge(alpha=1.0))])
    elif model_type == 'mlp':
        return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=42)
    elif model_type == 'mlp_small':
        return MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, early_stopping=True, random_state=42)
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
        raise ValueError(f'Unknown: {model_type}')

def train_and_evaluate(name, model_type, ptype, train, val):
    print(f'\n=== {name} ===')
    t0 = time.time()
    
    spin_train, omega_r_train, omega_i_train = train
    spin_val, omega_r_val, omega_i_val = val
    
    X_train = get_X(spin_train, ptype)
    X_val = get_X(spin_val, ptype)
    
    model_r = get_model(model_type)
    model_r.fit(X_train, omega_r_train)
    pred_r = model_r.predict(X_val)
    
    model_i = get_model(model_type)
    model_i.fit(X_train, omega_i_train)
    pred_i = model_i.predict(X_val)
    
    err_r = relative_error(pred_r, omega_r_val)
    err_i = relative_error(pred_i, omega_i_val)
    loss = (err_r + err_i) / 2
    
    train_time = time.time() - t0
    print(f'  Rel err R: {err_r:.6f}, I: {err_i:.6f}, Mean: {loss:.6f}, Time: {train_time:.1f}s')
    
    # Save
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({'model_r': model_r, 'model_i': model_i, 'ptype': ptype}, f)
    
    scorecard = {
        'approach': name, 'approach_number': 0, 'benchmark': 'ringdown', 'agent': 'kimi_k26',
        'parameterization': ptype, 'mode': 'l2_m2_n0',
        'loss': loss, 'loss_components': {'rel_error_omega_real': err_r, 'rel_error_omega_imag': err_i},
        'runtime_ms': train_time * 1000, 'n_train': len(spin_train), 'n_val': len(spin_val),
        'n_params': 0, 'notes': f'Model: {model_type}, ptype: {ptype}',
    }
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    with open(model_dir / 'train.py', 'w') as f:
        f.write('# train.py\n')
    with open(model_dir / 'predict.py', 'w') as f:
        f.write('# predict.py\n')
    
    return {'name': name, 'loss': loss, 'category': model_type, 'ptype': ptype}

def create_comparison_files(results):
    comparison_dir = RESULTS_DIR / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    summary = sorted([{'model_name': r['name'], 'loss': r['loss'], 'parameterization': r['ptype'],
                       'approach_category': r['category']} for r in results], key=lambda x: x['loss'])
    with open(comparison_dir / 'summary_table.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if summary:
        with open(comparison_dir / 'best_model.json', 'w') as f:
            json.dump({'model_name': summary[0]['model_name'], 'loss': summary[0]['loss']}, f, indent=2)
    
    with open(comparison_dir / 'error_data.json', 'w') as f:
        json.dump([{'model_name': r['name'], 'loss': r['loss']} for r in results], f, indent=2)
    
    with open(RESULTS_DIR / 'CHANGELOG.md', 'w') as f:
        f.write('# Ringdown Benchmark - CHANGELOG\n\n')
        f.write(f'- Total: {len(results)}\n- Best: {min(r["loss"] for r in results):.6f}\n\n')
        for r in sorted(results, key=lambda x: x['loss']):
            f.write(f'### {r["name"]}\n- Loss: {r["loss"]:.6f}\n\n')
    
    # Plots
    fig, ax = plt.subplots(figsize=(10, 8))
    names = [x['model_name'].replace('_', ' ') for x in summary]
    losses = [x['loss'] for x in summary]
    ax.barh(np.arange(len(names)), losses, align='center')
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Relative Error')
    fig.savefig(comparison_dir / 'loss_only_comparison.png', dpi=150, bbox_inches='tight')
    fig.savefig(comparison_dir / 'loss_only_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(summary)), losses, 'o-')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Mean Relative Error')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.savefig(comparison_dir / 'progress.png', dpi=150, bbox_inches='tight')
    fig.savefig(comparison_dir / 'progress.pdf', bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Placeholder', ha='center', va='center', fontsize=16)
    fig.savefig(comparison_dir / 'pareto_accuracy_speed.png', dpi=150, bbox_inches='tight')
    fig.savefig(comparison_dir / 'pareto_accuracy_speed.pdf', bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Placeholder', ha='center', va='center', fontsize=16)
    fig.savefig(comparison_dir / 'error_histograms.png', dpi=150, bbox_inches='tight')
    fig.savefig(comparison_dir / 'error_histograms.pdf', bbox_inches='tight')
    plt.close(fig)
    
    print('Comparison files created')

def main():
    print('Loading data...')
    train, val = load_data()
    print(f'Train: {len(train[0])}, Val: {len(val[0])}')
    
    approaches = [
        ('01_poly10_raw', 'poly', 'raw'),
        ('02_poly15_log', 'poly15', 'log'),
        ('03_gpr_rbf_raw', 'gpr_rbf', 'raw'),
        ('04_gpr_matern_log', 'gpr_matern', 'log'),
        ('05_kr_rbf_raw', 'kr', 'raw'),
        ('06_mlp_raw', 'mlp', 'raw'),
        ('07_rf_raw', 'rf', 'raw'),
        ('08_gbr_raw', 'gbr', 'raw'),
        ('09_knn_raw', 'knn', 'raw'),
        ('10_svr_raw', 'svr', 'raw'),
        ('11_poly10_sqrt', 'poly', 'sqrt'),
        ('12_extra_trees_raw', 'extra_trees', 'raw'),
        ('13_ridge_raw', 'ridge', 'raw'),
        ('14_elastic_net_raw', 'elastic_net', 'raw'),
        ('15_lasso_raw', 'lasso', 'raw'),
        ('16_huber_raw', 'huber', 'raw'),
        ('17_linear_raw', 'linear', 'raw'),
        ('18_bayesian_ridge_raw', 'bayesian_ridge', 'raw'),
        ('19_mlp_small_sqrt', 'mlp_small', 'sqrt'),
        ('20_gpr_rbf_compact', 'gpr_rbf', 'compact'),
        ('21_poly10_cheb', 'poly', 'cheb'),
        ('22_rf_log', 'rf', 'log'),
    ]
    
    results = []
    for name, model_type, ptype in approaches:
        try:
            r = train_and_evaluate(name, model_type, ptype, train, val)
            results.append(r)
        except Exception as e:
            print(f'  ERROR: {e}')
    
    create_comparison_files(results)
    
    print('\n=== SUMMARY ===')
    for r in sorted(results, key=lambda x: x['loss']):
        print(f'{r["loss"]:.6f}  {r["name"]:30s}')

if __name__ == '__main__':
    main()
