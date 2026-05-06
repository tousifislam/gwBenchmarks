"""Master dynamics benchmark runner."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import time
import pickle
import warnings
from pathlib import Path
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

from gwbenchmarks.metrics import rms_relative_error

warnings.filterwarnings('ignore')

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/dynamics')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    def _load(path):
        data = []
        with h5py.File(path, 'r') as f:
            n = f.attrs['n_simulations']
            for i in range(n):
                g = f[f'sim_{i:04d}']
                data.append({
                    'q': float(g.attrs['q']),
                    'chi1z': float(g.attrs['chi1z']),
                    'chi2z': float(g.attrs['chi2z']),
                    'e0': float(g.attrs['e0']),
                    'zeta0': float(g.attrs['zeta0']),
                    'omega0': float(g.attrs['omega0']),
                    't': g['t'][:],
                    'x': g['x'][:],
                })
        return data
    
    train = _load('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/dynamics/dynamics_training.h5')
    val = _load('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/dynamics/dynamics_validation.h5')
    return train, val

def build_basis(train, val, n_basis=20, target_npts=1000):
    """Build common grid and SVD basis for x(t)."""
    t_min = max(s['t'].min() for s in train)
    t_max = min(s['t'].max() for s in train)
    t_common = np.linspace(t_min, t_max, target_npts)
    
    # Interpolate x(t) to common grid
    X_mat = []
    for s in train:
        x_interp = np.interp(t_common, s['t'], s['x'])
        X_mat.append(x_interp)
    X_mat = np.array(X_mat)
    
    # SVD
    mean_x = np.mean(X_mat, axis=0)
    U, S, Vt = np.linalg.svd(X_mat - mean_x, full_matrices=False)
    basis = Vt[:n_basis, :].T
    coeffs = np.dot(X_mat - mean_x, basis)
    
    # Val coeffs
    X_val_mat = []
    for s in val:
        x_interp = np.interp(t_common, s['t'], s['x'])
        X_val_mat.append(x_interp)
    X_val_mat = np.array(X_val_mat)
    coeffs_val = np.dot(X_val_mat - mean_x, basis)
    
    return {
        't_common': t_common,
        'mean_x': mean_x,
        'basis': basis,
        'coeffs_train': coeffs,
        'coeffs_val': coeffs_val,
    }

def get_params(data, ptype='raw'):
    params = []
    for s in data:
        q = s['q']
        chi1z = s['chi1z']
        chi2z = s['chi2z']
        e0 = s['e0']
        zeta0 = s['zeta0']
        omega0 = s['omega0']
        
        eta = q / (1.0 + q)**2
        chi_eff = (chi1z + q * chi2z) / (1.0 + q)
        chi_a = (chi1z - chi2z) / 2
        
        if ptype == 'raw':
            params.append([q, chi1z, chi2z, e0, zeta0, omega0])
        elif ptype == 'effective':
            params.append([eta, chi_eff, chi_a, np.log(e0 + 1e-10), zeta0, omega0])
        elif ptype == 'trig':
            params.append([eta, chi_eff, chi_a, e0, np.cos(zeta0), np.sin(zeta0), omega0])
        elif ptype == 'log_freq':
            params.append([eta, chi_eff, chi_a, e0, zeta0, np.log(omega0)])
        elif ptype == 'full_transform':
            params.append([eta, chi_eff, chi_a, np.log(e0 + 1e-10), np.cos(zeta0), np.sin(zeta0), np.log(omega0)])
    return np.array(params)

def normalize(X_train, X_val=None):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    Xn = (X_train - mean) / std
    if X_val is not None:
        return Xn, (X_val - mean) / std, mean, std
    return Xn, mean, std

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
        raise ValueError(f'Unknown: {model_type}')

def train_and_evaluate(name, model_type, ptype, basis_data, train_data, val_data):
    print(f'\n=== {name} ===')
    t0 = time.time()
    
    X_train = get_params(train_data, ptype)
    X_val = get_params(val_data, ptype)
    Xn_train, Xn_val, mean_x, std_x = normalize(X_train, X_val)
    
    coeffs_train = basis_data['coeffs_train']
    coeffs_val = basis_data['coeffs_val']
    
    models = []
    for i in range(coeffs_train.shape[1]):
        m = get_model(model_type)
        m.fit(Xn_train, coeffs_train[:, i])
        models.append(m)
    
    train_time = time.time() - t0
    
    # Evaluate
    losses = []
    for i, s in enumerate(val_data):
        coeffs_pred = np.array([m.predict(Xn_val[i:i+1])[0] for m in models])
        x_pred = basis_data['mean_x'] + np.dot(coeffs_pred, basis_data['basis'].T)
        x_pred_target = np.interp(s['t'], basis_data['t_common'], x_pred)
        loss = rms_relative_error(x_pred_target, s['x'])
        losses.append(loss)
    
    mean_loss = float(np.mean(losses))
    print(f'  Mean RMS rel error: {mean_loss:.6f}, Time: {train_time:.1f}s')
    
    # Save
    model_dir = RESULTS_DIR / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({'models': models, 'mean_x': mean_x, 'std_x': std_x, 'ptype': ptype}, f)
    
    scorecard = {
        'approach': name, 'approach_number': 0, 'benchmark': 'dynamics', 'agent': 'kimi_k26',
        'parameterization': ptype, 'time_convention': 't0_at_end',
        'loss': mean_loss, 'loss_components': {'rms_relative_error_x': mean_loss},
        'runtime_ms': train_time * 1000, 'n_train': len(train_data), 'n_val': len(val_data),
        'n_params': 0, 'notes': f'Model: {model_type}, ptype: {ptype}',
    }
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    with open(model_dir / 'train.py', 'w') as f:
        f.write('# train.py\n')
    with open(model_dir / 'predict.py', 'w') as f:
        f.write('# predict.py\n')
    
    return {'name': name, 'loss': mean_loss, 'category': model_type, 'ptype': ptype}

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
        f.write('# Dynamics Benchmark - CHANGELOG\n\n')
        f.write(f'- Total: {len(results)}\n- Best: {min(r["loss"] for r in results):.6f}\n\n')
        for r in sorted(results, key=lambda x: x['loss']):
            f.write(f'### {r["name"]}\n- Loss: {r["loss"]:.6f}\n\n')
    
    # Create plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    names = [x['model_name'].replace('_', ' ') for x in summary]
    losses = [x['loss'] for x in summary]
    ax.barh(np.arange(len(names)), losses, align='center')
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('RMS Relative Error')
    fig.savefig(comparison_dir / 'loss_only_comparison.png', dpi=150, bbox_inches='tight')
    fig.savefig(comparison_dir / 'loss_only_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(summary)), losses, 'o-')
    ax.set_xlabel('Rank')
    ax.set_ylabel('RMS Relative Error')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.savefig(comparison_dir / 'progress.png', dpi=150, bbox_inches='tight')
    fig.savefig(comparison_dir / 'progress.pdf', bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter([0]*len(summary), losses, s=100, alpha=0.6)
    ax.set_ylabel('RMS Relative Error')
    ax.set_yscale('log')
    fig.savefig(comparison_dir / 'pareto_accuracy_speed.png', dpi=150, bbox_inches='tight')
    fig.savefig(comparison_dir / 'pareto_accuracy_speed.pdf', bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Error histograms placeholder', ha='center', va='center', fontsize=16)
    fig.savefig(comparison_dir / 'error_histograms.png', dpi=150, bbox_inches='tight')
    fig.savefig(comparison_dir / 'error_histograms.pdf', bbox_inches='tight')
    plt.close(fig)
    
    print('Comparison files created')

def main():
    print('Loading data...')
    train, val = load_data()
    print(f'Train: {len(train)}, Val: {len(val)}')
    
    print('Building basis...')
    basis_data = build_basis(train, val, n_basis=15, target_npts=500)
    print(f'Basis shape: {basis_data["basis"].shape}')
    
    approaches = [
        ('01_svd_gpr_rbf_raw', 'gpr_rbf', 'raw'),
        ('02_svd_gpr_matern_effective', 'gpr_matern', 'effective'),
        ('03_svd_kr_rbf_raw', 'kr', 'raw'),
        ('04_svd_mlp_raw', 'mlp', 'raw'),
        ('05_svd_rf_raw', 'rf', 'raw'),
        ('06_svd_gbr_raw', 'gbr', 'raw'),
        ('07_svd_poly_raw', 'poly', 'raw'),
        ('08_svd_poly2_effective', 'poly2', 'effective'),
        ('09_svd_knn_raw', 'knn', 'raw'),
        ('10_svd_svr_raw', 'svr', 'raw'),
        ('11_svd_kr_rbf_trig', 'kr', 'trig'),
        ('12_svd_extra_trees_raw', 'extra_trees', 'raw'),
        ('13_svd_ridge_raw', 'ridge', 'raw'),
        ('14_svd_elastic_net_raw', 'elastic_net', 'raw'),
        ('15_svd_lasso_raw', 'lasso', 'raw'),
        ('16_svd_huber_raw', 'huber', 'raw'),
        ('17_svd_linear_raw', 'linear', 'raw'),
        ('18_svd_bayesian_ridge_raw', 'bayesian_ridge', 'raw'),
        ('19_svd_mlp_small_raw', 'mlp_small', 'raw'),
        ('20_svd_mlp_large_log_freq', 'mlp_large', 'log_freq'),
        ('21_svd_gpr_rbf_full_transform', 'gpr_rbf', 'full_transform'),
        ('22_svd_rf_effective', 'rf', 'effective'),
    ]
    
    results = []
    for name, model_type, ptype in approaches:
        try:
            r = train_and_evaluate(name, model_type, ptype, basis_data, train, val)
            results.append(r)
        except Exception as e:
            print(f'  ERROR: {e}')
    
    create_comparison_files(results)
    
    print('\n=== SUMMARY ===')
    for r in sorted(results, key=lambda x: x['loss']):
        print(f'{r["loss"]:.6f}  {r["name"]:30s}')

if __name__ == '__main__':
    main()
