"""Comprehensive waveform benchmark runner.
Trains and evaluates multiple modeling approaches.
"""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import time
import pickle
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

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

from gwbenchmarks.metrics import mean_fd_mismatch

warnings.filterwarnings('ignore')

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/waveform')
BASIS_DIR = RESULTS_DIR / 'saved_basis'

def load_basis():
    return {
        't_common': np.load(BASIS_DIR / 't_common.npy'),
        'mean_real': np.load(BASIS_DIR / 'mean_real.npy'),
        'mean_imag': np.load(BASIS_DIR / 'mean_imag.npy'),
        'basis_real': np.load(BASIS_DIR / 'basis_real.npy'),
        'basis_imag': np.load(BASIS_DIR / 'basis_imag.npy'),
        'coeffs_real_train': np.load(BASIS_DIR / 'coeffs_real_train.npy'),
        'coeffs_imag_train': np.load(BASIS_DIR / 'coeffs_imag_train.npy'),
        'coeffs_real_val': np.load(BASIS_DIR / 'coeffs_real_val.npy'),
        'coeffs_imag_val': np.load(BASIS_DIR / 'coeffs_imag_val.npy'),
        'train_params': np.load(BASIS_DIR / 'train_params.npy'),
        'val_params': np.load(BASIS_DIR / 'val_params.npy'),
    }

def load_ref_sims():
    import h5py
    def load(h5_path):
        data = []
        with h5py.File(h5_path, 'r') as f:
            n = f.attrs['n_simulations']
            for i in range(n):
                g = f[f'sim_{i:04d}']
                t = g['t'][:]
                h22 = g['h22_real'][:] + 1j * g['h22_imag'][:]
                data.append({'t': t, 'h22': h22, 'dt': float(t[1] - t[0]) if len(t) > 1 else 1.0})
        return data
    train = load('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/waveform/waveform_training.h5')
    val = load('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/waveform/waveform_validation.h5')
    return train, val

def get_param_reparameterization(params, ptype='raw'):
    q = params[:, 0]
    chi1x, chi1y, chi1z = params[:, 1], params[:, 2], params[:, 3]
    chi2x, chi2y, chi2z = params[:, 4], params[:, 5], params[:, 6]
    omega0 = params[:, 7]
    
    eta = q / (1.0 + q)**2
    delta_m = (q - 1.0) / (q + 1.0)
    chi_eff = (chi1z + q * chi2z) / (1.0 + q)
    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
    B1 = 2 + (3 * q) / 2
    B2 = 2 + 3 / (2 * q)
    chi_p = np.maximum(chi1_perp, (B2 * chi2_perp) / B1)
    
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
        return np.column_stack([delta_m, chi_eff, chi_p, chi1_mag, chi2_mag, phi1, phi2])
    elif ptype == 'spherical':
        return np.column_stack([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])
    elif ptype == 'omega0':
        return np.column_stack([q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega0])
    else:
        raise ValueError(f'Unknown param type: {ptype}')

def normalize_features(X_train, X_val=None):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    Xn_train = (X_train - mean) / std
    if X_val is not None:
        Xn_val = (X_val - mean) / std
        return Xn_train, Xn_val, mean, std
    return Xn_train, mean, std

def fit_coefficient_models(X_train, coeffs_train, model_type):
    """Fit models to each coefficient."""
    n_coeffs = coeffs_train.shape[1]
    models = []
    for i in range(n_coeffs):
        y = coeffs_train[:, i]
        if model_type == 'gpr_rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
        elif model_type == 'gpr_matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
        elif model_type == 'poly':
            model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('ridge', Ridge(alpha=1.0))])
        elif model_type == 'mlp':
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=42)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        elif model_type == 'gbr':
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == 'kr':
            model = KernelRidge(alpha=1.0, kernel='rbf')
        elif model_type == 'knn':
            model = KNeighborsRegressor(n_neighbors=5)
        elif model_type == 'svr':
            model = SVR(kernel='rbf', C=1.0)
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'bayesian_ridge':
            model = BayesianRidge()
        elif model_type == 'elastic_net':
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_type == 'extra_trees':
            model = ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.1)
        elif model_type == 'huber':
            model = HuberRegressor()
        elif model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'mlp_small':
            model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, early_stopping=True, random_state=42)
        elif model_type == 'mlp_large':
            model = MLPRegressor(hidden_layer_sizes=(200, 100, 50), max_iter=1000, early_stopping=True, random_state=42)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        
        model.fit(X_train, y)
        models.append(model)
    return models

def predict_coefficients(models, X):
    """Predict coefficients using fitted models."""
    n_samples = X.shape[0]
    n_coeffs = len(models)
    preds = np.zeros((n_samples, n_coeffs))
    for i, model in enumerate(models):
        preds[:, i] = model.predict(X)
    return preds

def reconstruct_waveform(coeffs_real, coeffs_imag, basis_data, t_target):
    mean_real = basis_data['mean_real']
    mean_imag = basis_data['mean_imag']
    basis_real = basis_data['basis_real'][:, :coeffs_real.shape[0]]
    basis_imag = basis_data['basis_imag'][:, :coeffs_imag.shape[0]]
    t_common = basis_data['t_common']
    
    h_common_real = mean_real + np.dot(coeffs_real, basis_real.T)
    h_common_imag = mean_imag + np.dot(coeffs_imag, basis_imag.T)
    h_common = h_common_real + 1j * h_common_imag
    
    h_target = np.interp(t_target, t_common, h_common.real) + 1j * np.interp(t_target, t_common, h_common.imag)
    return h_target

def compute_single_mismatch(args):
    """Helper for parallel mismatch computation."""
    i, h_pred, h_ref, dt = args
    try:
        return mean_fd_mismatch(h_pred, h_ref, dt)
    except Exception as e:
        return 1.0

def evaluate_approach(models_real, models_imag, mean_x, std_x, basis_data, val_sims, 
                      val_params, ptype, n_basis_use, use_parallel=True):
    """Evaluate an approach on validation data."""
    X_val = get_param_reparameterization(val_params, ptype)
    Xn_val = (X_val - mean_x) / std_x
    
    coeffs_real_pred = predict_coefficients(models_real, Xn_val)[:, :n_basis_use]
    coeffs_imag_pred = predict_coefficients(models_imag, Xn_val)[:, :n_basis_use]
    
    losses = []
    args_list = []
    for i in range(len(val_sims)):
        h_pred = reconstruct_waveform(coeffs_real_pred[i], coeffs_imag_pred[i], basis_data, val_sims[i]['t'])
        args_list.append((i, h_pred, val_sims[i]['h22'], val_sims[i]['dt']))
    
    if use_parallel and len(args_list) > 1:
        n_procs = min(cpu_count(), 8)
        with Pool(n_procs) as pool:
            losses = pool.map(compute_single_mismatch, args_list)
    else:
        for args in args_list:
            losses.append(compute_single_mismatch(args))
    
    return losses

def train_and_save_approach(approach_name, model_type, ptype, n_basis_use, basis_data, 
                            train_sims, train_params, val_sims, val_params):
    """Train an approach and save it."""
    print(f'\n=== Training {approach_name} ===')
    t0 = time.time()
    
    X_train = get_param_reparameterization(train_params, ptype)
    Xn_train, mean_x, std_x = normalize_features(X_train)
    
    coeffs_real_train = basis_data['coeffs_real_train'][:, :n_basis_use]
    coeffs_imag_train = basis_data['coeffs_imag_train'][:, :n_basis_use]
    
    models_real = fit_coefficient_models(Xn_train, coeffs_real_train, model_type)
    models_imag = fit_coefficient_models(Xn_train, coeffs_imag_train, model_type)
    
    train_time = time.time() - t0
    print(f'  Training time: {train_time:.1f}s')
    
    # Save model
    model_dir = RESULTS_DIR / 'models' / approach_name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({'models_real': models_real, 'models_imag': models_imag, 
                     'mean_x': mean_x, 'std_x': std_x, 'ptype': ptype, 
                     'n_basis_use': n_basis_use}, f)
    
    return models_real, models_imag, mean_x, std_x

def evaluate_and_save_approach(approach_name, models_real, models_imag, mean_x, std_x,
                               basis_data, val_sims, val_params, ptype, n_basis_use, model_type='unknown'):
    """Evaluate an approach and save scorecard."""
    print(f'  Evaluating {approach_name}...')
    t0 = time.time()
    
    losses = evaluate_approach(models_real, models_imag, mean_x, std_x, basis_data,
                               val_sims, val_params, ptype, n_basis_use)
    
    eval_time = time.time() - t0
    mean_loss = float(np.mean(losses))
    print(f'  Mean loss: {mean_loss:.4f}, eval time: {eval_time:.1f}s')
    
    # Save scorecard
    model_dir = RESULTS_DIR / 'models' / approach_name
    scorecard = {
        'approach': approach_name,
        'approach_number': 0,
        'benchmark': 'waveform',
        'agent': 'kimi_k26',
        'parameterization': ptype,
        'time_convention': 't0_at_peak',
        'loss': mean_loss,
        'loss_components': {'mean_fd_mismatch': mean_loss},
        'runtime_ms': eval_time * 1000 / len(val_sims),
        'n_train': len(val_sims),
        'n_val': len(val_sims),
        'n_params': 0,
        'notes': f'Model: {model_type}, n_basis: {n_basis_use}',
        'val_losses': losses,
    }
    
    with open(model_dir / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)
    
    return scorecard

if __name__ == '__main__':
    print('Loading data...')
    basis_data = load_basis()
    train_sims, val_sims = load_ref_sims()
    train_params = basis_data['train_params']
    val_params = basis_data['val_params']
    
    print(f'Train: {len(train_sims)}, Val: {len(val_sims)}')
    print(f'Basis shape: {basis_data["basis_real"].shape}')
    
    # Test one approach
    test_approach = 'SVD_GPR_RBF_Raw'
    models_real, models_imag, mean_x, std_x = train_and_save_approach(
        test_approach, 'gpr_rbf', 'raw', 15, basis_data, train_sims, train_params, val_sims, val_params
    )
    scorecard = evaluate_and_save_approach(
        test_approach, models_real, models_imag, mean_x, std_x,
        basis_data, val_sims, val_params, 'raw', 15
    )
    print(f'Test complete: loss={scorecard["loss"]:.4f}')
