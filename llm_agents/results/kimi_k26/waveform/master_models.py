"""Master modeling script for waveform benchmark.
Implements multiple approaches for SVD coefficient modeling.
"""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import time
import pickle
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/waveform')
BASIS_DIR = RESULTS_DIR / 'saved_basis'

def load_basis():
    """Load precomputed SVD basis."""
    data = {
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
    return data

def get_param_reparameterization(params, ptype='raw'):
    """Reparameterize input parameters."""
    # params: (n, 8) with [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega0]
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
    """Z-normalize features."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    Xn_train = (X_train - mean) / std
    if X_val is not None:
        Xn_val = (X_val - mean) / std
        return Xn_train, Xn_val, mean, std
    return Xn_train, mean, std

def fit_model_to_coeffs(X_train, coeffs_train, X_val, coeffs_val, model_type='gpr_rbf', n_basis_use=20):
    """Fit a model to SVD coefficients.
    
    Returns predictions for val set.
    """
    # Only use top n_basis_use coefficients
    coeffs_train = coeffs_train[:, :n_basis_use]
    coeffs_val = coeffs_val[:, :n_basis_use]
    
    n_coeffs = coeffs_train.shape[1]
    
    # Normalize features
    Xn_train, Xn_val, mean_x, std_x = normalize_features(X_train, X_val)
    
    # Fit a separate model for each coefficient
    models = []
    val_preds = np.zeros_like(coeffs_val)
    train_preds = np.zeros_like(coeffs_train)
    
    for i in range(n_coeffs):
        y_train = coeffs_train[:, i]
        y_val = coeffs_val[:, i]
        
        if model_type == 'gpr_rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
        elif model_type == 'gpr_matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
        elif model_type == 'poly':
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('ridge', Ridge(alpha=1.0))
            ])
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
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        
        model.fit(Xn_train, y_train)
        train_preds[:, i] = model.predict(Xn_train)
        val_preds[:, i] = model.predict(Xn_val)
        models.append(model)
    
    return train_preds, val_preds, models, (mean_x, std_x)

def reconstruct_waveform(coeffs_real, coeffs_imag, basis_data, t_target, n_basis_use=20):
    """Reconstruct waveform from SVD coefficients on target time grid."""
    mean_real = basis_data['mean_real']
    mean_imag = basis_data['mean_imag']
    basis_real = basis_data['basis_real'][:, :n_basis_use]
    basis_imag = basis_data['basis_imag'][:, :n_basis_use]
    t_common = basis_data['t_common']
    
    # Reconstruct on common grid
    h_common_real = mean_real + np.dot(coeffs_real, basis_real.T)
    h_common_imag = mean_imag + np.dot(coeffs_imag, basis_imag.T)
    h_common = h_common_real + 1j * h_common_imag
    
    # Interpolate to target time grid
    h_target = np.interp(t_target, t_common, h_common.real) + 1j * np.interp(t_target, t_common, h_common.imag)
    
    return h_target

def compute_mismatch_for_approach(val_preds_real, val_preds_imag, basis_data, sims, max_samples=None):
    """Compute FD mismatch for validation samples.
    
    If max_samples is set, only evaluate on that many samples for speed.
    """
    from gwbenchmarks.metrics import mean_fd_mismatch
    
    n_val = len(sims)
    if max_samples is not None:
        n_val = min(n_val, max_samples)
    
    losses = []
    for i in range(n_val):
        try:
            h_pred = reconstruct_waveform(
                val_preds_real[i], val_preds_imag[i], basis_data, sims[i]['t']
            )
            h_ref = sims[i]['h22']
            dt = sims[i]['dt']
            loss = mean_fd_mismatch(h_pred, h_ref, dt)
            losses.append(loss)
        except Exception as e:
            print(f'  Error on val {i}: {e}')
            losses.append(1.0)
    
    return losses

def save_model(model_dir, models, mean_x, std_x, approach_info):
    """Save model artifacts."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    save_dir = model_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'mean_x': mean_x,
            'std_x': std_x,
            'approach_info': approach_info,
        }, f)

def save_train_predict_scripts(model_dir, approach_name, model_type, ptype, n_basis_use):
    """Save train.py and predict.py scripts."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    train_script = f'''"""Training script for {approach_name}."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')
import numpy as np
from pathlib import Path
from master_models import fit_model_to_coeffs, get_param_reparameterization, load_basis, normalize_features
import pickle

BASIS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/waveform/saved_basis')

def train():
    basis = load_basis()
    X_train = get_param_reparameterization(basis['train_params'], '{ptype}')
    coeffs_real_train = basis['coeffs_real_train'][:, :{n_basis_use}]
    coeffs_imag_train = basis['coeffs_imag_train'][:, :{n_basis_use}]
    
    Xn_train, mean_x, std_x = normalize_features(X_train)
    
    models_real = []
    for i in range(coeffs_real_train.shape[1]):
        y = coeffs_real_train[:, i]
        # Model fitting logic here (same as in master_models.py)
        models_real.append(None)  # Placeholder
    
    # Save models
    save_dir = Path(__file__).parent / 'saved_model'
    with open(save_dir / 'models.pkl', 'wb') as f:
        pickle.dump({{'models_real': models_real, 'mean_x': mean_x, 'std_x': std_x}}, f)

if __name__ == '__main__':
    train()
'''
    
    predict_script = f'''"""Prediction script for {approach_name}."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')
import numpy as np
from pathlib import Path
from master_models import get_param_reparameterization, reconstruct_waveform, load_basis
import pickle

BASIS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/waveform/saved_basis')

def load_model():
    model_path = Path(__file__).parent / 'saved_model' / 'models.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict(params):
    """Predict waveform for given parameters.
    params: array [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega0]
    """
    model_data = load_model()
    basis = load_basis()
    # Prediction logic here
    pass

if __name__ == '__main__':
    pass
'''
    
    with open(model_dir / 'train.py', 'w') as f:
        f.write(train_script)
    with open(model_dir / 'predict.py', 'w') as f:
        f.write(predict_script)

if __name__ == '__main__':
    print('Master models module loaded. Use train_and_evaluate.py to run approaches.')
