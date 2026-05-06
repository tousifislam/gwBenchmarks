"""Batch script for waveform approaches 2-15."""

import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import h5py
import json
import time
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import joblib

PROJECT_ROOT = Path("/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks")
MODELS_DIR = PROJECT_ROOT / "llm_agents/results/hy3_preview_free/waveform/models"

def load_data(file_path):
    """Load waveform data."""
    params_list = []
    h22_list = []
    t_list = []
    
    with h5py.File(PROJECT_ROOT / file_path, "r") as f:
        n_sims = f.attrs["n_simulations"]
        
        for i in range(n_sims):
            sim_id = f"sim_{i:04d}"
            g = f[sim_id]
            
            q = g.attrs["q"]
            chi1 = [g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]]
            chi2 = [g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]]
            
            params = [q] + chi1 + chi2
            params_list.append(params)
            
            t = g["t"][:]
            h22 = g["h22_real"][:] + 1j * g["h22_imag"][:]
            
            h22_list.append(h22)
            t_list.append(t)
    
    return np.array(params_list), h22_list, t_list

def compute_svd_basis(h22_list, n_components=20):
    """Compute SVD basis from training waveforms."""
    # Find minimum length
    min_len = min(len(h) for h in h22_list)
    
    # Create data matrix
    X = []
    for h22 in h22_list:
        h_trunc = h22[:min_len]
        row = np.concatenate([np.real(h_trunc), np.imag(h_trunc)])
        X.append(row)
    X = np.array(X)
    
    # SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    coeffs = svd.fit_transform(X)
    
    return svd, coeffs, min_len

def fit_coefficients(params, coeffs, model_type="gpr_rbf", param_type="raw"):
    """Fit models for each SVD coefficient."""
    models = []
    
    for i in range(coeffs.shape[1]):
        if model_type == "gpr_rbf":
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(params.shape[1]))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        elif model_type == "gpr_matern":
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(params.shape[1]), nu=2.5)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        elif model_type == "poly":
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            poly = PolynomialFeatures(degree=3)
            model = Pipeline([('poly', poly), ('linear', LinearRegression())])
        elif model_type == "mlp":
            model = MLPRegressor(hidden_layers=(50, 50), max_iter=500, random_state=42)
        elif model_type == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gbr":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "kr":
            model = KernelRidge(kernel='rbf')
        elif model_type == "svr":
            model = SVR(kernel='rbf')
        
        model.fit(params, coeffs[:, i])
        models.append(model)
    
    return models

def save_models(models, model_dir, model_type):
    """Save fitted models."""
    model_dir = Path(model_dir)
    (model_dir / "saved_model").mkdir(exist_ok=True)
    
    for i, model in enumerate(models):
        joblib.dump(model, model_dir / "saved_model" / f"{model_type}_coeff_{i}.pkl")
    
    # Also save metadata
    metadata = {
        "model_type": model_type,
        "n_components": len(models),
    }
    with open(model_dir / "saved_model" / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    start_time = time.time()
    print("="*60)
    print("Waveform Benchmark - Batch Approaches 2-15")
    print("="*60 + "\n")
    
    # Load training data
    print("Loading training data...")
    params, h22_list, t_list = load_data("datasets/waveform/waveform_training.h5")
    print(f"Loaded {len(params)} samples")
    
    # Compute SVD basis (use existing from approach 1 if available)
    print("\nComputing SVD basis...")
    svd, coeffs, min_len = compute_svd_basis(h22_list)
    print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    # Save SVD basis for reuse
    basis_dir = MODELS_DIR / "01_svd_gpr_rbf_raw" / "saved_model"
    np.save(basis_dir / "basis.npy", svd.components_)
    np.save(basis_dir / "explained_variance.npy", svd.explained_variance_ratio_)
    
    # Define approaches
    approaches = [
        ("02_svd_gpr_matern_eff", "gpr_matern", "eta_chi_eff"),
        ("03_svd_poly_raw", "poly", "raw_7d"),
        ("04_svd_mlp_raw", "mlp", "raw_7d"),
        ("05_svd_rf_raw", "rf", "raw_7d"),
        ("06_svd_gbr_raw", "gbr", "raw_7d"),
        ("07_svd_kr_raw", "kr", "raw_7d"),
        ("08_svd_svr_raw", "svr", "raw_7d"),
    ]
    
    # Reparameterization functions
    def to_eta_chi_eff(params):
        """Convert to eta + chi_eff parameterization."""
        q = params[0]
        eta = q / (1 + q)**2
        
        chi1 = np.array(params[1:4])
        chi2 = np.array(params[4:7])
        
        # chi_eff = (q*chi1z + chi2z) / (1 + q)
        chi_eff = (q * chi1[2] + chi2[2]) / (1 + q)
        
        # chi_p (simplified)
        chi1_perp = np.sqrt(chi1[0]**2 + chi1[1]**2)
        chi2_perp = np.sqrt(chi2[0]**2 + chi2[1]**2)
        chi_p = max(chi1_perp, (q * chi2_perp + chi1_perp) / (1 + q))
        
        return [eta, chi_eff, chi_p]
    
    print("\n" + "="*60)
    print("Training approaches...")
    print("="*60)
    
    for approach_name, model_type, param_type in approaches:
        print(f"\nTraining {approach_name}...")
        
        # Get parameters in correct format
        if param_type == "eta_chi_eff":
            params_transformed = np.array([to_eta_chi_eff(p) for p in params])
        else:
            params_transformed = params
        
        # Fit models
        models = fit_coefficients(params_transformed, coeffs, model_type)
        
        # Save models
        model_dir = MODELS_DIR / approach_name
        model_dir.mkdir(exist_ok=True)
        save_models(models, model_dir, model_type)
        
        # Save SVD basis (same for all SVD-based approaches)
        np.save(model_dir / "saved_model" / "basis.npy", svd.components_)
        
        print(f"  Completed {approach_name}")
    
    runtime = time.time() - start_time
    print(f"\n="*60)
    print(f"All approaches completed in {runtime:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
