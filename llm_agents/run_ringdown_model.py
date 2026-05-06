import h5py
import numpy as np
import os
import pickle
import json
import argparse
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn models and other libraries
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from scipy.interpolate import RBFInterpolator, CubicSpline
from sympy import symbols, sympify, lambdify
from scipy.optimize import curve_fit


# Symbolic Regression (PySR and gplearn) - Will attempt to use if possible
# from pysr import PySRRegressor # Blocked due to Julia dependency issues
# from gplearn.genetic import SymbolicRegressor # Blocked due to Julia dependency issues


# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Custom Loss Function for Ringdown ---
def calculate_ringdown_loss(pred_omega_r: np.ndarray, true_omega_r: np.ndarray,
                            pred_omega_i: np.ndarray, true_omega_i: np.ndarray) -> tuple:
    """Calculates the custom ringdown loss function.
    L = (mean(|pred - true| / |true|) for omega_R  +  same for omega_I) / 2
    Returns (total_loss, mean_rel_err_r, mean_rel_err_i).
    """
    pred_omega_r = np.asarray(pred_omega_r, dtype=float)
    true_omega_r = np.asarray(true_omega_r, dtype=float)
    pred_omega_i = np.asarray(pred_omega_i, dtype=float)
    true_omega_i = np.asarray(true_omega_i, dtype=float)

    # Calculate relative error for omega_R
    # Avoid division by zero: if true is zero, and pred is also zero, relative error is 0. Else, inf.
    # Using a small epsilon to avoid true_omega_r == 0 causing inf
    rel_err_r = np.abs(pred_omega_r - true_omega_r) / (np.abs(true_omega_r) + 1e-10) # Added epsilon
    rel_err_r[true_omega_r == 0] = 0.0 # This line is mostly for aesthetic, epsilon already helps.

    # Calculate relative error for omega_I
    rel_err_i = np.abs(pred_omega_i - true_omega_i) / (np.abs(true_omega_i) + 1e-10) # Added epsilon
    rel_err_i[true_omega_i == 0] = 0.0 # This line is mostly for aesthetic, epsilon already helps.
    
    mean_rel_err_r = np.mean(rel_err_r[np.isfinite(rel_err_r)]) # Exclude infs from mean
    mean_rel_err_i = np.mean(rel_err_i[np.isfinite(rel_err_i)]) # Exclude infs from mean

    total_loss = (mean_rel_err_r + mean_rel_err_i) / 2.0
    return total_loss, mean_rel_err_r, mean_rel_err_i


# --- Utility Functions ---
def load_ringdown_data(file_path, mode="l2/m+2/n0"):
    """Loads ringdown data for a specific mode from an HDF5 file."""
    with h5py.File(file_path, "r") as f:
        g = f[mode]
        spin = g["spin"][:]
        omega_r = g["omega_real"][:]
        omega_i = g["omega_imag"][:]
    return spin, omega_r, omega_i

# --- Reparameterization Functions ---
def reparameterize_spin(spin, param_type):
    if param_type == "raw_a":
        return spin.reshape(-1, 1)
    elif param_type == "log_compactified": # x = -log(1 - a)
        return -np.log(1 - spin).reshape(-1, 1)
    elif param_type == "sqrt_irreducible": # x = sqrt(1 - a^2)
        return np.sqrt(1 - spin**2).reshape(-1, 1)
    elif param_type == "compactified": # x = a / (1 - a)
        return (spin / (1 - spin)).reshape(-1, 1)
    elif param_type == "chebyshev_mapped": # x = 2*a - 1
        return (2 * spin - 1).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown reparameterization type: {param_type}")

# --- Rational Function for Approximation ---
def rational_func(x, a, b, c, d):
    return (a * x + b) / (c * x + d)

# --- Plotting Functions ---
def plot_progress(agent_name, benchmark_name):
    base_results_dir = os.path.join("llm_agents", "results", agent_name, benchmark_name)
    models_dir = os.path.join(base_results_dir, "models")
    comparison_dir = os.path.join(base_results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    scorecards = []
    for model_name in os.listdir(models_dir):
        scorecard_path = os.path.join(models_dir, model_name, 'scorecard.json')
        if os.path.exists(scorecard_path):
            with open(scorecard_path, 'r') as f:
                scorecards.append(json.load(f))

    if not scorecards:
        print("No scorecards found to plot progress.")
        return

    scorecards.sort(key=lambda x: x.get('approach_number', 0))

    approaches = [s['approach'] for s in scorecards]
    losses = [s['loss'] for s in scorecards]
    runtimes = [s['runtime_ms'] for s in scorecards]

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(approaches, losses, marker='o', linestyle='-')
    plt.xlabel('Approach')
    plt.ylabel(f'{benchmark_name.capitalize()} Loss (Mean Relative Error)')
    plt.title(f'Progress of {benchmark_name.capitalize()} Benchmark Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'progress_loss.png'))
    plt.savefig(os.path.join(comparison_dir, 'progress_loss.pdf'))
    plt.close()

    # Plot Runtime
    plt.figure(figsize=(10, 6))
    plt.plot(approaches, runtimes, marker='o', linestyle='-')
    plt.xlabel('Approach')
    plt.ylabel('Runtime (ms)')
    plt.title(f'Progress of {benchmark_name.capitalize()} Benchmark Runtime')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'progress_runtime.png'))
    plt.savefig(os.path.join(comparison_dir, 'progress_runtime.pdf'))
    plt.close()
    
    print("Progress plots generated: progress_loss.png/pdf, progress_runtime.png/pdf")


# --- Main Runner Logic ---
def run_model(args):
    agent_name = args.agent
    benchmark_name = args.benchmark
    model_name = args.model_name
    approach_number = args.approach_number
    parameterization_type = args.parameterization
    mode = args.mode

    model_dir = os.path.join("llm_agents", "results", agent_name, benchmark_name, "models", model_name)
    saved_model_dir = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    train_data_path = os.path.join("datasets", benchmark_name, f"{benchmark_name}_training.h5")
    val_data_path = os.path.join("datasets", benchmark_name, f"{benchmark_name}_validation.h5")

    # --- Load Data ---
    print("Loading training data...")
    train_spin_raw, train_omega_r, train_omega_i = load_ringdown_data(train_data_path, mode)
    print("Loading validation data...")
    val_spin_raw, val_omega_r, val_omega_i = load_ringdown_data(val_data_path, mode)

    # --- Reparameterize Spin ---
    train_spin_reparam = reparameterize_spin(train_spin_raw, parameterization_type)
    val_spin_reparam = reparameterize_spin(val_spin_raw, parameterization_type)

    # --- Scale Spin Parameters (optional, but good practice for some models) ---
    spin_scaler = StandardScaler()
    scaled_train_spin = spin_scaler.fit_transform(train_spin_reparam)
    scaled_val_spin = spin_scaler.transform(val_spin_reparam)

    # --- Model Training ---
    model_r = None
    model_i = None

    if model_name.startswith("Polynomial_Fit"):
        poly_degree = args.poly_degree
        poly_features = PolynomialFeatures(degree=poly_degree)
        train_spin_poly = poly_features.fit_transform(scaled_train_spin)
        
        lr_r = LinearRegression()
        lr_r.fit(train_spin_poly, train_omega_r)
        model_r = lr_r

        lr_i = LinearRegression()
        lr_i.fit(train_spin_poly, train_omega_i)
        model_i = lr_i
        
        with open(os.path.join(saved_model_dir, "poly_features.pkl"), "wb") as f:
            pickle.dump(poly_features, f)
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("GPR"):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gpr_r = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
        gpr_r.fit(scaled_train_spin, train_omega_r)
        model_r = gpr_r

        gpr_i = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
        gpr_i.fit(scaled_train_spin, train_omega_i)
        model_i = gpr_i
        
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("RBFInterp"):
        rbfi_r = RBFInterpolator(scaled_train_spin, train_omega_r)
        model_r = rbfi_r

        rbfi_i = RBFInterpolator(scaled_train_spin, train_omega_i)
        model_i = rbfi_i
        
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("MLP"):
        mlp_r = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=0, early_stopping=True)
        mlp_r.fit(scaled_train_spin, train_omega_r)
        model_r = mlp_r

        mlp_i = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=0, early_stopping=True)
        mlp_i.fit(scaled_train_spin, train_omega_i)
        model_i = mlp_i
        
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("RandomForest"):
        rf_r = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        rf_r.fit(scaled_train_spin, train_omega_r)
        model_r = rf_r

        rf_i = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        rf_i.fit(scaled_train_spin, train_omega_i)
        model_i = rf_i
        
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("XGBoost"):
        xgb_r = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0, n_jobs=-1)
        xgb_r.fit(scaled_train_spin, train_omega_r)
        model_r = xgb_r

        xgb_i = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0, n_jobs=-1)
        xgb_i.fit(scaled_train_spin, train_omega_i)
        model_i = xgb_i
        
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("LightGBM"):
        lgbm_r = LGBMRegressor(objective='regression', n_estimators=100, random_state=0, n_jobs=-1)
        lgbm_r.fit(scaled_train_spin, train_omega_r)
        model_r = lgbm_r

        lgbm_i = LGBMRegressor(objective='regression', n_estimators=100, random_state=0, n_jobs=-1)
        lgbm_i.fit(scaled_train_spin, train_omega_i)
        model_i = lgbm_i
        
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("Chebyshev_Poly"):
        chebyshev_degree = args.chebyshev_degree if hasattr(args, 'chebyshev_degree') else 10
        # Chebyshev.fit expects 1D arrays for X and Y, and maps X to [-1, 1] internally.
        # scaled_train_spin is already mapped to [-1,1] effectively by chebyshev_mapped reparam
        cheb_r = np.polynomial.chebyshev.Chebyshev.fit(scaled_train_spin.flatten(), train_omega_r, chebyshev_degree)
        model_r = cheb_r

        cheb_i = np.polynomial.chebyshev.Chebyshev.fit(scaled_train_spin.flatten(), train_omega_i, chebyshev_degree)
        model_i = cheb_i
        
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("CubicSpline"):
        # CubicSpline directly uses raw spin values, as it's an interpolation method
        cs_r = CubicSpline(train_spin_raw, train_omega_r)
        model_r = cs_r
        cs_i = CubicSpline(train_spin_raw, train_omega_i)
        model_i = cs_i
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("Rational_Fit"):
        # Rational_func is a global function, no need to define locally
        # Fit for omega_r
        popt_r, pcov_r = curve_fit(rational_func, train_spin_raw.flatten(), train_omega_r, p0=[1, 1, 1, 1])
        model_r = popt_r # Store coefficients

        # Fit for omega_i
        popt_i, pcov_i = curve_fit(rational_func, train_spin_raw.flatten(), train_omega_i, p0=[1, 1, 1, 1])
        model_i = popt_i # Store coefficients

        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    elif model_name.startswith("PhenomFit_Poly"): # New Phenom Fit
        poly_degree = args.poly_degree
        poly_features = PolynomialFeatures(degree=poly_degree)
        train_spin_poly = poly_features.fit_transform(scaled_train_spin)
        
        lr_r = LinearRegression()
        lr_r.fit(train_spin_poly, train_omega_r)
        model_r = lr_r

        lr_i = LinearRegression()
        lr_i.fit(train_spin_poly, train_omega_i)
        model_i = lr_i
        
        with open(os.path.join(saved_model_dir, "poly_features.pkl"), "wb") as f:
            pickle.dump(poly_features, f)
        with open(os.path.join(saved_model_dir, "model_r.pkl"), "wb") as f:
            pickle.dump(model_r, f)
        with open(os.path.join(saved_model_dir, "model_i.pkl"), "wb") as f:
            pickle.dump(model_i, f)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # --- Save Common Components ---
    with open(os.path.join(saved_model_dir, "spin_scaler.pkl"), "wb") as f:
        pickle.dump(spin_scaler, f)
    # No max_length for ringdown data as it's not time series based on SVD

    print(f"Model {model_name} training complete and saved.")

    # --- Prediction and Evaluation ---
    start_time = time.time()
    
    pred_omega_r = None
    pred_omega_i = None

    if model_name.startswith("Polynomial_Fit"):
        poly_features = pickle.load(open(os.path.join(saved_model_dir, "poly_features.pkl"), "rb"))
        val_spin_poly = poly_features.transform(scaled_val_spin)
        pred_omega_r = model_r.predict(val_spin_poly)
        pred_omega_i = model_i.predict(val_spin_poly)
    elif (model_name.startswith("GPR") or model_name.startswith("MLP") or 
          model_name.startswith("RandomForest") or model_name.startswith("XGBoost") or 
          model_name.startswith("LightGBM") or model_name.startswith("KRR")):
        # KRR will also use model_r.predict(scaled_val_spin)
        pred_omega_r = model_r.predict(scaled_val_spin)
        pred_omega_i = model_i.predict(scaled_val_spin)
    elif model_name.startswith("RBFInterp"):
        pred_omega_r = model_r(scaled_val_spin)
        pred_omega_i = model_i(scaled_val_spin)
    elif model_name.startswith("Chebyshev_Poly"):
        pred_omega_r = model_r(scaled_val_spin.flatten())
        pred_omega_i = model_i(scaled_val_spin.flatten())
    elif model_name.startswith("CubicSpline"):
        pred_omega_r = model_r(val_spin_raw)
        pred_omega_i = model_i(val_spin_raw)
    elif model_name.startswith("Rational_Fit"):
        model_r_coeffs = pickle.load(open(os.path.join(saved_model_dir, "model_r.pkl"), "rb"))
        model_i_coeffs = pickle.load(open(os.path.join(saved_model_dir, "model_i.pkl"), "rb"))
        pred_omega_r = rational_func(val_spin_raw, *model_r_coeffs)
        pred_omega_i = rational_func(val_spin_raw, *model_i_coeffs)
    elif model_name.startswith("PhenomFit_Poly"):
        poly_features = pickle.load(open(os.path.join(saved_model_dir, "poly_features.pkl"), "rb"))
        val_spin_poly = poly_features.transform(scaled_val_spin)
        pred_omega_r = model_r.predict(val_spin_poly)
        pred_omega_i = model_i.predict(val_spin_poly)
    # elif model_name.startswith("PySR") or model_name.startswith("gplearn"): # PySR/gplearn blocked
    #     print("Prediction for symbolic models not yet implemented.")
    #     pred_omega_r = np.zeros_like(val_omega_r)
    #     pred_omega_i = np.zeros_like(val_omega_i)
    else:
        raise ValueError(f"Unknown model name for prediction: {model_name}")

    end_time = time.time()
    runtime_ms = (end_time - start_time) * 1000

    total_loss, mean_rel_error_omega_r, mean_rel_error_omega_i = calculate_ringdown_loss(pred_omega_r, val_omega_r, pred_omega_i, val_omega_i)
    
    # --- Generate Scorecard ---
    scorecard = {
        "approach": model_name,
        "approach_number": approach_number,
        "benchmark": benchmark_name,
        "agent": agent_name,
        "parameterization": parameterization_type,
        "mode": mode,
        "loss": total_loss,
        "loss_components": {
            "mean_rel_error_omega_r": mean_rel_error_omega_r,
            "mean_rel_error_omega_i": mean_rel_error_omega_i,
        },
        "runtime_ms": runtime_ms,
        "n_train": len(train_spin_raw),
        "n_val": len(val_spin_raw),
        "n_params": args.n_params if hasattr(args, 'n_params') else None, # For future use if models have params
        "notes": f"Model trained with {parameterization_type} parameterization for mode {mode}. Evaluated loss: {total_loss:.4f}."
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    print(f"Scorecard generated for {model_name}. Loss: {total_loss:.4f}, Runtime: {runtime_ms:.2f} ms")

    # --- Generate Changelog Entry ---
    changelog_entry = f"""
## [Q-{approach_number:02d}] {model_name} ({parameterization_type} parameterization)
- **Time**: {time.strftime('%Y-%m-%d %H:%M', time.localtime())}
- **Benchmark**: {benchmark_name}
- **Method**: {model_name.replace('_', ' ').replace('Fit', ' Fit')}
- **Parameterization**: {parameterization_type}
- **Mode**: {mode}
- **Loss**: {total_loss:.4f} (Mean Relative Error)
- **Eval time**: {runtime_ms:.2f} ms
- **Key observations**:
  - Model trained with {parameterization_type} parameters for mode {mode}.
- **Next idea**: Continue implementing other models and reparameterizations.
"""
    return changelog_entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ringdown benchmark models.")
    parser.add_argument("--agent", type=str, default="gemini25_flash", help="Agent name.")
    parser.add_argument("--benchmark", type=str, default="ringdown", help="Benchmark name.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to run.")
    parser.add_argument("--approach_number", type=int, required=True, help="Approach number for changelog.")
    parser.add_argument("--parameterization", type=str, default="raw_a", help="Parameterization to use.")
    parser.add_argument("--mode", type=str, default="l2/m+2/n0", help="QNM mode to model.")
    parser.add_argument("--poly_degree", type=int, default=10, help="Degree for polynomial features.")
    parser.add_argument("--chebyshev_degree", type=int, default=10, help="Degree for Chebyshev polynomial features.")
    
    args = parser.parse_args()
    
    changelog_entry = run_model(args)
    
    # The main agent flow will collect the entries and append them to CHANGELOG.md at once.
    # So, for now, this script will just print the entry.
    print("\n--- Changelog Entry ---\n")
    print(changelog_entry)
