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
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from scipy.interpolate import RBFInterpolator

# Symbolic Regression (PySR and gplearn) - Will attempt to use if possible
# from pysr import PySRRegressor # Blocked due to Julia dependency issues
# from gplearn.genetic import SymbolicRegressor # Blocked due to Julia dependency issues


# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Custom Loss Function for Validity ---
def calculate_validity_loss(pred_log_mm: np.ndarray, true_log_mm: np.ndarray) -> float:
    """Calculates the RMSE in log10 space for mismatch.
    L = RMSE(log10(mm_pred), log10(mm_true))
    """
    pred_log_mm = np.asarray(pred_log_mm, dtype=float)
    true_log_mm = np.asarray(true_log_mm, dtype=float)
    return float(np.sqrt(np.mean((pred_log_mm - true_log_mm) ** 2)))


# --- Utility Functions ---
def load_validity_data(file_path):
    """Loads validity data from an HDF5 file and applies log10 transformation to mm_td."""
    with h5py.File(file_path, "r") as f:
        q = f["q"][:]
        chi1z = f["chi1z"][:]
        chi2z = f["chi2z"][:]
        omega0 = f["omega0"][:]
        mm_td = f["mm_td"][:]
    
    params = np.column_stack([q, chi1z, chi2z, omega0])
    log_mm_td = np.log10(mm_td) # Apply log10 transformation

    return params, log_mm_td

# --- Reparameterization Functions ---
def reparameterize_params(params_raw, param_type):
    q, chi1z, chi2z, omega0 = params_raw[:, 0], params_raw[:, 1], params_raw[:, 2], params_raw[:, 3]

    if param_type == "raw_4d":
        return params_raw
    elif param_type == "effective_spins": # (eta, chi_eff, chi_a, omega0)
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        eta = m1 * m2
        chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
        chi_a = (m1 * chi1z - m2 * chi2z) / (m1 + m2)
        return np.column_stack([eta, chi_eff, chi_a, omega0])
    elif param_type == "log_mass_ratio": # (log(q), chi_eff, chi_a, log(omega0))
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        eta = m1 * m2
        chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
        chi_a = (m1 * chi1z - m2 * chi2z) / (m1 + m2)
        return np.column_stack([np.log(q), chi_eff, chi_a, np.log(omega0)])
    elif param_type == "interaction_terms": # (eta, chi_eff, chi_a, omega0, q*chi_eff, eta*chi_a)
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        eta = m1 * m2
        chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
        chi_a = (m1 * chi1z - m2 * chi2z) / (m1 + m2)
        return np.column_stack([eta, chi_eff, chi_a, omega0, q * chi_eff, eta * chi_a])
    # elif param_type == "boundary_distance": # Features for proximity to NRHybSur3dq8 valid region
    #     # This would require more complex logic to define these features
    #     raise NotImplementedError("Boundary distance reparameterization not yet implemented.")
    else:
        raise ValueError(f"Unknown reparameterization type: {param_type}")

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
    plt.ylabel(f'{benchmark_name.capitalize()} Loss (RMSE on log10(mismatch))')
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

    model_dir = os.path.join("llm_agents", "results", agent_name, benchmark_name, "models", model_name)
    saved_model_dir = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    train_data_path = os.path.join("datasets", benchmark_name, f"{benchmark_name}_training.h5")
    val_data_path = os.path.join("datasets", benchmark_name, f"{benchmark_name}_validation.h5")

    # --- Load Data ---
    print("Loading training data...")
    train_params_raw, train_log_mm = load_validity_data(train_data_path)
    print("Loading validation data...")
    val_params_raw, val_log_mm = load_validity_data(val_data_path)

    # --- Reparameterize Parameters ---
    train_params = reparameterize_params(train_params_raw, parameterization_type)
    val_params = reparameterize_params(val_params_raw, parameterization_type)

    # --- Scale Parameters ---
    param_scaler = StandardScaler()
    scaled_train_params = param_scaler.fit_transform(train_params)
    scaled_val_params = param_scaler.transform(val_params)

    # --- Model Training ---
    model = None

    if model_name.startswith("GPR"):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) # Default RBF
        if "Matern" in model_name:
             kernel = C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5) # Example Matern
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
        gpr.fit(scaled_train_params, train_log_mm)
        model = gpr
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("KRR"):
        krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
        krr.fit(scaled_train_params, train_log_mm)
        model = krr
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("SVR"):
        svr = SVR(kernel='rbf', C=100, gamma=0.1) # Default SVR
        svr.fit(scaled_train_params, train_log_mm)
        model = svr
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("Polynomial_Fit"):
        poly_degree = args.poly_degree
        poly_features = PolynomialFeatures(degree=poly_degree)
        train_poly_params = poly_features.fit_transform(scaled_train_params)
        lr = LinearRegression()
        lr.fit(train_poly_params, train_log_mm)
        model = lr
        with open(os.path.join(saved_model_dir, "poly_features.pkl"), "wb") as f:
            pickle.dump(poly_features, f)
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    # elif model_name.startswith("PySR"): # Blocked
    #     print("PySR not implemented/blocked in runner script.")
    # elif model_name.startswith("gplearn"): # Blocked
    #     print("gplearn not implemented/blocked in runner script.")
    elif model_name.startswith("RBFInterp"):
        rbfi = RBFInterpolator(scaled_train_params, train_log_mm)
        model = rbfi
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("NNInterp"):
        knr = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        knr.fit(scaled_train_params, train_log_mm)
        model = knr
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("MLP"):
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=0, early_stopping=True)
        mlp.fit(scaled_train_params, train_log_mm)
        model = mlp
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("RandomForest"):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        rf_model.fit(scaled_train_params, train_log_mm)
        model = rf_model
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("XGBoost"):
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0, n_jobs=-1)
        xgb_model.fit(scaled_train_params, train_log_mm)
        model = xgb_model
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("LightGBM"):
        lgbm_model = LGBMRegressor(objective='regression', n_estimators=100, random_state=0, n_jobs=-1)
        lgbm_model.fit(scaled_train_params, train_log_mm)
        model = lgbm_model
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif model_name.startswith("PhenomFit_Poly"): # Alternative for Symbolic
        poly_degree = args.poly_degree
        poly_features = PolynomialFeatures(degree=poly_degree)
        train_poly_params = poly_features.fit_transform(scaled_train_params) # Use scaled parameters for fit
        lr = LinearRegression()
        lr.fit(train_poly_params, train_log_mm)
        model = lr
        with open(os.path.join(saved_model_dir, "poly_features.pkl"), "wb") as f:
            pickle.dump(poly_features, f)
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # --- Save Common Components ---
    with open(os.path.join(saved_model_dir, "param_scaler.pkl"), "wb") as f:
        pickle.dump(param_scaler, f)

    print(f"Model {model_name} training complete and saved.")

    # --- Prediction and Evaluation ---
    start_time = time.time()
    
    pred_log_mm = None

    if model_name.startswith("Polynomial_Fit") or model_name.startswith("PhenomFit_Poly"):
        poly_features = pickle.load(open(os.path.join(saved_model_dir, "poly_features.pkl"), "rb"))
        val_poly_params = poly_features.transform(scaled_val_params)
        model = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        pred_log_mm = model.predict(val_poly_params)
    elif model_name.startswith("RBFInterp"):
        model = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        pred_log_mm = model(scaled_val_params)
    # elif model_name.startswith("PySR") or model_name.startswith("gplearn"): # Blocked
    #     print("Prediction for symbolic models not yet implemented.")
    #     pred_log_mm = np.zeros_like(val_log_mm)
    else: # GPR, KRR, SVR, MLP, RandomForest, XGBoost, LightGBM, NNInterp
        model = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        pred_log_mm = model.predict(scaled_val_params)

    end_time = time.time()
    runtime_ms = (end_time - start_time) * 1000

    total_loss = calculate_validity_loss(pred_log_mm, val_log_mm)
    
    # --- Generate Scorecard ---
    scorecard = {
        "approach": model_name,
        "approach_number": approach_number,
        "benchmark": benchmark_name,
        "agent": agent_name,
        "parameterization": parameterization_type,
        "loss": total_loss,
        "loss_components": {"log_rmse": total_loss},
        "runtime_ms": runtime_ms,
        "n_train": len(train_params_raw),
        "n_val": len(val_params_raw),
        "n_params": train_params.shape[1], # Number of input parameters
        "notes": f"Model trained with {parameterization_type} parameterization. Evaluated loss: {total_loss:.4f}."
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    print(f"Scorecard generated for {model_name}. Loss: {total_loss:.4f}, Runtime: {runtime_ms:.2f} ms")

    # --- Generate Changelog Entry ---
    changelog_entry = f"""
## [V-{approach_number:02d}] {model_name} ({parameterization_type} parameterization)
- **Time**: {time.strftime('%Y-%m-%d %H:%M', time.localtime())}
- **Benchmark**: {benchmark_name}
- **Method**: {model_name.replace('_', ' ').replace('Fit', ' Fit').replace('NNInterp', 'Nearest Neighbor Interpolation')}
- **Parameterization**: {parameterization_type}
- **Loss**: {total_loss:.4f} (RMSE on log10(mismatch))
- **Eval time**: {runtime_ms:.2f} ms
- **Key observations**:
  - Model trained with {parameterization_type} parameters.
- **Next idea**: Continue implementing other models and reparameterizations.
"""
    return changelog_entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Validity benchmark models.")
    parser.add_argument("--agent", type=str, default="gemini25_flash", help="Agent name.")
    parser.add_argument("--benchmark", type=str, default="validity", help="Benchmark name.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to run.")
    parser.add_argument("--approach_number", type=int, required=True, help="Approach number for changelog.")
    parser.add_argument("--parameterization", type=str, default="raw_4d", help="Parameterization to use.")
    parser.add_argument("--poly_degree", type=int, default=2, help="Degree for polynomial features.")
    
    args = parser.parse_args()
    
    changelog_entry = run_model(args)
    
    # The main agent flow will collect the entries and append them to CHANGELOG.md at once.
    # So, for now, this script will just print the entry.
    print("\n--- Changelog Entry ---\n")
    print(changelog_entry)
