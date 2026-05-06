import h5py
import numpy as np
import os
import pickle
import json
import argparse
import time
import sys
import pandas as pd # Needed for PySR feature names
import matplotlib.pyplot as plt

# Scikit-learn models
from sklearn.decomposition import TruncatedSVD
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
from xgboost import XGBRegressor # Added this import
from scipy.interpolate import RBFInterpolator

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gwbenchmarks.metrics import rms_relative_error
import gwbenchmarks.plot_settings # Import for plot styling

# --- Utility Functions (Copied from previous scripts) ---
def load_dynamics_data(file_path):
    """Loads dynamics data from an HDF5 file."""
    params = []
    time_series_x = []
    ts = []
    with h5py.File(file_path, "r") as f:
        n_simulations = f.attrs["n_simulations"]
        for i in range(n_simulations):
            group_name = f"sim_{i:04d}"
            group = f[group_name]
            
            # Extract parameters
            q = group.attrs["q"]
            chi1z = group.attrs["chi1z"]
            chi2z = group.attrs["chi2z"]
            e0 = group.attrs["e0"]
            zeta0 = group.attrs["zeta0"]
            omega0 = group.attrs["omega0"]
            params.append([q, chi1z, chi2z, e0, zeta0, omega0])
            
            # Extract time series x(t) and t
            t = group["t"][:]
            x = group["x"][:]
            time_series_x.append(x)
            ts.append(t)
            
    return np.array(params), time_series_x, ts

def preprocess_time_series(time_series, target_length=None):
    """Pads time series to a common length for SVD."""
    if target_length is None:
        target_length = max(len(ts) for ts in time_series)
    
    padded_time_series = []
    for ts in time_series:
        if len(ts) < target_length:
            padded_ts = np.pad(ts, (0, target_length - len(ts)), 'edge')
        else:
            padded_ts = ts[:target_length]
        padded_time_series.append(padded_ts)
    return np.array(padded_time_series), target_length

def calculate_eta_chieff_chia_loge0(q, chi1z, chi2z, e0, zeta0, omega0):
    m1 = q / (1 + q)
    m2 = 1 / (1 + q)
    eta = m1 * m2
    chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
    chi_a = (m1 * chi1z - m2 * chi2z) / (m1 + m2)
    return np.array([eta, chi_eff, chi_a, np.log(e0), zeta0, omega0])

def calculate_m1m2_s1zs2z(q, chi1z, chi2z, e0, zeta0, omega0):
    m2 = 1.0 / (1.0 + q) # Assuming total mass M=1
    m1 = q * m2
    s1z = chi1z
    s2z = chi2z
    return np.array([m1, m2, s1z, s2z, e0, zeta0, omega0])

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
    plt.ylabel(f'{benchmark_name.capitalize()} Loss (RMS Relative Error)')
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
    parameterization = args.parameterization

    model_dir = os.path.join("llm_agents", "results", agent_name, benchmark_name, "models", model_name)
    saved_model_dir = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    train_data_path = os.path.join("datasets", benchmark_name, f"{benchmark_name}_training.h5")
    val_data_path = os.path.join("datasets", benchmark_name, f"{benchmark_name}_validation.h5")

    # --- Load Data ---
    print("Loading training data...")
    train_raw_params, train_x_series, train_t_series = load_dynamics_data(train_data_path)
    print("Loading validation data...")
    val_raw_params, val_x_series, val_t_series = load_dynamics_data(val_data_path)

    # --- Apply Parameterization ---
    train_params = train_raw_params
    val_params = val_raw_params
    if parameterization == "eta_chieff_chia_loge0_zeta0_omega0":
        train_params = np.array([calculate_eta_chieff_chia_loge0(*p) for p in train_raw_params])
        val_params = np.array([calculate_eta_chieff_chia_loge0(*p) for p in val_raw_params])
    elif parameterization == "m1m2_s1zs2z_e0_zeta0_omega0":
        train_params = np.array([calculate_m1m2_s1zs2z(*p) for p in train_raw_params])
        val_params = np.array([calculate_m1m2_s1zs2z(*p) for p in val_raw_params])

    # --- SVD Preprocessing ---
    max_train_length = max(len(ts) for ts in train_x_series)
    padded_train_x, _ = preprocess_time_series(train_x_series, max_train_length)
    padded_val_x, _ = preprocess_time_series(val_x_series, max_train_length)

    n_components = args.n_components # Use argument for n_components
    svd = TruncatedSVD(n_components=n_components)
    train_coeffs = svd.fit_transform(padded_train_x)
    val_coeffs = svd.transform(padded_val_x)

    # --- Parameter Scaling ---
    param_scaler = StandardScaler()
    scaled_train_params = param_scaler.fit_transform(train_params)
    scaled_val_params = param_scaler.transform(val_params)

    # --- Model Training ---
    model = None
    if model_name == "SVD_GPR_Raw" or model_name == "SVD_GPR_EtaChiEff":
        gpr_models = []
        for i in range(n_components):
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
            gpr.fit(scaled_train_params, train_coeffs[:, i])
            gpr_models.append(gpr)
        model = gpr_models
        with open(os.path.join(saved_model_dir, "gpr_models.pkl"), "wb") as f:
            pickle.dump(gpr_models, f)
    elif model_name == "SVD_Polynomial_Raw" or model_name.startswith("SVD_Polynomial_Raw_Deg"):
        poly = PolynomialFeatures(degree=args.poly_degree)
        poly_train_params = poly.fit_transform(scaled_train_params)
        poly_models = []
        for i in range(n_components):
            lr = LinearRegression()
            lr.fit(poly_train_params, train_coeffs[:, i])
            poly_models.append(lr)
        model = poly_models
        with open(os.path.join(saved_model_dir, "poly_features.pkl"), "wb") as f:
            pickle.dump(poly, f)
        with open(os.path.join(saved_model_dir, "poly_models.pkl"), "wb") as f:
            pickle.dump(poly_models, f)
    elif model_name == "SVD_MLP_Raw" or model_name == "SVD_MLP_EtaChiEff": # Group both MLP models
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=0, early_stopping=True)
        mlp.fit(scaled_train_params, train_coeffs)
        model = mlp
        with open(os.path.join(saved_model_dir, "mlp_model.pkl"), "wb") as f:
            pickle.dump(mlp, f)
    elif model_name == "SVD_RandomForest_Raw" or model_name == "SVD_RandomForest_EtaChiEff": # Group both RandomForest models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        rf_model.fit(scaled_train_params, train_coeffs)
        model = rf_model
        with open(os.path.join(saved_model_dir, "rf_model.pkl"), "wb") as f:
            pickle.dump(rf_model, f)
    elif model_name == "SVD_XGBoost_Raw" or model_name == "SVD_XGBoost_EtaChiEff": # Group both XGBoost models
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0, n_jobs=-1)
        xgb_model.fit(scaled_train_params, train_coeffs)
        model = xgb_model
        with open(os.path.join(saved_model_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
    elif model_name == "SVD_LightGBM_Raw" or model_name == "SVD_LightGBM_EtaChiEff": # Group both LightGBM models
        base_lgbm = LGBMRegressor(objective='regression', n_estimators=100, random_state=0, n_jobs=-1)
        lgbm_model = MultiOutputRegressor(base_lgbm, n_jobs=-1)
        lgbm_model.fit(scaled_train_params, train_coeffs)
        model = lgbm_model
        with open(os.path.join(saved_model_dir, "lgbm_model.pkl"), "wb") as f:
            pickle.dump(lgbm_model, f)
    elif model_name == "RBFInterp_Raw" or model_name == "RBFInterp_EtaChiEff":
        rbfi = RBFInterpolator(scaled_train_params, train_coeffs)
        model = rbfi
        with open(os.path.join(saved_model_dir, "rbf_interpolator.pkl"), "wb") as f:
            pickle.dump(rbfi, f)
    elif model_name == "NNInterp_Raw":
        knr = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        knr.fit(scaled_train_params, train_coeffs)
        model = knr
        with open(os.path.join(saved_model_dir, "knr_model.pkl"), "wb") as f:
            pickle.dump(knr, f)
    elif model_name == "SVD_KRR_Raw" or model_name == "SVD_KRR_EtaChiEff": # Group both KRR models
        krr_models = []
        for i in range(n_components):
            krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1) 
            krr.fit(scaled_train_params, train_coeffs[:, i])
            krr_models.append(krr)
        model = krr_models
        with open(os.path.join(saved_model_dir, "krr_models.pkl"), "wb") as f:
            pickle.dump(krr_models, f)
    elif model_name == "SVD_KRR_M1M2S1S2":
        krr_models = []
        for i in range(n_components):
            krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1) 
            krr.fit(scaled_train_params, train_coeffs[:, i])
            krr_models.append(krr)
        model = krr_models
        with open(os.path.join(saved_model_dir, "krr_models.pkl"), "wb") as f:
            pickle.dump(krr_models, f)
    elif model_name == "PhenomFit_Poly_EtaChiEff":
        # Specific logic for this model from train.py
        eta_chieff_params = np.array([calculate_eta_chieff_chia_loge0(*p)[:2] for p in train_raw_params]) # Only eta, chi_eff
        param_scaler_phenom = StandardScaler()
        scaled_eta_chieff = param_scaler_phenom.fit_transform(eta_chieff_params)

        poly = PolynomialFeatures(degree=args.poly_degree)
        poly_eta_chieff = poly.fit_transform(scaled_eta_chieff)
        
        c0_model = LinearRegression()
        c0_model.fit(poly_eta_chieff, train_coeffs[:, 0])
        mean_other_coeffs = np.mean(train_coeffs[:, 1:], axis=0)

        with open(os.path.join(saved_model_dir, "c0_model.pkl"), "wb") as f:
            pickle.dump(c0_model, f)
        with open(os.path.join(saved_model_dir, "poly_features.pkl"), "wb") as f:
            pickle.dump(poly, f)
        with open(os.path.join(saved_model_dir, "mean_other_coeffs.pkl"), "wb") as f:
            pickle.dump(mean_other_coeffs, f)
        with open(os.path.join(saved_model_dir, "param_scaler_phenom.pkl"), "wb") as f:
            pickle.dump(param_scaler_phenom, f)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # --- Save Common Components ---
    with open(os.path.join(saved_model_dir, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f)
    with open(os.path.join(saved_model_dir, "param_scaler.pkl"), "wb") as f:
        pickle.dump(param_scaler, f)
    with open(os.path.join(saved_model_dir, "max_length.pkl"), "wb") as f:
        pickle.dump(max_train_length, f)

    print(f"Model {model_name} training complete and saved.")

    # --- Prediction and Evaluation ---
    start_time = time.time()
    predicted_coeffs = np.zeros((scaled_val_params.shape[0], n_components))

    # Generic prediction logic for models that predict coeffs directly
    if model_name in ["SVD_MLP_Raw", "SVD_MLP_EtaChiEff", "SVD_RandomForest_Raw", "SVD_RandomForest_EtaChiEff", "SVD_XGBoost_Raw", "SVD_XGBoost_EtaChiEff", "SVD_LightGBM_Raw", "SVD_LightGBM_EtaChiEff", "NNInterp_Raw"]:
        predicted_coeffs = model.predict(scaled_val_params)
    elif model_name == "RBFInterp_Raw" or model_name == "RBFInterp_EtaChiEff":
        predicted_coeffs = model(scaled_val_params) # Call RBFInterpolator directly
    elif model_name == "SVD_GPR_Raw" or model_name == "SVD_GPR_EtaChiEff":
        for i, gpr in enumerate(model):
            predicted_coeffs[:, i] = gpr.predict(scaled_val_params)
    elif model_name == "SVD_Polynomial_Raw" or model_name == "SVD_Polynomial_Raw_Deg3":
        poly = pickle.load(open(os.path.join(saved_model_dir, "poly_features.pkl"), "rb"))
        poly_val_params = poly.transform(scaled_val_params)
        for i, lr in enumerate(model):
            predicted_coeffs[:, i] = lr.predict(poly_val_params)
    elif model_name == "SVD_KRR_Raw" or model_name == "SVD_KRR_M1M2S1S2" or model_name == "SVD_KRR_EtaChiEff": # Group all KRR models
        for i, krr in enumerate(model):
            predicted_coeffs[:, i] = krr.predict(scaled_val_params)
    elif model_name == "PhenomFit_Poly_EtaChiEff":
        # Specific prediction logic for this model
        eta_chieff_params_val = np.array([calculate_eta_chieff_chia_loge0(*p)[:2] for p in val_raw_params])
        param_scaler_phenom = pickle.load(open(os.path.join(saved_model_dir, "param_scaler_phenom.pkl"), "rb"))
        scaled_eta_chieff_val = param_scaler_phenom.transform(eta_chieff_params_val)
        poly = pickle.load(open(os.path.join(saved_model_dir, "poly_features.pkl"), "rb"))
        poly_eta_chieff_val = poly.transform(scaled_eta_chieff_val)
        c0_model = pickle.load(open(os.path.join(saved_model_dir, "c0_model.pkl"), "rb"))
        mean_other_coeffs = pickle.load(open(os.path.join(saved_model_dir, "mean_other_coeffs.pkl"), "rb"))

        predicted_c0 = c0_model.predict(poly_eta_chieff_val)
        predicted_coeffs[:, 0] = predicted_c0
        if n_components > 1:
            predicted_coeffs[:, 1:] = np.tile(mean_other_coeffs, (val_raw_params.shape[0], 1))
    
    end_time = time.time()
    runtime_ms = (end_time - start_time) * 1000

    predicted_x_padded = svd.inverse_transform(predicted_coeffs)

    per_sample_losses = []
    for i in range(len(val_raw_params)):
        true_len = len(val_x_series[i])
        loss = rms_relative_error(predicted_x_padded[i, :true_len], val_x_series[i])
        per_sample_losses.append(loss)
    
    total_loss = np.mean(per_sample_losses)

    # --- Generate Scorecard ---
    scorecard = {
        "approach": model_name,
        "approach_number": approach_number,
        "benchmark": benchmark_name,
        "agent": agent_name,
        "parameterization": parameterization,
        "time_convention": "t0_at_end",
        "loss": total_loss,
        "loss_components": {"rms_relative_error_x": total_loss},
        "runtime_ms": runtime_ms,
        "n_train": len(train_raw_params),
        "n_val": len(val_raw_params),
        "n_params": n_components,
        "notes": f"Model trained with {parameterization} parameterization. Evaluated loss: {total_loss:.4f}."
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    print(f"Scorecard generated for {model_name}. Loss: {total_loss:.4f}, Runtime: {runtime_ms:.2f} ms")

    # --- Generate Changelog Entry ---
    changelog_entry = f"""
## [D-{approach_number:02d}] {model_name} ({parameterization} parameterization)
- **Time**: {time.strftime('%Y-%m-%d %H:%M', time.localtime())}
- **Benchmark**: {benchmark_name}
- **Method**: SVD decomposition ({n_components} basis vectors) + {model_name.replace('SVD_', '').replace('_Raw', '').replace('_EtaChiEff', '').replace('_M1M2S1S2', '').replace('PhenomFit_Poly', 'PhenomenologicalFit_Poly')}
- **Parameterization**: {parameterization}
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: {total_loss:.4f} (rms_relative_error on x(t))
- **Eval time**: {runtime_ms:.2f} ms
- **Key observations**:
  - Model trained with {parameterization} parameters.
- **Next idea**: Continue implementing other models and reparameterizations.
"""
    return changelog_entry # Return the entry instead of writing directly

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamics benchmark models.")
    parser.add_argument("--agent", type=str, default="gemini25_flash", help="Agent name.")
    parser.add_argument("--benchmark", type=str, default="dynamics", help="Benchmark name.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to run.")
    parser.add_argument("--approach_number", type=int, required=True, help="Approach number for changelog.")
    parser.add_argument("--parameterization", type=str, default="raw_6d", help="Parameterization to use.")
    parser.add_argument("--n_components", type=int, default=10, help="Number of SVD components.")
    parser.add_argument("--poly_degree", type=int, default=2, help="Degree for polynomial features.")
    
    args = parser.parse_args()
    
    # Store changelog entries
    all_changelog_entries = []
    
    changelog_entry = run_model(args)
    
    # This will be replaced by iterative calls to run_model for each specific model
    # For now, just a placeholder to keep the structure.
    # The actual execution will be done in the main agent flow.
    
    # The main agent flow will collect the entries and append them to CHANGELOG.md at once.
    # So, for now, this script will just print the entry.
    print("\n--- Changelog Entry ---\n")
    print(changelog_entry)
