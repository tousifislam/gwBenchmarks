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
from scipy.optimize import curve_fit
from sklearn.decomposition import TruncatedSVD


# Symbolic Regression (PySR and gplearn) - Will attempt to use if possible
# from pysr import PySRRegressor # Blocked due to Julia dependency issues
# from gplearn.genetic import SymbolicRegressor # Blocked due to Julia dependency issues


# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the mean_fd_mismatch function from gwbenchmarks/metrics.py
from gwbenchmarks.metrics import mean_fd_mismatch

# --- Utility Functions ---
def load_analytic_data(file_path):
    """Loads analytic data from an HDF5 file and decomposes h22 into amplitude and phase."""
    q_params = []
    h22_series = []
    t_series = []
    sim_ids = []
    dt_geometric = None

    with h5py.File(file_path, "r") as f:
        dt_geometric = f.attrs["dt_geometric"]
        # Assuming simulations are under the "sims" group
        for sim_id in f["sims"].keys():
            group = f["sims"][sim_id]
            q_params.append(group.attrs["q"])
            
            t = group["t"][:]
            h22_real = group["h22_real"][:]
            h22_imag = group["h22_imag"][:]
            h22 = h22_real + 1j * h22_imag
            
            h22_series.append(h22)
            t_series.append(t)
            sim_ids.append(sim_id) # Store sim_id if needed later
            
    return np.array(q_params), h22_series, t_series, dt_geometric

def preprocess_h22_series(h22_series, t_series, target_length=None):
    """Aligns waveforms to peak amplitude and pads to a common length."""
    # This is a simplified version; proper alignment (e.g., using time-of-peak) is more complex.
    # For now, we'll pad to max length and assume rough alignment.
    if target_length is None:
        target_length = max(len(ts) for ts in h22_series)
    
    padded_h22_series = []
    padded_t_series = []
    for i, h22 in enumerate(h22_series):
        t = t_series[i]
        
        # Find peak amplitude index for rough alignment (simple for now)
        amp = np.abs(h22)
        peak_idx = np.argmax(amp)
        
        # Shift time to align peaks (simple alignment: make peak time relative)
        t_shifted = t - t[peak_idx]

        if len(h22) < target_length:
            pad_amount = target_length - len(h22)
            # Pad with zeros or edge values, or more sophisticated tapering
            padded_h22 = np.pad(h22, (0, pad_amount), 'constant') # Padding with zeros
            padded_t = np.pad(t_shifted, (0, pad_amount), 'edge') # Padding with last time value
        else:
            padded_h22 = h22[:target_length]
            padded_t = t_shifted[:target_length]

        padded_h22_series.append(padded_h22)
        padded_t_series.append(padded_t)
    
    return np.array(padded_h22_series), np.array(padded_t_series), target_length


# --- Reparameterization Functions for q ---
def reparameterize_q(q_raw, param_type):
    if param_type == "q":
        return q_raw.reshape(-1, 1)
    elif param_type == "eta": # Symmetric mass ratio
        return (q_raw / (1 + q_raw)**2).reshape(-1, 1)
    elif param_type == "delta_m": # Mass difference parameter
        return ((q_raw - 1) / (q_raw + 1)).reshape(-1, 1)
    elif param_type == "sqrt_eta":
        return np.sqrt(q_raw / (1 + q_raw)**2).reshape(-1, 1)
    elif param_type == "eta_power_fifth":
        return (q_raw / (1 + q_raw)**2)**(1/5.0).reshape(-1, 1)
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
    plt.ylabel(f'{benchmark_name.capitalize()} Loss (Mean FD Mismatch)')
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

    # --- Load Data and Preprocess ---
    print("Loading training data...")
    train_q_raw, train_h22_series, train_t_series, dt_geometric_val = load_analytic_data(train_data_path)
    print("Loading validation data...")
    val_q_raw, val_h22_series, val_t_series, _ = load_analytic_data(val_data_path)

    # Determine common length for SVD from training data
    max_train_length = max(len(ts) for ts in train_h22_series)

    # Preprocess h22 time series
    padded_train_h22, padded_train_t, _ = preprocess_h22_series(train_h22_series, train_t_series, max_train_length)
    padded_val_h22, padded_val_t, _ = preprocess_h22_series(val_h22_series, val_t_series, max_train_length)
    
    # Reparameterize q
    train_q_reparam = reparameterize_q(train_q_raw, parameterization_type)
    val_q_reparam = reparameterize_q(val_q_raw, parameterization_type)

    # --- SVD Preprocessing (for models using SVD) ---
    n_components = args.n_components # Number of SVD components
    svd_model = None
    train_features = None
    val_features = None

    if "SVD" in model_name: # Check if SVD is part of the model name
        print(f"Performing SVD with {n_components} components...")
        # Stack real and imaginary parts to perform SVD on both simultaneously
        train_h22_flat = np.hstack([np.real(padded_train_h22), np.imag(padded_train_h22)])
        val_h22_flat = np.hstack([np.real(padded_val_h22), np.imag(padded_val_h22)])

        svd_model = TruncatedSVD(n_components=n_components)
        train_features = svd_model.fit_transform(train_h22_flat)
        val_features = svd_model.transform(val_h22_flat)
        print(f"SVD explained variance ratio: {np.sum(svd_model.explained_variance_ratio_):.4f}")


    # --- Model Training ---
    model = None
    expression_string = "Not yet derived."

    # Define a generic damped sinusoid function for fitting to real part
    def damped_sinusoid_real(t, A, omega, phi, tau):
        # h(t) = A * exp(-t/tau) * cos(omega * t + phi)
        return A * np.exp(-t / tau) * np.cos(omega * t + phi)
    
    # Define a generic damped sinusoid function for complex fitting
    def damped_sinusoid_complex(t, A, omega, phi, tau):
        # h(t) = A * exp(-t/tau) * exp(-1j * (omega * t + phi))
        return A * np.exp(-t / tau) * np.exp(-1j * (omega * t + phi))


    if model_name.startswith("PhenomFit_PolyAP") or model_name.startswith("Polynomial_Fit_AP"): # Group both Amplitude/Phase Polynomial Fits
        print(f"Implementing Phenomenological Polynomial Amplitude and Phase Model for {model_name}...")
        
        # Decompose h22 into amplitude and phase
        train_amplitude_series = [np.abs(h22) for h22 in padded_train_h22]
        train_phase_series = [np.unwrap(np.angle(h22)) for h22 in padded_train_h22]

        # Ensure consistent time axis for all waveforms
        common_t_axis = padded_train_t[0]
        
        poly_q_degree = args.poly_degree
        poly_t_degree = args.poly_t_degree

        # --- Amplitude Model ---
        # Create (t, q) input for fitting coefficients
        X_amp_fit = np.array([(common_t_axis[t_idx], train_q_reparam.flatten()[q_idx]) 
                              for q_idx in range(len(train_q_reparam)) 
                              for t_idx in range(max_train_length)])
        y_amp_fit = np.array([train_amplitude_series[q_idx][t_idx] 
                              for q_idx in range(len(train_q_reparam)) 
                              for t_idx in range(max_train_length)])

        # Fit a simple linear regression for amplitude (very basic placeholder)
        amp_poly_features = PolynomialFeatures(degree=poly_q_degree + poly_t_degree) # Combined features
        X_amp_poly = amp_poly_features.fit_transform(X_amp_fit)
        lr_amp = LinearRegression()
        lr_amp.fit(X_amp_poly, y_amp_fit)
        
        model_amp = lr_amp
        model_amp_poly_features = amp_poly_features


        # --- Phase Model ---
        # Similar simplified approach for phase
        X_phase_fit = X_amp_fit # Same (t,q) pairs
        y_phase_fit = np.array([train_phase_series[q_idx][t_idx] 
                                for q_idx in range(len(train_q_reparam)) 
                                for t_idx in range(max_train_length)])
        
        phase_poly_features = PolynomialFeatures(degree=poly_q_degree + poly_t_degree) # Combined features
        X_phase_poly = phase_poly_features.fit_transform(X_phase_fit)
        lr_phase = LinearRegression()
        lr_phase.fit(X_phase_poly, y_phase_fit)

        model_phase = lr_phase
        model_phase_poly_features = phase_poly_features

        model = {
            "model_amp": model_amp,
            "model_amp_poly_features": model_amp_poly_features,
            "model_phase": model_phase,
            "model_phase_poly_features": model_phase_poly_features,
            "poly_q_degree": poly_q_degree,
            "poly_t_degree": poly_t_degree, 
            "common_t_axis": common_t_axis
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        # Construct the expression string
        expression_string = f"Phenomenological Polynomial Amplitude and Phase Model (A(t,q), Phi(t,q)): q-deg={poly_q_degree}, t-deg={poly_t_degree}"

    elif model_name.startswith("DampedSinusoid"):
        print(f"Implementing Damped Sinusoid Model for {model_name}...")

        # Store fitted parameters for each q
        q_to_damped_params_r = {} # A, omega, phi, tau for real part
        
        for q_idx, q_val in enumerate(train_q_raw):
            current_t = padded_train_t[q_idx]
            current_h22_real = np.real(padded_train_h22[q_idx])

            # Initial guess for parameters: A, omega, phi, tau
            initial_guess = [np.max(np.abs(current_h22_real)), 
                             2 * np.pi / (current_t[-1] - current_t[0] + 1e-10), # Approx freq
                             0.0, 
                             (current_t[-1] - current_t[0] + 1e-10) / 5.0] # Approx decay time

            try:
                popt_r, pcov_r = curve_fit(damped_sinusoid_real, current_t, current_h22_real, p0=initial_guess, maxfev=5000)
                q_to_damped_params_r[q_val] = popt_r
            except RuntimeError:
                print(f"Warning: curve_fit failed for q={q_val}. Using default parameters.")
                q_to_damped_params_r[q_val] = initial_guess # Use initial guess if fit fails
        
        # Now, fit each parameter (A, omega, phi, tau) as a polynomial of q
        q_params_array = np.array(list(q_to_damped_params_r.keys())).reshape(-1, 1)
        
        # Extract coefficients for A, omega, phi, tau
        A_coeffs = np.array([params[0] for params in q_to_damped_params_r.values()])
        omega_coeffs = np.array([params[1] for params in q_to_damped_params_r.values()])
        phi_coeffs = np.array([params[2] for params in q_to_damped_params_r.values()])
        tau_coeffs = np.array([params[3] for params in q_to_damped_params_r.values()])

        # Fit A, omega, phi, tau as polynomial of q
        poly_q_degree = args.poly_degree
        poly_features_q = PolynomialFeatures(degree=poly_q_degree)
        q_poly = poly_features_q.fit_transform(train_q_reparam) # Use reparameterized q

        lr_A = LinearRegression().fit(q_poly, A_coeffs)
        lr_omega = LinearRegression().fit(q_poly, omega_coeffs)
        lr_phi = LinearRegression().fit(q_poly, phi_coeffs)
        lr_tau = LinearRegression().fit(q_poly, tau_coeffs)
        
        model = {
            "lr_A": lr_A,
            "lr_omega": lr_omega,
            "lr_phi": lr_phi,
            "lr_tau": lr_tau,
            "poly_features_q": poly_features_q,
            "common_t_axis": padded_train_t[0]
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

        # Construct expression string
        expression_string = f"Damped Sinusoid Model: h(t,q) = A(q) * exp(-t/tau(q)) * cos(omega(q)*t + phi(q))"
    elif model_name.startswith("Composite_PNQNM"):
        print(f"Implementing Composite PN+QNM Model for {model_name}...")

        # This is a highly simplified composite model for demonstration purposes.
        # It will combine a polynomial fit for early times (PN-like)
        # and a damped sinusoid for late times (QNM-like)
        
        # Determine a rough transition point (e.g., half-way through the waveform)
        transition_idx = max_train_length // 2
        transition_time = padded_train_t[0][transition_idx]

        # Fit PN-like inspiral (e.g., polynomial of time)
        # We will fit the complex h22 directly as a polynomial in time for early times
        
        # Store fitted PN parameters for each q
        q_to_pn_params = {}

        for q_idx, q_val in enumerate(train_q_raw):
            current_t = padded_train_t[q_idx][:transition_idx]
            current_h22 = padded_train_h22[q_idx][:transition_idx]
            
            # Use complex polynomial fit
            pn_poly_real = np.polynomial.polynomial.polyfit(current_t, np.real(current_h22), args.poly_t_degree)
            pn_poly_imag = np.polynomial.polynomial.polyfit(current_t, np.imag(current_h22), args.poly_t_degree)
            q_to_pn_params[q_val] = (pn_poly_real, pn_poly_imag)
        
        # Fit PN coefficients as polynomials of q
        q_params_array = np.array(list(q_to_pn_params.keys())).reshape(-1, 1)
        
        pn_coeffs_real_list = [params[0] for params in q_to_pn_params.values()] # list of arrays
        pn_coeffs_imag_list = [params[1] for params in q_to_pn_params.values()] # list of arrays

        # Convert to numpy array to fit coefficients for each power of t
        pn_coeffs_real = np.array(pn_coeffs_real_list) # Shape: (n_simulations, poly_t_degree + 1)
        pn_coeffs_imag = np.array(pn_coeffs_imag_list) # Shape: (n_simulations, poly_t_degree + 1)

        lr_pn_real = []
        lr_pn_imag = []
        poly_features_q_pn = PolynomialFeatures(degree=args.poly_degree)
        q_poly_pn = poly_features_q_pn.fit_transform(train_q_reparam)

        for i in range(args.poly_t_degree + 1):
            lr_real = LinearRegression().fit(q_poly_pn, pn_coeffs_real[:, i])
            lr_imag = LinearRegression().fit(q_poly_pn, pn_coeffs_imag[:, i])
            lr_pn_real.append(lr_real)
            lr_pn_imag.append(lr_imag)
        
        # Fit QNM-like ringdown (damped sinusoid) for late times
        q_to_qnm_params = {} # A, omega, phi, tau for complex h22

        for q_idx, q_val in enumerate(train_q_raw):
            current_t = padded_train_t[q_idx][transition_idx:]
            current_h22 = padded_train_h22[q_idx][transition_idx:]

            # Initial guess for parameters: A, omega, phi, tau
            # Complex fit is hard, simplifying to fit to real part
            current_h22_amp = np.abs(current_h22)
            current_h22_phase = np.unwrap(np.angle(current_h22))
            
            initial_guess = [np.max(current_h22_amp),
                             2 * np.pi / (current_t[-1] - current_t[0] + 1e-10),
                             current_h22_phase[0],
                             (current_t[-1] - current_t[0] + 1e-10) / 5.0]

            try:
                # We fit the real part's parameters
                # This is a simplification; ideally, we fit amplitude and phase
                popt, pcov = curve_fit(damped_sinusoid_real, current_t, np.real(current_h22), p0=initial_guess, maxfev=5000)
                q_to_qnm_params[q_val] = popt
            except RuntimeError:
                print(f"Warning: curve_fit (QNM) failed for q={q_val}. Using default parameters.")
                q_to_qnm_params[q_val] = initial_guess

        # Fit QNM parameters (A, omega, phi, tau) as polynomials of q
        QNM_params_array = np.array(list(q_to_qnm_params.keys())).reshape(-1, 1)

        A_qnm_coeffs = np.array([params[0] for params in q_to_qnm_params.values()])
        omega_qnm_coeffs = np.array([params[1] for params in q_to_qnm_params.values()])
        phi_qnm_coeffs = np.array([params[2] for params in q_to_qnm_params.values()])
        tau_qnm_coeffs = np.array([params[3] for params in q_to_qnm_params.values()])

        lr_A_qnm = LinearRegression().fit(q_poly_pn, A_qnm_coeffs)
        lr_omega_qnm = LinearRegression().fit(q_poly_pn, omega_qnm_coeffs)
        lr_phi_qnm = LinearRegression().fit(q_poly_pn, phi_qnm_coeffs)
        lr_tau_qnm = LinearRegression().fit(q_poly_pn, tau_qnm_coeffs)

        model = {
            "lr_pn_real": lr_pn_real,
            "lr_pn_imag": lr_pn_imag,
            "poly_features_q_pn": poly_features_q_pn,
            "lr_A_qnm": lr_A_qnm,
            "lr_omega_qnm": lr_omega_qnm,
            "lr_phi_qnm": lr_phi_qnm,
            "lr_tau_qnm": lr_tau_qnm,
            "transition_idx": transition_idx,
            "common_t_axis": padded_train_t[0]
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        expression_string = f"Composite PN+QNM Model: h(t,q) = PN_inspiral(t,q) + transition(t) * QNM_ringdown(t,q)"

    elif model_name.startswith("GPR_Waveform"):
        print(f"Implementing Gaussian Process Regressor on waveforms for {model_name}...")
        
        # For simplicity, we'll train two independent GPRs: one for real part, one for imaginary part.
        # This is not a multi-output GPR but two single-output GPRs.
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # Train GPR for real part
        gpr_real = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2, n_restarts_optimizer=10, random_state=0)
        gpr_real.fit(train_q_reparam, np.real(padded_train_h22)) # Input q, output is array of real parts
        
        # Train GPR for imaginary part
        gpr_imag = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2, n_restarts_optimizer=10, random_state=0)
        gpr_imag.fit(train_q_reparam, np.imag(padded_train_h22)) # Input q, output is array of imag parts

        model = {
            "gpr_real": gpr_real,
            "gpr_imag": gpr_imag
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

        expression_string = "Gaussian Process Regression on (Real, Imag) waveform components"
    elif model_name.startswith("MLP_Waveform"):
        print(f"Implementing MLP Regressor on waveforms for {model_name}...")
        
        # Train MLP for real part
        mlp_real = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0)
        mlp_real.fit(train_q_reparam, np.real(padded_train_h22))
        
        # Train MLP for imaginary part
        mlp_imag = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0)
        mlp_imag.fit(train_q_reparam, np.imag(padded_train_h22))

        model = {
            "mlp_real": mlp_real,
            "mlp_imag": mlp_imag
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

        expression_string = "Multi-Layer Perceptron Regression on (Real, Imag) waveform components"
    elif model_name.startswith("RandomForest_Waveform"):
        print(f"Implementing RandomForest Regressor on waveforms for {model_name}...")
        
        # Train RandomForest for real part
        rf_real = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_real.fit(train_q_reparam, np.real(padded_train_h22))
        
        # Train RandomForest for imaginary part
        rf_imag = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_imag.fit(train_q_reparam, np.imag(padded_train_h22))

        model = {
            "rf_real": rf_real,
            "rf_imag": rf_imag
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

        expression_string = "RandomForest Regression on (Real, Imag) waveform components"
    elif model_name.startswith("XGBoost_SVD"):
        print(f"Implementing XGBoost Regressor on SVD coefficients for {model_name}...")
        
        # Train XGBoost for SVD coefficients
        xgb_svd = XGBRegressor(n_estimators=100, random_state=0)
        xgb_svd.fit(train_q_reparam, train_features)

        model = {
            "xgb_svd": xgb_svd,
            "svd_model": svd_model # Need to save SVD model for reconstruction
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        expression_string = "XGBoost Regression on SVD coefficients"
    elif model_name.startswith("LightGBM_SVD"):
        print(f"Implementing LightGBM Regressor on SVD coefficients for {model_name}...")
        
        # Train LightGBM for SVD coefficients
        lgbm_base = LGBMRegressor(n_estimators=100, random_state=0)
        lgbm_svd = MultiOutputRegressor(lgbm_base) # Wrap with MultiOutputRegressor
        lgbm_svd.fit(train_q_reparam, train_features)

        model = {
            "lgbm_svd": lgbm_svd,
            "svd_model": svd_model # Need to save SVD model for reconstruction
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        expression_string = "LightGBM Regression on SVD coefficients"
    elif model_name.startswith("KNR_SVD"):
        print(f"Implementing K-Nearest Neighbors Regressor on SVD coefficients for {model_name}...")
        
        # Train KNR for SVD coefficients
        knr_svd = KNeighborsRegressor(n_neighbors=3) # Default to 3 neighbors
        knr_svd.fit(train_q_reparam, train_features)

        model = {
            "knr_svd": knr_svd,
            "svd_model": svd_model # Need to save SVD model for reconstruction
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        expression_string = "K-Nearest Neighbors Regression on SVD coefficients"
    elif model_name.startswith("KRR_SVD"):
        print(f"Implementing Kernel Ridge Regressor on SVD coefficients for {model_name}...")
        
        # Train KRR for SVD coefficients
        krr_svd = KernelRidge(alpha=1.0, kernel='rbf') # Default alpha and RBF kernel
        krr_svd.fit(train_q_reparam, train_features)

        model = {
            "krr_svd": krr_svd,
            "svd_model": svd_model # Need to save SVD model for reconstruction
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        expression_string = "Kernel Ridge Regression on SVD coefficients"
    elif model_name.startswith("SVR_SVD"):
        print(f"Implementing Support Vector Regressor on SVD coefficients for {model_name}...")
        
        # Train SVR for SVD coefficients
        svr_base = SVR(kernel='rbf', C=100, gamma=0.1) # Default parameters, can be tuned
        svr_svd = MultiOutputRegressor(svr_base) # Wrap with MultiOutputRegressor
        svr_svd.fit(train_q_reparam, train_features)

        model = {
            "svr_svd": svr_svd,
            "svd_model": svd_model # Need to save SVD model for reconstruction
        }
        with open(os.path.join(saved_model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        expression_string = "Support Vector Regression on SVD coefficients"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # --- Save Common Components ---
    with open(os.path.join(saved_model_dir, "q_scaler.pkl"), "wb") as f: # Renamed from param_scaler
        pickle.dump(None, f) # No scaling of q_reparam for now as we use it directly in PolynomialFeatures
    with open(os.path.join(saved_model_dir, "max_length.pkl"), "wb") as f:
        pickle.dump(max_train_length, f)
    with open(os.path.join(saved_model_dir, "padded_t.pkl"), "wb") as f:
        pickle.dump(padded_train_t[0], f) # Save one padded time array for reconstruction


    print(f"Model {model_name} training complete and saved.")

    # --- Prediction and Evaluation ---
    start_time = time.time()
    
    pred_h22_series = [] # This will store the predicted complex waveforms

    # This prediction logic will need to be significantly expanded for full analytic waveforms
    # For now, it will reconstruct a simple h22 based on the dummy polynomial fit
    if model_name.startswith("PhenomFit_PolyAP") or model_name.startswith("Polynomial_Fit_AP"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        model_amp = loaded_model_data["model_amp"]
        model_amp_poly_features = loaded_model_data["model_amp_poly_features"]
        model_phase = loaded_model_data["model_phase"]
        model_phase_poly_features = loaded_model_data["model_phase_poly_features"]
        common_t_axis = loaded_model_data["common_t_axis"]

        # Predict for each validation q and time
        for q_idx in range(len(val_q_reparam)):
            current_q = val_q_reparam[q_idx]
            
            # Create (t, q) input for prediction
            X_val_amp_fit = np.array([(t_val, current_q[0]) for t_val in common_t_axis])
            X_val_amp_poly = model_amp_poly_features.transform(X_val_amp_fit)
            predicted_amplitude = model_amp.predict(X_val_amp_poly)

            X_val_phase_fit = X_val_amp_fit # Same (t,q) pairs
            X_val_phase_poly = model_phase_poly_features.transform(X_val_amp_poly) # Use amp_poly_features for consistency
            predicted_phase = model_phase.predict(X_val_phase_poly)

            predicted_h22 = predicted_amplitude * np.exp(-1j * predicted_phase)
            pred_h22_series.append(predicted_h22)
        
        pred_h22_series = np.array(pred_h22_series)
    elif model_name.startswith("DampedSinusoid"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        lr_A = loaded_model_data["lr_A"]
        lr_omega = loaded_model_data["lr_omega"]
        lr_phi = loaded_model_data["lr_phi"]
        lr_tau = loaded_model_data["lr_tau"]
        poly_features_q = loaded_model_data["poly_features_q"]
        common_t_axis = loaded_model_data["common_t_axis"]

        for q_idx in range(len(val_q_reparam)):
            current_q_reparam = val_q_reparam[q_idx].reshape(1, -1)
            q_poly_val = poly_features_q.transform(current_q_reparam)

            # Predict parameters for current q
            predicted_A = lr_A.predict(q_poly_val)[0]
            predicted_omega = lr_omega.predict(q_poly_val)[0]
            predicted_phi = lr_phi.predict(q_poly_val)[0]
            predicted_tau = lr_tau.predict(q_poly_val)[0]

            # Reconstruct waveform
            predicted_h22_real = damped_sinusoid_real(common_t_axis, predicted_A, predicted_omega, predicted_phi, predicted_tau)
            # For simplicity, assume imaginary part is 0. This needs refinement.
            predicted_h22 = predicted_h22_real + 1j * np.zeros_like(predicted_h22_real) 
            pred_h22_series.append(predicted_h22)
        pred_h22_series = np.array(pred_h22_series)
    elif model_name.startswith("Composite_PNQNM"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        lr_pn_real = loaded_model_data["lr_pn_real"]
        lr_pn_imag = loaded_model_data["lr_pn_imag"]
        poly_features_q_pn = loaded_model_data["poly_features_q_pn"]
        lr_A_qnm = loaded_model_data["lr_A_qnm"]
        lr_omega_qnm = loaded_model_data["lr_omega_qnm"]
        lr_phi_qnm = loaded_model_data["lr_phi_qnm"]
        lr_tau_qnm = loaded_model_data["lr_tau_qnm"]
        transition_idx = loaded_model_data["transition_idx"]
        common_t_axis = loaded_model_data["common_t_axis"]
        
        for q_idx in range(len(val_q_reparam)):
            current_q_reparam = val_q_reparam[q_idx].reshape(1, -1)
            q_poly_val = poly_features_q_pn.transform(current_q_reparam)

            # Predict PN coefficients
            pn_poly_coeffs_real = np.array([lr.predict(q_poly_val)[0] for lr in lr_pn_real])
            pn_poly_coeffs_imag = np.array([lr.predict(q_poly_val)[0] for lr in lr_pn_imag])
            
            # Reconstruct PN inspiral
            predicted_pn_h22_real = np.polynomial.polynomial.polyval(common_t_axis[:transition_idx], pn_poly_coeffs_real)
            predicted_pn_h22_imag = np.polynomial.polynomial.polyval(common_t_axis[:transition_idx], pn_poly_coeffs_imag)
            predicted_pn_h22 = predicted_pn_h22_real + 1j * predicted_pn_h22_imag

            # Predict QNM parameters
            predicted_A_qnm = lr_A_qnm.predict(q_poly_val)[0]
            predicted_omega_qnm = lr_omega_qnm.predict(q_poly_val)[0]
            predicted_phi_qnm = lr_phi_qnm.predict(q_poly_val)[0]
            predicted_tau_qnm = lr_tau_qnm.predict(q_poly_val)[0]

            # Reconstruct QNM ringdown
            predicted_qnm_h22 = damped_sinusoid_complex(common_t_axis[transition_idx:], 
                                                          predicted_A_qnm, predicted_omega_qnm, 
                                                          predicted_phi_qnm, predicted_tau_qnm)
            
            # Combine PN and QNM (simple stitching for now, could use tanh)
            predicted_h22 = np.concatenate([predicted_pn_h22, predicted_qnm_h22])
            pred_h22_series.append(predicted_h22)
        pred_h22_series = np.array(pred_h22_series)
    elif model_name.startswith("GPR_Waveform"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        gpr_real = loaded_model_data["gpr_real"]
        gpr_imag = loaded_model_data["gpr_imag"]
        
        # Predict real and imaginary parts of the waveform
        predicted_real_h22 = gpr_real.predict(val_q_reparam)
        predicted_imag_h22 = gpr_imag.predict(val_q_reparam)
        
        # Combine into complex waveform
        pred_h22_series = predicted_real_h22 + 1j * predicted_imag_h22

    elif model_name.startswith("MLP_Waveform"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        mlp_real = loaded_model_data["mlp_real"]
        mlp_imag = loaded_model_data["mlp_imag"]
        
        # Predict real and imaginary parts of the waveform
        predicted_real_h22 = mlp_real.predict(val_q_reparam)
        predicted_imag_h22 = mlp_imag.predict(val_q_reparam)
        
        # Combine into complex waveform
        pred_h22_series = predicted_real_h22 + 1j * predicted_imag_h22
    elif model_name.startswith("RandomForest_Waveform"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        rf_real = loaded_model_data["rf_real"]
        rf_imag = loaded_model_data["rf_imag"]
        
        # Predict real and imaginary parts of the waveform
        predicted_real_h22 = rf_real.predict(val_q_reparam)
        predicted_imag_h22 = rf_imag.predict(val_q_reparam)
        
        # Combine into complex waveform
        pred_h22_series = predicted_real_h22 + 1j * predicted_imag_h22
    elif model_name.startswith("XGBoost_SVD"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        xgb_svd = loaded_model_data["xgb_svd"]
        svd_model = loaded_model_data["svd_model"] # Load SVD model

        # Predict SVD coefficients
        predicted_svd_features = xgb_svd.predict(val_q_reparam)
        
        # Reconstruct waveform from SVD coefficients
        reconstructed_h22_flat = svd_model.inverse_transform(predicted_svd_features)
        
        # Split into real and imaginary parts
        predicted_real_h22 = reconstructed_h22_flat[:, :max_train_length]
        predicted_imag_h22 = reconstructed_h22_flat[:, max_train_length:]

        pred_h22_series = predicted_real_h22 + 1j * predicted_imag_h22
    elif model_name.startswith("LightGBM_SVD"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        lgbm_svd = loaded_model_data["lgbm_svd"]
        svd_model = loaded_model_data["svd_model"] # Load SVD model

        # Predict SVD coefficients
        predicted_svd_features = lgbm_svd.predict(val_q_reparam)
        
        # Reconstruct waveform from SVD coefficients
        reconstructed_h22_flat = svd_model.inverse_transform(predicted_svd_features)
        
        # Split into real and imaginary parts
        predicted_real_h22 = reconstructed_h22_flat[:, :max_train_length]
        predicted_imag_h22 = reconstructed_h22_flat[:, max_train_length:]

        pred_h22_series = predicted_real_h22 + 1j * predicted_imag_h22
    elif model_name.startswith("KNR_SVD"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        knr_svd = loaded_model_data["knr_svd"]
        svd_model = loaded_model_data["svd_model"] # Load SVD model

        # Predict SVD coefficients
        predicted_svd_features = knr_svd.predict(val_q_reparam)
        
        # Reconstruct waveform from SVD coefficients
        reconstructed_h22_flat = svd_model.inverse_transform(predicted_svd_features)
        
        # Split into real and imaginary parts
        predicted_real_h22 = reconstructed_h22_flat[:, :max_train_length]
        predicted_imag_h22 = reconstructed_h22_flat[:, max_train_length:]

        pred_h22_series = predicted_real_h22 + 1j * predicted_imag_h22
    elif model_name.startswith("KRR_SVD"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        krr_svd = loaded_model_data["krr_svd"]
        svd_model = loaded_model_data["svd_model"] # Load SVD model

        # Predict SVD coefficients
        predicted_svd_features = krr_svd.predict(val_q_reparam)
        
        # Reconstruct waveform from SVD coefficients
        reconstructed_h22_flat = svd_model.inverse_transform(predicted_svd_features)
        
        # Split into real and imaginary parts
        predicted_real_h22 = reconstructed_h22_flat[:, :max_train_length]
        predicted_imag_h22 = reconstructed_h22_flat[:, max_train_length:]

        pred_h22_series = predicted_real_h22 + 1j * predicted_imag_h22
    elif model_name.startswith("SVR_SVD"):
        loaded_model_data = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
        svr_svd = loaded_model_data["svr_svd"]
        svd_model = loaded_model_data["svd_model"] # Load SVD model

        # Predict SVD coefficients
        predicted_svd_features = svr_svd.predict(val_q_reparam)
        
        # Reconstruct waveform from SVD coefficients
        reconstructed_h22_flat = svd_model.inverse_transform(predicted_svd_features)
        
        # Split into real and imaginary parts
        predicted_real_h22 = reconstructed_h22_flat[:, :max_train_length]
        predicted_imag_h22 = reconstructed_h22_flat[:, max_train_length:]

        pred_h22_series = predicted_real_h22 + 1j * predicted_imag_h22
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    end_time = time.time()
    runtime_ms = (end_time - start_time) * 1000

    # Calculate loss using mean_fd_mismatch
    losses_per_sample = []
    
    # Masses to average over for mismatch calculation (from prompt)
    FD_MASSES_MSUN = [40.0, 80.0, 120.0, 160.0, 200.0]

    for i in range(len(val_q_raw)):
        loss = mean_fd_mismatch(pred_h22_series[i], padded_val_h22[i], 
                                dt_geometric=dt_geometric_val,
                                masses=FD_MASSES_MSUN)
        losses_per_sample.append(loss)
    
    total_loss = np.mean(losses_per_sample)
    
    # Save the derived expression string
    with open(os.path.join(model_dir, "expression.txt"), "w") as f:
        f.write(expression_string)

    # --- Generate Scorecard ---
    scorecard = {
        "approach": model_name,
        "approach_number": approach_number,
        "benchmark": benchmark_name,
        "agent": agent_name,
        "parameterization": parameterization_type,
        "loss": total_loss,
        "loss_components": {"mean_fd_mismatch": total_loss}, # Placeholder for more detailed components
        "runtime_ms": runtime_ms,
        "n_train": len(train_q_raw),
        "n_val": len(val_q_raw),
        "n_params": train_q_reparam.shape[1], # Number of input parameters
        "notes": f"Model trained with {parameterization_type} parameterization. Evaluated loss: {total_loss:.4f}."
    }
    with open(os.path.join(model_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=4)
    print(f"Scorecard generated for {model_name}. Loss: {total_loss:.4f}, Runtime: {runtime_ms:.2f} ms")

    # --- Generate Changelog Entry ---
    changelog_entry = f"""
## [A-{approach_number:02d}] {model_name} ({parameterization_type} parameterization)
- **Time**: {time.strftime('%Y-%m-%d %H:%M', time.localtime())}
- **Benchmark**: {benchmark_name}
- **Method**: {model_name.replace('_', ' ').replace('Fit', ' Fit')}
- **Parameterization**: {parameterization_type}
- **Loss**: {total_loss:.4f} (Mean FD Mismatch)
- **Eval time**: {runtime_ms:.2f} ms
- **Key observations**:
  - Model trained with {parameterization_type} parameters.
  - Expression: {expression_string}
- **Next idea**: Continue implementing other models and reparameterizations, focusing on full analytic h22(t,q) forms.
"""
    return changelog_entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Analytic benchmark models.")
    parser.add_argument("--agent", type=str, default="gemini25_flash", help="Agent name.")
    parser.add_argument("--benchmark", type=str, default="analytic", help="Benchmark name.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to run.")
    parser.add_argument("--approach_number", type=int, required=True, help="Approach number for changelog.")
    parser.add_argument("--parameterization", type=str, default="q", help="Parameterization to use (q, eta, delta_m, etc.).")
    parser.add_argument("--poly_degree", type=int, default=2, help="Degree for polynomial features.")
    parser.add_argument("--poly_t_degree", type=int, default=3, help="Degree for polynomial features (for time-dependence).")
    parser.add_argument("--n_components", type=int, default=10, help="Number of SVD components for dimensionality reduction.")
    
    args = parser.parse_args()
    
    changelog_entry = run_model(args)
    
    # The main agent flow will collect the entries and append them to CHANGELOG.md at once.
    # So, for now, this script will just print the entry.
    print("\n--- Changelog Entry ---\n")
    print(changelog_entry)
