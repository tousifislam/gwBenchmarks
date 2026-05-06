import numpy as np
import joblib
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from llm_agents.results.gemini3_flash_preview.validity.utils import evaluate_model, save_approach, get_reparameterized_params

# Load data
X_train_raw, y_train, X_val_raw, y_val = joblib.load("llm_agents/results/gemini3_flash_preview/validity/data_cache/data.pkl")

def train_and_eval(model_name, approach_num, params_type, model_obj, notes):
    model_dir = f"llm_agents/results/gemini3_flash_preview/validity/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_raw, type=params_type)
    X_val = get_reparameterized_params(X_val_raw, type=params_type)
    
    model_obj.fit(X_train, y_train)
    joblib.dump(model_obj, f"{model_dir}/saved_model/model.pkl")
    
    predict_fn = lambda X: model_obj.predict(get_reparameterized_params(X, type=params_type))
    loss, runtime_ms, val_errors, _ = evaluate_model(model_dir, predict_fn, X_val_raw, y_val)
    _, _, train_errors, _ = evaluate_model(model_dir, predict_fn, X_train_raw, y_train)
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 100, notes, model_dir, train_errors, val_errors)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # 6. KRR (RBF, raw)
    krr6 = KernelRidge(kernel='rbf', alpha=0.1)
    train_and_eval("krr_rbf_raw", 6, "raw", krr6, "Kernel Ridge Regression with RBF kernel.")

    # 7. Poly (deg 3, raw)
    poly7 = Pipeline([("poly", PolynomialFeatures(degree=3)), ("linear", LinearRegression())])
    train_and_eval("poly3_raw", 7, "raw", poly7, "Polynomial Regression (degree 3) on raw params.")

    # 9. MLP (eff)
    mlp9 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
    train_and_eval("mlp_eff", 9, "effective_spins", mlp9, "MLP on effective spins.")

    # 10. GPR (Matern 2.5, interaction)
    kernel10 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
    gpr10 = GaussianProcessRegressor(kernel=kernel10, n_restarts_optimizer=5, random_state=42)
    train_and_eval("gpr_matern25_interaction", 10, "interaction", gpr10, "GPR with Matern 2.5 on interaction terms.")
