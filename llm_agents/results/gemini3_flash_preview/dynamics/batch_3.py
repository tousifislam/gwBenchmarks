import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import RBFInterpolator
from llm_agents.results.gemini3_flash_preview.dynamics.utils import evaluate_model, save_approach, get_reparameterized_params

# Load data
X_train_raw, y_train, X_val_raw, y_val, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/dynamics/data_cache/data.pkl")

def train_and_eval(model_name, approach_num, params_type, model_obj, notes):
    model_dir = f"llm_agents/results/gemini3_flash_preview/dynamics/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_raw, type=params_type)
    X_val = get_reparameterized_params(X_val_raw, type=params_type)
    
    model_obj.fit(X_train, y_train_reduced)
    joblib.dump(model_obj, f"{model_dir}/saved_model/model.pkl")
    joblib.dump(svd, f"{model_dir}/saved_model/svd.pkl")
    
    def predict(X):
        X_re = get_reparameterized_params(X, type=params_type)
        coeffs = model_obj.predict(X_re)
        y_pred = svd.inverse_transform(coeffs)
        return y_pred
    
    loss, runtime_ms, val_losses, _ = evaluate_model(model_dir, predict, X_val_raw, y_val)
    _, _, train_losses, _ = evaluate_model(model_dir, predict, X_train_raw, y_train)
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 100, notes, model_dir, train_losses, val_losses)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

class RBFModel:
    def __init__(self, kernel='thin_plate_spline'):
        self.kernel = kernel
        self.rbf = None
    def fit(self, X, y):
        self.rbf = RBFInterpolator(X, y, kernel=self.kernel)
    def predict(self, X):
        return self.rbf(X)

if __name__ == "__main__":
    # 11. RBF Interpolation (raw)
    rbf11 = RBFModel()
    train_and_eval("rbf_raw", 11, "raw", rbf11, "RBF Interpolation on raw params.")

    # 12. Nearest Neighbor (raw)
    knn12 = KNeighborsRegressor(n_neighbors=5)
    train_and_eval("knn_raw", 12, "raw", knn12, "K-Nearest Neighbors on raw params.")

    # 13. MLP (eff_log_e)
    mlp13 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    train_and_eval("mlp_eff_log_e", 13, "eff_log_e", mlp13, "MLP on effective spin + log eccentricity.")

    # 14. RF (eff_log_e)
    rf14 = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_eval("rf_eff_log_e", 14, "eff_log_e", rf14, "Random Forest on effective spin + log eccentricity.")

    # 15. GPR (Matern 2.5, fully_transformed)
    kernel15 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
    gpr15 = GaussianProcessRegressor(kernel=kernel15, n_restarts_optimizer=2, random_state=42)
    train_and_eval("gpr_matern25_fully", 15, "fully_transformed", gpr15, "GPR with Matern 2.5 on fully transformed parameters.")
