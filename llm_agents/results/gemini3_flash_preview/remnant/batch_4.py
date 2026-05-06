import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from llm_agents.results.gemini3_flash_preview.remnant.utils import evaluate_model, save_approach, get_reparameterized_params

# Load data
X_train_raw, y_train, X_val_raw, y_val = joblib.load("llm_agents/results/gemini3_flash_preview/remnant/data_cache/data.pkl")

def train_and_eval(model_name, approach_num, params_type, model_obj, notes):
    model_dir = f"llm_agents/results/gemini3_flash_preview/remnant/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_raw, type=params_type)
    X_val = get_reparameterized_params(X_val_raw, type=params_type)
    
    model_obj.fit(X_train, y_train)
    joblib.dump(model_obj, f"{model_dir}/saved_model/model.pkl")
    
    predict_fn = lambda X: model_obj.predict(get_reparameterized_params(X, type=params_type))
    loss, runtime_ms, val_errors, _ = evaluate_model(model_dir, predict_fn, X_val_raw, y_val)
    _, _, train_errors, _ = evaluate_model(model_dir, predict_fn, X_train_raw, y_train)
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 10000, notes, model_dir, train_errors, val_errors)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # 19. GPR (RBF, delta_m_chi_a)
    kernel19 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr19 = GaussianProcessRegressor(kernel=kernel19, n_restarts_optimizer=2, random_state=42)
    
    model_dir = f"llm_agents/results/gemini3_flash_preview/remnant/models/gpr_rbf_deltam"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    X_train = get_reparameterized_params(X_train_raw, type='delta_m_chi_a')
    gpr19.fit(X_train, y_train)
    joblib.dump(gpr19, f"{model_dir}/saved_model/model.pkl")
    predict_fn = lambda X: gpr19.predict(get_reparameterized_params(X, type='delta_m_chi_a'))
    loss, runtime_ms, val_errors, _ = evaluate_model(model_dir, predict_fn, X_val_raw, y_val)
    _, _, train_errors, _ = evaluate_model(model_dir, predict_fn, X_train_raw, y_train)
    save_approach("gpr_rbf_deltam", 19, "delta_m_chi_a", loss, runtime_ms, 1000, "GPR with RBF on delta_m + chi_a.", model_dir, train_errors, val_errors)
    print(f"Approach 19 (gpr_rbf_deltam) completed. Loss: {loss}")
    
    # 20. XGBoost (eff)
    xgb20 = XGBRegressor(n_estimators=100, random_state=42)
    train_and_eval("xgb_eff", 20, "effective_spins", xgb20, "XGBoost on effective spins.")

    # 21. LightGBM (eff)
    lgbm21 = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    train_and_eval("lgbm_eff", 21, "effective_spins", lgbm21, "LightGBM on effective spins.")

    # 22. RF (delta_m_chi_a)
    rf22 = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_eval("rf_deltam", 22, "delta_m_chi_a", rf22, "Random Forest on delta_m + chi_a.")

    # 23. MLP (delta_m_chi_a)
    mlp23 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    train_and_eval("mlp_deltam", 23, "delta_m_chi_a", mlp23, "MLP on delta_m + chi_a.")
