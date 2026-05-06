import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
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
    
    with open(f"{model_dir}/predict.py", "w") as f:
        f.write(f"""import numpy as np
import joblib
import os
from llm_agents.results.gemini3_flash_preview.validity.utils import get_reparameterized_params

model_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(model_dir, "saved_model/model.pkl"))

def predict(X):
    X_reparam = get_reparameterized_params(np.atleast_2d(X), type='{params_type}')
    return model.predict(X_reparam)
""")
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # 1. GPR (RBF, raw)
    kernel1 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=5, random_state=42)
    train_and_eval("gpr_rbf_raw", 1, "raw", gpr1, "GPR with RBF kernel on raw params.")

    # 2. GPR (Matern 1.5, eff)
    kernel2 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
    gpr2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=5, random_state=42)
    train_and_eval("gpr_matern15_eff", 2, "effective_spins", gpr2, "GPR with Matern 1.5 on effective spins.")

    # 3. RF (raw)
    rf3 = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_eval("rf_raw", 3, "raw", rf3, "Random Forest on raw params.")

    # 4. XGBoost (raw)
    xgb4 = XGBRegressor(n_estimators=100, random_state=42)
    train_and_eval("xgb_raw", 4, "raw", xgb4, "XGBoost on raw params.")

    # 5. MLP (raw)
    mlp5 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
    train_and_eval("mlp_raw", 5, "raw", mlp5, "MLP on raw params.")
