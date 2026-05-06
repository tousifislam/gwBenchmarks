import numpy as np
import joblib
import os
import json
from pysr import PySRRegressor
from llm_agents.results.gemini3_flash_preview.dynamics.utils import evaluate_model, save_approach, get_reparameterized_params

# Load data
X_train_raw, y_train, X_val_raw, y_val, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/dynamics/data_cache/data.pkl")

def train_pysr(model_name, approach_num, params_type, n_coeffs=1):
    model_dir = f"llm_agents/results/gemini3_flash_preview/dynamics/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_raw[:100], type=params_type)
    
    model = PySRRegressor(
        niterations=2,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log"],
        maxsize=10,
        populations=5,
        procs=1,
        loss="loss(prediction, target) = (prediction - target)^2",
    )
    
    model.fit(X_train, y_train_reduced[:100, 0])
    joblib.dump(model, f"{model_dir}/saved_model/model.pkl")
    
    # Backup for rest of coeffs
    from sklearn.linear_model import LinearRegression
    backup = LinearRegression()
    backup.fit(get_reparameterized_params(X_train_raw, type=params_type), y_train_reduced[:, 1:])
    joblib.dump(backup, f"{model_dir}/saved_model/backup.pkl")
    joblib.dump(svd, f"{model_dir}/saved_model/svd.pkl")
    
    def predict(X):
        X_re = get_reparameterized_params(X, type=params_type)
        res1 = model.predict(X_re).reshape(-1, 1)
        res_rest = backup.predict(X_re)
        coeffs = np.column_stack([res1, res_rest])
        return svd.inverse_transform(coeffs)
        
    loss, runtime_ms, val_losses, _ = evaluate_model(model_dir, predict, X_val_raw, y_val)
    _, _, train_losses, _ = evaluate_model(model_dir, predict, X_train_raw, y_train)
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 1, "PySR on first coefficient.", model_dir, train_losses, val_losses)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    train_pysr("pysr_raw", 18, "raw")
