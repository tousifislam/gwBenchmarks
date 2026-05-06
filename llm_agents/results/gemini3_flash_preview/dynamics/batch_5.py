import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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

if __name__ == "__main__":
    # 19. Linear (raw)
    train_and_eval("lr_raw", 19, "raw", LinearRegression(), "Linear Regression on raw params.")

    # 20. Ridge (raw)
    train_and_eval("ridge_raw", 20, "raw", Ridge(alpha=1.0), "Ridge Regression on raw params.")

    # 21. Lasso (raw)
    train_and_eval("lasso_raw", 21, "raw", Lasso(alpha=0.01), "Lasso Regression on raw params.")
