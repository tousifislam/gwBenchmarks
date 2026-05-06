import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
    # 18. Linear (raw)
    train_and_eval("lr_raw", 18, "raw", LinearRegression(), "Linear Regression on raw params.")

    # 19. Ridge (raw)
    train_and_eval("ridge_raw", 19, "raw", Ridge(alpha=1.0), "Ridge Regression on raw params.")

    # 20. Lasso (raw)
    train_and_eval("lasso_raw", 20, "raw", Lasso(alpha=0.1), "Lasso Regression on raw params.")
