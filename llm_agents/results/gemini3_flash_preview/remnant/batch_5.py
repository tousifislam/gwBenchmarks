import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR, NuSVR
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
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 1000, notes, model_dir, train_errors, val_errors)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # 18. Linear (raw)
    train_and_eval("lr_raw", 18, "raw", LinearRegression(), "Linear Regression on raw params.")

    # 19. Ridge (raw)
    train_and_eval("ridge_raw", 19, "raw", Ridge(alpha=1.0), "Ridge Regression on raw params.")

    # 20. SVR (RBF, raw)
    train_and_eval("svr_rbf_raw", 20, "raw", SVR(kernel='rbf'), "SVR with RBF kernel on raw params.")

    # 21. NuSVR (raw)
    train_and_eval("nusvr_raw", 21, "raw", NuSVR(), "NuSVR on raw params.")
    
    # 22. Linear (eff)
    train_and_eval("lr_eff", 22, "effective_spins", LinearRegression(), "Linear Regression on effective spins.")
