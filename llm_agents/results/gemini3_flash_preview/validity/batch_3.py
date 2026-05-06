import numpy as np
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import RBFInterpolator
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

class RBFModel:
    def fit(self, x, y):
        self.rbf = RBFInterpolator(x, y)
    def predict(self, x):
        return self.rbf(x)

if __name__ == "__main__":
    # 11. RBF (raw)
    rbf11 = RBFModel()
    train_and_eval("rbf_raw", 11, "raw", rbf11, "RBF Interpolation on raw params.")

    # 12. KNN (raw)
    knn12 = KNeighborsRegressor(n_neighbors=5)
    train_and_eval("knn_raw", 12, "raw", knn12, "K-Nearest Neighbors on raw params.")

    # 13. Ridge (boundary)
    ridge13 = Ridge(alpha=1.0)
    train_and_eval("ridge_boundary", 13, "boundary", ridge13, "Ridge Regression on boundary distance features.")

    # 14. RF (interaction)
    rf14 = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_eval("rf_interaction", 14, "interaction", rf14, "Random Forest on interaction terms.")

    # 15. XGBoost (log_q)
    xgb15 = XGBRegressor(n_estimators=100, random_state=42)
    train_and_eval("xgb_log_q", 15, "log_q", xgb15, "XGBoost on log mass ratio parameters.")
