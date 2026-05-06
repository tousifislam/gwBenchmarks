import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
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
    # 6. XGBoost (raw)
    xgb6 = XGBRegressor(n_estimators=100, random_state=42)
    train_and_eval("xgb_raw", 6, "raw", xgb6, "XGBoost on raw params.")

    # 7. LightGBM (raw)
    lgbm7 = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    train_and_eval("lgbm_raw", 7, "raw", lgbm7, "LightGBM on raw params.")

    # 8. CatBoost (raw)
    cat8 = CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
    train_and_eval("cat_raw", 8, "raw", cat8, "CatBoost on raw params.")

    # 9. Poly (deg 2, raw)
    poly9 = Pipeline([("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())])
    train_and_eval("poly2_raw", 9, "raw", poly9, "Polynomial Regression (degree 2) on raw params.")

    # 10. Poly (deg 2, delta_m_chi_a)
    poly10 = Pipeline([("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())])
    train_and_eval("poly2_deltam", 10, "delta_m_chi_a", poly10, "Polynomial Regression (degree 2) on delta_m + chi_a.")
