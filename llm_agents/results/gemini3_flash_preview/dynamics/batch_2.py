import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
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
    # 6. XGBoost (raw)
    xgb6 = MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42))
    train_and_eval("svd_xgb_raw", 6, "raw", xgb6, "SVD + XGBoost on raw params.")

    # 7. LightGBM (raw)
    lgbm7 = MultiOutputRegressor(LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    train_and_eval("svd_lgbm_raw", 7, "raw", lgbm7, "SVD + LightGBM on raw params.")

    # 8. CatBoost (raw)
    cat8 = MultiOutputRegressor(CatBoostRegressor(n_estimators=100, random_state=42, verbose=False))
    train_and_eval("svd_cat_raw", 8, "raw", cat8, "SVD + CatBoost on raw params.")

    # 9. Poly (deg 3, raw)
    poly9 = Pipeline([("poly", PolynomialFeatures(degree=3)), ("linear", LinearRegression())])
    train_and_eval("svd_poly3_raw", 9, "raw", poly9, "SVD + Polynomial Regression (degree 3) on raw params.")

    # 10. Poly (deg 2, trig_anomaly)
    poly10 = Pipeline([("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())])
    train_and_eval("svd_poly2_trig", 10, "trig_anomaly", poly10, "SVD + Polynomial Regression (degree 2) on trig anomaly params.")
