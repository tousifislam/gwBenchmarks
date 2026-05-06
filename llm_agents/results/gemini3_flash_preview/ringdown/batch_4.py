import numpy as np
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from llm_agents.results.gemini3_flash_preview.ringdown.utils import evaluate_model, save_approach, get_reparameterized_spin, transform_target, inverse_transform_target

# Load data
spin_train, r_train, i_train, spin_val, r_val, i_val = joblib.load("llm_agents/results/gemini3_flash_preview/ringdown/data_cache/data.pkl")

def train_and_eval(model_name, approach_num, params_type, model_obj_r, model_obj_i, notes, target_transform='none'):
    model_dir = f"llm_agents/results/gemini3_flash_preview/ringdown/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    x_train = get_reparameterized_spin(spin_train, type=params_type).reshape(-1, 1)
    y_train_r = transform_target(r_train, type=target_transform)
    y_train_i = transform_target(i_train, type=target_transform)
    model_obj_r.fit(x_train, y_train_r)
    model_obj_i.fit(x_train, y_train_i)
    joblib.dump((model_obj_r, model_obj_i), f"{model_dir}/saved_model/model.pkl")
    def predict(spin_array):
        x = get_reparameterized_spin(spin_array, type=params_type).reshape(-1, 1)
        return inverse_transform_target(model_obj_r.predict(x), type=target_transform, sign=1), inverse_transform_target(model_obj_i.predict(x), type=target_transform, sign=-1)
    loss, runtime_ms, val_losses, _ = evaluate_model(model_dir, predict, spin_val, r_val, i_val)
    _, _, train_losses, _ = evaluate_model(model_dir, predict, spin_train, r_train, i_train)
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 20, notes, model_dir, train_losses, val_losses)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # 15. Poly 20 (raw)
    p20_r = Pipeline([("poly", PolynomialFeatures(degree=20)), ("linear", LinearRegression())])
    p20_i = Pipeline([("poly", PolynomialFeatures(degree=20)), ("linear", LinearRegression())])
    train_and_eval("poly20_raw", 15, "raw", p20_r, p20_i, "Polynomial degree 20 on raw spin.")

    # 19. XGBoost (raw)
    xgb_r = XGBRegressor(n_estimators=100, random_state=42)
    xgb_i = XGBRegressor(n_estimators=100, random_state=42)
    train_and_eval("xgb_raw", 19, "raw", xgb_r, xgb_i, "XGBoost on raw spin.")

    # 20. LightGBM (raw)
    lgbm_r = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    lgbm_i = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    train_and_eval("lgbm_raw", 20, "raw", lgbm_r, lgbm_i, "LightGBM on raw spin.")

    # 21. CatBoost (raw)
    cat_r = CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
    cat_i = CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
    train_and_eval("cat_raw", 21, "raw", cat_r, cat_i, "CatBoost on raw spin.")

    # 22. Ridge (deg 10, raw)
    ridge_r = Pipeline([("poly", PolynomialFeatures(degree=10)), ("ridge", Ridge(alpha=1.0))])
    ridge_i = Pipeline([("poly", PolynomialFeatures(degree=10)), ("ridge", Ridge(alpha=1.0))])
    train_and_eval("ridge_poly10", 22, "raw", ridge_r, ridge_i, "Ridge Regression (deg 10) on raw spin.")

    # 23. Lasso (deg 10, raw)
    lasso_r = Pipeline([("poly", PolynomialFeatures(degree=10)), ("lasso", Lasso(alpha=0.01))])
    lasso_i = Pipeline([("poly", PolynomialFeatures(degree=10)), ("lasso", Lasso(alpha=0.01))])
    train_and_eval("lasso_poly10", 23, "raw", lasso_r, lasso_i, "Lasso Regression (deg 10) on raw spin.")

    # 24. SVR (raw)
    svr_r = SVR(kernel='rbf')
    svr_i = SVR(kernel='rbf')
    train_and_eval("svr_raw", 24, "raw", svr_r, svr_i, "SVR with RBF kernel on raw spin.")
    
    # 25. Linear Regression (log_compact, log targets)
    lr_r = LinearRegression()
    lr_i = LinearRegression()
    train_and_eval("lr_log_logt", 25, "log_compact", lr_r, lr_i, "Linear Regression on log-compactified spin and log targets.", target_transform='log')
