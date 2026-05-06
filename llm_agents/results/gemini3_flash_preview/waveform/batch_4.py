import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
from llm_agents.results.gemini3_flash_preview.waveform.utils import evaluate_model, save_approach
from llm_agents.results.gemini3_flash_preview.waveform.predictors import SVDPredictor

# Load data
X_train_raw, y_train, X_val_raw, y_val, t, dt, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")

def train_and_eval(model_name, approach_num, params_type, model_obj, notes):
    model_dir = f"llm_agents/results/gemini3_flash_preview/waveform/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_raw, type=params_type)
    X_val = get_reparameterized_params(X_val_raw, type=params_type)
    
    model_obj.fit(X_train, y_train_reduced)
    joblib.dump(model_obj, f"{model_dir}/saved_model/model.pkl")
    joblib.dump(svd, f"{model_dir}/saved_model/svd.pkl")
    
    predict_fn = SVDPredictor(model_obj, svd, t, params_type)
    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict_fn, X_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict_fn, X_train, y_train, dt)
    
    save_approach(
        approach_name=model_name,
        approach_number=approach_num,
        params_type=params_type,
        time_conv="t0_at_peak",
        loss=loss,
        components=components,
        runtime_ms=runtime_ms,
        n_params=y_train_reduced.shape[1],
        notes=notes,
        model_dir=model_dir,
        train_losses=train_losses,
        val_losses=val_losses
    )
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # Approach 17: SVD + Poly-3 (raw)
    poly3_raw = Pipeline([("poly", PolynomialFeatures(degree=3)), ("linear", LinearRegression())])
    train_and_eval("svd_poly3_raw", 17, "raw", poly3_raw, "SVD + Poly (deg 3) on raw params.")

    # Approach 18: SVD + Poly-2 (eff)
    poly2_eff = Pipeline([("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())])
    train_and_eval("svd_poly2_eff", 18, "effective_spins", poly2_eff, "SVD + Poly (deg 2) on effective spins.")

    # Approach 19: SVD + MLP (eff)
    mlp_eff = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    train_and_eval("svd_mlp_eff", 19, "effective_spins", mlp_eff, "SVD + MLP on effective spins.")

    # Approach 20: SVD + RF (eff)
    rf_eff = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_eval("svd_rf_eff", 20, "effective_spins", rf_eff, "SVD + Random Forest on effective spins.")

    # Approach 23: SVD + Linear (raw)
    lr_raw = LinearRegression()
    train_and_eval("svd_lr_raw", 23, "raw", lr_raw, "SVD + Linear Regression on raw params.")

    # Approach 24: SVD + Ridge (raw)
    from sklearn.linear_model import Ridge
    ridge_raw = Ridge(alpha=1.0)
    train_and_eval("svd_ridge_raw", 24, "raw", ridge_raw, "SVD + Ridge Regression on raw params.")
