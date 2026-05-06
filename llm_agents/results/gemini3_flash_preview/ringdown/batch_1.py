import numpy as np
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.interpolate import CubicSpline
from llm_agents.results.gemini3_flash_preview.ringdown.utils import evaluate_model, save_approach, get_reparameterized_spin, transform_target, inverse_transform_target

# Load data
spin_train, r_train, i_train, spin_val, r_val, i_val = joblib.load("llm_agents/results/gemini3_flash_preview/ringdown/data_cache/data.pkl")

def train_and_eval(model_name, approach_num, params_type, model_obj_r, model_obj_i, notes, target_transform='none'):
    model_dir = f"llm_agents/results/gemini3_flash_preview/ringdown/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    x_train = get_reparameterized_spin(spin_train, type=params_type).reshape(-1, 1)
    x_val = get_reparameterized_spin(spin_val, type=params_type).reshape(-1, 1)
    
    y_train_r = transform_target(r_train, type=target_transform)
    y_train_i = transform_target(i_train, type=target_transform)
    
    model_obj_r.fit(x_train, y_train_r)
    model_obj_i.fit(x_train, y_train_i)
    
    joblib.dump((model_obj_r, model_obj_i), f"{model_dir}/saved_model/model.pkl")
    
    def predict(spin_array):
        x = get_reparameterized_spin(spin_array, type=params_type).reshape(-1, 1)
        p_r = model_obj_r.predict(x)
        p_i = model_obj_i.predict(x)
        return inverse_transform_target(p_r, type=target_transform, sign=1), inverse_transform_target(p_i, type=target_transform, sign=-1)
    
    loss, runtime_ms, val_losses, _ = evaluate_model(model_dir, predict, spin_val, r_val, i_val)
    _, _, train_losses, _ = evaluate_model(model_dir, predict, spin_train, r_train, i_train)
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 20, notes, model_dir, train_losses, val_losses)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

class SplineModel:
    def fit(self, x, y):
        self.cs = CubicSpline(x.flatten(), y)
    def predict(self, x):
        return self.cs(x.flatten())

if __name__ == "__main__":
    # 1. Poly 10 (raw)
    p10_r = Pipeline([("poly", PolynomialFeatures(degree=10)), ("linear", LinearRegression())])
    p10_i = Pipeline([("poly", PolynomialFeatures(degree=10)), ("linear", LinearRegression())])
    train_and_eval("poly10_raw", 1, "raw", p10_r, p10_i, "Polynomial degree 10 on raw spin.")

    # 4. Poly 10 (raw, log targets)
    p10_r_log = Pipeline([("poly", PolynomialFeatures(degree=10)), ("linear", LinearRegression())])
    p10_i_log = Pipeline([("poly", PolynomialFeatures(degree=10)), ("linear", LinearRegression())])
    train_and_eval("poly10_raw_logtarget", 4, "raw", p10_r_log, p10_i_log, "Polynomial degree 10 on raw spin with log-targets.", target_transform='log')

    # 5. Spline (raw, log targets)
    s_r_log = SplineModel()
    s_i_log = SplineModel()
    train_and_eval("spline_raw_logtarget", 5, "raw", s_r_log, s_i_log, "Cubic spline on raw spin with log-targets.", target_transform='log')
