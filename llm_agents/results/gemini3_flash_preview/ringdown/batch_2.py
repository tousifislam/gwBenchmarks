import numpy as np
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
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
    # 6. Chebyshev (deg 15, chebyshev)
    # Using PolynomialFeatures on mapped spin is equivalent to Chebyshev if using the right basis
    # but regular polynomial basis works too if mapped to [-1, 1].
    p15_cheb_r = Pipeline([("poly", PolynomialFeatures(degree=15)), ("linear", LinearRegression())])
    p15_cheb_i = Pipeline([("poly", PolynomialFeatures(degree=15)), ("linear", LinearRegression())])
    train_and_eval("poly15_cheb", 6, "chebyshev", p15_cheb_r, p15_cheb_i, "Polynomial degree 15 on Chebyshev mapped spin.")

    # 8. Poly 15 (log_compact, log targets)
    p15_log_logt_r = Pipeline([("poly", PolynomialFeatures(degree=15)), ("linear", LinearRegression())])
    p15_log_logt_i = Pipeline([("poly", PolynomialFeatures(degree=15)), ("linear", LinearRegression())])
    train_and_eval("poly15_log_logt", 8, "log_compact", p15_log_logt_r, p15_log_logt_i, "Polynomial degree 15 on log-compactified spin and log targets.", target_transform='log')

    # 9. GPR (RBF, raw)
    kernel9 = C(1.0) * RBF(0.1)
    gpr9_r = GaussianProcessRegressor(kernel=kernel9, n_restarts_optimizer=2, random_state=42)
    gpr9_i = GaussianProcessRegressor(kernel=kernel9, n_restarts_optimizer=2, random_state=42)
    train_and_eval("gpr_rbf_raw", 9, "raw", gpr9_r, gpr9_i, "GPR with RBF kernel on raw spin.")

    # 10. GPR (Matern 2.5, raw, log targets)
    kernel10 = C(1.0) * Matern(length_scale=0.1, nu=2.5)
    gpr10_r = GaussianProcessRegressor(kernel=kernel10, n_restarts_optimizer=2, random_state=42)
    gpr10_i = GaussianProcessRegressor(kernel=kernel10, n_restarts_optimizer=2, random_state=42)
    train_and_eval("gpr_matern25_logt", 10, "raw", gpr10_r, gpr10_i, "GPR with Matern 2.5 on raw spin and log targets.", target_transform='log')
