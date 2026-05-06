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
    
    with open(f"{model_dir}/predict.py", "w") as f:
        f.write(f"""import numpy as np
import joblib
import os
from llm_agents.results.gemini3_flash_preview.dynamics.utils import get_reparameterized_params

model_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(model_dir, "saved_model/model.pkl"))
svd = joblib.load(os.path.join(model_dir, "saved_model/svd.pkl"))

def predict(X):
    X_reparam = get_reparameterized_params(np.atleast_2d(X), type='{params_type}')
    coeffs = model.predict(X_reparam)
    return svd.inverse_transform(coeffs)
""")
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # 1. SVD + GPR (raw)
    kernel1 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=2, random_state=42)
    train_and_eval("svd_gpr_raw", 1, "raw", gpr1, "SVD + GPR with RBF kernel on raw params.")

    # 2. SVD + GPR (eff_log_e)
    kernel2 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
    gpr2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=2, random_state=42)
    train_and_eval("svd_gpr_eff_log_e", 2, "eff_log_e", gpr2, "SVD + GPR with Matern 1.5 on effective spin + log eccentricity.")

    # 3. SVD + Poly (deg 2, raw)
    poly3 = Pipeline([("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())])
    train_and_eval("svd_poly2_raw", 3, "raw", poly3, "SVD + Polynomial Regression (degree 2) on raw params.")

    # 4. SVD + MLP (raw)
    mlp4 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    train_and_eval("svd_mlp_raw", 4, "raw", mlp4, "SVD + MLP on raw params.")

    # 5. SVD + RF (raw)
    rf5 = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_eval("svd_rf_raw", 5, "raw", rf5, "SVD + Random Forest on raw params.")
