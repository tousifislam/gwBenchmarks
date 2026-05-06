import numpy as np
import joblib
import os
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
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
    
    with open(f"{model_dir}/predict.py", "w") as f:
        f.write(f"""import numpy as np
import joblib
import os
from llm_agents.results.gemini3_flash_preview.waveform.predictors import SVDPredictor

model_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(model_dir, "saved_model/model.pkl"))
svd = joblib.load(os.path.join(model_dir, "saved_model/svd.pkl"))
t = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")[4]

predict_fn = SVDPredictor(model, svd, t, '{params_type}')

def predict(X):
    return predict_fn(X)
""")
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # Approach 6: SVD + Kernel Ridge (raw)
    kr = KernelRidge(kernel='rbf', alpha=0.1)
    train_and_eval("svd_kr_raw", 6, "raw", kr, "SVD + Kernel Ridge on raw params.")

    # Approach 7: SVD + XGBoost (raw)
    xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    train_and_eval("svd_xgb_raw", 7, "raw", xgb, "SVD + XGBoost on raw params.")

    # Approach 8: SVD + GPR (spherical)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    kernel8 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr8 = GaussianProcessRegressor(kernel=kernel8, n_restarts_optimizer=5, random_state=42)
    train_and_eval("svd_gpr_sph", 8, "spherical", gpr8, "SVD + GPR with spherical spin parameters.")

    # Approach 10: SVD + GPR (Matern 2.5, effective spins)
    from sklearn.gaussian_process.kernels import Matern
    kernel10 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    gpr10 = GaussianProcessRegressor(kernel=kernel10, n_restarts_optimizer=5, random_state=42)
    train_and_eval("svd_gpr_eff_matern25", 10, "effective_spins", gpr10, "SVD + GPR with effective spins and Matern 2.5 kernel.")
