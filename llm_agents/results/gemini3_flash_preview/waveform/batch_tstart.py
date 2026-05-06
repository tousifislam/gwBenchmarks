import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
from llm_agents.results.gemini3_flash_preview.waveform.utils import evaluate_model, save_approach
from llm_agents.results.gemini3_flash_preview.waveform.predictors import SVDPredictor

# Load data (tstart)
X_train_raw, y_train, X_val_raw, y_val, t, dt, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data_tstart.pkl")

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
        time_conv="t0_at_start",
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
t = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data_tstart.pkl")[4]

predict_fn = SVDPredictor(model, svd, t, '{params_type}')

def predict(X):
    return predict_fn(X)
""")
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    # Approach 15: SVD + GPR (tstart, raw)
    kernel15 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr15 = GaussianProcessRegressor(kernel=kernel15, n_restarts_optimizer=5, random_state=42)
    train_and_eval("svd_gpr_raw_tstart", 15, "raw", gpr15, "SVD + GPR on raw params with t=0 at start.")

    # Approach 16: SVD + GPR (tstart, effective spins)
    kernel16 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr16 = GaussianProcessRegressor(kernel=kernel16, n_restarts_optimizer=5, random_state=42)
    train_and_eval("svd_gpr_eff_tstart", 16, "effective_spins", gpr16, "SVD + GPR on effective spins with t=0 at start.")
