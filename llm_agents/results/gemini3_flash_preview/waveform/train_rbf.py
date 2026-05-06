import numpy as np
import joblib
import os
from scipy.interpolate import RBFInterpolator
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
from llm_agents.results.gemini3_flash_preview.waveform.utils import evaluate_model, save_approach
from llm_agents.results.gemini3_flash_preview.waveform.predictors import SVDPredictor

class RBFModel:
    def __init__(self, kernel='thin_plate_spline'):
        self.kernel = kernel
        self.rbf = None
        
    def fit(self, X, y):
        self.rbf = RBFInterpolator(X, y, kernel=self.kernel)
        
    def predict(self, X):
        return self.rbf(X)

if __name__ == "__main__":
    X_train_raw, y_train, X_val_raw, y_val, t, dt, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")

    model_name = "svd_rbf_raw"
    model_dir = f"llm_agents/results/gemini3_flash_preview/waveform/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_raw, type='raw')
    X_val = get_reparameterized_params(X_val_raw, type='raw')
    
    rbf = RBFModel()
    rbf.fit(X_train, y_train_reduced)
    
    joblib.dump(rbf, f"{model_dir}/saved_model/model.pkl")
    joblib.dump(svd, f"{model_dir}/saved_model/svd.pkl")
    
    predict_fn = SVDPredictor(rbf, svd, t, 'raw')
    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict_fn, X_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict_fn, X_train, y_train, dt)
    
    save_approach(
        approach_name=model_name,
        approach_number=22,
        params_type="raw",
        time_conv="t0_at_peak",
        loss=loss,
        components=components,
        runtime_ms=runtime_ms,
        n_params=y_train_reduced.shape[1],
        notes="SVD + RBF Interpolation on raw parameters.",
        model_dir=model_dir,
        train_losses=train_losses,
        val_losses=val_losses
    )
    print(f"Approach 22 (svd_rbf_raw) completed. Loss: {loss}")
