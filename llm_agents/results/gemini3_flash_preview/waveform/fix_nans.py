import joblib
import numpy as np
import os
import json
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
from llm_agents.results.gemini3_flash_preview.waveform.predictors import SVDPredictor
from llm_agents.results.gemini3_flash_preview.waveform.utils import evaluate_model, save_approach

def re_evaluate(model_name, params_type, time_conv, data_cache_name):
    model_dir = f"llm_agents/results/gemini3_flash_preview/waveform/models/{model_name}"
    print(f"Re-evaluating {model_name}...")
    
    # Load data
    X_train_raw, y_train, X_val_raw, y_val, t, dt, svd, y_train_reduced = joblib.load(f"llm_agents/results/gemini3_flash_preview/waveform/data_cache/{data_cache_name}")
    
    model = joblib.load(os.path.join(model_dir, "saved_model/model.pkl"))
    svd = joblib.load(os.path.join(model_dir, "saved_model/svd.pkl"))

    predict_fn = SVDPredictor(model, svd, t, params_type)
    
    X_val = get_reparameterized_params(X_val_raw, type=params_type)
    X_train = get_reparameterized_params(X_train_raw, type=params_type)
    
    # Use a subset for speed if needed, but here we want the full scorecard
    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict_fn, X_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict_fn, X_train, y_train, dt)
    
    # Load old scorecard to keep number
    with open(os.path.join(model_dir, "scorecard.json"), "r") as f:
        old_data = json.load(f)
    
    save_approach(
        approach_name=model_name,
        approach_number=old_data["approach_number"],
        params_type=params_type,
        time_conv=time_conv,
        loss=loss,
        components=components,
        runtime_ms=runtime_ms,
        n_params=old_data["n_params"],
        notes=old_data["notes"],
        model_dir=model_dir,
        train_losses=train_losses,
        val_losses=val_losses
    )
    print(f"Re-evaluation of {model_name} completed. Loss: {loss}")

if __name__ == "__main__":
    nan_models = [
        ("svd_gpr_eff", "effective_spins", "t0_at_peak", "svd_data.pkl"),
        ("svd_gpr_eff_matern25", "effective_spins", "t0_at_peak", "svd_data.pkl"),
        ("svd_gpr_eff_tstart", "effective_spins", "t0_at_start", "svd_data_tstart.pkl"),
        ("svd_gpr_raw_tstart", "raw", "t0_at_start", "svd_data_tstart.pkl"),
        ("svd_gpr_sph", "spherical", "t0_at_peak", "svd_data.pkl")
    ]
    for m, p, t, d in nan_models:
        try:
            re_evaluate(m, p, t, d)
        except Exception as e:
            print(f"Failed to re-evaluate {m}: {e}")
