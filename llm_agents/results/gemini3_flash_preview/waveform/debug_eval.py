import joblib
import numpy as np
import os
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
from llm_agents.results.gemini3_flash_preview.waveform.predictors import SVDPredictor
from gwbenchmarks.metrics import mean_fd_mismatch

model_dir = "llm_agents/results/gemini3_flash_preview/waveform/models/svd_gpr_eff"
model = joblib.load(os.path.join(model_dir, "saved_model/model.pkl"))
svd = joblib.load(os.path.join(model_dir, "saved_model/svd.pkl"))
X_raw, y_true, _, _, t, dt, _, _ = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")

predict_fn = SVDPredictor(model, svd, t, 'effective_spins')
h_pred = predict_fn(X_raw[0])
h_true = y_true[0]

mismatch = mean_fd_mismatch(h_pred, h_true, dt, masses=[40])
print("Mismatch:", mismatch)
