import joblib
import numpy as np
import os

model_dir = "llm_agents/results/gemini3_flash_preview/waveform/models/svd_gpr_eff_matern25"
model = joblib.load(os.path.join(model_dir, "saved_model/model.pkl"))
svd = joblib.load(os.path.join(model_dir, "saved_model/svd.pkl"))
X_train_raw, _, _, _, _, _, _, _ = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")

from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
X_train = get_reparameterized_params(X_train_raw, type='effective_spins')

coeffs = model.predict(X_train[:1])
print("Coeffs sum:", np.sum(np.abs(coeffs)))
y_flat = svd.inverse_transform(coeffs)
print("Waveform sum:", np.sum(np.abs(y_flat)))
