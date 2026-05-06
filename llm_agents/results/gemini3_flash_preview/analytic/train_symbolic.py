import numpy as np
import joblib
import os
import json
from gplearn.genetic import SymbolicRegressor
from llm_agents.results.gemini3_flash_preview.analytic.utils import evaluate_model, save_approach, get_reparameterized_q

# Load data
qs_train, y_train, qs_val, y_val, common_t, dt = joblib.load("llm_agents/results/gemini3_flash_preview/analytic/data_cache/data.pkl")

# Patch gplearn
import gplearn.genetic
def dummy_validate(self, X, y=None, **kwargs):
    self.n_features_in_ = X.shape[1]
    return X, y
gplearn.genetic.SymbolicRegressor._validate_data = dummy_validate

def train_gplearn(model_name, approach_num, params_type):
    model_dir = f"llm_agents/results/gemini3_flash_preview/analytic/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    # Prepare training data: (n_sim * n_grid, 2) features: [t, eta]
    X_train = []
    y_amp = []
    y_phase = []
    for i, q in enumerate(qs_train):
        eta = q / (1 + q)**2
        for j, t in enumerate(common_t):
            X_train.append([t, eta])
            y_amp.append(np.abs(y_train[i, j]))
            y_phase.append(np.unwrap(np.angle(y_train[i]))[j])
    
    X_train = np.array(X_train)
    y_amp = np.array(y_amp)
    y_phase = np.array(y_phase)
    
    # Downsample for speed
    idx = np.random.choice(len(X_train), 5000, replace=False)
    X_sub = X_train[idx]
    y_amp_sub = y_amp[idx]
    y_phase_sub = y_phase[idx]
    
    est_amp = SymbolicRegressor(population_size=1000, generations=10, random_state=42)
    est_phase = SymbolicRegressor(population_size=1000, generations=10, random_state=43)
    
    est_amp.fit(X_sub, y_amp_sub)
    est_phase.fit(X_sub, y_phase_sub)
    
    joblib.dump((est_amp, est_phase), f"{model_dir}/saved_model/model.pkl")
    
    def predict(q):
        eta = q / (1 + q)**2
        X_test = np.column_stack([common_t, np.full_like(common_t, eta)])
        amp = est_amp.predict(X_test)
        phase = est_phase.predict(X_test)
        return amp * np.exp(1j * phase)
        
    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict, qs_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict, qs_train, y_train, dt)
    
    expr = f"A(t, eta) = {est_amp._program}, Phi(t, eta) = {est_phase._program}"
    save_approach(model_name, approach_num, params_type, loss, components, runtime_ms, 0, f"gplearn on (t, {params_type}).", model_dir, train_losses, val_losses, expression=expr)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    train_gplearn("gplearn_t_eta", 2, "eta")
