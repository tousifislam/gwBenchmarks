import numpy as np
import joblib
import os
import json
from gplearn.genetic import SymbolicRegressor
from llm_agents.results.gemini3_flash_preview.ringdown.utils import evaluate_model, save_approach, get_reparameterized_spin, transform_target, inverse_transform_target

# Load data
spin_train, r_train, i_train, spin_val, r_val, i_val = joblib.load("llm_agents/results/gemini3_flash_preview/ringdown/data_cache/data.pkl")

# Patch gplearn
import gplearn.genetic
def dummy_validate(self, X, y=None, **kwargs):
    self.n_features_in_ = X.shape[1]
    return X, y
gplearn.genetic.SymbolicRegressor._validate_data = dummy_validate

def train_gplearn(model_name, approach_num, params_type, target_transform='none'):
    model_dir = f"llm_agents/results/gemini3_flash_preview/ringdown/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    x_train = get_reparameterized_spin(spin_train, type=params_type).reshape(-1, 1)
    y_train_r = transform_target(r_train, type=target_transform)
    y_train_i = transform_target(i_train, type=target_transform)
    
    est_r = SymbolicRegressor(
        population_size=1000,
        generations=20,
        tournament_size=20,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
        metric='mse',
        parsimony_coefficient=0.001,
        max_samples=1.0,
        verbose=0,
        random_state=42,
    )
    est_i = SymbolicRegressor(
        population_size=1000,
        generations=20,
        tournament_size=20,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
        metric='mse',
        parsimony_coefficient=0.001,
        max_samples=1.0,
        verbose=0,
        random_state=43,
    )
    
    est_r.fit(x_train, y_train_r)
    est_i.fit(x_train, y_train_i)
    joblib.dump((est_r, est_i), f"{model_dir}/saved_model/model.pkl")
    
    # Save expression
    with open(f"{model_dir}/saved_model/expressions.json", "w") as f:
        json.dump([{"expression_r": str(est_r._program), "expression_i": str(est_i._program)}], f, indent=4)
        
    def predict(spin_array):
        x = get_reparameterized_spin(spin_array, type=params_type).reshape(-1, 1)
        p_r = est_r.predict(x)
        p_i = est_i.predict(x)
        return inverse_transform_target(p_r, type=target_transform, sign=1), inverse_transform_target(p_i, type=target_transform, sign=-1)
        
    loss, runtime_ms, val_losses, _ = evaluate_model(model_dir, predict, spin_val, r_val, i_val)
    _, _, train_losses, _ = evaluate_model(model_dir, predict, spin_train, r_train, i_train)
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 100, f"gplearn on {params_type} with {target_transform} targets.", model_dir, train_losses, val_losses)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    train_gplearn("gplearn_raw_logt", 16, "raw", target_transform='log')
    train_gplearn("gplearn_log_logt", 17, "log_compact", target_transform='log')
