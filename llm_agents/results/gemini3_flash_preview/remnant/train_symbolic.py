import numpy as np
import joblib
import os
import json
from gplearn.genetic import SymbolicRegressor
from llm_agents.results.gemini3_flash_preview.remnant.utils import evaluate_model, save_approach, get_reparameterized_params

# Load data
X_train_raw, y_train, X_val_raw, y_val = joblib.load("llm_agents/results/gemini3_flash_preview/remnant/data_cache/data.pkl")

# Use a small subset for symbolic regression to keep it fast
idx = np.random.choice(len(X_train_raw), 500, replace=False)
X_train_sub = X_train_raw[idx]
y_train_sub = y_train[idx]

import gplearn.genetic
def dummy_validate(self, X, y=None, **kwargs):
    self.n_features_in_ = X.shape[1]
    return X, y
gplearn.genetic.SymbolicRegressor._validate_data = dummy_validate

def train_gplearn(model_name, approach_num, params_type):
    model_dir = f"llm_agents/results/gemini3_flash_preview/remnant/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_sub, type=params_type)
    
    est = SymbolicRegressor(
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
    
    est.fit(X_train, y_train_sub)
    joblib.dump(est, f"{model_dir}/saved_model/model.pkl")
    
    # Save expression
    with open(f"{model_dir}/saved_model/expressions.json", "w") as f:
        json.dump([{"expression": str(est._program)}], f, indent=4)
        
    predict_fn = lambda X: est.predict(get_reparameterized_params(X, type=params_type))
    loss, runtime_ms, val_errors, _ = evaluate_model(model_dir, predict_fn, X_val_raw, y_val)
    _, _, train_errors, _ = evaluate_model(model_dir, predict_fn, X_train_raw, y_train)
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 500, f"gplearn on {params_type} (500 subset).", model_dir, train_errors, val_errors)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    train_gplearn("gplearn_raw", 17, "raw")
    train_gplearn("gplearn_eff", 16, "effective_spins")
