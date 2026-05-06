import numpy as np
import joblib
import os
import json
from gplearn.genetic import SymbolicRegressor
from llm_agents.results.gemini3_flash_preview.dynamics.utils import evaluate_model, save_approach, get_reparameterized_params

# Load data
X_train_raw, y_train, X_val_raw, y_val, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/dynamics/data_cache/data.pkl")

# Patch gplearn
import gplearn.genetic
def dummy_validate(self, X, y=None, **kwargs):
    self.n_features_in_ = X.shape[1]
    return X, y
gplearn.genetic.SymbolicRegressor._validate_data = dummy_validate

def train_gplearn(model_name, approach_num, params_type, n_coeffs=5):
    model_dir = f"llm_agents/results/gemini3_flash_preview/dynamics/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_raw, type=params_type)
    
    # gplearn doesn't handle multi-output well, so we fit each coeff
    regressors = []
    for i in range(n_coeffs):
        est = SymbolicRegressor(
            population_size=1000,
            generations=10,
            tournament_size=20,
            function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
            metric='mse',
            parsimony_coefficient=0.001,
            max_samples=1.0,
            verbose=0,
            random_state=42+i,
        )
        est.fit(X_train, y_train_reduced[:, i])
        regressors.append(est)
        
    # Backup for rest of coeffs
    from sklearn.linear_model import LinearRegression
    backup = LinearRegression()
    backup.fit(X_train, y_train_reduced[:, n_coeffs:])
    
    joblib.dump((regressors, backup), f"{model_dir}/saved_model/model.pkl")
    joblib.dump(svd, f"{model_dir}/saved_model/svd.pkl")
    
    def predict(X):
        X_re = get_reparameterized_params(X, type=params_type)
        res = [est.predict(X_re).reshape(-1, 1) for est in regressors]
        res_backup = backup.predict(X_re)
        coeffs = np.column_stack(res + [res_backup])
        return svd.inverse_transform(coeffs)
        
    loss, runtime_ms, val_losses, _ = evaluate_model(model_dir, predict, X_val_raw, y_val)
    _, _, train_losses, _ = evaluate_model(model_dir, predict, X_train_raw, y_train)
    
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 500, f"gplearn on {params_type} (first {n_coeffs} coeffs).", model_dir, train_losses, val_losses)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

if __name__ == "__main__":
    train_gplearn("gplearn_raw", 16, "raw")
    train_gplearn("gplearn_trig", 17, "trig_anomaly")
