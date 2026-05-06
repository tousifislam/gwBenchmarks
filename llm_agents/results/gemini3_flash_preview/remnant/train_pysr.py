import numpy as np
import joblib
import os
import json
from pysr import PySRRegressor
from llm_agents.results.gemini3_flash_preview.remnant.utils import evaluate_model, save_approach, get_reparameterized_params

# Load data
X_train_raw, y_train, X_val_raw, y_val = joblib.load("llm_agents/results/gemini3_flash_preview/remnant/data_cache/data.pkl")

# Use a small subset for symbolic regression to keep it fast
idx = np.random.choice(len(X_train_raw), 500, replace=False)
X_train_sub = X_train_raw[idx]
y_train_sub = y_train[idx]

def train_pysr():
    model_name = "pysr_raw"
    model_dir = f"llm_agents/results/gemini3_flash_preview/remnant/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    X_train = get_reparameterized_params(X_train_sub, type='raw')
    
    model = PySRRegressor(
        niterations=5,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log"],
        maxsize=20,
        populations=5,
        procs=1,
        loss="loss(prediction, target) = (prediction - target)^2",
    )
    
    model.fit(X_train, y_train_sub)
    joblib.dump(model, f"{model_dir}/saved_model/model.pkl")
    
    # Save expressions
    equations = model.equations_.to_dict(orient="records")
    # Convert sympy to string for JSON
    for eq in equations:
        eq["sympy_format"] = str(eq["sympy_format"])
        eq["lambda_format"] = str(eq["lambda_format"])
    with open(f"{model_dir}/saved_model/expressions.json", "w") as f:
        json.dump(equations, f, indent=4)
        
    predict_fn = lambda X: model.predict(get_reparameterized_params(X, type='raw'))
    loss, runtime_ms, val_errors, _ = evaluate_model(model_dir, predict_fn, X_val_raw, y_val)
    _, _, train_errors, _ = evaluate_model(model_dir, predict_fn, X_train_raw, y_train)
    
    save_approach(model_name, 16, "raw", loss, runtime_ms, 500, "PySR on raw params (500 subset).", model_dir, train_errors, val_errors)
    print(f"Approach 16 (pysr_raw) completed. Loss: {loss}")

if __name__ == "__main__":
    train_pysr()
