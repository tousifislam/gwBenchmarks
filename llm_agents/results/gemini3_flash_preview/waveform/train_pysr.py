import numpy as np
import joblib
import os
from pysr import PySRRegressor
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
from llm_agents.results.gemini3_flash_preview.waveform.utils import evaluate_model, save_approach

# Load data
X_train_raw, y_train, X_val_raw, y_val, t, dt, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")

X_train = get_reparameterized_params(X_train_raw, type='effective_spins')
X_val = get_reparameterized_params(X_val_raw, type='effective_spins')

# Run PySR on the first SVD coefficient
model = PySRRegressor(
    niterations=2, # Extremely small
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "log"],
    maxsize=10,
    populations=5,
    procs=1,
    loss="loss(prediction, target) = (prediction - target)^2",
)

# We need to fit each coefficient. For now, let's just do 1.
n_pysr = 1
class PySRPredictor:
    def __init__(self, pysr_models, backup_model, svd, t, params_type):
        self.pysr_models = pysr_models
        self.backup_model = backup_model
        self.svd = svd
        self.t = t
        self.params_type = params_type
        
    def __call__(self, X):
        from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
        X = get_reparameterized_params(np.atleast_2d(X), type=self.params_type)
        coeffs = []
        for m in self.pysr_models:
            coeffs.append(m.predict(X))
        coeffs_backup = self.backup_model.predict(X)
        # Handle 1D coeffs
        coeffs_list = [c.reshape(-1, 1) if c.ndim == 1 else c for c in coeffs]
        coeffs_all = np.column_stack(coeffs_list + [coeffs_backup])
        y_flat = self.svd.inverse_transform(coeffs_all)
        y = y_flat[0, :len(self.t)] + 1j * y_flat[0, len(self.t):]
        return y

if __name__ == "__main__":
    # Load data
    X_train_raw, y_train, X_val_raw, y_val, t, dt, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")

    X_train = get_reparameterized_params(X_train_raw, type='effective_spins')
    X_val = get_reparameterized_params(X_val_raw, type='effective_spins')

    # Run PySR on the first SVD coefficient
    model = PySRRegressor(
        niterations=2, # Extremely small
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log"],
        maxsize=10,
        populations=5,
        procs=1,
        loss="loss(prediction, target) = (prediction - target)^2",
    )

    # We need to fit each coefficient. For now, let's just do 1.
    n_pysr = 1
    pysr_models = []
    for i in range(n_pysr):
        print(f"Fitting coefficient {i} with PySR...")
        model.fit(X_train, y_train_reduced[:, i])
        pysr_models.append(model)
        # Save equations
        model_dir_app = f"llm_agents/results/gemini3_flash_preview/waveform/models/svd_pysr_eff"
        os.makedirs(f"{model_dir_app}/saved_model", exist_ok=True)
        model.equations_.to_json(f"{model_dir_app}/saved_model/equations_{i}.json")

    # For the rest of the coefficients, use a simple linear regression
    from sklearn.linear_model import LinearRegression
    backup_model = LinearRegression()
    backup_model.fit(X_train, y_train_reduced[:, n_pysr:])

    model_name = "svd_pysr_eff"
    model_dir = f"llm_agents/results/gemini3_flash_preview/waveform/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)

    joblib.dump((pysr_models, backup_model, svd), f"{model_dir}/saved_model/model.pkl")

    predict_fn = PySRPredictor(pysr_models, backup_model, svd, t, 'effective_spins')

    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict_fn, X_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict_fn, X_train, y_train, dt)

    save_approach(
        approach_name=model_name,
        approach_number=9, # gplearn was 9 in my plan, let's use 21
        params_type="effective_spins",
        time_conv="t0_at_peak",
        loss=loss,
        components=components,
        runtime_ms=runtime_ms,
        n_params=n_pysr,
        notes="SVD + PySR on first coefficient, Linear Regression on others.",
        model_dir=model_dir,
        train_losses=train_losses,
        val_losses=val_losses
    )
    print(f"Approach 21 (svd_pysr_eff) completed. Loss: {loss}")
