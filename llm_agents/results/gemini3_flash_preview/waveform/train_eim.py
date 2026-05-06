import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
from llm_agents.results.gemini3_flash_preview.waveform.utils import evaluate_model, save_approach
from llm_agents.results.gemini3_flash_preview.waveform.predictors import EIMPredictor

def build_eim(waveforms, n_points):
    n_sim, n_times = waveforms.shape
    nodes = []
    basis = []
    
    # First node: point where first waveform is maximum
    i0 = np.argmax(np.abs(waveforms[0]))
    nodes.append(i0)
    basis.append(waveforms[0] / waveforms[0, i0])
    
    for i in range(1, n_points):
        # Find best waveform to add
        errors = []
        for j in range(n_sim):
            # Solve for coeffs at current nodes
            B = np.array([b[nodes] for b in basis]).T
            y = waveforms[j, nodes]
            coeffs = np.linalg.solve(B, y)
            pred = sum(c * b for c, b in zip(coeffs, basis))
            errors.append(np.max(np.abs(waveforms[j] - pred)))
        
        j_best = np.argmax(errors)
        # Find next node
        B = np.array([b[nodes] for b in basis]).T
        y = waveforms[j_best, nodes]
        coeffs = np.linalg.solve(B, y)
        pred = sum(c * b for c, b in zip(coeffs, basis))
        i_next = np.argmax(np.abs(waveforms[j_best] - pred))
        
        nodes.append(i_next)
        new_basis = waveforms[j_best] - pred
        basis.append(new_basis / new_basis[i_next])
        
    return np.array(nodes), np.array(basis)

if __name__ == "__main__":
    X_train_raw, y_train, X_val_raw, y_val, t, dt, svd, y_train_reduced = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")
    
    n_eim = 50
    nodes, basis = build_eim(y_train, n_eim)
    
    y_train_eim = y_train[:, nodes]
    
    model_name = "eim_gpr_raw"
    model_dir = f"llm_agents/results/gemini3_flash_preview/waveform/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr_real = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr_imag = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    
    X_train = get_reparameterized_params(X_train_raw, type='raw')
    gpr_real.fit(X_train, np.real(y_train_eim))
    gpr_imag.fit(X_train, np.imag(y_train_eim))
    
    joblib.dump((gpr_real, gpr_imag, nodes, basis), f"{model_dir}/saved_model/model.pkl")
    
    predict_fn = EIMPredictor(gpr_real, gpr_imag, nodes, basis, 'raw')
    
    X_val = get_reparameterized_params(X_val_raw, type='raw')
    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict_fn, X_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict_fn, X_train, y_train, dt)
    
    save_approach(
        approach_name=model_name,
        approach_number=11,
        params_type="raw",
        time_conv="t0_at_peak",
        loss=loss,
        components=components,
        runtime_ms=runtime_ms,
        n_params=n_eim,
        notes="EIM + GPR on raw parameters.",
        model_dir=model_dir,
        train_losses=train_losses,
        val_losses=val_losses
    )
    
    with open(f"{model_dir}/predict.py", "w") as f:
        f.write(f"""import numpy as np
import joblib
import os
from llm_agents.results.gemini3_flash_preview.waveform.predictors import EIMPredictor

model_dir = os.path.dirname(__file__)
gpr_real, gpr_imag, nodes, basis = joblib.load(os.path.join(model_dir, "saved_model/model.pkl"))

predict_fn = EIMPredictor(gpr_real, gpr_imag, nodes, basis, 'raw')

def predict(X):
    return predict_fn(X)
""")
    print(f"Approach 11 (eim_gpr_raw) completed. Loss: {loss}")
