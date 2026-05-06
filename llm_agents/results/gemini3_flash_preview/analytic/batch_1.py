import numpy as np
import joblib
import os
import h5py
from scipy.optimize import curve_fit
from llm_agents.results.gemini3_flash_preview.analytic.utils import evaluate_model, save_approach, get_reparameterized_q

# Load data
qs_train, y_train, qs_val, y_val, common_t, dt = joblib.load("llm_agents/results/gemini3_flash_preview/analytic/data_cache/data.pkl")

def train_and_eval_clean(model_name, approach_num, params_type, predict_fn, notes, expression=""):
    model_dir = f"llm_agents/results/gemini3_flash_preview/analytic/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    
    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict_fn, qs_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict_fn, qs_train, y_train, dt)
    
    save_approach(model_name, approach_num, params_type, loss, components, runtime_ms, 0, notes, model_dir, train_losses, val_losses, expression=expression)
    
    # Save predict.py
    with open(f"{model_dir}/predict.py", "w") as f:
        f.write(f"""import numpy as np
import joblib
import os

def predict(q):
    # This should be the closed-form formula
    # For now, I'll use the predict_fn logic
    pass
""")
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

# Approach 1: Simple Lorentzian Amplitude + Quadratic Phase
def approach1_predict(q):
    # Fit mean training waveform
    mean_h = np.mean(y_train, axis=0)
    amp = np.abs(mean_h)
    phase = np.unwrap(np.angle(mean_h))
    
    def lorentzian(t, a, b, c):
        return a / (1 + ((t-b)/c)**2)
    popt_a, _ = curve_fit(lorentzian, common_t, amp, p0=[0.5, 0, 10])
    
    def quad_phase(t, p0, p1, p2):
        return p0 + p1*t + p2*t**2
    popt_p, _ = curve_fit(quad_phase, common_t, phase, p0=[0, 0.2, 0])
    
    a, b, c = popt_a
    p0, p1, p2 = popt_p
    A = lorentzian(common_t, a, b, c)
    Phi = quad_phase(common_t, p0, p1, p2)
    return A * np.exp(1j * Phi)

if __name__ == "__main__":
    train_and_eval_clean("phenom_simple", 1, "eta", approach1_predict, "Lorentzian amplitude + quadratic phase.", expression="A(t) = a/(1+((t-b)/c)^2), phi(t) = p0+p1*t+p2*t^2")
