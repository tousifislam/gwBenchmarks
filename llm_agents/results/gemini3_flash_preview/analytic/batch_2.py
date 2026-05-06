import numpy as np
import joblib
import os
from scipy.optimize import curve_fit
from llm_agents.results.gemini3_flash_preview.analytic.utils import evaluate_model, save_approach, get_reparameterized_q

# Load data
qs_train, y_train, qs_val, y_val, common_t, dt = joblib.load("llm_agents/results/gemini3_flash_preview/analytic/data_cache/data.pkl")

def train_and_eval(model_name, approach_num, params_type, predict_fn, notes, expression=""):
    model_dir = f"llm_agents/results/gemini3_flash_preview/analytic/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict_fn, qs_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict_fn, qs_train, y_train, dt)
    save_approach(model_name, approach_num, params_type, loss, components, runtime_ms, 0, notes, model_dir, train_losses, val_losses, expression=expression)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

# Generalized Lorentzian + Poly Phase
def get_predict_fn(params_type):
    # Fit coefficients as function of reparameterized q
    coefs_a = []
    coefs_p = []
    
    def lorentzian(t, a, b, c):
        return a / (1 + ((t-b)/c)**2)
    def quad_phase(t, p0, p1, p2):
        return p0 + p1*t + p2*t**2

    for i in range(len(qs_train)):
        amp = np.abs(y_train[i])
        phase = np.unwrap(np.angle(y_train[i]))
        pa, _ = curve_fit(lorentzian, common_t, amp, p0=[0.5, 0, 10])
        pp, _ = curve_fit(quad_phase, common_t, phase, p0=[0, 0.2, 0])
        coefs_a.append(pa)
        coefs_p.append(pp)
    
    coefs_a = np.array(coefs_a)
    coefs_p = np.array(coefs_p)
    xq = get_reparameterized_q(qs_train, type=params_type)
    
    # Fit each coefficient as linear function of xq
    models_a = [np.polyfit(xq, coefs_a[:, j], 1) for j in range(3)]
    models_p = [np.polyfit(xq, coefs_p[:, j], 1) for j in range(3)]
    
    def predict(q):
        x = get_reparameterized_q(q, type=params_type)
        a = np.polyval(models_a[0], x)
        b = np.polyval(models_a[1], x)
        c = np.polyval(models_a[2], x)
        p0 = np.polyval(models_p[0], x)
        p1 = np.polyval(models_p[1], x)
        p2 = np.polyval(models_p[2], x)
        A = lorentzian(common_t, a, b, c)
        Phi = quad_phase(common_t, p0, p1, p2)
        return A * np.exp(1j * Phi)
    
    return predict

if __name__ == "__main__":
    # 3. Lorentzian params linear in q
    train_and_eval("lorentz_linear_q", 3, "q", get_predict_fn("q"), "Lorentzian params linear in q.")
    
    # 4. Lorentzian params linear in eta
    train_and_eval("lorentz_linear_eta", 4, "eta", get_predict_fn("eta"), "Lorentzian params linear in eta.")
    
    # 5. Lorentzian params linear in delta_m
    train_and_eval("lorentz_linear_deltam", 5, "delta_m", get_predict_fn("delta_m"), "Lorentzian params linear in delta_m.")

    # 6. Lorentzian params quadratic in eta
    def get_predict_fn_quad(params_type):
        coefs_a = []
        coefs_p = []
        def lorentzian(t, a, b, c): return a / (1 + ((t-b)/c)**2)
        def quad_phase(t, p0, p1, p2): return p0 + p1*t + p2*t**2
        for i in range(len(qs_train)):
            amp = np.abs(y_train[i]); phase = np.unwrap(np.angle(y_train[i]))
            pa, _ = curve_fit(lorentzian, common_t, amp, p0=[0.5, 0, 10])
            pp, _ = curve_fit(quad_phase, common_t, phase, p0=[0, 0.2, 0])
            coefs_a.append(pa); coefs_p.append(pp)
        xq = get_reparameterized_q(qs_train, type=params_type)
        models_a = [np.polyfit(xq, np.array(coefs_a)[:, j], 2) for j in range(3)]
        models_p = [np.polyfit(xq, np.array(coefs_p)[:, j], 2) for j in range(3)]
        def predict(q):
            x = get_reparameterized_q(q, type=params_type)
            a, b, c = [np.polyval(m, x) for m in models_a]
            p0, p1, p2 = [np.polyval(m, x) for m in models_p]
            return lorentzian(common_t, a, b, c) * np.exp(1j * quad_phase(common_t, p0, p1, p2))
        return predict

    train_and_eval("lorentz_quad_eta", 6, "eta", get_predict_fn_quad("eta"), "Lorentzian params quadratic in eta.")
    
    # 7. Same for sqrt_eta
    train_and_eval("lorentz_quad_sqrteta", 7, "sqrt_eta", get_predict_fn_quad("sqrt_eta"), "Lorentzian params quadratic in sqrt_eta.")
