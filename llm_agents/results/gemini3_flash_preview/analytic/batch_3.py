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

if __name__ == "__main__":
    # 8. Gaussian Mixture Amplitude
    def gmm_amp(t, *p):
        res = 0
        for i in range(0, len(p), 3):
            a, b, c = p[i:i+3]
            res += a * np.exp(-((t-b)/c)**2)
        return res
    
    mean_amp = np.mean(np.abs(y_train), axis=0)
    popt_gmm, _ = curve_fit(lambda t, *p: gmm_amp(t, *p), common_t, mean_amp, p0=[0.5, 0, 10, 0.2, -100, 50, 0.1, 50, 20], maxfev=2000)
    
    def approach8_predict(q):
        phase = np.unwrap(np.angle(np.mean(y_train, axis=0)))
        popt_p, _ = curve_fit(lambda t, p0, p1, p2: p0 + p1*t + p2*t**2, common_t, phase, p0=[0, 0.2, 0])
        A = gmm_amp(common_t, *popt_gmm)
        Phi = popt_p[0] + popt_p[1]*common_t + popt_p[2]*common_t**2
        return A * np.exp(1j * Phi)
    
    train_and_eval("gmm_amp", 8, "eta", approach8_predict, "3-Gaussian mixture for amplitude.", expression="A(t) = sum(a_i * exp(-((t-b_i)/c_i)^2))")

    # 9. Rational Phase
    def rational_phase(t, a0, a1, a2, b1, b2):
        return (a0 + a1*t + a2*t**2) / (1 + b1*t + b2*t**2)
    
    mean_h = np.mean(y_train, axis=0)
    mean_phase = np.unwrap(np.angle(mean_h))
    # Avoid zero in denominator
    popt_rp, _ = curve_fit(rational_phase, common_t, mean_phase, p0=[0, 0.2, 0, 0.01, 0.001], maxfev=5000)
    
    def approach9_predict(q):
        amp = np.abs(np.mean(y_train, axis=0))
        Phi = rational_phase(common_t, *popt_rp)
        return amp * np.exp(1j * Phi)
    
    train_and_eval("rational_phase", 9, "eta", approach9_predict, "Rational function for phase.", expression="Phi(t) = (a0+a1*t+a2*t^2)/(1+b1*t+b2*t^2)")

    # 10. Lorentzian + Polynomial (degree 4)
    def poly4(t, *p): return np.polyval(p, t)
    mean_phase = np.unwrap(np.angle(np.mean(y_train, axis=0)))
    popt_p4, _ = curve_fit(lambda t, *p: poly4(t, *p), common_t, mean_phase, p0=[0, 0, 0, 0.2, 0])
    
    def approach10_predict(q):
        amp = np.abs(np.mean(y_train, axis=0))
        Phi = poly4(common_t, *popt_p4)
        return amp * np.exp(1j * Phi)
    
    train_and_eval("poly4_phase", 10, "eta", approach10_predict, "Polynomial degree 4 for phase.", expression="Phi(t) = p0 + p1*t + p2*t^2 + p3*t^3 + p4*t^4")
