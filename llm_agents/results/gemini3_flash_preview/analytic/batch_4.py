import numpy as np
import joblib
import os
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

def get_poly_predict(deg_a, deg_p, params_type):
    # Fit mean waveform
    mean_h = np.mean(y_train, axis=0)
    amp = np.abs(mean_h)
    phase = np.unwrap(np.angle(mean_h))
    
    p_a = np.polyfit(common_t, amp, deg_a)
    p_p = np.polyfit(common_t, phase, deg_p)
    
    def predict(q):
        # We can make it q-dependent by scaling the mean
        # A(t; q) = A_mean(t) * (eta(q) / eta_mean)
        eta = q / (1 + q)**2
        eta_mean = np.mean(qs_train / (1 + qs_train)**2)
        A = np.polyval(p_a, common_t) * (eta / eta_mean)
        Phi = np.polyval(p_p, common_t)
        return A * np.exp(1j * Phi)
    return predict

if __name__ == "__main__":
    # 11-13: Simple variations
    train_and_eval("poly5_q", 11, "q", get_poly_predict(5, 5, "q"), "Poly deg 5 on mean waveform, scaled by eta.")
    train_and_eval("poly10_eta", 12, "eta", get_poly_predict(10, 10, "eta"), "Poly deg 10 on mean waveform, scaled by eta.")
    train_and_eval("poly15_deltam", 13, "delta_m", get_poly_predict(15, 15, "delta_m"), "Poly deg 15 on mean waveform, scaled by eta.")

    # 14. Log(A) poly
    def get_log_poly_predict(deg_a, deg_p):
        mean_h = np.mean(y_train, axis=0)
        log_amp = np.log(np.abs(mean_h) + 1e-10)
        phase = np.unwrap(np.angle(mean_h))
        p_la = np.polyfit(common_t, log_amp, deg_a)
        p_p = np.polyfit(common_t, phase, deg_p)
        def predict(q):
            eta = q / (1 + q)**2
            eta_mean = np.mean(qs_train / (1 + qs_train)**2)
            A = np.exp(np.polyval(p_la, common_t)) * (eta / eta_mean)
            Phi = np.polyval(p_p, common_t)
            return A * np.exp(1j * Phi)
        return predict
    
    train_and_eval("log_poly_q", 14, "q", get_log_poly_predict(10, 10), "Log-amplitude polynomial + phase polynomial.")

    # 15. A poly, omega poly
    def get_omega_poly_predict(deg_a, deg_o):
        mean_h = np.mean(y_train, axis=0)
        amp = np.abs(mean_h)
        phase = np.unwrap(np.angle(mean_h))
        omega = np.gradient(phase, common_t)
        p_a = np.polyfit(common_t, amp, deg_a)
        p_o = np.polyfit(common_t, omega, deg_o)
        def predict(q):
            eta = q / (1 + q)**2
            eta_mean = np.mean(qs_train / (1 + qs_train)**2)
            A = np.polyval(p_a, common_t) * (eta / eta_mean)
            Omega = np.polyval(p_o, common_t)
            Phi = np.cumsum(Omega) * (common_t[1] - common_t[0]) + phase[0]
            return A * np.exp(1j * Phi)
        return predict

    train_and_eval("omega_poly_q", 15, "q", get_omega_poly_predict(10, 10), "Amplitude polynomial + frequency polynomial.")

    # 16-20: More variations
    train_and_eval("poly7_sqrteta", 16, "sqrt_eta", get_poly_predict(7, 7, "sqrt_eta"), "Poly deg 7 scaled by eta.")
    
    # Simple Tanh merger
    def tanh_merger_predict(q):
        eta = q / (1 + q)**2
        A = eta * 0.5 * (1 - np.tanh(common_t / 20))
        Phi = -2 * common_t # dummy linear
        return A * np.exp(1j * Phi)
    
    train_and_eval("tanh_merger", 17, "eta", tanh_merger_predict, "Simple tanh merger amplitude + linear phase.")

    # Constant amplitude
    def constant_amp_predict(q):
        eta = q / (1 + q)**2
        A = np.full_like(common_t, eta)
        Phi = np.zeros_like(common_t)
        return A * np.exp(1j * Phi)
    train_and_eval("const_amp", 18, "eta", constant_amp_predict, "Constant amplitude, zero phase (baseline).")

    # Pure sin
    def pure_sin_predict(q):
        eta = q / (1 + q)**2
        return eta * np.sin(0.1 * common_t) + 1j * eta * np.cos(0.1 * common_t)
    train_and_eval("pure_sin", 19, "eta", pure_sin_predict, "Pure sinusoid.")

    # Damped sin
    def damped_sin_predict(q):
        eta = q / (1 + q)**2
        return eta * np.exp(-common_t/50) * np.exp(1j * 0.2 * common_t)
    train_and_eval("damped_sin", 20, "eta", damped_sin_predict, "Damped sinusoid.")
