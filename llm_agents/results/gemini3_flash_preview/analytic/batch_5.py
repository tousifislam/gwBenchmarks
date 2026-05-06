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
    mean_h = np.mean(y_train, axis=0)
    amp = np.abs(mean_h); phase = np.unwrap(np.angle(mean_h))
    p_a = np.polyfit(common_t, amp, deg_a); p_p = np.polyfit(common_t, phase, deg_p)
    def predict(q):
        eta = q / (1 + q)**2
        eta_mean = np.mean(qs_train / (1 + qs_train)**2)
        A = np.polyval(p_a, common_t) * (eta / eta_mean)
        Phi = np.polyval(p_p, common_t)
        return A * np.exp(1j * Phi)
    return predict

if __name__ == "__main__":
    train_and_eval("poly20_q", 21, "q", get_poly_predict(20, 20, "q"), "Poly deg 20.")
    train_and_eval("poly5_sqrteta", 22, "sqrt_eta", get_poly_predict(5, 5, "sqrt_eta"), "Poly deg 5 scaled by eta.")
    
    # PN-like
    def pn_like_predict(q):
        eta = q / (1 + q)**2
        # t is negative, goes to 0 at peak
        # tau = -t + epsilon
        tau = np.maximum(-common_t, 1e-3)
        A = eta * tau**(-1/4)
        Phi = - (tau**(5/8)) / (32 * eta) # very crude
        return A * np.exp(1j * Phi)
    train_and_eval("pn_like", 23, "eta", pn_like_predict, "PN-like power laws.")

    # More polys
    train_and_eval("poly3_deltam", 24, "delta_m", get_poly_predict(3, 3, "delta_m"), "Poly deg 3.")
    train_and_eval("poly12_eta", 25, "eta", get_poly_predict(12, 12, "eta"), "Poly deg 12.")
    
    # Exponential A
    def exp_a_predict(q):
        eta = q / (1 + q)**2
        A = eta * np.exp(common_t / 100) # common_t is negative
        Phi = -0.2 * common_t
        return A * np.exp(1j * Phi)
    train_and_eval("exp_a", 26, "eta", exp_a_predict, "Exponential amplitude.")

    # Linear everything
    def linear_predict(q):
        eta = q / (1 + q)**2
        A = eta * (1 + 0.01 * common_t)
        Phi = -0.1 * common_t
        return A * np.exp(1j * Phi)
    train_and_eval("linear_all", 27, "eta", linear_predict, "Linear amplitude and phase.")

    # Gaussian A
    def gaussian_a_predict(q):
        eta = q / (1 + q)**2
        A = eta * np.exp(-(common_t/50)**2)
        Phi = -0.1 * common_t
        return A * np.exp(1j * Phi)
    train_and_eval("gaussian_a", 28, "eta", gaussian_a_predict, "Gaussian amplitude.")
