import numpy as np
import joblib
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
from llm_agents.results.gemini3_flash_preview.waveform.utils import evaluate_model, save_approach

class AmpPhasePredictor:
    def __init__(self, gpr_amp, gpr_phase, svd_amp, svd_phase, params_type):
        self.gpr_amp = gpr_amp
        self.gpr_phase = gpr_phase
        self.svd_amp = svd_amp
        self.svd_phase = svd_phase
        self.params_type = params_type
        
    def __call__(self, X):
        from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
        X = get_reparameterized_params(np.atleast_2d(X), type=self.params_type)
        log_amp_coeffs = self.gpr_amp.predict(X)
        phase_coeffs = self.gpr_phase.predict(X)
        log_amp = self.svd_amp.inverse_transform(log_amp_coeffs)
        phase = self.svd_phase.inverse_transform(phase_coeffs)
        amp = np.exp(log_amp)
        y = amp * np.exp(1j * phase)
        return y[0]

if __name__ == "__main__":
    # Load data
    X_train_raw, y_train, X_val_raw, y_val, t, dt, _, _ = joblib.load("llm_agents/results/gemini3_flash_preview/waveform/data_cache/svd_data.pkl")

    # Amplitude and Phase
    amp_train = np.abs(y_train)
    # Avoid log(0)
    amp_train = np.maximum(amp_train, 1e-10)
    log_amp_train = np.log(amp_train)
    phase_train = np.unwrap(np.angle(y_train), axis=1)

    # SVD on log_amp and phase separately
    svd_amp = TruncatedSVD(n_components=20)
    log_amp_reduced = svd_amp.fit_transform(log_amp_train)

    svd_phase = TruncatedSVD(n_components=20)
    phase_reduced = svd_phase.fit_transform(phase_train)

    # Train GPRs
    X_train = get_reparameterized_params(X_train_raw, type='raw')
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    gpr_amp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr_amp.fit(X_train, log_amp_reduced)

    gpr_phase = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpr_phase.fit(X_train, phase_reduced)

    model_name = "amp_phase_svd_gpr_raw"
    model_dir = f"llm_agents/results/gemini3_flash_preview/waveform/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)

    joblib.dump((gpr_amp, gpr_phase, svd_amp, svd_phase), f"{model_dir}/saved_model/model.pkl")

    predict_fn = AmpPhasePredictor(gpr_amp, gpr_phase, svd_amp, svd_phase, 'raw')

    X_val = get_reparameterized_params(X_val_raw, type='raw')
    loss, components, runtime_ms, val_losses = evaluate_model(model_dir, predict_fn, X_val, y_val, dt)
    _, _, _, train_losses = evaluate_model(model_dir, predict_fn, X_train, y_train, dt)

    save_approach(
        approach_name=model_name,
        approach_number=14,
        params_type="raw",
        time_conv="t0_at_peak",
        loss=loss,
        components=components,
        runtime_ms=runtime_ms,
        n_params=40,
        notes="Separate SVD+GPR for log-amplitude and phase.",
        model_dir=model_dir,
        train_losses=train_losses,
        val_losses=val_losses
    )
