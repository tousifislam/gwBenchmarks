import numpy as np
import joblib
import os

class SVDPredictor:
    def __init__(self, model, svd, t, params_type):
        self.model = model
        self.svd = svd
        self.t = t
        self.params_type = params_type
        
    def __call__(self, X):
        from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
        X = get_reparameterized_params(np.atleast_2d(X), type=self.params_type)
        coeffs = self.model.predict(X)
        y_flat = self.svd.inverse_transform(coeffs)
        y = y_flat[0, :len(self.t)] + 1j * y_flat[0, len(self.t):]
        return y

class EIMPredictor:
    def __init__(self, gpr_real, gpr_imag, nodes, basis, params_type):
        self.gpr_real = gpr_real
        self.gpr_imag = gpr_imag
        self.nodes = nodes
        self.basis = basis
        self.params_type = params_type
        
    def __call__(self, X):
        from llm_agents.results.gemini3_flash_preview.waveform.prepare_data import get_reparameterized_params
        X = get_reparameterized_params(np.atleast_2d(X), type=self.params_type)
        vals_real = self.gpr_real.predict(X)
        vals_imag = self.gpr_imag.predict(X)
        vals = vals_real + 1j * vals_imag
        B = np.array([b[self.nodes] for b in self.basis]).T
        coeffs = np.linalg.solve(B, vals.T).T
        y = coeffs @ self.basis
        return y[0]
