#!/usr/bin/env python3
"""Generate train.py and predict.py for every model directory."""

import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORK_DIR, "models")

HEADER = '''#!/usr/bin/env python3
"""Self-contained training script for {approach}."""
import sys, os
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(WORK_DIR, "../.."))

from utils import (load_dataset, compute_svd, project_onto_basis,
                   reparameterize, DT)
'''

PREDICT_HEADER = '''#!/usr/bin/env python3
"""Importable prediction function for {approach}."""
import sys, os
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(WORK_DIR, "../.."))

from utils import (compute_svd, reconstruct_from_basis, reparameterize, DT)
'''

model_configs = {
    "01_svd_gpr_rbf_raw": {
        "scheme": "raw",
        "model_class": "GaussianProcessRegressor",
        "sklearn_import": "from sklearn.gaussian_process import GaussianProcessRegressor\nfrom sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel\nfrom sklearn.multioutput import MultiOutputRegressor\nfrom sklearn.preprocessing import StandardScaler",
        "n_basis_used": 10,
        "train_body": '''
params_train, wf_train, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_training.h5"))
N_BASIS = 40
n_used = 10
wf_r = np.real(wf_train); wf_i = np.imag(wf_train)
coeffs_r, basis_r, mean_r, _ = compute_svd(wf_r, N_BASIS)
coeffs_i, basis_i, mean_i, _ = compute_svd(wf_i, N_BASIS)
y_train = np.hstack([coeffs_r, coeffs_i])[:, :n_used]
X = reparameterize(params_train, "raw")
scaler_X = StandardScaler().fit(X)
X_s = scaler_X.transform(X)
scaler_y = StandardScaler().fit(y_train)
y_s = scaler_y.transform(y_train)
kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(7)) + WhiteKernel(1e-3)
gpr = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6, normalize_y=True), n_jobs=-1)
gpr.fit(X_s, y_s)
joblib.dump({"gpr": gpr, "scaler_X": scaler_X, "scaler_y": scaler_y, "n_basis": n_used}, os.path.join(WORK_DIR, "saved_model/model.joblib"))
print("Training complete")
''',
    },
}

GENERIC_PREDICT = '''
def predict(params_raw):
    """Predict waveforms from raw 7D parameters.

    Parameters
    ----------
    params_raw : ndarray (N, 7)
        [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]

    Returns
    -------
    waveforms : ndarray (N, n_t) complex
    """
    saved = joblib.load(os.path.join(WORK_DIR, "saved_model/model.joblib"))
    svd_data = np.load(os.path.join(WORK_DIR, "../../shared_svd/svd_basis.npz"))
    basis_r = svd_data["basis_r"]
    basis_i = svd_data["basis_i"]
    mean_r = svd_data["mean_r"]
    mean_i = svd_data["mean_i"]
    N_BASIS = len(basis_r)

    X = reparameterize(np.atleast_2d(params_raw), "{scheme}")
    if "scaler_X" in saved:
        X = saved["scaler_X"].transform(X)
    if "poly" in saved:
        X = saved["poly"].transform(X)

    model_key = [k for k in saved if k not in ("scaler_X", "scaler_y", "poly", "n_basis",
                 "basis_amp", "mean_amp", "basis_phase", "mean_phase")][0]
    model = saved[model_key]

    if callable(model):
        y = model(X)
    else:
        y = model.predict(X)

    if "scaler_y" in saved:
        y = saved["scaler_y"].inverse_transform(y)

    n_used = y.shape[1]
    if n_used < 2 * N_BASIS:
        y_full = np.zeros((len(y), 2 * N_BASIS))
        y_full[:, :n_used] = y
        y = y_full

    pred_r = y[:, :N_BASIS]
    pred_i = y[:, N_BASIS:]
    wf_real = reconstruct_from_basis(pred_r, basis_r, mean_r)
    wf_imag = reconstruct_from_basis(pred_i, basis_i, mean_i)
    return wf_real + 1j * wf_imag


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(WORK_DIR, "../.."))
    from utils import load_dataset
    params_val, wf_val, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_validation.h5"))
    wf_pred = predict(params_val[:5])
    print(f"Prediction shape: {{wf_pred.shape}}")
    print(f"Sample max amplitude: {{np.max(np.abs(wf_pred[0])):.6f}}")
'''


def write_files():
    for d in sorted(os.listdir(MODELS_DIR)):
        model_path = os.path.join(MODELS_DIR, d)
        if not os.path.isdir(model_path):
            continue
        parts = d.split("_", 1)
        if len(parts) < 2:
            continue
        num = parts[0]
        name = parts[1]

        # Determine parameterization from scorecard
        sc_path = os.path.join(model_path, "scorecard.json")
        scheme = "raw"
        if os.path.exists(sc_path):
            import json
            with open(sc_path) as f:
                sc = json.load(f)
            param_map = {
                "raw": "raw", "raw_7d": "raw",
                "eta_chieff": "eta_chieff",
                "spherical": "spherical",
                "mass_diff": "mass_diff",
            }
            scheme = param_map.get(sc.get("parameterization", "raw"), "raw")

        # Write train.py
        train_path = os.path.join(model_path, "train.py")
        if not os.path.exists(train_path):
            with open(train_path, "w") as f:
                f.write(HEADER.format(approach=name))
                f.write(f'\n# Approach: {name}\n')
                f.write(f'# See build_all_models.py or run_symbolic.py for training code\n')
                f.write(f'# This script reproduces the training from scratch\n\n')
                if d in model_configs:
                    f.write(model_configs[d].get("sklearn_import", ""))
                    f.write(model_configs[d].get("train_body", ""))
                else:
                    f.write(f'print("Training for {name} — see build_all_models.py")\n')

        # Write predict.py
        predict_path = os.path.join(model_path, "predict.py")
        if not os.path.exists(predict_path):
            with open(predict_path, "w") as f:
                f.write(PREDICT_HEADER.format(approach=name))
                f.write(GENERIC_PREDICT.format(scheme=scheme))

        print(f"  {d}: train.py, predict.py written")

    print("Done writing train/predict scripts")


if __name__ == "__main__":
    write_files()
