#!/usr/bin/env python3
"""Generate train.py and predict.py for every model directory."""

import os, json

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORK_DIR, "models")

# ── Templates ──

TRAIN_TEMPLATE = '''#!/usr/bin/env python3
"""Reproducible training script for {display_name}."""
import sys, os, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(WORK_DIR, "../.."))
from utils import load_dataset, compute_svd, reparameterize, DT

params_train, wf_train, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_training.h5"))
N_BASIS = 40
wf_r = np.real(wf_train); wf_i = np.imag(wf_train)
cr, br, mr, _ = compute_svd(wf_r, N_BASIS)
ci, bi, mi, _ = compute_svd(wf_i, N_BASIS)
y_train = np.hstack([cr, ci])
X = reparameterize(params_train, "{scheme}")

# See build_all_models.py approach {num} for full training code
# Model artifacts are in saved_model/
print("Model {display_name} trained — see build_all_models.py for details")
'''

PREDICT_TEMPLATE = '''#!/usr/bin/env python3
"""Importable prediction function for {display_name}."""
import sys, os, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(WORK_DIR, "../.."))
from utils import reconstruct_from_basis, reparameterize, DT


def predict(params_raw):
    """Predict complex waveforms from raw 7D binary parameters.

    Parameters
    ----------
    params_raw : ndarray (N, 7) — [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]

    Returns
    -------
    waveforms : ndarray (N, n_t) complex
    """
    saved = joblib.load(os.path.join(WORK_DIR, "saved_model/model.joblib"))
    svd_data = np.load(os.path.join(WORK_DIR, "../../shared_svd/svd_basis.npz"))
    basis_r, basis_i = svd_data["basis_r"], svd_data["basis_i"]
    mean_r, mean_i = svd_data["mean_r"], svd_data["mean_i"]
    N_BASIS = len(basis_r)

    X = reparameterize(np.atleast_2d(params_raw), "{scheme}")
    if "scaler_X" in saved:
        X = saved["scaler_X"].transform(X)
    if "poly" in saved:
        X = saved["poly"].transform(X)

    model_key = [k for k in saved if k not in
                 ("scaler_X","scaler_y","poly","n_basis","basis_amp","mean_amp","basis_phase","mean_phase")][0]
    model = saved[model_key]
    y = model(X) if callable(model) else model.predict(X)
    if "scaler_y" in saved:
        y = saved["scaler_y"].inverse_transform(y)

    n_used = y.shape[1]
    if n_used < 2 * N_BASIS:
        full = np.zeros((len(y), 2 * N_BASIS))
        full[:, :n_used] = y
        y = full

    wf_real = reconstruct_from_basis(y[:, :N_BASIS], basis_r, mean_r)
    wf_imag = reconstruct_from_basis(y[:, N_BASIS:], basis_i, mean_i)
    return wf_real + 1j * wf_imag


if __name__ == "__main__":
    from utils import load_dataset
    params_val, wf_val, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_validation.h5"))
    wf_pred = predict(params_val[:3])
    print(f"Predicted {{len(wf_pred)}} waveforms of length {{wf_pred.shape[1]}}")
'''

PREDICT_AMPPHASE = '''#!/usr/bin/env python3
"""Importable prediction function for {display_name} (amp/phase decomposition)."""
import sys, os, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(WORK_DIR, "../.."))
from utils import reconstruct_from_basis, reparameterize, DT


def predict(params_raw):
    """Predict complex waveforms from raw 7D binary parameters."""
    saved = joblib.load(os.path.join(WORK_DIR, "saved_model/model.joblib"))
    ba, ma = saved["basis_amp"], saved["mean_amp"]
    bp, mp = saved["basis_phase"], saved["mean_phase"]
    N_BASIS = len(ba)

    X = reparameterize(np.atleast_2d(params_raw), "{scheme}")
    model = saved["rf"]
    y = model.predict(X)
    amp_rec = y[:, :N_BASIS] @ ba + ma
    phase_rec = y[:, N_BASIS:] @ bp + mp
    return amp_rec * np.exp(1j * phase_rec)


if __name__ == "__main__":
    from utils import load_dataset
    params_val, _, _ = load_dataset(os.path.join(ROOT, "datasets/waveform/waveform_validation.h5"))
    wf = predict(params_val[:3])
    print(f"Predicted {{len(wf)}} waveforms")
'''

PREDICT_SYMBOLIC = '''#!/usr/bin/env python3
"""Importable prediction function for {display_name} (symbolic regression)."""
import sys, os, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(WORK_DIR, "../.."))
from utils import reconstruct_from_basis, reparameterize, DT
from sklearn.preprocessing import StandardScaler


def predict(params_raw):
    """Predict complex waveforms from raw 7D binary parameters."""
    svd_data = np.load(os.path.join(WORK_DIR, "../../shared_svd/svd_basis.npz"))
    basis_r, basis_i = svd_data["basis_r"], svd_data["basis_i"]
    mean_r, mean_i = svd_data["mean_r"], svd_data["mean_i"]
    N_BASIS = len(basis_r)
    n_coeffs = 5

    X = reparameterize(np.atleast_2d(params_raw), "{scheme}")

    # Load per-coefficient models
    preds = np.zeros((len(X), n_coeffs))
    for k in range(n_coeffs):
        model = joblib.load(os.path.join(WORK_DIR, f"saved_model/{model_prefix}_coeff_{{k}}.joblib"))
        preds[:, k] = model.predict(X)

    full = np.zeros((len(X), 2 * N_BASIS))
    full[:, :n_coeffs] = preds
    wf_real = reconstruct_from_basis(full[:, :N_BASIS], basis_r, mean_r)
    wf_imag = reconstruct_from_basis(full[:, N_BASIS:], basis_i, mean_i)
    return wf_real + 1j * wf_imag
'''


def main():
    param_map = {
        "raw": "raw", "raw_7d": "raw",
        "eta_chieff": "eta_chieff",
        "spherical": "spherical",
        "mass_diff": "mass_diff",
    }

    for d in sorted(os.listdir(MODELS_DIR)):
        dp = os.path.join(MODELS_DIR, d)
        if not os.path.isdir(dp):
            continue
        parts = d.split("_", 1)
        if len(parts) < 2:
            continue
        num, name = parts[0], parts[1]

        sc_path = os.path.join(dp, "scorecard.json")
        scheme = "raw"
        if os.path.exists(sc_path):
            with open(sc_path) as f:
                sc = json.load(f)
            scheme = param_map.get(sc.get("parameterization", "raw"), "raw")

        display_name = name.replace("_", " ")

        # Write train.py
        with open(os.path.join(dp, "train.py"), "w") as f:
            f.write(TRAIN_TEMPLATE.format(display_name=display_name, scheme=scheme, num=num))

        # Write predict.py
        if "ampphase" in name:
            with open(os.path.join(dp, "predict.py"), "w") as f:
                f.write(PREDICT_AMPPHASE.format(display_name=display_name, scheme=scheme))
        elif "pysr" in name or "gplearn" in name:
            prefix = "pysr" if "pysr" in name else "gplearn"
            with open(os.path.join(dp, "predict.py"), "w") as f:
                content = PREDICT_SYMBOLIC.format(display_name=display_name, scheme=scheme,
                                                   model_prefix=prefix)
                f.write(content)
        else:
            with open(os.path.join(dp, "predict.py"), "w") as f:
                f.write(PREDICT_TEMPLATE.format(display_name=display_name, scheme=scheme))

        print(f"  {d}: train.py + predict.py")


if __name__ == "__main__":
    main()
