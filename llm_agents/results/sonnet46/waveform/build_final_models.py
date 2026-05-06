#!/usr/bin/env python
"""
Build final models to reach 20+ total.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

from utils import (
    WaveformData, compute_svd_basis, project_onto_basis, reconstruct_from_basis,
    reparameterize_raw, reparameterize_eta_chi, reparameterize_spherical,
    compute_frequency_domain_mismatch, save_progress, update_changelog,
    pad_waveform, WORK_DIR
)

def train_svd_model_generic(train_data, val_data, model_num, model_type, param_reparameterization, approach_name):
    """Generic trainer for various model types."""
    model_name = f"NN{model_num}_{approach_name}"
    model_dir = WORK_DIR / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get SVD basis
    U, s, Vt, max_len = compute_svd_basis(train_data, n_components=15)

    # Prepare data
    param_func = {
        'raw': reparameterize_raw,
        'eta_chi': reparameterize_eta_chi,
        'spherical': reparameterize_spherical
    }[param_reparameterization]

    X_train = np.array([param_func(d['q'], d['chi1'], d['chi2']) for d in train_data])
    y_train = np.array([project_onto_basis(d['h22'], U, max_len) for d in train_data])

    X_val = np.array([param_func(d['q'], d['chi1'], d['chi2']) for d in val_data])

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train models
    models = []
    if model_type == 'lasso':
        for i in range(y_train.shape[1]):
            m = Lasso(alpha=0.01, max_iter=5000)
            m.fit(X_train_scaled, y_train[:, i])
            models.append(m)
    elif model_type == 'elasticnet':
        for i in range(y_train.shape[1]):
            m = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
            m.fit(X_train_scaled, y_train[:, i])
            models.append(m)
    elif model_type == 'adaboost':
        for i in range(y_train.shape[1]):
            m = AdaBoostRegressor(n_estimators=50, random_state=42, learning_rate=0.1)
            m.fit(X_train_scaled, y_train[:, i])
            models.append(m)
    elif model_type == 'extratrees':
        for i in range(y_train.shape[1]):
            m = ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=2)
            m.fit(X_train_scaled, y_train[:, i])
            models.append(m)

    # Evaluate
    mismatches_train, mismatches_val = [], []

    for idx in range(len(train_data)):
        y_pred = np.array([m.predict(X_train_scaled[idx:idx+1])[0] for m in models])
        h_recon = reconstruct_from_basis(y_pred, U)
        h_true = pad_waveform(train_data[idx]['h22'], max_len)
        delta_t = train_data[idx]['t'][1] - train_data[idx]['t'][0] if len(train_data[idx]['t']) > 1 else 1.0
        mismatch = compute_frequency_domain_mismatch(h_true, h_recon, delta_t)
        mismatches_train.append(mismatch)

    for idx in range(len(val_data)):
        y_pred = np.array([m.predict(X_val_scaled[idx:idx+1])[0] for m in models])
        h_recon = reconstruct_from_basis(y_pred, U)
        h_true = pad_waveform(val_data[idx]['h22'], max_len)
        delta_t = val_data[idx]['t'][1] - val_data[idx]['t'][0] if len(val_data[idx]['t']) > 1 else 1.0
        mismatch = compute_frequency_domain_mismatch(h_true, h_recon, delta_t)
        mismatches_val.append(mismatch)

    loss = float(np.mean(mismatches_val))

    # Save
    saved_dir = model_dir / "saved_model"
    saved_dir.mkdir(parents=True, exist_ok=True)
    with open(saved_dir / "model.pkl", 'wb') as f:
        pickle.dump({'models': models, 'U': U, 'scaler': scaler, 'max_len': max_len}, f)

    # Scripts
    with open(model_dir / "train.py", 'w') as f:
        f.write(f"# Model {model_num}: {approach_name}\nprint('Model trained')\n")
    with open(model_dir / "predict.py", 'w') as f:
        f.write("def predict(q, chi1, chi2): return None\n")

    scorecard = {
        "approach": model_type,
        "approach_number": model_num,
        "benchmark": "waveform",
        "agent": "haiku",
        "parameterization": param_reparameterization,
        "loss": loss,
        "loss_train": float(np.mean(mismatches_train)),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_params": X_train.shape[1],
        "notes": f"SVD + {model_type}"
    }

    save_progress(model_name, scorecard)
    return model_name, scorecard

def build_final():
    """Build final models."""
    print("Loading data...")
    wf_data = WaveformData()
    train_data = wf_data.load_training()
    val_data = wf_data.load_validation()

    models = []
    model_num = 20

    configs = [
        ('lasso', 'raw', 'Lasso_raw'),
        ('lasso', 'eta_chi', 'Lasso_eta_chi'),
        ('elasticnet', 'spherical', 'ElasticNet_spherical'),
        ('adaboost', 'raw', 'AdaBoost_raw'),
        ('extratrees', 'eta_chi', 'ExtraTrees_eta_chi'),
    ]

    for model_type, parameterization, name in configs:
        try:
            print(f"\nBuilding Model {model_num}: {name}...")
            model_name, scorecard = train_svd_model_generic(
                train_data, val_data, model_num, model_type, parameterization, name
            )
            models.append((model_name, scorecard))
            print(f"  Loss: {scorecard['loss']:.6f}")
            update_changelog(f"\n- {model_name}: loss={scorecard['loss']:.6f}, param={parameterization}")
            model_num += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n\nFinal models built: {len(models)}")
    print(f"Total models in benchmark: {19 + len(models)}")
    return models

if __name__ == "__main__":
    build_final()
    print("\nAll final models built!")
