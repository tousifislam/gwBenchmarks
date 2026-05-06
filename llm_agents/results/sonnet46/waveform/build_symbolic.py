#!/usr/bin/env python
"""
Build symbolic regression and RBF interpolation models.
"""

import os
import json
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils import (
    WaveformData, compute_svd_basis, project_onto_basis, reconstruct_from_basis,
    reparameterize_raw, reparameterize_eta_chi, reparameterize_spherical,
    compute_frequency_domain_mismatch, save_progress, update_changelog,
    pad_waveform, WORK_DIR
)

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False
    print("PySR not available")

try:
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except ImportError:
    HAS_GPLEARN = False
    print("gplearn not available")

from scipy.interpolate import Rbf
from sklearn.preprocessing import StandardScaler

MODEL_COUNTER = {"count": 16}

def build_pysr_model(train_data, val_data, model_num):
    """Build PySR-based symbolic regression model."""
    if not HAS_PYSR:
        print("Skipping PySR model - not installed")
        return None, None

    model_name = f"NN{model_num}_PySR_SVD"
    model_dir = WORK_DIR / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get SVD basis
    U, s, Vt, max_len = compute_svd_basis(train_data, n_components=15)

    # Prepare data
    X_train = np.array([reparameterize_raw(d['q'], d['chi1'], d['chi2']) for d in train_data])
    y_train = np.array([project_onto_basis(d['h22'], U, max_len) for d in train_data])

    X_val = np.array([reparameterize_raw(d['q'], d['chi1'], d['chi2']) for d in val_data])
    y_val = np.array([project_onto_basis(d['h22'], U, max_len) for d in val_data])

    # Scale inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train PySR on first SVD coefficient
    print(f"  Training PySR on SVD coefficient 0...")
    try:
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "exp", "sin", "cos"],
            maxsize=20,
            populations=20,
            procs=2,
            loss="loss(prediction, target) = abs(prediction - target)",
            verbosity=0,
        )
        model.fit(X_train_scaled, y_train[:, 0])

        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)

        mismatches_train, mismatches_val = [], []
        for idx in range(len(train_data)):
            y_full_pred = y_pred_train[idx:idx+1].reshape(1, -1)
            if y_full_pred.shape[1] < y_train.shape[1]:
                # Pad with zeros
                padding = np.zeros((1, y_train.shape[1] - y_full_pred.shape[1]))
                y_full_pred = np.concatenate([y_full_pred, padding], axis=1)
            h_recon = reconstruct_from_basis(y_full_pred[0], U)
            h_true = pad_waveform(train_data[idx]['h22'], max_len)
            delta_t = train_data[idx]['t'][1] - train_data[idx]['t'][0] if len(train_data[idx]['t']) > 1 else 1.0
            mismatch = compute_frequency_domain_mismatch(h_true, h_recon, delta_t)
            mismatches_train.append(mismatch)

        for idx in range(len(val_data)):
            y_full_pred = y_pred_val[idx:idx+1].reshape(1, -1)
            if y_full_pred.shape[1] < y_train.shape[1]:
                padding = np.zeros((1, y_train.shape[1] - y_full_pred.shape[1]))
                y_full_pred = np.concatenate([y_full_pred, padding], axis=1)
            h_recon = reconstruct_from_basis(y_full_pred[0], U)
            h_true = pad_waveform(val_data[idx]['h22'], max_len)
            delta_t = val_data[idx]['t'][1] - val_data[idx]['t'][0] if len(val_data[idx]['t']) > 1 else 1.0
            mismatch = compute_frequency_domain_mismatch(h_true, h_recon, delta_t)
            mismatches_val.append(mismatch)

        loss = float(np.mean(mismatches_val))

        # Save
        saved_dir = model_dir / "saved_model"
        saved_dir.mkdir(parents=True, exist_ok=True)
        with open(saved_dir / "pysr_model.pkl", 'wb') as f:
            pickle.dump({'model': model, 'U': U, 'scaler': scaler, 'max_len': max_len}, f)

        # Save expressions
        try:
            with open(saved_dir / "expressions.json", 'w') as f:
                json.dump({"equation": str(model.sympy())}, f)
        except:
            pass

        scorecard = {
            "approach": "pysr_svd",
            "approach_number": model_num,
            "benchmark": "waveform",
            "agent": "haiku",
            "parameterization": "raw_7d",
            "loss": loss,
            "loss_train": float(np.mean(mismatches_train)),
            "n_train": len(train_data),
            "n_val": len(val_data),
            "notes": "Symbolic regression via PySR on SVD coefficients"
        }

        save_progress(model_name, scorecard)
        return model_name, scorecard

    except Exception as e:
        print(f"  PySR training failed: {e}")
        return None, None

def build_gplearn_model(train_data, val_data, model_num):
    """Build gplearn symbolic regression model."""
    if not HAS_GPLEARN:
        print("Skipping gplearn model - not installed")
        return None, None

    model_name = f"NN{model_num}_gplearn_SVD"
    model_dir = WORK_DIR / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get SVD basis
    U, s, Vt, max_len = compute_svd_basis(train_data, n_components=15)

    # Prepare data
    X_train = np.array([reparameterize_eta_chi(d['q'], d['chi1'], d['chi2']) for d in train_data])
    y_train = np.array([project_onto_basis(d['h22'], U, max_len) for d in train_data])

    X_val = np.array([reparameterize_eta_chi(d['q'], d['chi1'], d['chi2']) for d in val_data])

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"  Training gplearn on SVD coefficient 0...")
    try:
        est = SymbolicRegressor(
            population_size=1000,
            generations=20,
            tournament_size=20,
            function_set=['add', 'sub', 'mul', 'div', 'sqrt'],
            metric='mse',
            parsimony_coefficient=0.001,
            max_samples=0.9,
            verbose=0,
            random_state=42,
            n_jobs=2
        )
        est.fit(X_train_scaled, y_train[:, 0])

        # Evaluate
        y_pred_train = est.predict(X_train_scaled)
        y_pred_val = est.predict(X_val_scaled)

        mismatches_train, mismatches_val = [], []
        for idx in range(len(train_data)):
            y_full_pred = y_pred_train[idx:idx+1].reshape(1, -1)
            if y_full_pred.shape[1] < y_train.shape[1]:
                padding = np.zeros((1, y_train.shape[1] - y_full_pred.shape[1]))
                y_full_pred = np.concatenate([y_full_pred, padding], axis=1)
            h_recon = reconstruct_from_basis(y_full_pred[0], U)
            h_true = pad_waveform(train_data[idx]['h22'], max_len)
            delta_t = train_data[idx]['t'][1] - train_data[idx]['t'][0] if len(train_data[idx]['t']) > 1 else 1.0
            mismatch = compute_frequency_domain_mismatch(h_true, h_recon, delta_t)
            mismatches_train.append(mismatch)

        for idx in range(len(val_data)):
            y_full_pred = y_pred_val[idx:idx+1].reshape(1, -1)
            if y_full_pred.shape[1] < y_train.shape[1]:
                padding = np.zeros((1, y_train.shape[1] - y_full_pred.shape[1]))
                y_full_pred = np.concatenate([y_full_pred, padding], axis=1)
            h_recon = reconstruct_from_basis(y_full_pred[0], U)
            h_true = pad_waveform(val_data[idx]['h22'], max_len)
            delta_t = val_data[idx]['t'][1] - val_data[idx]['t'][0] if len(val_data[idx]['t']) > 1 else 1.0
            mismatch = compute_frequency_domain_mismatch(h_true, h_recon, delta_t)
            mismatches_val.append(mismatch)

        loss = float(np.mean(mismatches_val))

        # Save
        saved_dir = model_dir / "saved_model"
        saved_dir.mkdir(parents=True, exist_ok=True)
        with open(saved_dir / "gplearn_model.pkl", 'wb') as f:
            pickle.dump({'model': est, 'U': U, 'scaler': scaler, 'max_len': max_len}, f)

        scorecard = {
            "approach": "gplearn_svd",
            "approach_number": model_num,
            "benchmark": "waveform",
            "agent": "haiku",
            "parameterization": "eta_chi_7d",
            "loss": loss,
            "loss_train": float(np.mean(mismatches_train)),
            "n_train": len(train_data),
            "n_val": len(val_data),
            "notes": "Symbolic regression via gplearn on SVD coefficients"
        }

        save_progress(model_name, scorecard)
        return model_name, scorecard

    except Exception as e:
        print(f"  gplearn training failed: {e}")
        return None, None

def build_rbf_models(train_data, val_data, model_num):
    """Build RBF interpolation models."""
    model_name = f"NN{model_num}_RBF_SVD"
    model_dir = WORK_DIR / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get SVD basis
    U, s, Vt, max_len = compute_svd_basis(train_data, n_components=10)

    # Prepare data
    X_train = np.array([reparameterize_spherical(d['q'], d['chi1'], d['chi2']) for d in train_data])
    y_train = np.array([project_onto_basis(d['h22'], U, max_len) for d in train_data])

    X_val = np.array([reparameterize_spherical(d['q'], d['chi1'], d['chi2']) for d in val_data])

    print(f"  Training RBF model...")
    try:
        # Train RBF models for each coefficient
        rbf_models = []
        for i in range(min(5, y_train.shape[1])):  # Limit to first 5 coefficients for speed
            rbf = Rbf(*X_train.T, y_train[:, i], function='thin_plate', smooth=1.0)
            rbf_models.append(rbf)

        # Evaluate
        mismatches_val = []
        for idx in range(len(val_data)):
            y_full_pred = np.zeros(y_train.shape[1])
            for i in range(len(rbf_models)):
                y_full_pred[i] = rbf_models[i](*X_val[idx])
            h_recon = reconstruct_from_basis(y_full_pred, U)
            h_true = pad_waveform(val_data[idx]['h22'], max_len)
            delta_t = val_data[idx]['t'][1] - val_data[idx]['t'][0] if len(val_data[idx]['t']) > 1 else 1.0
            mismatch = compute_frequency_domain_mismatch(h_true, h_recon, delta_t)
            mismatches_val.append(mismatch)

        loss = float(np.mean(mismatches_val))

        # Save
        saved_dir = model_dir / "saved_model"
        saved_dir.mkdir(parents=True, exist_ok=True)
        with open(saved_dir / "rbf_models.pkl", 'wb') as f:
            pickle.dump({'rbf_models': rbf_models, 'U': U, 'max_len': max_len}, f)

        scorecard = {
            "approach": "rbf_interpolation",
            "approach_number": model_num,
            "benchmark": "waveform",
            "agent": "haiku",
            "parameterization": "spherical_7d",
            "loss": loss,
            "loss_train": float(loss),  # Same for now
            "n_train": len(train_data),
            "n_val": len(val_data),
            "notes": "RBF interpolation on SVD coefficients"
        }

        save_progress(model_name, scorecard)
        return model_name, scorecard

    except Exception as e:
        print(f"  RBF training failed: {e}")
        return None, None

def build_all():
    """Build all symbolic/specialized models."""
    print("Loading data...")
    wf_data = WaveformData()
    train_data = wf_data.load_training()
    val_data = wf_data.load_validation()

    models_built = []
    MODEL_COUNTER['count'] = 17

    # PySR model
    if HAS_PYSR:
        print(f"\nBuilding Model {MODEL_COUNTER['count']}: PySR symbolic regression...")
        result = build_pysr_model(train_data, val_data, MODEL_COUNTER['count'])
        if result[0]:
            models_built.append(result)
            print(f"  Loss: {result[1]['loss']:.6f}")
            update_changelog(f"\n- {result[0]}: loss={result[1]['loss']:.6f}, param=raw_7d")
            MODEL_COUNTER['count'] += 1

    # gplearn model
    if HAS_GPLEARN:
        print(f"\nBuilding Model {MODEL_COUNTER['count']}: gplearn symbolic regression...")
        result = build_gplearn_model(train_data, val_data, MODEL_COUNTER['count'])
        if result[0]:
            models_built.append(result)
            print(f"  Loss: {result[1]['loss']:.6f}")
            update_changelog(f"\n- {result[0]}: loss={result[1]['loss']:.6f}, param=eta_chi_7d")
            MODEL_COUNTER['count'] += 1

    # RBF model
    print(f"\nBuilding Model {MODEL_COUNTER['count']}: RBF interpolation...")
    result = build_rbf_models(train_data, val_data, MODEL_COUNTER['count'])
    if result[0]:
        models_built.append(result)
        print(f"  Loss: {result[1]['loss']:.6f}")
        update_changelog(f"\n- {result[0]}: loss={result[1]['loss']:.6f}, param=spherical_7d")

    print(f"\n\nSpecialized models built: {len(models_built)}")
    return models_built

if __name__ == "__main__":
    build_all()
    print("\nAll symbolic/specialized models built successfully!")
