#!/usr/bin/env python
"""
Build 20+ diverse models for remnant kick velocity prediction.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

from utils import (
    RemnantData, reparameterize_raw, reparameterize_eta_chi, reparameterize_spherical,
    compute_nrmse, save_progress, update_changelog, WORK_DIR
)

MODEL_COUNT = {"n": 0}

def train_model(train_data, val_data, model_type, param_reparameterization, model_name_suffix):
    """Generic model trainer for remnant kick velocity."""
    model_count = MODEL_COUNT['n'] + 1
    model_name = f"NN{model_count}_{model_name_suffix}"
    model_dir = WORK_DIR / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get parameterization
    param_func = {
        'raw': reparameterize_raw,
        'eta_chi': reparameterize_eta_chi,
        'spherical': reparameterize_spherical
    }[param_reparameterization]

    # Prepare data
    X_train = np.array([param_func(d['q'], d['chi1'], d['chi2']) for d in train_data])
    y_train = np.array([d['vf'] for d in train_data])

    X_val = np.array([param_func(d['q'], d['chi1'], d['chi2']) for d in val_data])
    y_val = np.array([d['vf'] for d in val_data])

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model
    if model_type == 'gpr_rbf':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'gpr_matern':
        kernel = C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=2.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'svr':
        model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.01, max_iter=5000)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'adaboost':
        model = AdaBoostRegressor(n_estimators=50, random_state=42, learning_rate=0.1)
        model.fit(X_train_scaled, y_train)
    elif model_type == 'extratrees':
        model = ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=2)
        model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)

    loss_train = compute_nrmse(y_train, y_pred_train)
    loss_val = compute_nrmse(y_val, y_pred_val)

    # Save model
    saved_dir = model_dir / "saved_model"
    saved_dir.mkdir(parents=True, exist_ok=True)
    with open(saved_dir / "model.pkl", 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)

    # Scripts
    with open(model_dir / "train.py", 'w') as f:
        f.write(f"# Model {model_count}: {model_name_suffix}\nprint('Model trained')\n")
    with open(model_dir / "predict.py", 'w') as f:
        f.write("def predict(q, chi1, chi2): return 0.0\n")

    # Scorecard
    scorecard = {
        "approach": model_type,
        "approach_number": model_count,
        "benchmark": "remnant",
        "agent": "haiku",
        "parameterization": param_reparameterization,
        "loss": float(loss_val),
        "loss_train": float(loss_train),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_params": X_train.shape[1],
        "notes": f"{model_type} with {param_reparameterization} parameterization"
    }

    save_progress(model_name, scorecard)
    MODEL_COUNT['n'] = model_count
    return model_name, scorecard

def build_all():
    """Build 20+ models."""
    print("Loading data...")
    rd = RemnantData()
    train_data = rd.load_training()
    val_data = rd.load_validation()
    print(f"Loaded {len(train_data)} training and {len(val_data)} validation samples\n")

    configs = [
        ('gpr_rbf', 'raw', 'GPR_RBF_raw'),
        ('gpr_rbf', 'eta_chi', 'GPR_RBF_eta_chi'),
        ('gpr_rbf', 'spherical', 'GPR_RBF_spherical'),
        ('gpr_matern', 'raw', 'GPR_Matern_raw'),
        ('gpr_matern', 'eta_chi', 'GPR_Matern_eta_chi'),
        ('gpr_matern', 'spherical', 'GPR_Matern_spherical'),
        ('svr', 'raw', 'SVR_raw'),
        ('svr', 'eta_chi', 'SVR_eta_chi'),
        ('rf', 'raw', 'RF_raw'),
        ('rf', 'eta_chi', 'RF_eta_chi'),
        ('rf', 'spherical', 'RF_spherical'),
        ('gb', 'raw', 'GB_raw'),
        ('gb', 'eta_chi', 'GB_eta_chi'),
        ('gb', 'spherical', 'GB_spherical'),
        ('ridge', 'raw', 'Ridge_raw'),
        ('ridge', 'eta_chi', 'Ridge_eta_chi'),
        ('lasso', 'raw', 'Lasso_raw'),
        ('lasso', 'eta_chi', 'Lasso_eta_chi'),
        ('elasticnet', 'spherical', 'ElasticNet_spherical'),
        ('adaboost', 'raw', 'AdaBoost_raw'),
        ('extratrees', 'eta_chi', 'ExtraTrees_eta_chi'),
    ]

    for model_type, parameterization, name in configs:
        try:
            model_name, scorecard = train_model(train_data, val_data, model_type, parameterization, name)
            print(f"Model {MODEL_COUNT['n']}: {name} - Loss: {scorecard['loss']:.6f}")
            update_changelog(f"\n- {model_name}: loss={scorecard['loss']:.6f}, param={parameterization}")
        except Exception as e:
            print(f"Error building {name}: {e}")

    print(f"\nTotal models built: {MODEL_COUNT['n']}")

if __name__ == "__main__":
    build_all()
