"""Evaluate PySR model."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import pickle
from pathlib import Path
from train_evaluate import load_basis, load_ref_sims, evaluate_and_save_approach, RESULTS_DIR

model_dir = RESULTS_DIR / 'models' / '23_svd_pysr_raw'

with open(model_dir / 'saved_model' / 'models.pkl', 'rb') as f:
    data = pickle.load(f)

basis_data = load_basis()
train_sims, val_sims = load_ref_sims()
train_params = basis_data['train_params']
val_params = basis_data['val_params']

models_real = data['models_real']
models_imag = data['models_imag']
mean_x = data['mean_x']
std_x = data['std_x']
ptype = data['ptype']
n_basis_use = data['n_basis_use']

scorecard = evaluate_and_save_approach(
    '23_svd_pysr_raw', models_real, models_imag, mean_x, std_x,
    basis_data, val_sims, val_params, ptype, n_basis_use, 'pysr'
)

print(f'PySR loss: {scorecard["loss"]:.4f}')
