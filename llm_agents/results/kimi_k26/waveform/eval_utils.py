"""Evaluation utilities for waveform benchmark."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import time
from pathlib import Path
from gwbenchmarks.metrics import mean_fd_mismatch

# Pre-load reference data to avoid reloading
def load_ref_data():
    from data_utils import load_simulations
    train = load_simulations(Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/waveform/waveform_training.h5'))
    val = load_simulations(Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/waveform/waveform_validation.h5'))
    return train, val

_ref_train = None
_ref_val = None

def get_ref_data():
    global _ref_train, _ref_val
    if _ref_train is None:
        _ref_train, _ref_val = load_ref_data()
    return _ref_train, _ref_val

def compute_fd_mismatch_for_sample(h_pred, h_ref, dt):
    """Compute mean FD mismatch for a single sample."""
    try:
        return mean_fd_mismatch(h_pred, h_ref, dt)
    except Exception as e:
        print(f'Error computing mismatch: {e}')
        return 1.0

def evaluate_model_on_dataset(predict_fn, sims, label='val', verbose=True):
    """Evaluate a model on a dataset.
    
    predict_fn: function that takes a sim dict and returns predicted h22
    Returns: list of losses, mean loss
    """
    losses = []
    for i, sim in enumerate(sims):
        try:
            h_pred = predict_fn(sim)
            h_ref = sim['h22']
            dt = sim['dt']
            loss = compute_fd_mismatch_for_sample(h_pred, h_ref, dt)
            losses.append(loss)
        except Exception as e:
            print(f'Error on {label} sample {i}: {e}')
            losses.append(1.0)
        if verbose and (i + 1) % 10 == 0:
            print(f'  {label}: {i+1}/{len(sims)} done, current mean={np.mean(losses):.4f}')
    return losses, float(np.mean(losses))

def save_scorecard(model_dir, approach, approach_number, parameterization, time_convention,
                   train_losses, val_losses, runtime_ms, notes=''):
    """Save scorecard.json for a model."""
    scorecard = {
        'approach': approach,
        'approach_number': approach_number,
        'benchmark': 'waveform',
        'agent': 'kimi_k26',
        'parameterization': parameterization,
        'time_convention': time_convention,
        'loss': float(np.mean(val_losses)),
        'loss_components': {
            'mismatch_40Msun': float(np.mean(val_losses)),  # Simplified - actual per-mass breakdown not stored
        },
        'runtime_ms': runtime_ms,
        'n_train': len(train_losses),
        'n_val': len(val_losses),
        'n_params': 0,
        'notes': notes,
        'train_loss_mean': float(np.mean(train_losses)),
        'train_loss_median': float(np.median(train_losses)),
        'val_loss_mean': float(np.mean(val_losses)),
        'val_loss_median': float(np.median(val_losses)),
    }
    scorecard_path = Path(model_dir) / 'scorecard.json'
    with open(scorecard_path, 'w') as f:
        json.dump(scorecard, f, indent=2)
    return scorecard

if __name__ == '__main__':
    train, val = get_ref_data()
    print(f'Reference data loaded: {len(train)} train, {len(val)} val')
