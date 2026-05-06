"""Build SVD basis and common time grid for waveform modeling."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import h5py
from pathlib import Path
import pickle

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/waveform')
DATA_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/waveform')

def load_simulations(h5_path):
    data = []
    with h5py.File(h5_path, 'r') as f:
        n = f.attrs['n_simulations']
        for i in range(n):
            g = f[f'sim_{i:04d}']
            t = g['t'][:]
            h22 = g['h22_real'][:] + 1j * g['h22_imag'][:]
            data.append({
                'q': float(g.attrs['q']),
                'chi1': np.array([g.attrs['chi1x'], g.attrs['chi1y'], g.attrs['chi1z']]),
                'chi2': np.array([g.attrs['chi2x'], g.attrs['chi2y'], g.attrs['chi2z']]),
                'omega0': float(g.attrs['omega0']),
                't': t,
                'h22': h22,
                'dt': float(t[1] - t[0]) if len(t) > 1 else 1.0,
                'n_pts': len(t),
            })
    return data

def build_common_grid_and_svd(n_basis=50, target_npts=5000):
    """Build common time grid and SVD basis."""
    print('Loading training data...')
    train = load_simulations(DATA_DIR / 'waveform_training.h5')
    val = load_simulations(DATA_DIR / 'waveform_validation.h5')
    
    # Find common time range
    t_min = max(s['t'].min() for s in train)
    t_max = min(s['t'].max() for s in train)
    
    # Use target_npts or minimum length
    min_len = min(s['n_pts'] for s in train)
    npts = min(target_npts, min_len)
    
    t_common = np.linspace(t_min, t_max, npts)
    print(f'Common grid: [{t_min:.2f}, {t_max:.2f}], {npts} points')
    
    # Interpolate training waveforms to common grid
    print('Interpolating waveforms to common grid...')
    H_train = []
    for s in train:
        h_real = np.interp(t_common, s['t'], s['h22'].real)
        h_imag = np.interp(t_common, s['t'], s['h22'].imag)
        H_train.append(h_real + 1j * h_imag)
    H_train = np.array(H_train)  # (n_train, npts)
    
    # Also interpolate validation
    H_val = []
    for s in val:
        h_real = np.interp(t_common, s['t'], s['h22'].real)
        h_imag = np.interp(t_common, s['t'], s['h22'].imag)
        H_val.append(h_real + 1j * h_imag)
    H_val = np.array(H_val)
    
    # Build SVD on real and imaginary parts separately
    print('Building SVD basis...')
    H_real = H_train.real
    H_imag = H_train.imag
    
    # Center the data
    mean_real = np.mean(H_real, axis=0)
    mean_imag = np.mean(H_imag, axis=0)
    
    U_real, S_real, Vt_real = np.linalg.svd(H_real - mean_real, full_matrices=False)
    U_imag, S_imag, Vt_imag = np.linalg.svd(H_imag - mean_imag, full_matrices=False)
    
    # Keep top n_basis
    basis_real = Vt_real[:n_basis, :].T  # (npts, n_basis)
    basis_imag = Vt_imag[:n_basis, :].T
    
    # Compute coefficients
    coeffs_real_train = np.dot(H_real - mean_real, basis_real)  # (n_train, n_basis)
    coeffs_imag_train = np.dot(H_imag - mean_imag, basis_imag)
    
    coeffs_real_val = np.dot(H_val.real - mean_real, basis_real)
    coeffs_imag_val = np.dot(H_val.imag - mean_imag, basis_imag)
    
    # Save everything
    save_dir = RESULTS_DIR / 'saved_basis'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / 't_common.npy', t_common)
    np.save(save_dir / 'mean_real.npy', mean_real)
    np.save(save_dir / 'mean_imag.npy', mean_imag)
    np.save(save_dir / 'basis_real.npy', basis_real)
    np.save(save_dir / 'basis_imag.npy', basis_imag)
    np.save(save_dir / 'coeffs_real_train.npy', coeffs_real_train)
    np.save(save_dir / 'coeffs_imag_train.npy', coeffs_imag_train)
    np.save(save_dir / 'coeffs_real_val.npy', coeffs_real_val)
    np.save(save_dir / 'coeffs_imag_val.npy', coeffs_imag_val)
    
    # Save parameters
    train_params = []
    for s in train:
        train_params.append([
            s['q'],
            s['chi1'][0], s['chi1'][1], s['chi1'][2],
            s['chi2'][0], s['chi2'][1], s['chi2'][2],
            s['omega0'],
        ])
    train_params = np.array(train_params)
    
    val_params = []
    for s in val:
        val_params.append([
            s['q'],
            s['chi1'][0], s['chi1'][1], s['chi1'][2],
            s['chi2'][0], s['chi2'][1], s['chi2'][2],
            s['omega0'],
        ])
    val_params = np.array(val_params)
    
    np.save(save_dir / 'train_params.npy', train_params)
    np.save(save_dir / 'val_params.npy', val_params)
    
    # Save reconstruction info for original grids
    train_info = []
    for s in train:
        train_info.append({'t': s['t'], 'dt': s['dt']})
    val_info = []
    for s in val:
        val_info.append({'t': s['t'], 'dt': s['dt']})
    
    with open(save_dir / 'train_info.pkl', 'wb') as f:
        pickle.dump(train_info, f)
    with open(save_dir / 'val_info.pkl', 'wb') as f:
        pickle.dump(val_info, f)
    
    print(f'Saved basis to {save_dir}')
    print(f'SVD real singular values: {S_real[:10]}')
    print(f'SVD imag singular values: {S_imag[:10]}')
    print(f'Coefficients shape: {coeffs_real_train.shape}')
    
    return {
        't_common': t_common,
        'mean_real': mean_real, 'mean_imag': mean_imag,
        'basis_real': basis_real, 'basis_imag': basis_imag,
        'coeffs_real_train': coeffs_real_train, 'coeffs_imag_train': coeffs_imag_train,
        'coeffs_real_val': coeffs_real_val, 'coeffs_imag_val': coeffs_imag_val,
        'train_params': train_params, 'val_params': val_params,
        'train': train, 'val': val,
    }

if __name__ == '__main__':
    build_common_grid_and_svd(n_basis=50, target_npts=5000)
