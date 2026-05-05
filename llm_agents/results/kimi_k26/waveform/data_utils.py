"""Data loading and preprocessing for waveform benchmark."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import h5py
import numpy as np
from pathlib import Path

DATA_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/waveform')

def load_simulations(h5_path):
    """Load all simulations from an HDF5 file.
    
    Returns list of dicts with keys: q, chi1, chi2, omega0, t, h22, dt, n_pts, sxs_id
    """
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
                'sxs_id': str(g.attrs.get('sxs_id', '')),
            })
    return data

def compute_eta(q):
    """Symmetric mass ratio."""
    return q / (1.0 + q)**2

def compute_delta_m(q):
    """Mass difference parameter."""
    return (q - 1.0) / (q + 1.0)

def compute_chi_eff(q, chi1z, chi2z):
    """Effective spin parameter."""
    eta = compute_eta(q)
    return (chi1z + q * chi2z) / (1.0 + q)

def compute_chi_p(q, chi1x, chi1y, chi2x, chi2y):
    """Precession parameter."""
    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
    B1 = 2 + (3 * q) / 2
    B2 = 2 + 3 / (2 * q)
    chi_eff_perp = np.maximum(chi1_perp, (B2 * chi2_perp) / B1)
    return chi_eff_perp

def cartesian_to_spherical(chi):
    """Convert cartesian spin to spherical: |chi|, theta, phi."""
    norm = np.linalg.norm(chi)
    if norm == 0:
        return 0.0, 0.0, 0.0
    theta = np.arccos(np.clip(chi[2] / norm, -1.0, 1.0))
    phi = np.arctan2(chi[1], chi[0])
    return norm, theta, phi

def get_raw_params(sim):
    """Raw 7D parameters."""
    return np.array([
        sim['q'],
        sim['chi1'][0], sim['chi1'][1], sim['chi1'][2],
        sim['chi2'][0], sim['chi2'][1], sim['chi2'][2],
    ])

def get_effective_params(sim):
    """Effective spin parameterization."""
    q = sim['q']
    chi1 = sim['chi1']
    chi2 = sim['chi2']
    eta = compute_eta(q)
    chi_eff = compute_chi_eff(q, chi1[2], chi2[2])
    chi_p = compute_chi_p(q, chi1[0], chi1[1], chi2[0], chi2[1])
    chi1_mag = np.linalg.norm(chi1)
    chi2_mag = np.linalg.norm(chi2)
    _, theta1, _ = cartesian_to_spherical(chi1)
    _, theta2, _ = cartesian_to_spherical(chi2)
    return np.array([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])

def get_mass_diff_params(sim):
    """Mass difference + spins parameterization."""
    q = sim['q']
    chi1 = sim['chi1']
    chi2 = sim['chi2']
    delta_m = compute_delta_m(q)
    chi_eff = compute_chi_eff(q, chi1[2], chi2[2])
    chi_p = compute_chi_p(q, chi1[0], chi1[1], chi2[0], chi2[1])
    chi1_mag = np.linalg.norm(chi1)
    chi2_mag = np.linalg.norm(chi2)
    _, _, phi1 = cartesian_to_spherical(chi1)
    _, _, phi2 = cartesian_to_spherical(chi2)
    return np.array([delta_m, chi_eff, chi_p, chi1_mag, chi2_mag, phi1, phi2])

def get_spherical_params(sim):
    """Spherical spin parameterization."""
    q = sim['q']
    chi1 = sim['chi1']
    chi2 = sim['chi2']
    eta = compute_eta(q)
    chi1_mag, theta1, phi1 = cartesian_to_spherical(chi1)
    chi2_mag, theta2, phi2 = cartesian_to_spherical(chi2)
    return np.array([eta, chi1_mag, theta1, phi1, chi2_mag, theta2, phi2])

def get_omega0_params(sim):
    """Raw params + omega0."""
    raw = get_raw_params(sim)
    return np.append(raw, sim['omega0'])

def align_waveforms(waveforms, t_ref=None):
    """Align waveforms to a common time grid by interpolation.
    
    If t_ref is None, use the first waveform's time grid.
    """
    if t_ref is None:
        t_ref = waveforms[0]['t']
    
    aligned = []
    for w in waveforms:
        h_interp = np.interp(t_ref, w['t'], w['h22'].real) + 1j * np.interp(t_ref, w['t'], w['h22'].imag)
        aligned.append(h_interp)
    return np.array(aligned), t_ref

def build_parameter_matrix(sims, param_type='raw'):
    """Build parameter matrix for all simulations.
    
    param_type: 'raw', 'effective', 'mass_diff', 'spherical', 'omega0'
    """
    getters = {
        'raw': get_raw_params,
        'effective': get_effective_params,
        'mass_diff': get_mass_diff_params,
        'spherical': get_spherical_params,
        'omega0': get_omega0_params,
    }
    getter = getters[param_type]
    return np.array([getter(s) for s in sims])

def normalize_params(X, X_mean=None, X_std=None):
    """Z-normalize parameters."""
    if X_mean is None:
        X_mean = np.mean(X, axis=0)
    if X_std is None:
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0
    return (X - X_mean) / X_std, X_mean, X_std

if __name__ == '__main__':
    train = load_simulations(DATA_DIR / 'waveform_training.h5')
    val = load_simulations(DATA_DIR / 'waveform_validation.h5')
    print(f'Loaded {len(train)} training, {len(val)} validation simulations')
    print(f'Train waveform shape: {train[0]["h22"].shape}')
    print(f'Train q range: [{min(s["q"] for s in train):.2f}, {max(s["q"] for s in train):.2f}]')
    
    # Test parameterizations
    for ptype in ['raw', 'effective', 'mass_diff', 'spherical', 'omega0']:
        X = build_parameter_matrix(train, ptype)
        print(f'{ptype}: shape={X.shape}, range=[{X.min():.3f}, {X.max():.3f}]')
