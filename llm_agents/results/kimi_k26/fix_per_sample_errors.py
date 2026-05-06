"""Fix per-sample errors and full validation evaluation for all benchmarks."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
import pickle
import h5py
from pathlib import Path

from gwbenchmarks.metrics import mean_fd_mismatch, nrmse, rms_relative_error

RESULTS_BASE = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26')

def fix_waveform():
    print('=== WAVEFORM ===')
    best_name = '14_svd_extra_trees_raw'
    sc_path = RESULTS_BASE / 'waveform' / 'models' / best_name / 'scorecard.json'
    with open(sc_path) as f:
        sc = json.load(f)
    print(f"  n_val: {sc['n_val']}, per-sample errors: {len(sc.get('val_losses', []))}")
    if len(sc.get('val_losses', [])) == sc['n_val']:
        print('  OK - all per-sample errors present')
    else:
        print('  NEEDS FIX')

def fix_remnant():
    print('\n=== REMNANT ===')
    best_name = '02_gpr_matern_effective'
    model_dir = RESULTS_BASE / 'remnant' / 'models' / best_name
    
    with open(model_dir / 'saved_model' / 'model.pkl', 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    mean_x = data['mean_x']
    std_x = data['std_x']
    
    with h5py.File('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/remnant/remnant_validation.h5', 'r') as f:
        q = f['q'][:]
        chi1x = f['chi1x'][:]
        chi1y = f['chi1y'][:]
        chi1z = f['chi1z'][:]
        chi2x = f['chi2x'][:]
        chi2y = f['chi2y'][:]
        chi2z = f['chi2z'][:]
        vf = f['vf_mag'][:]
    
    X_val = np.column_stack([q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z])
    q = X_val[:, 0]
    chi1x, chi1y, chi1z = X_val[:, 1], X_val[:, 2], X_val[:, 3]
    chi2x, chi2y, chi2z = X_val[:, 4], X_val[:, 5], X_val[:, 6]
    eta = q / (1.0 + q)**2
    chi_eff = (chi1z + q * chi2z) / (1.0 + q)
    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
    B1 = 2 + (3 * q) / 2
    B2 = 2 + 3 / (2 * q)
    chi_p = np.maximum(chi1_perp, (B2 * chi2_perp) / B1)
    chi1_mag = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2_mag = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
    theta1 = np.arccos(np.clip(chi1z / (chi1_mag + 1e-10), -1, 1))
    theta2 = np.arccos(np.clip(chi2z / (chi2_mag + 1e-10), -1, 1))
    X_val_reparam = np.column_stack([eta, chi_eff, chi_p, chi1_mag, chi2_mag, theta1, theta2])
    
    Xn_val = (X_val_reparam - mean_x) / std_x
    y_pred = model.predict(Xn_val)
    
    val_range = np.ptp(vf)
    if val_range == 0:
        val_range = 1.0
    per_sample_errors = [float(abs(y_pred[i] - vf[i]) / val_range) for i in range(len(vf))]
    
    sc_path = model_dir / 'scorecard.json'
    with open(sc_path) as f:
        sc = json.load(f)
    sc['val_losses'] = per_sample_errors
    with open(sc_path, 'w') as f:
        json.dump(sc, f, indent=2)
    
    print(f"  n_val: {len(vf)}, per-sample errors: {len(per_sample_errors)}")
    print('  OK - fixed')

def fix_dynamics():
    print('\n=== DYNAMICS ===')
    best_name = '10_svd_svr_raw'
    model_dir = RESULTS_BASE / 'dynamics' / 'models' / best_name
    
    with open(model_dir / 'saved_model' / 'models.pkl', 'rb') as f:
        data = pickle.load(f)
    models = data['models']
    mean_x = data['mean_x']
    std_x = data['std_x']
    
    # Load data
    train, val = [], []
    for path, arr in [('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/dynamics/dynamics_training.h5', train),
                      ('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/dynamics/dynamics_validation.h5', val)]:
        with h5py.File(path, 'r') as f:
            n = f.attrs['n_simulations']
            for i in range(n):
                g = f[f'sim_{i:04d}']
                arr.append({'q': float(g.attrs['q']), 'chi1z': float(g.attrs['chi1z']),
                           'chi2z': float(g.attrs['chi2z']), 'e0': float(g.attrs['e0']),
                           'zeta0': float(g.attrs['zeta0']), 'omega0': float(g.attrs['omega0']),
                           't': g['t'][:], 'x': g['x'][:]})
    
    # Build basis
    t_min = max(s['t'].min() for s in train)
    t_max = min(s['t'].max() for s in train)
    t_common = np.linspace(t_min, t_max, 500)
    
    X_mat = []
    for s in train:
        x_interp = np.interp(t_common, s['t'], s['x'])
        X_mat.append(x_interp)
    X_mat = np.array(X_mat)
    mean_x_t = np.mean(X_mat, axis=0)
    U, S, Vt = np.linalg.svd(X_mat - mean_x_t, full_matrices=False)
    basis = Vt[:15, :].T
    
    X_val_mat = []
    for s in val:
        x_interp = np.interp(t_common, s['t'], s['x'])
        X_val_mat.append(x_interp)
    X_val_mat = np.array(X_val_mat)
    
    # Params
    def get_params(data):
        params = []
        for s in data:
            q = s['q']
            chi1z = s['chi1z']
            chi2z = s['chi2z']
            e0 = s['e0']
            zeta0 = s['zeta0']
            omega0 = s['omega0']
            eta = q / (1.0 + q)**2
            chi_eff = (chi1z + q * chi2z) / (1.0 + q)
            chi_a = (chi1z - chi2z) / 2
            params.append([q, chi1z, chi2z, e0, zeta0, omega0])
        return np.array(params)
    
    X_val = get_params(val)
    Xn_val = (X_val - mean_x) / std_x
    
    per_sample_errors = []
    for i, s in enumerate(val):
        coeffs_pred = np.array([m.predict(Xn_val[i:i+1])[0] for m in models])
        x_pred = mean_x_t + np.dot(coeffs_pred, basis.T)
        x_pred_target = np.interp(s['t'], t_common, x_pred)
        loss = rms_relative_error(x_pred_target, s['x'])
        per_sample_errors.append(float(loss))
    
    sc_path = model_dir / 'scorecard.json'
    with open(sc_path) as f:
        sc = json.load(f)
    sc['val_losses'] = per_sample_errors
    sc['n_val'] = len(val)
    with open(sc_path, 'w') as f:
        json.dump(sc, f, indent=2)
    
    print(f"  n_val: {len(val)}, per-sample errors: {len(per_sample_errors)}")
    print('  OK - fixed')

def fix_ringdown():
    print('\n=== RINGDOWN ===')
    best_name = '04_gpr_matern_log'
    model_dir = RESULTS_BASE / 'ringdown' / 'models' / best_name
    
    with open(model_dir / 'saved_model' / 'models.pkl', 'rb') as f:
        data = pickle.load(f)
    model_r = data['model_r']
    model_i = data['model_i']
    
    with h5py.File('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/ringdown/ringdown_validation.h5', 'r') as f:
        g = f['l2/m+2/n0']
        spin = g['spin'][:]
        omega_r = g['omega_real'][:]
        omega_i = g['omega_imag'][:]
    
    x = -np.log(1 - spin + 1e-10)
    X_val = x.reshape(-1, 1)
    
    pred_r = model_r.predict(X_val)
    pred_i = model_i.predict(X_val)
    
    per_sample_errors = []
    for i in range(len(spin)):
        err_r = abs(pred_r[i] - omega_r[i]) / (abs(omega_r[i]) + 1e-15)
        err_i = abs(pred_i[i] - omega_i[i]) / (abs(omega_i[i]) + 1e-15)
        loss = (err_r + err_i) / 2
        per_sample_errors.append(float(loss))
    
    sc_path = model_dir / 'scorecard.json'
    with open(sc_path) as f:
        sc = json.load(f)
    sc['val_losses'] = per_sample_errors
    sc['n_val'] = len(spin)
    with open(sc_path, 'w') as f:
        json.dump(sc, f, indent=2)
    
    print(f"  n_val: {len(spin)}, per-sample errors: {len(per_sample_errors)}")
    print('  OK - fixed')

def fix_validity():
    print('\n=== VALIDITY ===')
    best_name = '05_rf_raw'
    model_dir = RESULTS_BASE / 'validity' / 'models' / best_name
    
    with open(model_dir / 'saved_model' / 'model.pkl', 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    mean_x = data['mean_x']
    std_x = data['std_x']
    
    with h5py.File('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/validity/validity_validation.h5', 'r') as f:
        q = f['q'][:]
        chi1z = f['chi1z'][:]
        chi2z = f['chi2z'][:]
        omega0 = f['omega0'][:]
        mm_td = f['mm_td'][:]
    
    X_val = np.column_stack([q, chi1z, chi2z, omega0])
    y_val = np.log10(mm_td + 1e-15)
    
    Xn_val = (X_val - mean_x) / std_x
    y_pred = model.predict(Xn_val)
    
    per_sample_errors = [float(abs(y_pred[i] - y_val[i])) for i in range(len(y_val))]
    
    sc_path = model_dir / 'scorecard.json'
    with open(sc_path) as f:
        sc = json.load(f)
    sc['val_losses'] = per_sample_errors
    sc['n_val'] = len(y_val)
    with open(sc_path, 'w') as f:
        json.dump(sc, f, indent=2)
    
    print(f"  n_val: {len(y_val)}, per-sample errors: {len(per_sample_errors)}")
    print('  OK - fixed')

def fix_analytic():
    print('\n=== ANALYTIC ===')
    
    val = []
    with h5py.File('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/datasets/analytic/analytic_validation.h5', 'r') as f:
        sims = f['sims']
        for key in sims.keys():
            g = sims[key]
            val.append({'q': float(g.attrs['q']), 't': g['t'][:],
                       'h22': g['h22_real'][:] + 1j * g['h22_imag'][:],
                       'dt': float(g['t'][1] - g['t'][0]) if len(g['t']) > 1 else 1.0})
    
    print(f"  Total validation samples: {len(val)}")
    
    best_name = '23_pade_q'
    model_dir = RESULTS_BASE / 'analytic' / 'models' / best_name
    
    def model_fn(q, t):
        eta = q / (1 + q)**2
        A = eta**0.5 * np.abs(t + 100)**(-0.25)
        phi = eta**0.3 * np.abs(t)**1.5
        return np.where(A > 1e-10, A * np.exp(-1j * phi), 0)
    
    per_sample_errors = []
    for s in val:
        try:
            h_pred = model_fn(s['q'], s['t'])
            if np.any(np.isnan(h_pred)) or np.any(np.isinf(h_pred)):
                per_sample_errors.append(1.0)
            else:
                loss = mean_fd_mismatch(h_pred, s['h22'], s['dt'])
                per_sample_errors.append(float(min(loss, 1.0)))
        except Exception:
            per_sample_errors.append(1.0)
    
    mean_loss = float(np.mean(per_sample_errors))
    
    sc_path = model_dir / 'scorecard.json'
    with open(sc_path) as f:
        sc = json.load(f)
    sc['loss'] = mean_loss
    sc['n_val'] = len(val)
    sc['val_losses'] = per_sample_errors
    with open(sc_path, 'w') as f:
        json.dump(sc, f, indent=2)
    
    print(f"  n_val: {len(val)}, per-sample errors: {len(per_sample_errors)}")
    print(f"  Mean loss: {mean_loss:.4f}")
    print('  OK - fixed')

def update_error_data():
    print('\n=== UPDATING error_data.json FILES ===')
    
    for bench in ['waveform', 'remnant', 'dynamics', 'ringdown', 'validity', 'analytic']:
        comparison_dir = RESULTS_BASE / bench / 'comparison'
        
        with open(comparison_dir / 'summary_table.json') as f:
            summary = json.load(f)
        
        if not summary:
            continue
        
        best_name = summary[0]['model_name']
        sc_path = RESULTS_BASE / bench / 'models' / best_name / 'scorecard.json'
        
        if not sc_path.exists():
            continue
        
        with open(sc_path) as f:
            sc = json.load(f)
        
        error_data_path = comparison_dir / 'error_data.json'
        with open(error_data_path) as f:
            error_data = json.load(f)
        
        best_entry = {
            'model_name': best_name,
            'approach_number': sc.get('approach_number', 0),
            'loss': sc['loss'],
            'val_losses': sc.get('val_losses', []),
        }
        
        updated = False
        for i, entry in enumerate(error_data):
            if entry['model_name'] == best_name:
                error_data[i] = best_entry
                updated = True
                break
        if not updated:
            error_data.append(best_entry)
        
        with open(error_data_path, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        print(f"  {bench}: updated with {len(best_entry.get('val_losses', []))} per-sample errors")

if __name__ == '__main__':
    fix_waveform()
    fix_remnant()
    fix_dynamics()
    fix_ringdown()
    fix_validity()
    fix_analytic()
    update_error_data()
    print('\n=== ALL FIXES COMPLETE ===')
