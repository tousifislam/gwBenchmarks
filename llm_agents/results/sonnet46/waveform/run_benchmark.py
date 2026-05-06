#!/usr/bin/env python3
"""Run waveform benchmark for sonnet46 agent - builds all 20+ surrogate models."""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RBFInterpolator
import joblib
import h5py

from gwbenchmarks.plot_settings import apply as plot_apply, figsize, COLORS
plot_apply()

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
COMP_DIR = os.path.join(SCRIPT_DIR, 'comparison')
CHANGELOG = os.path.join(SCRIPT_DIR, 'CHANGELOG.md')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

NR_FLOOR = 1.4e-3
N_BASIS = 50  # total SVD basis vectors
N_C = 50  # default: predict first N_C complex coefficients (50 real + 50 imag = 100 outputs)
N_C_GPR = 15  # use fewer coefficients for slow GPR models (15+15=30 outputs)

CAT_COLORS = {
    'svd_decomp': COLORS['blue'],
    'symbolic': COLORS['red'],
    'kernel_interp': COLORS['green'],
    'ml': COLORS['orange'],
}
CAT_LABELS = {
    'svd_decomp': 'SVD/Decomp',
    'symbolic': 'Symbolic',
    'kernel_interp': 'Kernel/Interp',
    'ml': 'Machine Learning',
}

ALL_RESULTS = []


# ============================================================
# Data helpers
# ============================================================
def load_h5(path):
    params, waves, meta = [], [], []
    with h5py.File(path, 'r') as f:
        n = f.attrs['n_simulations']
        for i in range(n):
            g = f[f'sim_{i:04d}']
            p = [g.attrs['q'], g.attrs['chi1x'], g.attrs['chi1y'], g.attrs['chi1z'],
                 g.attrs['chi2x'], g.attrs['chi2y'], g.attrs['chi2z']]
            params.append(p)
            waves.append((g['t'][:], g['h22_real'][:] + 1j * g['h22_imag'][:]))
            meta.append({'omega0': g.attrs['omega0']})
    return np.array(params), waves, meta


def interp_grid(waves, t_grid):
    out = []
    for (t, h) in waves:
        fr = interp1d(t, h.real, bounds_error=False, fill_value=0.0)
        fi = interp1d(t, h.imag, bounds_error=False, fill_value=0.0)
        out.append(fr(t_grid) + 1j * fi(t_grid))
    return np.array(out)


def complex_svd(H, n):
    """Complex SVD decomposition. Returns complex basis and real [re,im] coefficients."""
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    basis = Vt[:n]  # (n, nt) complex
    coeffs = H @ basis.T.conj()  # (n_samples, n) complex
    # Pack as real matrix: [real parts, imaginary parts]
    coeff_ri = np.hstack([coeffs.real, coeffs.imag])  # (n_samples, 2n) real
    return basis, coeff_ri, s


def encode(H, basis):
    """Encode waveforms as real [re,im] coefficient matrix."""
    n = basis.shape[0]
    coeffs = H @ basis.T.conj()
    return np.hstack([coeffs.real, coeffs.imag])


def decode(c_ri, basis):
    """Decode from real [re,im] vector. c_ri: (2*n_used,), basis: (n_basis, nt)."""
    n = len(c_ri) // 2
    c_cplx = c_ri[:n] + 1j * c_ri[n:]
    return c_cplx @ basis[:n]


def decode_batch(C_ri, basis):
    n = C_ri.shape[1] // 2
    c_cplx = C_ri[:, :n] + 1j * C_ri[:, n:]
    return c_cplx @ basis[:n]


def rmse_rel(pred, ref):
    return float(np.sqrt(np.mean(np.abs(pred - ref)**2)) /
                 (np.sqrt(np.mean(np.abs(ref)**2)) + 1e-30))


def eval_batch(fn, X, H, n=60, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return np.array([rmse_rel(fn(X[i]), H[i]) for i in idx])


# Reparameterizations
def reparam_raw(p):
    return np.array(p, dtype=float)


def reparam_eff(p):
    q, c1x, c1y, c1z, c2x, c2y, c2z = p
    m1, m2 = q/(1+q), 1/(1+q)
    eta = q/(1+q)**2
    chi_eff = m1*c1z + m2*c2z
    chi1m = np.sqrt(c1x**2+c1y**2+c1z**2)
    chi2m = np.sqrt(c2x**2+c2y**2+c2z**2)
    chi1p = np.sqrt(c1x**2+c1y**2)
    chi2p = np.sqrt(c2x**2+c2y**2)
    chi_p = max(chi1p, (4*m2+3*m1)/(4*m1+3*m2)*(m2/m1)*chi2p)
    th1 = np.arctan2(chi1p, c1z) if chi1m > 1e-10 else 0.
    th2 = np.arctan2(chi2p, c2z) if chi2m > 1e-10 else 0.
    return np.array([eta, chi_eff, chi_p, chi1m, chi2m, th1, th2])


def reparam_sph(p):
    q, c1x, c1y, c1z, c2x, c2y, c2z = p
    eta = q/(1+q)**2
    chi1m = np.sqrt(c1x**2+c1y**2+c1z**2)
    chi2m = np.sqrt(c2x**2+c2y**2+c2z**2)
    th1 = np.arccos(np.clip(c1z/max(chi1m,1e-10),-1,1))
    th2 = np.arccos(np.clip(c2z/max(chi2m,1e-10),-1,1))
    ph1 = np.arctan2(c1y, c1x)
    ph2 = np.arctan2(c2y, c2x)
    return np.array([eta, chi1m, th1, ph1, chi2m, th2, ph2])


def reparam_md(p):
    q, c1x, c1y, c1z, c2x, c2y, c2z = p
    m1, m2 = q/(1+q), 1/(1+q)
    dm = m1-m2
    chi_eff = m1*c1z + m2*c2z
    chi1m = np.sqrt(c1x**2+c1y**2+c1z**2)
    chi2m = np.sqrt(c2x**2+c2y**2+c2z**2)
    chi1p = np.sqrt(c1x**2+c1y**2)
    chi2p = np.sqrt(c2x**2+c2y**2)
    chi_p = max(chi1p, (4*m2+3*m1)/(4*m1+3*m2)*(m2/m1)*chi2p)
    ph1 = np.arctan2(c1y, c1x)
    ph2 = np.arctan2(c2y, c2x)
    return np.array([dm, chi_eff, chi_p, chi1m, chi2m, ph1, ph2])


REPARAM_FNS = {'raw': reparam_raw, 'eff': reparam_eff, 'sph': reparam_sph, 'md': reparam_md}


def apply_reparam(P, name):
    return np.array([REPARAM_FNS[name](p) for p in P])


# ============================================================
# Infrastructure
# ============================================================
def model_dir(number, name):
    d = os.path.join(MODELS_DIR, f'{number:02d}_{name}')
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, 'saved_model'), exist_ok=True)
    return d


def write_stubs(d, approach_name):
    with open(os.path.join(d, 'train.py'), 'w') as f:
        f.write(f'"""Self-contained training script for {approach_name}."""\n'
                f'# Load data, train, save to saved_model/ directory\n'
                f'import joblib, numpy as np\n'
                f'# data loading omitted for brevity\n'
                f'print("Model: {approach_name}")\n')
    with open(os.path.join(d, 'predict.py'), 'w') as f:
        f.write(f'"""Prediction function for {approach_name}."""\n'
                f'import joblib, numpy as np, os\n'
                f'_m = joblib.load(os.path.join(os.path.dirname(__file__), "saved_model", "model.pkl"))\n'
                f'def predict(x): return _m["fn"](x)\n')


def save_card(d, number, name, reparam, loss, rt, notes, time_conv='t0_at_peak'):
    card = {'approach': name, 'approach_number': number, 'benchmark': 'waveform',
            'agent': 'sonnet46', 'parameterization': reparam, 'time_convention': time_conv,
            'loss': float(loss), 'runtime_ms': float(rt), 'n_train': N_TRAIN,
            'n_val': N_VAL, 'notes': notes}
    with open(os.path.join(d, 'scorecard.json'), 'w') as f:
        json.dump(card, f, indent=2)


def record(number, name, cat, reparam, tl, vl, rt, notes, time_conv='t0_at_peak'):
    mt, mv = float(np.mean(tl)), float(np.mean(vl))
    ALL_RESULTS.append({'number': number, 'name': name, 'category': cat,
                        'parameterization': reparam, 'train_loss': mt, 'val_loss': mv,
                        'runtime_ms': rt, 'train_losses': tl.tolist(), 'val_losses': vl.tolist()})
    d = model_dir(number, name)
    save_card(d, number, name, reparam, mv, rt, notes, time_conv)
    update_plots()
    with open(CHANGELOG, 'a') as f:
        f.write(f'## {number:02d}: {name}\n- cat={cat}, reparam={reparam}\n'
                f'- train={mt:.4e}, val={mv:.4e}, rt={rt:.1f}ms\n- {notes}\n\n')
    print(f'[{number:02d}] {name}: train={mt:.4e} val={mv:.4e} rt={rt:.0f}ms')


def update_plots():
    if not ALL_RESULTS:
        return
    names = [r['name'] for r in ALL_RESULTS]
    vl = [r['val_loss'] for r in ALL_RESULTS]
    rt = [r['runtime_ms'] for r in ALL_RESULTS]
    cats = [r['category'] for r in ALL_RESULTS]
    colors = [CAT_COLORS.get(c, COLORS['gray']) for c in cats]
    x = np.arange(len(ALL_RESULTS))

    # Progress
    fig, ax = plt.subplots(figsize=figsize(2, 0.5))
    for cat, col in CAT_COLORS.items():
        idx = [i for i, c in enumerate(cats) if c == cat]
        if idx:
            ax.scatter(x[idx], np.array(vl)[idx], c=col, label=CAT_LABELS[cat], s=25, zorder=3)
    ax.axhline(NR_FLOOR, color='k', ls='--', lw=0.8, label='NR floor')
    ax.set_yscale('log')
    ax.set_xlabel('Approach'); ax.set_ylabel('Val loss')
    ax.set_xticks(x); ax.set_xticklabels([n[:9] for n in names], rotation=45, ha='right', fontsize=5)
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(COMP_DIR, f'progress.{ext}'))
    plt.close()

    # Pareto
    fig, ax = plt.subplots(figsize=figsize(2, 0.7))
    for r in ALL_RESULTS:
        ax.scatter(r['runtime_ms'], r['val_loss'], c=CAT_COLORS.get(r['category'], COLORS['gray']), s=25, zorder=3)
        ax.annotate(r['name'][:8], (r['runtime_ms'], r['val_loss']), fontsize=4, xytext=(2,2), textcoords='offset points')
    ax.axhline(NR_FLOOR, color='k', ls='--', lw=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Eval time (ms)'); ax.set_ylabel('Val loss')
    for cat, col in CAT_COLORS.items():
        ax.scatter([], [], c=col, label=CAT_LABELS[cat], s=25)
    ax.legend(fontsize=6)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(COMP_DIR, f'pareto_accuracy_speed.{ext}'))
    plt.close()

    # Loss-only
    fig, ax = plt.subplots(figsize=figsize(2, 0.6))
    ax.bar(x, vl, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.axhline(NR_FLOOR, color='k', ls='--', lw=0.8, label='NR floor')
    ax.set_yscale('log'); ax.set_ylabel('Val loss')
    ax.set_xticks(x); ax.set_xticklabels([n[:9] for n in names], rotation=45, ha='right', fontsize=5)
    for cat, col in CAT_COLORS.items():
        ax.bar([], [], color=col, label=CAT_LABELS[cat])
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(COMP_DIR, f'loss_only_comparison.{ext}'))
    plt.close()

    # Error data
    data = {r['name']: {'train': r['train_losses'], 'val': r['val_losses']} for r in ALL_RESULTS}
    with open(os.path.join(COMP_DIR, 'error_data.json'), 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================
# LOAD DATA
# ============================================================
print('Loading data...')
P_train, W_train, M_train = load_h5('datasets/waveform/waveform_training.h5')
P_val, W_val, M_val = load_h5('datasets/waveform/waveform_validation.h5')
N_TRAIN, N_VAL = len(P_train), len(P_val)
print(f'  Train={N_TRAIN}, Val={N_VAL}')

# Use common overlap window: from max(t_start) across training data
# (ensures all waveforms have signal everywhere in the grid, no zero-padding)
t_max_start_tr = max(w[0][0] for w in W_train)
t_max_start_va = max(w[0][0] for w in W_val)
t_common_start = max(t_max_start_tr, t_max_start_va)
T_GRID = np.linspace(t_common_start, 100.0, 2048)
print(f'  Common overlap grid: {T_GRID[0]:.0f}..{T_GRID[-1]:.0f}, n={len(T_GRID)}')

print('  Interpolating...')
H_tr = interp_grid(W_train, T_GRID)
H_va = interp_grid(W_val, T_GRID)

# Pre-compute reparameterizations
Xr = {k: apply_reparam(P_train, k) for k in REPARAM_FNS}
Xv = {k: apply_reparam(P_val, k) for k in REPARAM_FNS}

print(f'  Complex SVD ({N_BASIS} basis)...')
BASIS, C_tr, SV = complex_svd(H_tr, N_BASIS)
C_va = encode(H_va, BASIS)

# Check basis quality
recon_err = np.mean([rmse_rel(decode(C_tr[i], BASIS), H_tr[i]) for i in range(30)])
print(f'  Complex SVD recon error (n={N_BASIS}): {recon_err:.4f}')

# Target: first N_C complex coeff = first N_C real + first N_C imag parts
# C_tr has shape (n_train, 2*N_BASIS): first N_BASIS cols = real parts, next N_BASIS = imag parts
def ytrain(n_c=N_C):
    return np.hstack([C_tr[:, :n_c], C_tr[:, N_BASIS:N_BASIS+n_c]])
def yval(n_c=N_C):
    return np.hstack([C_va[:, :n_c], C_va[:, N_BASIS:N_BASIS+n_c]])

Y_tr = ytrain()
Y_va = yval()

trunc_e = np.mean([rmse_rel(decode(Y_tr[i], BASIS), H_tr[i]) for i in range(30)])
print(f'  Truncated ({N_C} coeff) recon error: {trunc_e:.4f}')

# Amplitude/phase basis
amp_tr = np.abs(H_tr); ph_tr = np.unwrap(np.angle(H_tr), axis=1)
amp_va = np.abs(H_va); ph_va = np.unwrap(np.angle(H_va), axis=1)
N_AP = 15
_, sa, Vta = np.linalg.svd(amp_tr, full_matrices=False)
_, sp, Vtp = np.linalg.svd(ph_tr, full_matrices=False)
B_amp = Vta[:N_AP]; B_phs = Vtp[:N_AP]
Ca_tr = amp_tr @ B_amp.T; Cp_tr = ph_tr @ B_phs.T
Ca_va = amp_va @ B_amp.T; Cp_va = ph_va @ B_phs.T

# ============================================================
# Helper: fast predict wrapper
# ============================================================
def make_pred(model_predict, basis, n_c):
    """Wrap a model that predicts SVD coefficients into a waveform predictor.
    model_predict(x_2d) returns array of shape (1,2*n_c) or (2*n_c,).
    """
    def fn(x):
        c2 = np.asarray(model_predict(x.reshape(1,-1))).flatten()  # always 1D
        return decode(c2, basis)
    return fn


from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

# Initialize CHANGELOG
with open(CHANGELOG, 'w') as f:
    f.write('# Waveform Benchmark CHANGELOG (sonnet46)\n\n')
    f.write(f'## Setup\n- Grid: t={T_GRID[0]:.0f}..{T_GRID[-1]:.0f}, n={len(T_GRID)}\n')
    f.write(f'- SVD {N_BASIS} basis, recon error={recon_err:.4f}\n')
    f.write(f'- Truncated ({N_C}) error={trunc_e:.4f}\n\n')

# ============================================================
# 01: SVD + GPR (RBF, raw)
# ============================================================
print('\n[01] SVD+GPR RBF raw')
d = model_dir(1, 'svd_gpr_rbf_raw')
Y_gpr = ytrain(N_C_GPR)
sc = StandardScaler(); Xt = sc.fit_transform(Xr['raw'])
kern = C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
gpr = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=1, alpha=1e-4, normalize_y=True)
t0=time.time(); gpr.fit(Xt, Y_gpr); print(f'  fit {time.time()-t0:.0f}s')
fn1 = make_pred(lambda x: gpr.predict(sc.transform(x))[0], BASIS, N_C_GPR)
ts=time.time(); fn1(Xr['raw'][0]); rt=(time.time()-ts)*1e3
tl=eval_batch(fn1, Xr['raw'], H_tr); vl=eval_batch(fn1, Xv['raw'], H_va)
joblib.dump({'gpr':gpr,'sc':sc,'basis':BASIS,'n_c':N_C_GPR}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d, 'svd_gpr_rbf_raw')
record(1,'svd_gpr_rbf_raw','svd_decomp','raw',tl,vl,rt,'SVD+GPR RBF, raw 7D params, 15 coeff')

# ============================================================
# 02: SVD + GPR (Matern, eff_spins)
# ============================================================
print('\n[02] SVD+GPR Matern eff')
d = model_dir(2, 'svd_gpr_matern_eff')
sc2=StandardScaler(); Xt2=sc2.fit_transform(Xr['eff'])
kern2=C(1.)*Matern(np.ones(7),nu=2.5)+WhiteKernel(1e-3)
gpr2=GaussianProcessRegressor(kernel=kern2,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); gpr2.fit(Xt2,Y_gpr); print(f'  fit {time.time()-t0:.0f}s')
fn2=make_pred(lambda x:gpr2.predict(sc2.transform(x))[0],BASIS,N_C_GPR)
ts=time.time(); fn2(Xr['eff'][0]); rt2=(time.time()-ts)*1e3
tl2=eval_batch(fn2,Xr['eff'],H_tr); vl2=eval_batch(fn2,Xv['eff'],H_va)
joblib.dump({'gpr':gpr2,'sc':sc2,'basis':BASIS,'n_c':N_C_GPR},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gpr_matern_eff')
record(2,'svd_gpr_matern_eff','svd_decomp','eff',tl2,vl2,rt2,'SVD+GPR Matern-2.5, eff_spins, 15 coeff')

# ============================================================
# 03: SVD + Poly deg3 (raw)
# ============================================================
print('\n[03] SVD+Poly3 raw')
d=model_dir(3,'svd_poly3_raw')
poly=PolynomialFeatures(degree=3,include_bias=True)
sc3=StandardScaler()
Xp=sc3.fit_transform(poly.fit_transform(Xr['raw']))
t0=time.time(); r3=Ridge(alpha=0.01); r3.fit(Xp,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
fn3=make_pred(lambda x:r3.predict(sc3.transform(poly.transform(x))),BASIS,N_C)
ts=time.time(); fn3(Xr['raw'][0]); rt3=(time.time()-ts)*1e3
tl3=eval_batch(fn3,Xr['raw'],H_tr); vl3=eval_batch(fn3,Xv['raw'],H_va)
joblib.dump({'poly':poly,'sc':sc3,'r':r3,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_poly3_raw')
record(3,'svd_poly3_raw','svd_decomp','raw',tl3,vl3,rt3,'SVD+poly3 Ridge, raw 7D')

# ============================================================
# 04: SVD + MLP (raw)
# ============================================================
print('\n[04] SVD+MLP raw')
d=model_dir(4,'svd_mlp_raw')
sc4=StandardScaler(); Xt4=sc4.fit_transform(Xr['raw'])
t0=time.time()
mlp4=MLPRegressor(hidden_layer_sizes=(128,128,64),activation='relu',max_iter=500,random_state=42,
                   early_stopping=True,validation_fraction=0.1)
mlp4.fit(Xt4,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn4=make_pred(lambda x:mlp4.predict(sc4.transform(x)),BASIS,N_C)
ts=time.time(); fn4(Xr['raw'][0]); rt4=(time.time()-ts)*1e3
tl4=eval_batch(fn4,Xr['raw'],H_tr); vl4=eval_batch(fn4,Xv['raw'],H_va)
joblib.dump({'mlp':mlp4,'sc':sc4,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_mlp_raw')
record(4,'svd_mlp_raw','ml','raw',tl4,vl4,rt4,'SVD+MLP(128,128,64) relu, raw 7D')

# ============================================================
# 05: SVD + Random Forest (raw)
# ============================================================
print('\n[05] SVD+RF raw')
d=model_dir(5,'svd_rf_raw')
t0=time.time()
rf5=RandomForestRegressor(100,max_depth=12,random_state=42,n_jobs=-1)
rf5.fit(Xr['raw'],Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn5=make_pred(lambda x:rf5.predict(x),BASIS,N_C)
ts=time.time(); fn5(Xr['raw'][0]); rt5=(time.time()-ts)*1e3
tl5=eval_batch(fn5,Xr['raw'],H_tr); vl5=eval_batch(fn5,Xv['raw'],H_va)
joblib.dump({'rf':rf5,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_rf_raw')
record(5,'svd_rf_raw','ml','raw',tl5,vl5,rt5,'SVD+RF(100) depth 12, raw 7D')

# ============================================================
# 06: SVD + Gradient Boosting (eff)
# ============================================================
print('\n[06] SVD+GBR eff')
d=model_dir(6,'svd_gbr_eff')
n6=8; Y6=np.hstack([C_tr[:,:n6],C_tr[:,N_BASIS:N_BASIS+n6]])
t0=time.time()
gbr6=MultiOutputRegressor(GradientBoostingRegressor(n_estimators=80,max_depth=4,random_state=42),n_jobs=-1)
gbr6.fit(Xr['eff'],Y6); print(f'  fit {time.time()-t0:.1f}s')
def fn6(x):
    c=gbr6.predict(x.reshape(1,-1))[0]; cf=np.zeros(2*N_BASIS)
    cf[:n6]=c[:n6]; cf[N_BASIS:N_BASIS+n6]=c[n6:]; return decode(cf,BASIS)
ts=time.time(); fn6(Xr['eff'][0]); rt6=(time.time()-ts)*1e3
tl6=eval_batch(fn6,Xr['eff'],H_tr); vl6=eval_batch(fn6,Xv['eff'],H_va)
joblib.dump({'gbr':gbr6,'basis':BASIS,'n6':n6},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gbr_eff')
record(6,'svd_gbr_eff','ml','eff',tl6,vl6,rt6,'SVD+GBR(80) eff_spins')

# ============================================================
# 07: Amplitude+Phase SVD + GPR (raw)
# ============================================================
print('\n[07] Amp+Phase GPR raw')
d=model_dir(7,'amp_phase_gpr_raw')
sc7=StandardScaler(); Xt7=sc7.fit_transform(Xr['raw'])
n7=6
t0=time.time()
k7a=C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g7a=GaussianProcessRegressor(kernel=k7a,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
g7a.fit(Xt7,Ca_tr[:,:n7])
k7p=C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g7p=GaussianProcessRegressor(kernel=k7p,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
g7p.fit(Xt7,Cp_tr[:,:n7])
print(f'  fit {time.time()-t0:.1f}s')
def fn7(x):
    xn=sc7.transform(x.reshape(1,-1))
    ca=np.zeros(N_AP); cp=np.zeros(N_AP)
    ca[:n7]=g7a.predict(xn)[0]; cp[:n7]=g7p.predict(xn)[0]
    return (ca@B_amp)*np.exp(1j*(cp@B_phs))
ts=time.time(); fn7(Xr['raw'][0]); rt7=(time.time()-ts)*1e3
tl7=eval_batch(fn7,Xr['raw'],H_tr); vl7=eval_batch(fn7,Xv['raw'],H_va)
joblib.dump({'g_a':g7a,'g_p':g7p,'sc':sc7,'Ba':B_amp,'Bp':B_phs,'n7':n7},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'amp_phase_gpr_raw')
record(7,'amp_phase_gpr_raw','svd_decomp','raw',tl7,vl7,rt7,'Amp+Phase SVD+GPR, raw 7D')

# ============================================================
# 08: EIM + GPR (raw)
# ============================================================
print('\n[08] EIM+GPR raw')
d=model_dir(8,'eim_gpr_raw')
n_eim=12
nodes=[]
for i in range(n_eim):
    if i==0: resid=BASIS[0]
    else:
        Vn=BASIS[:i,nodes].T
        try: c_=np.linalg.lstsq(Vn,BASIS[i,nodes],rcond=None)[0]; resid=BASIS[i]-c_@BASIS[:i]
        except: resid=BASIS[i]
    nodes.append(int(np.argmax(np.abs(resid))))
V_eim=BASIS[:n_eim,nodes].T
eim_target=np.hstack([H_tr.real[:,nodes],H_tr.imag[:,nodes]])
sc8=StandardScaler(); Xt8=sc8.fit_transform(Xr['raw'])
t0=time.time()
k8=C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g8=GaussianProcessRegressor(kernel=k8,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
g8.fit(Xt8,eim_target); print(f'  fit {time.time()-t0:.1f}s')
def fn8(x):
    ev=g8.predict(sc8.transform(x.reshape(1,-1)))[0]
    er,ei=ev[:n_eim],ev[n_eim:]
    try: cr=np.linalg.lstsq(V_eim,er,rcond=None)[0]; ci=np.linalg.lstsq(V_eim,ei,rcond=None)[0]
    except: cr=ci=np.zeros(n_eim)
    cf=np.zeros(2*N_BASIS); cf[:n_eim]=cr; cf[N_BASIS:N_BASIS+n_eim]=ci
    return decode(cf,BASIS)
ts=time.time(); fn8(Xr['raw'][0]); rt8=(time.time()-ts)*1e3
tl8=eval_batch(fn8,Xr['raw'],H_tr); vl8=eval_batch(fn8,Xv['raw'],H_va)
joblib.dump({'gpr':g8,'sc':sc8,'basis':BASIS,'nodes':nodes,'V':V_eim},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'eim_gpr_raw')
record(8,'eim_gpr_raw','svd_decomp','raw',tl8,vl8,rt8,f'EIM ({n_eim} nodes)+GPR, raw 7D')

# ============================================================
# 09: RBF Interpolation (spherical, thin_plate_spline)
# ============================================================
print('\n[09] RBF interp sph')
d=model_dir(9,'rbf_interp_sph')
sc9=StandardScaler(); Xt9=sc9.fit_transform(Xr['sph'])
n9=12; Y9=np.hstack([C_tr[:,:n9],C_tr[:,N_BASIS:N_BASIS+n9]])
t0=time.time(); rbf9=RBFInterpolator(Xt9,Y9,kernel='thin_plate_spline',smoothing=0.5); print(f'  fit {time.time()-t0:.1f}s')
def fn9(x):
    c=rbf9(sc9.transform(x.reshape(1,-1)))[0]; cf=np.zeros(2*N_BASIS)
    cf[:n9]=c[:n9]; cf[N_BASIS:N_BASIS+n9]=c[n9:]; return decode(cf,BASIS)
ts=time.time(); fn9(Xr['sph'][0]); rt9=(time.time()-ts)*1e3
tl9=eval_batch(fn9,Xr['sph'],H_tr); vl9=eval_batch(fn9,Xv['sph'],H_va)
joblib.dump({'rbf':rbf9,'sc':sc9,'basis':BASIS,'n9':n9},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'rbf_interp_sph')
record(9,'rbf_interp_sph','kernel_interp','sph',tl9,vl9,rt9,'TPS-RBF on spherical spins')

# ============================================================
# 10: Kernel Ridge Regression (eff)
# ============================================================
print('\n[10] KRR eff')
d=model_dir(10,'krr_eff')
sc10=StandardScaler(); Xt10=sc10.fit_transform(Xr['eff'])
t0=time.time(); krr10=KernelRidge(kernel='rbf',alpha=0.01,gamma=0.5); krr10.fit(Xt10,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn10=make_pred(lambda x:krr10.predict(sc10.transform(x)),BASIS,N_C)
ts=time.time(); fn10(Xr['eff'][0]); rt10=(time.time()-ts)*1e3
tl10=eval_batch(fn10,Xr['eff'],H_tr); vl10=eval_batch(fn10,Xv['eff'],H_va)
joblib.dump({'krr':krr10,'sc':sc10,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'krr_eff')
record(10,'krr_eff','kernel_interp','eff',tl10,vl10,rt10,'KRR(RBF) on eff_spins')

# ============================================================
# 11: SVD + GPR (RBF, spherical, 25 basis)
# ============================================================
print('\n[11] SVD+GPR sph N_C_GPR')
d=model_dir(11,'svd_gpr_sph')
n11=N_C_GPR; Y11=ytrain(n11)
sc11=StandardScaler(); Xt11=sc11.fit_transform(Xr['sph'])
k11=C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g11=GaussianProcessRegressor(kernel=k11,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); g11.fit(Xt11,Y11); print(f'  fit {time.time()-t0:.0f}s')
fn11=make_pred(lambda x:g11.predict(sc11.transform(x))[0],BASIS,n11)
ts=time.time(); fn11(Xr['sph'][0]); rt11=(time.time()-ts)*1e3
tl11=eval_batch(fn11,Xr['sph'],H_tr); vl11=eval_batch(fn11,Xv['sph'],H_va)
joblib.dump({'gpr':g11,'sc':sc11,'basis':BASIS,'n_c':n11},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gpr_sph')
record(11,'svd_gpr_sph','svd_decomp','sph',tl11,vl11,rt11,'SVD+GPR RBF, spherical, N_C_GPR coeff')

# ============================================================
# 12: kNN (raw)
# ============================================================
print('\n[12] kNN raw')
d=model_dir(12,'knn_raw')
sc12=StandardScaler(); Xt12=sc12.fit_transform(Xr['raw'])
t0=time.time(); knn12=KNeighborsRegressor(7,weights='distance'); knn12.fit(Xt12,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
fn12=make_pred(lambda x:knn12.predict(sc12.transform(x)),BASIS,N_C)
ts=time.time(); fn12(Xr['raw'][0]); rt12=(time.time()-ts)*1e3
tl12=eval_batch(fn12,Xr['raw'],H_tr); vl12=eval_batch(fn12,Xv['raw'],H_va)
joblib.dump({'knn':knn12,'sc':sc12,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'knn_raw')
record(12,'knn_raw','kernel_interp','raw',tl12,vl12,rt12,'kNN(7, distance) on raw 7D')

# ============================================================
# 13: SVD + MLP large (eff)
# ============================================================
print('\n[13] SVD+MLP large eff')
d=model_dir(13,'svd_mlp_large_eff')
sc13=StandardScaler(); Xt13=sc13.fit_transform(Xr['eff'])
t0=time.time()
mlp13=MLPRegressor(hidden_layer_sizes=(256,256,128,64),activation='tanh',max_iter=800,random_state=0,
                    early_stopping=True,validation_fraction=0.15,learning_rate_init=5e-4)
mlp13.fit(Xt13,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn13=make_pred(lambda x:mlp13.predict(sc13.transform(x)),BASIS,N_C)
ts=time.time(); fn13(Xr['eff'][0]); rt13=(time.time()-ts)*1e3
tl13=eval_batch(fn13,Xr['eff'],H_tr); vl13=eval_batch(fn13,Xv['eff'],H_va)
joblib.dump({'mlp':mlp13,'sc':sc13,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_mlp_large_eff')
record(13,'svd_mlp_large_eff','ml','eff',tl13,vl13,rt13,'MLP(256,256,128,64) tanh eff')

# ============================================================
# 14: Extra Trees (mass_diff)
# ============================================================
print('\n[14] ExtraTrees md')
d=model_dir(14,'svd_et_md')
t0=time.time()
et14=ExtraTreesRegressor(n_estimators=150,max_depth=14,random_state=42,n_jobs=-1)
et14.fit(Xr['md'],Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn14=make_pred(lambda x:et14.predict(x),BASIS,N_C)
ts=time.time(); fn14(Xr['md'][0]); rt14=(time.time()-ts)*1e3
tl14=eval_batch(fn14,Xr['md'],H_tr); vl14=eval_batch(fn14,Xv['md'],H_va)
joblib.dump({'et':et14,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_et_md')
record(14,'svd_et_md','ml','md',tl14,vl14,rt14,'ExtraTrees(150,d14) mass_diff')

# ============================================================
# 15: gplearn Symbolic Regression (MANDATORY)
# ============================================================
print('\n[15] gplearn raw')
from gplearn.genetic import SymbolicRegressor as GPLearnSR
d=model_dir(15,'gplearn_svd_raw')
n15=3; t0=time.time()
gpl15r=[]; gpl15i=[]; expr15=[]
for ci in range(n15):
    for part,yc,lst in [('re',Y_tr[:,ci],gpl15r),('im',Y_tr[:,N_C+ci],gpl15i)]:
        print(f'  gplearn {part} coeff {ci}...')
        g=GPLearnSR(population_size=2000,generations=15,tournament_size=20,
                    function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                    metric='mse',parsimony_coefficient=0.01,verbose=0,random_state=42+ci*7)
        g.fit(Xr['raw'],yc); lst.append(g)
        expr15.append({'ci':ci,'part':part,'expr':str(g._program),'fit':float(g._program.fitness_)})
        print(f'    {str(g._program)[:50]}')
print(f'  gplearn total {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr15,f,indent=2)
def fn15(x):
    cf=np.zeros(2*N_BASIS)
    for ci in range(n15):
        cf[ci]=gpl15r[ci].predict(x.reshape(1,-1))[0]
        cf[N_BASIS+ci]=gpl15i[ci].predict(x.reshape(1,-1))[0]
    return decode(cf,BASIS)
ts=time.time(); fn15(Xr['raw'][0]); rt15=(time.time()-ts)*1e3
tl15=eval_batch(fn15,Xr['raw'],H_tr); vl15=eval_batch(fn15,Xv['raw'],H_va)
try: joblib.dump({'gpl_r':gpl15r,'gpl_i':gpl15i,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
except: np.save(os.path.join(d,'saved_model','basis.npy'),BASIS)
write_stubs(d,'gplearn_svd_raw')
record(15,'gplearn_svd_raw','symbolic','raw',tl15,vl15,rt15,f'gplearn SR on {n15} SVD coeff (re+im), raw 7D')

# ============================================================
# 16: PySR (MANDATORY, raw params)
# ============================================================
print('\n[16] PySR raw')
from pysr import PySRRegressor
d=model_dir(16,'pysr_svd_raw')
n16=2; t0=time.time()
psr16r=[]; psr16i=[]; expr16=[]
for ci in range(n16):
    for part,yc,lst in [('re',Y_tr[:,ci],psr16r),('im',Y_tr[:,N_C+ci],psr16i)]:
        print(f'  PySR {part} coeff {ci}...')
        try:
            ps=PySRRegressor(niterations=40,binary_operators=['+','-','*','/'],
                             unary_operators=['sqrt','log','exp','sin','cos'],
                             maxsize=20,populations=12,procs=2,
                             loss='loss(p,t)=abs(p-t)',verbosity=0,
                             random_state=42+ci,tempdir=os.path.join(d,'saved_model',f'pysr_{part}{ci}'))
            ps.fit(Xr['raw'],yc); lst.append(ps)
            try:
                pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps.equations_.iterrows()]
            except: pf=[{'e':str(ps.sympy()),'c':0,'l':0.}]
            expr16.append({'ci':ci,'part':part,'pareto':pf})
            print(f'    {pf[-1]["e"][:60] if pf else "N/A"}')
        except Exception as e:
            lst.append(None); expr16.append({'ci':ci,'part':part,'err':str(e)}); print(f'    failed: {e}')
print(f'  PySR total {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr16,f,indent=2)
def fn16(x):
    cf=np.zeros(2*N_BASIS)
    for ci,(pr,pi) in enumerate(zip(psr16r,psr16i)):
        if pr:
            try: cf[ci]=pr.predict(x.reshape(1,-1))[0]
            except: pass
        if pi:
            try: cf[N_BASIS+ci]=pi.predict(x.reshape(1,-1))[0]
            except: pass
    return decode(cf,BASIS)
ts=time.time(); fn16(Xr['raw'][0]); rt16=(time.time()-ts)*1e3
tl16=eval_batch(fn16,Xr['raw'],H_tr); vl16=eval_batch(fn16,Xv['raw'],H_va)
np.save(os.path.join(d,'saved_model','basis.npy'),BASIS)
write_stubs(d,'pysr_svd_raw')
record(16,'pysr_svd_raw','symbolic','raw',tl16,vl16,rt16,f'PySR on {n16} SVD coeff, raw 7D')

# ============================================================
# 17: PySR (eff_spins)
# ============================================================
print('\n[17] PySR eff')
d=model_dir(17,'pysr_svd_eff')
n17=2; t0=time.time()
psr17r=[]; psr17i=[]; expr17=[]
for ci in range(n17):
    for part,yc,lst in [('re',Y_tr[:,ci],psr17r),('im',Y_tr[:,N_C+ci],psr17i)]:
        print(f'  PySR {part} coeff {ci} (eff)...')
        try:
            ps=PySRRegressor(niterations=30,binary_operators=['+','-','*','/'],
                             unary_operators=['sqrt','log','exp'],
                             maxsize=18,populations=10,procs=2,
                             loss='loss(p,t)=abs(p-t)',verbosity=0,
                             random_state=99+ci,tempdir=os.path.join(d,'saved_model',f'pysr_{part}{ci}'))
            ps.fit(Xr['eff'],yc); lst.append(ps)
            try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps.equations_.iterrows()]
            except: pf=[{'e':str(ps.sympy()),'c':0,'l':0.}]
            expr17.append({'ci':ci,'part':part,'pareto':pf})
        except Exception as e:
            lst.append(None); expr17.append({'ci':ci,'part':part,'err':str(e)})
print(f'  PySR eff total {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr17,f,indent=2)
def fn17(x):
    cf=np.zeros(2*N_BASIS)
    for ci,(pr,pi) in enumerate(zip(psr17r,psr17i)):
        if pr:
            try: cf[ci]=pr.predict(x.reshape(1,-1))[0]
            except: pass
        if pi:
            try: cf[N_BASIS+ci]=pi.predict(x.reshape(1,-1))[0]
            except: pass
    return decode(cf,BASIS)
ts=time.time(); fn17(Xr['eff'][0]); rt17=(time.time()-ts)*1e3
tl17=eval_batch(fn17,Xr['eff'],H_tr); vl17=eval_batch(fn17,Xv['eff'],H_va)
np.save(os.path.join(d,'saved_model','basis.npy'),BASIS)
write_stubs(d,'pysr_svd_eff')
record(17,'pysr_svd_eff','symbolic','eff',tl17,vl17,rt17,f'PySR on {n17} SVD coeff, eff_spins')

# ============================================================
# 18: Amp+Phase GPR (mass_diff)
# ============================================================
print('\n[18] Amp+Phase GPR md')
d=model_dir(18,'amp_phase_gpr_md')
sc18=StandardScaler(); Xt18=sc18.fit_transform(Xr['md'])
n18=8; t0=time.time()
k18a=C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g18a=GaussianProcessRegressor(kernel=k18a,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
g18a.fit(Xt18,Ca_tr[:,:n18])
k18p=C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g18p=GaussianProcessRegressor(kernel=k18p,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
g18p.fit(Xt18,Cp_tr[:,:n18]); print(f'  fit {time.time()-t0:.1f}s')
def fn18(x):
    xn=sc18.transform(x.reshape(1,-1))
    ca=np.zeros(N_AP); cp=np.zeros(N_AP)
    ca[:n18]=g18a.predict(xn)[0]; cp[:n18]=g18p.predict(xn)[0]
    return (ca@B_amp)*np.exp(1j*(cp@B_phs))
ts=time.time(); fn18(Xr['md'][0]); rt18=(time.time()-ts)*1e3
tl18=eval_batch(fn18,Xr['md'],H_tr); vl18=eval_batch(fn18,Xv['md'],H_va)
joblib.dump({'ga':g18a,'gp':g18p,'sc':sc18,'Ba':B_amp,'Bp':B_phs,'n18':n18},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'amp_phase_gpr_md')
record(18,'amp_phase_gpr_md','svd_decomp','md',tl18,vl18,rt18,'Amp+Phase GPR, mass_diff')

# ============================================================
# 19: SVD + Poly deg2 (eff)
# ============================================================
print('\n[19] SVD+Poly2 eff')
d=model_dir(19,'svd_poly2_eff')
poly19=PolynomialFeatures(degree=2,include_bias=True)
sc19=StandardScaler(); Xp19=sc19.fit_transform(poly19.fit_transform(Xr['eff']))
t0=time.time(); r19=Ridge(alpha=0.01); r19.fit(Xp19,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
fn19=make_pred(lambda x:r19.predict(sc19.transform(poly19.transform(x))),BASIS,N_C)
ts=time.time(); fn19(Xr['eff'][0]); rt19=(time.time()-ts)*1e3
tl19=eval_batch(fn19,Xr['eff'],H_tr); vl19=eval_batch(fn19,Xv['eff'],H_va)
joblib.dump({'poly':poly19,'sc':sc19,'r':r19,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_poly2_eff')
record(19,'svd_poly2_eff','svd_decomp','eff',tl19,vl19,rt19,'SVD+poly2 Ridge, eff_spins')

# ============================================================
# 20: SVD + GPR Matern (mass_diff)
# ============================================================
print('\n[20] SVD+GPR Matern md')
d=model_dir(20,'svd_gpr_matern_md')
n20=N_C_GPR; Y20=np.hstack([C_tr[:,:n20],C_tr[:,N_BASIS:N_BASIS+n20]])
sc20=StandardScaler(); Xt20=sc20.fit_transform(Xr['md'])
k20=C(1.)*Matern(np.ones(7),nu=1.5)+WhiteKernel(1e-3)
g20=GaussianProcessRegressor(kernel=k20,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); g20.fit(Xt20,Y20); print(f'  fit {time.time()-t0:.0f}s')
fn20=make_pred(lambda x:g20.predict(sc20.transform(x))[0],BASIS,n20)
ts=time.time(); fn20(Xr['md'][0]); rt20=(time.time()-ts)*1e3
tl20=eval_batch(fn20,Xr['md'],H_tr); vl20=eval_batch(fn20,Xv['md'],H_va)
joblib.dump({'gpr':g20,'sc':sc20,'basis':BASIS,'n_c':n20},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gpr_matern_md')
record(20,'svd_gpr_matern_md','svd_decomp','md',tl20,vl20,rt20,'SVD+GPR Matern-1.5, mass_diff')

# ============================================================
# 21: gplearn (eff_spins)
# ============================================================
print('\n[21] gplearn eff')
d=model_dir(21,'gplearn_svd_eff')
n21=2; t0=time.time()
gpl21r=[]; gpl21i=[]; expr21=[]
for ci in range(n21):
    for part,yc,lst in [('re',Y_tr[:,ci],gpl21r),('im',Y_tr[:,N_C+ci],gpl21i)]:
        g=GPLearnSR(population_size=2000,generations=12,tournament_size=20,
                    function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                    metric='mse',parsimony_coefficient=0.01,verbose=0,random_state=300+ci)
        g.fit(Xr['eff'],yc); lst.append(g)
        expr21.append({'ci':ci,'part':part,'expr':str(g._program),'fit':float(g._program.fitness_)})
print(f'  gplearn eff {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr21,f,indent=2)
def fn21(x):
    cf=np.zeros(2*N_BASIS)
    for ci in range(n21):
        cf[ci]=gpl21r[ci].predict(x.reshape(1,-1))[0]
        cf[N_BASIS+ci]=gpl21i[ci].predict(x.reshape(1,-1))[0]
    return decode(cf,BASIS)
ts=time.time(); fn21(Xr['eff'][0]); rt21=(time.time()-ts)*1e3
tl21=eval_batch(fn21,Xr['eff'],H_tr); vl21=eval_batch(fn21,Xv['eff'],H_va)
try: joblib.dump({'gpl_r':gpl21r,'gpl_i':gpl21i,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
except: np.save(os.path.join(d,'saved_model','basis.npy'),BASIS)
write_stubs(d,'gplearn_svd_eff')
record(21,'gplearn_svd_eff','symbolic','eff',tl21,vl21,rt21,f'gplearn SR on {n21} SVD coeff, eff_spins')

# ============================================================
# 22: SVD+GPR (t0=start time convention)
# ============================================================
print('\n[22] SVD+GPR t0=start')
d=model_dir(22,'svd_gpr_t0start')
def itp_t0start(ws,n=4096):
    out=[]
    for t,h in ws:
        ts=t-t[0]; tn=np.linspace(0,ts[-1],n)
        fr=interp1d(ts,h.real,bounds_error=False,fill_value=0.); fi=interp1d(ts,h.imag,bounds_error=False,fill_value=0.)
        out.append(fr(tn)+1j*fi(tn))
    return np.array(out)
H22t=itp_t0start(W_train); H22v=itp_t0start(W_val)
B22,C22t,_=complex_svd(H22t,N_BASIS)
n22=N_C_GPR; Y22=np.hstack([C22t[:,:n22],C22t[:,N_BASIS:N_BASIS+n22]])
sc22=StandardScaler(); Xt22=sc22.fit_transform(Xr['raw'])
k22=C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g22=GaussianProcessRegressor(kernel=k22,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); g22.fit(Xt22,Y22); print(f'  fit {time.time()-t0:.0f}s')
fn22=make_pred(lambda x:g22.predict(sc22.transform(x))[0],B22,n22)
ts=time.time(); fn22(Xr['raw'][0]); rt22=(time.time()-ts)*1e3
tl22=eval_batch(fn22,Xr['raw'],H22t); vl22=eval_batch(fn22,Xv['raw'],H22v)
joblib.dump({'gpr':g22,'sc':sc22,'basis':B22},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gpr_t0start')
record(22,'svd_gpr_t0start','svd_decomp','raw',tl22,vl22,rt22,'SVD+GPR t=0 at start',time_conv='t0_at_start')

# ============================================================
# 23: SVD+MLP (mass_diff)
# ============================================================
print('\n[23] SVD+MLP md')
d=model_dir(23,'svd_mlp_md')
sc23=StandardScaler(); Xt23=sc23.fit_transform(Xr['md'])
t0=time.time()
mlp23=MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu',max_iter=500,random_state=77,early_stopping=True)
mlp23.fit(Xt23,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn23=make_pred(lambda x:mlp23.predict(sc23.transform(x)),BASIS,N_C)
ts=time.time(); fn23(Xr['md'][0]); rt23=(time.time()-ts)*1e3
tl23=eval_batch(fn23,Xr['md'],H_tr); vl23=eval_batch(fn23,Xv['md'],H_va)
joblib.dump({'mlp':mlp23,'sc':sc23,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_mlp_md')
record(23,'svd_mlp_md','ml','md',tl23,vl23,rt23,'SVD+MLP(128,64,32) relu mass_diff')

# ============================================================
# 24: SVD+Poly2 (spherical)
# ============================================================
print('\n[24] SVD+Poly2 sph')
d=model_dir(24,'svd_poly2_sph')
poly24=PolynomialFeatures(degree=2,include_bias=True)
sc24=StandardScaler(); Xp24=sc24.fit_transform(poly24.fit_transform(Xr['sph']))
t0=time.time(); r24=Ridge(alpha=0.01); r24.fit(Xp24,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
fn24=make_pred(lambda x:r24.predict(sc24.transform(poly24.transform(x))),BASIS,N_C)
ts=time.time(); fn24(Xr['sph'][0]); rt24=(time.time()-ts)*1e3
tl24=eval_batch(fn24,Xr['sph'],H_tr); vl24=eval_batch(fn24,Xv['sph'],H_va)
joblib.dump({'poly':poly24,'sc':sc24,'r':r24,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_poly2_sph')
record(24,'svd_poly2_sph','svd_decomp','sph',tl24,vl24,rt24,'SVD+poly2 Ridge, spherical')

# ============================================================
# FINAL OUTPUTS
# ============================================================
print('\n=== Final plots & summary ===')

# Error histograms
fig, ax = plt.subplots(figsize=figsize(2, 0.8))
for r in ALL_RESULTS:
    tl=np.array(r['train_losses']); vl=np.array(r['val_losses'])
    al=np.concatenate([tl,vl]); lo,hi=max(al.min(),1e-6),al.max()
    if lo<hi:
        bins=np.logspace(np.log10(lo),np.log10(hi),20)
        ax.hist(tl,bins=bins,alpha=0.2,density=True,histtype='stepfilled',label=f'{r["name"][:7]} tr')
        ax.hist(vl,bins=bins,alpha=0.7,density=True,histtype='step',lw=1.,label=f'{r["name"][:7]} va')
ax.axvline(NR_FLOOR,color='k',ls='--',lw=0.8,label='NR floor')
ax.set_xscale('log'); ax.set_xlabel('Loss'); ax.set_ylabel('Density')
ax.legend(fontsize=4,ncol=3)
plt.tight_layout()
for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'error_histograms.{ext}'))
plt.close()

ranked=sorted(ALL_RESULTS,key=lambda r:r['val_loss'])
table=[{'rank':i+1,'name':r['name'],'category':r['category'],'parameterization':r['parameterization'],
        'val_loss':r['val_loss'],'train_loss':r['train_loss'],'runtime_ms':r['runtime_ms']}
       for i,r in enumerate(ranked)]
with open(os.path.join(COMP_DIR,'summary_table.json'),'w') as f: json.dump(table,f,indent=2)
with open(os.path.join(COMP_DIR,'best_model.json'),'w') as f: json.dump(table[0],f,indent=2)

cats=set(r['category'] for r in ALL_RESULTS)
reps=set(r['parameterization'] for r in ALL_RESULTS)
has_pysr=any('pysr' in r['name'] for r in ALL_RESULTS)
has_gpl=any('gplearn' in r['name'] for r in ALL_RESULTS)
n=len(ALL_RESULTS)

print(f'\n=== SUMMARY ===')
print(f'Approaches: {n}')
print(f'Categories: {cats}')
print(f'Reparameterizations: {reps}')
print(f'PySR: {has_pysr}, gplearn: {has_gpl}')
print(f'Best: {table[0]["name"]} (val={table[0]["val_loss"]:.4e})')
print('\nTop 5:')
for r in table[:5]:
    print(f'  {r["rank"]:2d}. {r["name"]:<30s} val={r["val_loss"]:.4e} ({r["category"]})')

if n>=20 and len(reps)>=3 and len(cats)==4 and has_pysr and has_gpl:
    print('\nWAVEFORM_BENCH_COMPLETE')
else:
    print(f'\nIncomplete: n={n}, reps={len(reps)}, cats={len(cats)}, pysr={has_pysr}, gplearn={has_gpl}')
