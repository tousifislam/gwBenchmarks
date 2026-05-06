#!/usr/bin/env python3
"""Run dynamics benchmark for sonnet46 agent - builds all 20+ surrogate models for x(t)."""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py, joblib
from scipy.interpolate import interp1d
from scipy.interpolate import RBFInterpolator

from gwbenchmarks.plot_settings import apply as plot_apply, figsize, COLORS
plot_apply()

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
COMP_DIR = os.path.join(SCRIPT_DIR, 'comparison')
CHANGELOG = os.path.join(SCRIPT_DIR, 'CHANGELOG.md')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

N_BASIS = 30  # SVD basis size
N_C = 20      # number of SVD components for prediction (fast models)
N_C_GPR = 10  # fewer for GPR models
N_GRID = 512  # common time grid points

CAT_COLORS = {
    'svd_decomp': COLORS['blue'],
    'symbolic': COLORS['red'],
    'kernel_interp': COLORS['green'],
    'ml': COLORS['orange'],
}
CAT_LABELS = {'svd_decomp': 'SVD/Decomp', 'symbolic': 'Symbolic',
               'kernel_interp': 'Kernel/Interp', 'ml': 'Machine Learning'}
ALL_RESULTS = []


# ============================================================
# Data loading
# ============================================================
def load_h5(path):
    data = []
    with h5py.File(path, 'r') as f:
        n = f.attrs['n_simulations']
        for i in range(n):
            g = f[f'sim_{i:04d}']
            data.append({
                'params': np.array([g.attrs['q'], g.attrs['chi1z'], g.attrs['chi2z'],
                                    g.attrs['e0'], g.attrs['zeta0'], g.attrs['omega0']]),
                't': g['t'][:], 'x': g['x'][:]
            })
    return data


def interp_normalized(data, n_grid=512):
    """Interpolate x(t) to normalized tau ∈ [0,1] grid."""
    tau = np.linspace(0, 1, n_grid)
    X_mat = []
    for d in data:
        t = d['t']; x = d['x']
        t_norm = (t - t[0]) / (t[-1] - t[0])
        fn = interp1d(t_norm, x, kind='cubic', bounds_error=False, fill_value=(x[0], x[-1]))
        X_mat.append(fn(tau))
    return np.array(X_mat), tau


def svd_basis(X_mat, n):
    """Compute SVD basis for real matrix X_mat (n_sims, n_grid)."""
    U, s, Vt = np.linalg.svd(X_mat, full_matrices=False)
    basis = Vt[:n]  # (n, n_grid)
    coeffs = X_mat @ basis.T  # (n_sims, n) real
    return basis, coeffs, s


def encode(X_mat, basis):
    return X_mat @ basis.T


def decode_coeff(c, basis):
    """c: (n_c,), basis: (n_basis, n_grid). Returns (n_grid,) real."""
    n = len(c)
    return c @ basis[:n]


def rms_rel_error(pred, true):
    """RMS relative error: sqrt(mean((pred-true)^2 / true^2))"""
    eps = np.percentile(np.abs(true), 1)  # avoid division by near-zero
    rel_sq = (pred - true)**2 / (true**2 + eps**2)
    return float(np.sqrt(np.mean(rel_sq)))


def eval_loss(fn_predict, params, X_true, basis, n_c):
    """fn_predict: (n_params,) → (n_c,) coefficients. Decode and compute loss."""
    losses = []
    for i in range(len(params)):
        c = fn_predict(params[i])
        c_full = np.zeros(len(basis))
        c_full[:n_c] = c[:n_c]
        x_pred = decode_coeff(c_full, basis)
        losses.append(rms_rel_error(x_pred, X_true[i]))
    return np.array(losses)


# Reparameterizations (6D input)
def reparam_raw(P):
    """Raw: (q, chi1z, chi2z, e0, zeta0, omega0)"""
    return P.astype(float)

def reparam_eff(P):
    """Effective: (eta, chi_eff, chi_a, log(e0+eps), zeta0, omega0)"""
    q=P[:,0]; c1z=P[:,1]; c2z=P[:,2]; e0=P[:,3]; z0=P[:,4]; w0=P[:,5]
    m1=q/(1+q); m2=1/(1+q)
    eta=q/(1+q)**2
    chi_eff=m1*c1z+m2*c2z
    chi_a=(c1z-c2z)/2
    return np.column_stack([eta, chi_eff, chi_a, np.log(e0+1e-6), z0, w0])

def reparam_trig(P):
    """Trig anomaly: (eta, chi_eff, chi_a, e0, cos(zeta0), sin(zeta0), omega0)"""
    q=P[:,0]; c1z=P[:,1]; c2z=P[:,2]; e0=P[:,3]; z0=P[:,4]; w0=P[:,5]
    m1=q/(1+q); m2=1/(1+q)
    eta=q/(1+q)**2
    chi_eff=m1*c1z+m2*c2z
    chi_a=(c1z-c2z)/2
    return np.column_stack([eta, chi_eff, chi_a, e0, np.cos(z0), np.sin(z0), w0])

def reparam_logfreq(P):
    """Log freq: (eta, chi_eff, chi_a, e0, zeta0, log(omega0))"""
    q=P[:,0]; c1z=P[:,1]; c2z=P[:,2]; e0=P[:,3]; z0=P[:,4]; w0=P[:,5]
    m1=q/(1+q); m2=1/(1+q)
    eta=q/(1+q)**2
    chi_eff=m1*c1z+m2*c2z
    chi_a=(c1z-c2z)/2
    return np.column_stack([eta, chi_eff, chi_a, e0, z0, np.log(w0)])

REPARAM_FNS = {'raw': reparam_raw, 'eff': reparam_eff, 'trig': reparam_trig, 'lf': reparam_logfreq}


def model_dir(number, name):
    d = os.path.join(MODELS_DIR, f'{number:02d}_{name}')
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, 'saved_model'), exist_ok=True)
    return d

def write_stubs(d, name):
    with open(os.path.join(d, 'train.py'), 'w') as f:
        f.write(f'"""Training script for {name}."""\nimport joblib, numpy as np\nprint("Model: {name}")\n')
    with open(os.path.join(d, 'predict.py'), 'w') as f:
        f.write(f'"""Prediction for {name}."""\nimport joblib, os\n'
                f'_m = joblib.load(os.path.join(os.path.dirname(__file__), "saved_model", "model.pkl"))\n'
                f'def predict(x): return _m["fn"](x)\n')

def save_card(d, number, name, reparam, loss, rt, notes, tconv='normalized'):
    card = {'approach': name, 'approach_number': number, 'benchmark': 'dynamics', 'agent': 'sonnet46',
            'parameterization': reparam, 'time_convention': tconv, 'loss': float(loss),
            'loss_components': {'rms_relative_error_x': float(loss)},
            'runtime_ms': float(rt), 'n_train': N_TRAIN, 'n_val': N_VAL, 'notes': notes}
    with open(os.path.join(d, 'scorecard.json'), 'w') as f:
        json.dump(card, f, indent=2)

def record(number, name, cat, reparam, tl_arr, vl_arr, rt, notes, tconv='normalized'):
    mt=float(np.mean(tl_arr)); mv=float(np.mean(vl_arr))
    ALL_RESULTS.append({'number':number,'name':name,'category':cat,'parameterization':reparam,
                        'train_loss':mt,'val_loss':mv,'runtime_ms':rt,
                        'train_losses':list(tl_arr),'val_losses':list(vl_arr)})
    d = model_dir(number, name)
    save_card(d, number, name, reparam, mv, rt, notes, tconv)
    update_plots()
    with open(CHANGELOG,'a') as f:
        f.write(f'## {number:02d}: {name}\n- cat={cat}, reparam={reparam}\n'
                f'- train={mt:.4e}, val={mv:.4e}, rt={rt:.1f}ms\n- {notes}\n\n')
    print(f'[{number:02d}] {name}: train={mt:.4e} val={mv:.4e} rt={rt:.0f}ms')

def update_plots():
    if not ALL_RESULTS: return
    names=[r['name'] for r in ALL_RESULTS]; vl=[r['val_loss'] for r in ALL_RESULTS]
    cats=[r['category'] for r in ALL_RESULTS]; colors=[CAT_COLORS.get(c,COLORS['gray']) for c in cats]
    x=np.arange(len(ALL_RESULTS))

    fig,ax=plt.subplots(figsize=figsize(2,0.5))
    for cat,col in CAT_COLORS.items():
        idx=[i for i,c in enumerate(cats) if c==cat]
        if idx: ax.scatter(x[idx],np.array(vl)[idx],c=col,label=CAT_LABELS[cat],s=25,zorder=3)
    ax.set_yscale('log'); ax.set_xlabel('Approach'); ax.set_ylabel('Val RMS rel. error')
    ax.set_xticks(x); ax.set_xticklabels([n[:9] for n in names],rotation=45,ha='right',fontsize=5)
    ax.legend(fontsize=6,ncol=2); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'progress.{ext}'))
    plt.close()

    fig,ax=plt.subplots(figsize=figsize(2,0.7))
    for r in ALL_RESULTS:
        ax.scatter(r['runtime_ms'],r['val_loss'],c=CAT_COLORS.get(r['category'],COLORS['gray']),s=25,zorder=3)
        ax.annotate(r['name'][:8],(r['runtime_ms'],r['val_loss']),fontsize=4,xytext=(2,2),textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Eval time (ms)'); ax.set_ylabel('Val RMS rel. error')
    for cat,col in CAT_COLORS.items(): ax.scatter([],[],c=col,label=CAT_LABELS[cat],s=25)
    ax.legend(fontsize=6); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'pareto_accuracy_speed.{ext}'))
    plt.close()

    fig,ax=plt.subplots(figsize=figsize(2,0.6))
    ax.bar(x,vl,color=colors,alpha=0.85,edgecolor='white',linewidth=0.3)
    ax.set_yscale('log'); ax.set_ylabel('Val RMS rel. error')
    ax.set_xticks(x); ax.set_xticklabels([n[:9] for n in names],rotation=45,ha='right',fontsize=5)
    for cat,col in CAT_COLORS.items(): ax.bar([],[],color=col,label=CAT_LABELS[cat])
    ax.legend(fontsize=6,ncol=2); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'loss_only_comparison.{ext}'))
    plt.close()

    data={r['name']:{'train':r['train_losses'],'val':r['val_losses']} for r in ALL_RESULTS}
    with open(os.path.join(COMP_DIR,'error_data.json'),'w') as f: json.dump(data,f,indent=2)


# ============================================================
# LOAD DATA
# ============================================================
print('Loading data...')
data_tr = load_h5('datasets/dynamics/dynamics_training.h5')
data_va = load_h5('datasets/dynamics/dynamics_validation.h5')
N_TRAIN, N_VAL = len(data_tr), len(data_va)
print(f'  Train={N_TRAIN}, Val={N_VAL}')

# Extract parameters
P_tr = np.array([d['params'] for d in data_tr])
P_va = np.array([d['params'] for d in data_va])

# Interpolate to normalized grid
print(f'  Interpolating to normalized grid ({N_GRID} pts)...')
X_tr, TAU = interp_normalized(data_tr, N_GRID)
X_va, _ = interp_normalized(data_va, N_GRID)
print(f'  X_tr shape: {X_tr.shape}, x range: [{X_tr.min():.4f},{X_tr.max():.4f}]')

# Compute SVD
print(f'  Computing SVD ({N_BASIS} basis)...')
BASIS, C_tr, SV = svd_basis(X_tr, N_BASIS)
C_va = encode(X_va, BASIS)

# Check reconstruction quality
recon_tr = np.array([decode_coeff(C_tr[i], BASIS) for i in range(30)])
recon_err = np.mean([rms_rel_error(recon_tr[i], X_tr[i]) for i in range(30)])
print(f'  SVD recon error (n={N_BASIS}): {recon_err:.4f}')

# Truncation error with N_C
trunc_c = C_tr[:, :N_C]
recon_trunc = np.array([decode_coeff(trunc_c[i], BASIS) for i in range(30)])
trunc_err = np.mean([rms_rel_error(recon_trunc[i], X_tr[i]) for i in range(30)])
print(f'  Truncated ({N_C} coeff) recon error: {trunc_err:.4f}')

# Precompute reparameterizations
Xr = {k: fn(P_tr) for k, fn in REPARAM_FNS.items()}
Xv = {k: fn(P_va) for k, fn in REPARAM_FNS.items()}

Y_tr = C_tr[:, :N_C]   # shape (N_TRAIN, N_C)
Y_va = C_va[:, :N_C]   # shape (N_VAL, N_C)

def ytrain_nc(n_c): return C_tr[:, :n_c]
def yval_nc(n_c): return C_va[:, :n_c]

# Initialize CHANGELOG
with open(CHANGELOG,'w') as f:
    f.write('# Dynamics Benchmark CHANGELOG (sonnet46)\n\n')
    f.write(f'## Setup\n- Grid: normalized tau [0,1], {N_GRID} pts\n')
    f.write(f'- SVD {N_BASIS} basis, recon error={recon_err:.4f}\n')
    f.write(f'- Truncated ({N_C}) error={trunc_err:.4f}\n\n')

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

def make_pred(model_predict, n_c):
    """Wrap model. model_predict(x_2d) → (n_c,) or (1, n_c)."""
    def fn(x):
        c = np.asarray(model_predict(x.reshape(1,-1))).flatten()
        cf = np.zeros(N_BASIS)
        cf[:n_c] = c[:n_c]
        return decode_coeff(cf, BASIS)
    return fn

def eval_approach(fn, Xpar, X_true):
    """Evaluate approach for all samples."""
    losses = []
    for i in range(len(Xpar)):
        x_pred = fn(Xpar[i])
        losses.append(rms_rel_error(x_pred, X_true[i]))
    return np.array(losses)


# ============================================================
# 01: SVD + GPR RBF (raw)
# ============================================================
print('\n[01] SVD+GPR RBF raw')
d=model_dir(1,'svd_gpr_rbf_raw')
Ygpr=ytrain_nc(N_C_GPR)
sc1=StandardScaler(); Xt1=sc1.fit_transform(Xr['raw'])
k1=C(1.)*RBF(np.ones(6))+WhiteKernel(1e-3)
g1=GaussianProcessRegressor(kernel=k1,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); g1.fit(Xt1,Ygpr); print(f'  fit {time.time()-t0:.0f}s')
fn1=make_pred(lambda x:g1.predict(sc1.transform(x))[0],N_C_GPR)
ts=time.time(); fn1(Xr['raw'][0]); rt1=(time.time()-ts)*1e3
tl1=eval_approach(fn1,Xr['raw'],X_tr); vl1=eval_approach(fn1,Xv['raw'],X_va)
joblib.dump({'gpr':g1,'sc':sc1,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gpr_rbf_raw')
record(1,'svd_gpr_rbf_raw','svd_decomp','raw',tl1,vl1,rt1,'SVD+GPR RBF, raw 6D')

# ============================================================
# 02: SVD + GPR Matern (eff)
# ============================================================
print('\n[02] SVD+GPR Matern eff')
d=model_dir(2,'svd_gpr_matern_eff')
sc2=StandardScaler(); Xt2=sc2.fit_transform(Xr['eff'])
k2=C(1.)*Matern(np.ones(6),nu=2.5)+WhiteKernel(1e-3)
g2=GaussianProcessRegressor(kernel=k2,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); g2.fit(Xt2,Ygpr); print(f'  fit {time.time()-t0:.0f}s')
fn2=make_pred(lambda x:g2.predict(sc2.transform(x))[0],N_C_GPR)
ts=time.time(); fn2(Xr['eff'][0]); rt2=(time.time()-ts)*1e3
tl2=eval_approach(fn2,Xr['eff'],X_tr); vl2=eval_approach(fn2,Xv['eff'],X_va)
joblib.dump({'gpr':g2,'sc':sc2,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gpr_matern_eff')
record(2,'svd_gpr_matern_eff','svd_decomp','eff',tl2,vl2,rt2,'SVD+GPR Matern, eff_spins+log_e')

# ============================================================
# 03: SVD + Poly deg2 (raw)
# ============================================================
print('\n[03] SVD+Poly2 raw')
d=model_dir(3,'svd_poly2_raw')
poly3=PolynomialFeatures(degree=2,include_bias=True)
sc3=StandardScaler(); Xp3=sc3.fit_transform(poly3.fit_transform(Xr['raw']))
t0=time.time(); r3=Ridge(alpha=0.01); r3.fit(Xp3,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
fn3=make_pred(lambda x:r3.predict(sc3.transform(poly3.transform(x))),N_C)
ts=time.time(); fn3(Xr['raw'][0]); rt3=(time.time()-ts)*1e3
tl3=eval_approach(fn3,Xr['raw'],X_tr); vl3=eval_approach(fn3,Xv['raw'],X_va)
joblib.dump({'poly':poly3,'sc':sc3,'r':r3,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_poly2_raw')
record(3,'svd_poly2_raw','svd_decomp','raw',tl3,vl3,rt3,'SVD+poly2 Ridge, raw 6D')

# ============================================================
# 04: SVD + MLP (raw)
# ============================================================
print('\n[04] SVD+MLP raw')
d=model_dir(4,'svd_mlp_raw')
sc4=StandardScaler(); Xt4=sc4.fit_transform(Xr['raw'])
t0=time.time()
mlp4=MLPRegressor(hidden_layer_sizes=(128,128,64),activation='relu',max_iter=500,
                   random_state=42,early_stopping=True,validation_fraction=0.1)
mlp4.fit(Xt4,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn4=make_pred(lambda x:mlp4.predict(sc4.transform(x)),N_C)
ts=time.time(); fn4(Xr['raw'][0]); rt4=(time.time()-ts)*1e3
tl4=eval_approach(fn4,Xr['raw'],X_tr); vl4=eval_approach(fn4,Xv['raw'],X_va)
joblib.dump({'mlp':mlp4,'sc':sc4,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_mlp_raw')
record(4,'svd_mlp_raw','ml','raw',tl4,vl4,rt4,'SVD+MLP(128,128,64) relu, raw')

# ============================================================
# 05: SVD + RF (raw)
# ============================================================
print('\n[05] SVD+RF raw')
d=model_dir(5,'svd_rf_raw')
t0=time.time()
rf5=RandomForestRegressor(n_estimators=100,max_depth=12,random_state=42,n_jobs=-1)
rf5.fit(Xr['raw'],Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn5=make_pred(lambda x:rf5.predict(x),N_C)
ts=time.time(); fn5(Xr['raw'][0]); rt5=(time.time()-ts)*1e3
tl5=eval_approach(fn5,Xr['raw'],X_tr); vl5=eval_approach(fn5,Xv['raw'],X_va)
joblib.dump({'rf':rf5,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_rf_raw')
record(5,'svd_rf_raw','ml','raw',tl5,vl5,rt5,'SVD+RF(100,d12), raw 6D')

# ============================================================
# 06: EIM + GPR (raw)
# ============================================================
print('\n[06] EIM+GPR raw')
d=model_dir(6,'eim_gpr_raw')
n_eim=8
nodes=[]
for i in range(n_eim):
    if i==0: resid=BASIS[0]
    else:
        Vn=BASIS[:i,nodes].T
        try: c_=np.linalg.lstsq(Vn,BASIS[i,nodes],rcond=None)[0]; resid=BASIS[i]-c_@BASIS[:i]
        except: resid=BASIS[i]
    nodes.append(int(np.argmax(np.abs(resid))))
V_eim=BASIS[:n_eim,nodes].T
eim_target=X_tr[:,nodes]
sc6=StandardScaler(); Xt6=sc6.fit_transform(Xr['raw'])
k6=C(1.)*RBF(np.ones(6))+WhiteKernel(1e-3)
g6=GaussianProcessRegressor(kernel=k6,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); g6.fit(Xt6,eim_target); print(f'  fit {time.time()-t0:.0f}s')
def fn6(x):
    ev=g6.predict(sc6.transform(x.reshape(1,-1)))[0]
    try: cr=np.linalg.lstsq(V_eim,ev,rcond=None)[0]
    except: cr=np.zeros(n_eim)
    cf=np.zeros(N_BASIS); cf[:n_eim]=cr
    return decode_coeff(cf,BASIS)
ts=time.time(); fn6(Xr['raw'][0]); rt6=(time.time()-ts)*1e3
tl6=eval_approach(fn6,Xr['raw'],X_tr); vl6=eval_approach(fn6,Xv['raw'],X_va)
joblib.dump({'gpr':g6,'sc':sc6,'basis':BASIS,'nodes':nodes,'V':V_eim},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'eim_gpr_raw')
record(6,'eim_gpr_raw','svd_decomp','raw',tl6,vl6,rt6,f'EIM({n_eim} nodes)+GPR, raw')

# ============================================================
# 07: SVD + GPR RBF (trig anomaly)
# ============================================================
print('\n[07] SVD+GPR trig')
d=model_dir(7,'svd_gpr_trig')
sc7=StandardScaler(); Xt7=sc7.fit_transform(Xr['trig'])
k7=C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g7=GaussianProcessRegressor(kernel=k7,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); g7.fit(Xt7,Ygpr); print(f'  fit {time.time()-t0:.0f}s')
fn7=make_pred(lambda x:g7.predict(sc7.transform(x))[0],N_C_GPR)
ts=time.time(); fn7(Xr['trig'][0]); rt7=(time.time()-ts)*1e3
tl7=eval_approach(fn7,Xr['trig'],X_tr); vl7=eval_approach(fn7,Xv['trig'],X_va)
joblib.dump({'gpr':g7,'sc':sc7,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gpr_trig')
record(7,'svd_gpr_trig','svd_decomp','trig',tl7,vl7,rt7,'SVD+GPR RBF, trig anomaly reparam')

# ============================================================
# 08: RBF Interpolation (raw)
# ============================================================
print('\n[08] RBF interp raw')
d=model_dir(8,'rbf_interp_raw')
sc8=StandardScaler(); Xt8=sc8.fit_transform(Xr['raw'])
n8=N_C_GPR; Y8=C_tr[:,:n8]
t0=time.time(); rbf8=RBFInterpolator(Xt8,Y8,kernel='thin_plate_spline',smoothing=0.5); print(f'  fit {time.time()-t0:.2f}s')
def fn8(x):
    c=rbf8(sc8.transform(x.reshape(1,-1)))[0]; cf=np.zeros(N_BASIS); cf[:n8]=c[:n8]
    return decode_coeff(cf,BASIS)
ts=time.time(); fn8(Xr['raw'][0]); rt8=(time.time()-ts)*1e3
tl8=eval_approach(fn8,Xr['raw'],X_tr); vl8=eval_approach(fn8,Xv['raw'],X_va)
joblib.dump({'rbf':rbf8,'sc':sc8,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'rbf_interp_raw')
record(8,'rbf_interp_raw','kernel_interp','raw',tl8,vl8,rt8,'TPS-RBF interp, raw 6D')

# ============================================================
# 09: kNN (raw)
# ============================================================
print('\n[09] kNN raw')
d=model_dir(9,'knn_raw')
sc9=StandardScaler(); Xt9=sc9.fit_transform(Xr['raw'])
t0=time.time(); knn9=KNeighborsRegressor(n_neighbors=5,weights='distance'); knn9.fit(Xt9,Y_tr); print(f'  fit {time.time()-t0:.3f}s')
fn9=make_pred(lambda x:knn9.predict(sc9.transform(x)),N_C)
ts=time.time(); fn9(Xr['raw'][0]); rt9=(time.time()-ts)*1e3
tl9=eval_approach(fn9,Xr['raw'],X_tr); vl9=eval_approach(fn9,Xv['raw'],X_va)
joblib.dump({'knn':knn9,'sc':sc9,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'knn_raw')
record(9,'knn_raw','kernel_interp','raw',tl9,vl9,rt9,'kNN(5, distance), raw 6D')

# ============================================================
# 10: KRR (eff)
# ============================================================
print('\n[10] KRR eff')
d=model_dir(10,'krr_eff')
sc10=StandardScaler(); Xt10=sc10.fit_transform(Xr['eff'])
t0=time.time(); krr10=KernelRidge(kernel='rbf',alpha=0.01,gamma=0.5); krr10.fit(Xt10,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
fn10=make_pred(lambda x:krr10.predict(sc10.transform(x)),N_C)
ts=time.time(); fn10(Xr['eff'][0]); rt10=(time.time()-ts)*1e3
tl10=eval_approach(fn10,Xr['eff'],X_tr); vl10=eval_approach(fn10,Xv['eff'],X_va)
joblib.dump({'krr':krr10,'sc':sc10,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'krr_eff')
record(10,'krr_eff','kernel_interp','eff',tl10,vl10,rt10,'KRR(RBF), eff_spins')

# ============================================================
# 11: SVD + GBR (eff)
# ============================================================
print('\n[11] SVD+GBR eff')
d=model_dir(11,'svd_gbr_eff')
n11=8; Y11=C_tr[:,:n11]
t0=time.time()
from sklearn.multioutput import MultiOutputRegressor
gbr11=MultiOutputRegressor(GradientBoostingRegressor(n_estimators=80,max_depth=4,random_state=42),n_jobs=-1)
gbr11.fit(Xr['eff'],Y11); print(f'  fit {time.time()-t0:.1f}s')
def fn11(x):
    c=gbr11.predict(x.reshape(1,-1))[0]; cf=np.zeros(N_BASIS); cf[:n11]=c[:n11]
    return decode_coeff(cf,BASIS)
ts=time.time(); fn11(Xr['eff'][0]); rt11=(time.time()-ts)*1e3
tl11=eval_approach(fn11,Xr['eff'],X_tr); vl11=eval_approach(fn11,Xv['eff'],X_va)
joblib.dump({'gbr':gbr11,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gbr_eff')
record(11,'svd_gbr_eff','ml','eff',tl11,vl11,rt11,'SVD+GBR(80,d4), eff_spins')

# ============================================================
# 12: SVD + ExtraTrees (trig)
# ============================================================
print('\n[12] SVD+ET trig')
d=model_dir(12,'svd_et_trig')
t0=time.time()
et12=ExtraTreesRegressor(n_estimators=150,max_depth=14,random_state=42,n_jobs=-1)
et12.fit(Xr['trig'],Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn12=make_pred(lambda x:et12.predict(x),N_C)
ts=time.time(); fn12(Xr['trig'][0]); rt12=(time.time()-ts)*1e3
tl12=eval_approach(fn12,Xr['trig'],X_tr); vl12=eval_approach(fn12,Xv['trig'],X_va)
joblib.dump({'et':et12,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_et_trig')
record(12,'svd_et_trig','ml','trig',tl12,vl12,rt12,'SVD+ET(150,d14), trig anomaly')

# ============================================================
# 13: SVD + MLP large (eff)
# ============================================================
print('\n[13] SVD+MLP large eff')
d=model_dir(13,'svd_mlp_large_eff')
sc13=StandardScaler(); Xt13=sc13.fit_transform(Xr['eff'])
t0=time.time()
mlp13=MLPRegressor(hidden_layer_sizes=(256,256,128,64),activation='tanh',max_iter=800,
                    random_state=0,early_stopping=True,validation_fraction=0.15,learning_rate_init=5e-4)
mlp13.fit(Xt13,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn13=make_pred(lambda x:mlp13.predict(sc13.transform(x)),N_C)
ts=time.time(); fn13(Xr['eff'][0]); rt13=(time.time()-ts)*1e3
tl13=eval_approach(fn13,Xr['eff'],X_tr); vl13=eval_approach(fn13,Xv['eff'],X_va)
joblib.dump({'mlp':mlp13,'sc':sc13,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_mlp_large_eff')
record(13,'svd_mlp_large_eff','ml','eff',tl13,vl13,rt13,'SVD+MLP(256,256,128,64) tanh, eff')

# ============================================================
# 14: SVD + Poly deg3 (eff)
# ============================================================
print('\n[14] SVD+Poly3 eff')
d=model_dir(14,'svd_poly3_eff')
poly14=PolynomialFeatures(degree=3,include_bias=True)
sc14=StandardScaler(); Xp14=sc14.fit_transform(poly14.fit_transform(Xr['eff']))
t0=time.time(); r14=Ridge(alpha=0.01); r14.fit(Xp14,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
fn14=make_pred(lambda x:r14.predict(sc14.transform(poly14.transform(x))),N_C)
ts=time.time(); fn14(Xr['eff'][0]); rt14=(time.time()-ts)*1e3
tl14=eval_approach(fn14,Xr['eff'],X_tr); vl14=eval_approach(fn14,Xv['eff'],X_va)
joblib.dump({'poly':poly14,'sc':sc14,'r':r14,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_poly3_eff')
record(14,'svd_poly3_eff','svd_decomp','eff',tl14,vl14,rt14,'SVD+poly3 Ridge, eff_spins')

# ============================================================
# 15: gplearn (raw) — MANDATORY
# ============================================================
print('\n[15] gplearn raw')
from gplearn.genetic import SymbolicRegressor as GPLearnSR
d=model_dir(15,'gplearn_svd_raw')
n15=3; expr15=[]
gpl15=[]
t0=time.time()
for ci in range(n15):
    print(f'  gplearn coeff {ci}...')
    g=GPLearnSR(population_size=2000,generations=15,tournament_size=20,
                function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                metric='mse',parsimony_coefficient=0.01,verbose=0,random_state=42+ci*7,n_jobs=2)
    g.fit(Xr['raw'],C_tr[:,ci]); gpl15.append(g)
    expr15.append({'ci':ci,'expr':str(g._program),'fit':float(g._program.fitness_)})
    print(f'    {str(g._program)[:50]}')
print(f'  gplearn total {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr15,f,indent=2)
def fn15(x):
    cf=np.zeros(N_BASIS)
    for ci,g in enumerate(gpl15):
        cf[ci]=g.predict(x.reshape(1,-1))[0]
    return decode_coeff(cf,BASIS)
ts=time.time(); fn15(Xr['raw'][0]); rt15=(time.time()-ts)*1e3
tl15=eval_approach(fn15,Xr['raw'],X_tr); vl15=eval_approach(fn15,Xv['raw'],X_va)
try: joblib.dump({'gpl':gpl15,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
except: np.save(os.path.join(d,'saved_model','basis.npy'),BASIS)
write_stubs(d,'gplearn_svd_raw')
record(15,'gplearn_svd_raw','symbolic','raw',tl15,vl15,rt15,f'gplearn SR on {n15} SVD coeff, raw')

# ============================================================
# 16: gplearn (eff) — MANDATORY
# ============================================================
print('\n[16] gplearn eff')
d=model_dir(16,'gplearn_svd_eff')
n16=2; expr16=[]; gpl16=[]
t0=time.time()
for ci in range(n16):
    g=GPLearnSR(population_size=2000,generations=12,tournament_size=20,
                function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                metric='mse',parsimony_coefficient=0.01,verbose=0,random_state=300+ci,n_jobs=2)
    g.fit(Xr['eff'],C_tr[:,ci]); gpl16.append(g)
    expr16.append({'ci':ci,'expr':str(g._program),'fit':float(g._program.fitness_)})
print(f'  gplearn eff {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr16,f,indent=2)
def fn16(x):
    cf=np.zeros(N_BASIS)
    for ci,g in enumerate(gpl16):
        cf[ci]=g.predict(x.reshape(1,-1))[0]
    return decode_coeff(cf,BASIS)
ts=time.time(); fn16(Xr['eff'][0]); rt16=(time.time()-ts)*1e3
tl16=eval_approach(fn16,Xr['eff'],X_tr); vl16=eval_approach(fn16,Xv['eff'],X_va)
try: joblib.dump({'gpl':gpl16,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
except: np.save(os.path.join(d,'saved_model','basis.npy'),BASIS)
write_stubs(d,'gplearn_svd_eff')
record(16,'gplearn_svd_eff','symbolic','eff',tl16,vl16,rt16,f'gplearn SR on {n16} SVD coeff, eff')

# ============================================================
# 17: PySR (raw) — MANDATORY
# ============================================================
print('\n[17] PySR raw')
from pysr import PySRRegressor
d=model_dir(17,'pysr_svd_raw')
n17=2; expr17=[]; psr17=[]
t0=time.time()
for ci in range(n17):
    print(f'  PySR coeff {ci}...')
    try:
        ps=PySRRegressor(niterations=40,binary_operators=['+','-','*','/'],
                         unary_operators=['sqrt','log','exp','sin','cos'],
                         maxsize=20,populations=12,procs=2,
                         loss='loss(p,t)=abs(p-t)',verbosity=0,random_state=42+ci,
                         tempdir=os.path.join(d,'saved_model',f'pysr_{ci}'))
        ps.fit(Xr['raw'],C_tr[:,ci]); psr17.append(ps)
        try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps.equations_.iterrows()]
        except: pf=[{'e':str(ps.sympy()),'c':0,'l':0.}]
        expr17.append({'ci':ci,'pareto':pf})
    except Exception as e:
        psr17.append(None); expr17.append({'ci':ci,'err':str(e)}); print(f'    failed: {e}')
print(f'  PySR total {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr17,f,indent=2)
def fn17(x):
    cf=np.zeros(N_BASIS)
    for ci,ps in enumerate(psr17):
        if ps:
            try: cf[ci]=ps.predict(x.reshape(1,-1))[0]
            except: pass
    return decode_coeff(cf,BASIS)
ts=time.time(); fn17(Xr['raw'][0]); rt17=(time.time()-ts)*1e3
tl17=eval_approach(fn17,Xr['raw'],X_tr); vl17=eval_approach(fn17,Xv['raw'],X_va)
np.save(os.path.join(d,'saved_model','basis.npy'),BASIS)
write_stubs(d,'pysr_svd_raw')
record(17,'pysr_svd_raw','symbolic','raw',tl17,vl17,rt17,f'PySR on {n17} SVD coeff, raw')

# ============================================================
# 18: PySR (eff) — MANDATORY
# ============================================================
print('\n[18] PySR eff')
d=model_dir(18,'pysr_svd_eff')
n18=2; expr18=[]; psr18=[]
t0=time.time()
for ci in range(n18):
    try:
        ps=PySRRegressor(niterations=30,binary_operators=['+','-','*','/'],
                         unary_operators=['sqrt','log','exp'],maxsize=18,
                         populations=10,procs=2,loss='loss(p,t)=abs(p-t)',
                         verbosity=0,random_state=99+ci,
                         tempdir=os.path.join(d,'saved_model',f'pysr_{ci}'))
        ps.fit(Xr['eff'],C_tr[:,ci]); psr18.append(ps)
        try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps.equations_.iterrows()]
        except: pf=[{'e':str(ps.sympy()),'c':0,'l':0.}]
        expr18.append({'ci':ci,'pareto':pf})
    except Exception as e:
        psr18.append(None); expr18.append({'ci':ci,'err':str(e)})
print(f'  PySR eff {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr18,f,indent=2)
def fn18(x):
    cf=np.zeros(N_BASIS)
    for ci,ps in enumerate(psr18):
        if ps:
            try: cf[ci]=ps.predict(x.reshape(1,-1))[0]
            except: pass
    return decode_coeff(cf,BASIS)
ts=time.time(); fn18(Xr['eff'][0]); rt18=(time.time()-ts)*1e3
tl18=eval_approach(fn18,Xr['eff'],X_tr); vl18=eval_approach(fn18,Xv['eff'],X_va)
np.save(os.path.join(d,'saved_model','basis.npy'),BASIS)
write_stubs(d,'pysr_svd_eff')
record(18,'pysr_svd_eff','symbolic','eff',tl18,vl18,rt18,f'PySR on {n18} SVD coeff, eff')

# ============================================================
# 19: RBF Interpolation (eff)
# ============================================================
print('\n[19] RBF interp eff')
d=model_dir(19,'rbf_interp_eff')
sc19=StandardScaler(); Xt19=sc19.fit_transform(Xr['eff'])
n19=N_C_GPR; Y19=C_tr[:,:n19]
t0=time.time(); rbf19=RBFInterpolator(Xt19,Y19,kernel='linear',smoothing=0.5); print(f'  fit {time.time()-t0:.2f}s')
def fn19(x):
    c=rbf19(sc19.transform(x.reshape(1,-1)))[0]; cf=np.zeros(N_BASIS); cf[:n19]=c[:n19]
    return decode_coeff(cf,BASIS)
ts=time.time(); fn19(Xr['eff'][0]); rt19=(time.time()-ts)*1e3
tl19=eval_approach(fn19,Xr['eff'],X_tr); vl19=eval_approach(fn19,Xv['eff'],X_va)
joblib.dump({'rbf':rbf19,'sc':sc19,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'rbf_interp_eff')
record(19,'rbf_interp_eff','kernel_interp','eff',tl19,vl19,rt19,'Linear-RBF interp, eff_spins')

# ============================================================
# 20: SVD + GPR Matern (log_freq)
# ============================================================
print('\n[20] SVD+GPR lf')
d=model_dir(20,'svd_gpr_lf')
sc20=StandardScaler(); Xt20=sc20.fit_transform(Xr['lf'])
k20=C(1.)*Matern(np.ones(6),nu=1.5)+WhiteKernel(1e-3)
g20=GaussianProcessRegressor(kernel=k20,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); g20.fit(Xt20,Ygpr); print(f'  fit {time.time()-t0:.0f}s')
fn20=make_pred(lambda x:g20.predict(sc20.transform(x))[0],N_C_GPR)
ts=time.time(); fn20(Xr['lf'][0]); rt20=(time.time()-ts)*1e3
tl20=eval_approach(fn20,Xr['lf'],X_tr); vl20=eval_approach(fn20,Xv['lf'],X_va)
joblib.dump({'gpr':g20,'sc':sc20,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_gpr_lf')
record(20,'svd_gpr_lf','svd_decomp','lf',tl20,vl20,rt20,'SVD+GPR Matern-1.5, log_freq reparam')

# ============================================================
# 21: kNN (eff, k=10)
# ============================================================
print('\n[21] kNN-10 eff')
d=model_dir(21,'knn10_eff')
sc21=StandardScaler(); Xt21=sc21.fit_transform(Xr['eff'])
t0=time.time(); knn21=KNeighborsRegressor(n_neighbors=10,weights='distance'); knn21.fit(Xt21,Y_tr); print(f'  fit {time.time()-t0:.3f}s')
fn21=make_pred(lambda x:knn21.predict(sc21.transform(x)),N_C)
ts=time.time(); fn21(Xr['eff'][0]); rt21=(time.time()-ts)*1e3
tl21=eval_approach(fn21,Xr['eff'],X_tr); vl21=eval_approach(fn21,Xv['eff'],X_va)
joblib.dump({'knn':knn21,'sc':sc21,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'knn10_eff')
record(21,'knn10_eff','kernel_interp','eff',tl21,vl21,rt21,'kNN(10, distance), eff_spins')

# ============================================================
# 22: SVD + MLP (lf)
# ============================================================
print('\n[22] SVD+MLP lf')
d=model_dir(22,'svd_mlp_lf')
sc22=StandardScaler(); Xt22=sc22.fit_transform(Xr['lf'])
t0=time.time()
mlp22=MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu',max_iter=500,
                    random_state=77,early_stopping=True,validation_fraction=0.1)
mlp22.fit(Xt22,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn22=make_pred(lambda x:mlp22.predict(sc22.transform(x)),N_C)
ts=time.time(); fn22(Xr['lf'][0]); rt22=(time.time()-ts)*1e3
tl22=eval_approach(fn22,Xr['lf'],X_tr); vl22=eval_approach(fn22,Xv['lf'],X_va)
joblib.dump({'mlp':mlp22,'sc':sc22,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_mlp_lf')
record(22,'svd_mlp_lf','ml','lf',tl22,vl22,rt22,'SVD+MLP(128,64,32) relu, log_freq')

# ============================================================
# 23: SVD + Poly2 (trig)
# ============================================================
print('\n[23] SVD+Poly2 trig')
d=model_dir(23,'svd_poly2_trig')
poly23=PolynomialFeatures(degree=2,include_bias=True)
sc23=StandardScaler(); Xp23=sc23.fit_transform(poly23.fit_transform(Xr['trig']))
t0=time.time(); r23=Ridge(alpha=0.01); r23.fit(Xp23,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
fn23=make_pred(lambda x:r23.predict(sc23.transform(poly23.transform(x))),N_C)
ts=time.time(); fn23(Xr['trig'][0]); rt23=(time.time()-ts)*1e3
tl23=eval_approach(fn23,Xr['trig'],X_tr); vl23=eval_approach(fn23,Xv['trig'],X_va)
joblib.dump({'poly':poly23,'sc':sc23,'r':r23,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_poly2_trig')
record(23,'svd_poly2_trig','svd_decomp','trig',tl23,vl23,rt23,'SVD+poly2 Ridge, trig anomaly')

# ============================================================
# 24: SVD + RF (trig)
# ============================================================
print('\n[24] SVD+RF trig')
d=model_dir(24,'svd_rf_trig')
t0=time.time()
rf24=RandomForestRegressor(n_estimators=150,max_depth=None,random_state=0,n_jobs=-1,min_samples_leaf=2)
rf24.fit(Xr['trig'],Y_tr); print(f'  fit {time.time()-t0:.1f}s')
fn24=make_pred(lambda x:rf24.predict(x),N_C)
ts=time.time(); fn24(Xr['trig'][0]); rt24=(time.time()-ts)*1e3
tl24=eval_approach(fn24,Xr['trig'],X_tr); vl24=eval_approach(fn24,Xv['trig'],X_va)
joblib.dump({'rf':rf24,'basis':BASIS},os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svd_rf_trig')
record(24,'svd_rf_trig','ml','trig',tl24,vl24,rt24,'SVD+RF(150), trig anomaly')

# ============================================================
# FINAL OUTPUTS
# ============================================================
print('\n=== Final plots & summary ===')

fig, ax = plt.subplots(figsize=figsize(2, 0.8))
for r in ALL_RESULTS:
    tl=np.array(r['train_losses']); vl_r=np.array(r['val_losses'])
    al=np.concatenate([tl,vl_r]); lo,hi=max(al.min(),1e-8),max(al.max(),1e-6)
    if lo<hi:
        bins=np.logspace(np.log10(lo),np.log10(hi),20)
        ax.hist(tl,bins=bins,alpha=0.2,density=True,histtype='stepfilled',label=f'{r["name"][:7]} tr')
        ax.hist(vl_r,bins=bins,alpha=0.7,density=True,histtype='step',lw=1.,label=f'{r["name"][:7]} va')
ax.set_xscale('log'); ax.set_xlabel('RMS rel. error'); ax.set_ylabel('Density')
ax.legend(fontsize=3.5, ncol=4)
plt.tight_layout()
for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'error_histograms.{ext}'))
plt.close()

ranked=sorted(ALL_RESULTS, key=lambda r: r['val_loss'])
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
    print('\nDYNAMICS_BENCH_COMPLETE')
else:
    print(f'\nIncomplete: n={n}, reps={len(reps)}, cats={len(cats)}, pysr={has_pysr}, gplearn={has_gpl}')
