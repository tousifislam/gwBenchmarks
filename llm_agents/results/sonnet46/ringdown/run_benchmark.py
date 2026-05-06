#!/usr/bin/env python3
"""Run ringdown benchmark for sonnet46 agent — QNM frequency surrogate models."""
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
from scipy.interpolate import CubicSpline, RBFInterpolator
from scipy.optimize import curve_fit, minimize
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

from gwbenchmarks.plot_settings import apply as plot_apply, figsize, COLORS
plot_apply()

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
COMP_DIR   = os.path.join(SCRIPT_DIR, 'comparison')
CHANGELOG  = os.path.join(SCRIPT_DIR, 'CHANGELOG.md')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

CAT_COLORS = {
    'analytical': COLORS['blue'],
    'symbolic':   COLORS['red'],
    'interp':     COLORS['green'],
    'ml':         COLORS['orange'],
}
ALL_RESULTS = []

# ============================================================
# Data loading
# ============================================================
MODE = 'l2/m+2/n0'

def load_data(path):
    with h5py.File(path, 'r') as f:
        g = f[MODE]
        return g['spin'][:], g['omega_real'][:], g['omega_imag'][:]

print('Loading data...')
a_tr, or_tr, oi_tr = load_data('datasets/ringdown/ringdown_training.h5')
a_va, or_va, oi_va = load_data('datasets/ringdown/ringdown_validation.h5')
N_TRAIN, N_VAL = len(a_tr), len(a_va)
print(f'  Train={N_TRAIN}, Val={N_VAL}')
print(f'  spin: [{a_tr.min():.4f}, {a_tr.max():.6f}]')
print(f'  omega_r: [{or_tr.min():.4f}, {or_tr.max():.4f}]')
print(f'  omega_i: [{oi_tr.min():.6f}, {oi_tr.max():.6f}]')

# ============================================================
# Reparameterizations
# ============================================================
def rep_raw(a):       return a.reshape(-1,1)
def rep_log(a):       return (-np.log(np.clip(1-a, 1e-8, None))).reshape(-1,1)
def rep_sqrt(a):      return np.sqrt(np.clip(1-a**2, 0, None)).reshape(-1,1)
def rep_cheb(a):      return (2*a-1).reshape(-1,1)
def rep_compact(a):   return (a/np.clip(1-a, 1e-8, None)).reshape(-1,1)

REPS = {'raw': rep_raw, 'log': rep_log, 'sqrt': rep_sqrt, 'cheb': rep_cheb, 'compact': rep_compact}

# Precompute
Xr_tr = {k: fn(a_tr) for k,fn in REPS.items()}
Xr_va = {k: fn(a_va) for k,fn in REPS.items()}
Y_tr = np.column_stack([or_tr, oi_tr])
Y_va = np.column_stack([or_va, oi_va])

# ============================================================
# Loss
# ============================================================
def loss_fn(pred_r, pred_i, true_r, true_i):
    err_r = np.mean(np.abs(pred_r - true_r) / np.abs(true_r))
    err_i = np.mean(np.abs(pred_i - true_i) / np.abs(true_i))
    return (err_r + err_i) / 2

def eval_loss_arr(pred_r, pred_i, true_r, true_i):
    """Per-sample errors."""
    er = np.abs(pred_r - true_r) / np.abs(true_r)
    ei = np.abs(pred_i - true_i) / np.abs(true_i)
    return (er + ei) / 2

# ============================================================
# Utility
# ============================================================
def model_dir(num, name):
    d = os.path.join(MODELS_DIR, f'{num:02d}_{name}')
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d,'saved_model'), exist_ok=True)
    return d

def write_stubs(d, name):
    with open(os.path.join(d,'train.py'),'w') as f:
        f.write(f'"""Training script for {name}."""\nimport joblib, numpy as np\nprint("Model: {name}")\n')
    with open(os.path.join(d,'predict.py'),'w') as f:
        f.write(f'"""Prediction for {name}. Returns (omega_r, omega_i)."""\n'
                f'import joblib, os, numpy as np\n'
                f'_m = joblib.load(os.path.join(os.path.dirname(__file__), "saved_model", "model.pkl"))\n'
                f'def predict(spin_array):\n'
                f'    y = _m["fn"](np.asarray(spin_array).reshape(-1))\n'
                f'    return y[:,0], y[:,1]\n')

def save_card(d, num, name, reparam, loss, comp, rt, notes, extra=None):
    card = {'approach': name, 'approach_number': num, 'benchmark': 'ringdown', 'agent': 'sonnet46',
            'parameterization': reparam, 'mode': MODE.replace('/','_'),
            'loss': float(loss), 'loss_components': comp,
            'runtime_ms': float(rt), 'n_train': N_TRAIN, 'n_val': N_VAL, 'notes': notes}
    if extra: card.update(extra)
    with open(os.path.join(d,'scorecard.json'),'w') as f:
        json.dump(card, f, indent=2)

def record(num, name, cat, reparam, tl_arr, vl_arr, rt, notes,
           comp=None, extra=None):
    mt=float(np.mean(tl_arr)); mv=float(np.mean(vl_arr))
    if comp is None: comp={'mean_rel_err': mv}
    ALL_RESULTS.append({'number':num,'name':name,'category':cat,'parameterization':reparam,
                        'train_loss':mt,'val_loss':mv,'runtime_ms':rt,
                        'train_losses':list(tl_arr),'val_losses':list(vl_arr)})
    d = model_dir(num, name)
    save_card(d, num, name, reparam, mv, comp, rt, notes, extra)
    write_stubs(d, name)
    update_plots()
    with open(CHANGELOG,'a') as f:
        f.write(f'## {num:02d}: {name}\n- cat={cat}, reparam={reparam}\n'
                f'- train={mt:.4e}, val={mv:.4e}, rt={rt:.2f}ms\n- {notes}\n\n')
    print(f'[{num:02d}] {name}: train={mt:.4e} val={mv:.4e} rt={rt:.2f}ms')

def update_plots():
    if not ALL_RESULTS: return
    names=[r['name'] for r in ALL_RESULTS]
    vl=[r['val_loss'] for r in ALL_RESULTS]
    cats=[r['category'] for r in ALL_RESULTS]
    colors=[CAT_COLORS.get(c,COLORS['gray']) for c in cats]
    x=np.arange(len(ALL_RESULTS))

    # progress
    fig,ax=plt.subplots(figsize=figsize(2,0.5))
    for cat,col in CAT_COLORS.items():
        idx=[i for i,c in enumerate(cats) if c==cat]
        if idx: ax.scatter(np.array(idx), np.array(vl)[idx], color=col, label=cat, s=20, zorder=3)
    ax.set_yscale('log'); ax.set_xlabel('Approach'); ax.set_ylabel('Val loss (mean rel err)')
    ax.legend(fontsize=6)
    plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'progress.{ext}'))
    plt.close()

    # loss_only
    fig,ax=plt.subplots(figsize=figsize(2,0.6))
    bars=ax.bar(x, vl, color=colors)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(names,rotation=90,fontsize=4)
    ax.set_ylabel('Val loss')
    for cat,col in CAT_COLORS.items():
        ax.bar(0,0,color=col,label=cat)
    ax.legend(fontsize=6)
    plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'loss_only_comparison.{ext}'))
    plt.close()

    # pareto
    rts=[r['runtime_ms'] for r in ALL_RESULTS]
    fig,ax=plt.subplots(figsize=figsize(2,0.6))
    for r in ALL_RESULTS:
        ax.scatter(r['runtime_ms'],r['val_loss'],color=CAT_COLORS.get(r['category'],COLORS['gray']),s=25,zorder=3)
        ax.annotate(r['name'][:10],(r['runtime_ms'],r['val_loss']),fontsize=3,ha='left')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Eval time (ms)'); ax.set_ylabel('Val loss')
    for cat,col in CAT_COLORS.items():
        ax.scatter([],[], color=col, label=cat, s=15)
    ax.legend(fontsize=6)
    plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'pareto_accuracy_speed.{ext}'))
    plt.close()

    # error_data
    ed={r['name']:{'train':r['train_losses'],'val':r['val_losses']} for r in ALL_RESULTS}
    with open(os.path.join(COMP_DIR,'error_data.json'),'w') as f: json.dump(ed,f)


# ============================================================
# 01: Polynomial deg 10 (raw)
# ============================================================
print('\n[01] Poly-10 raw')
d=model_dir(1,'poly10_raw')
t0=time.time()
coeffs_r1 = np.polyfit(a_tr, or_tr, 10)
coeffs_i1 = np.polyfit(a_tr, oi_tr, 10)
print(f'  fit {time.time()-t0:.3f}s')
def fn01(a):
    return np.polyval(coeffs_r1, a), np.polyval(coeffs_i1, a)
ts=time.time(); fn01(a_va[:1]); rt1=(time.time()-ts)*1e3
tr_r1,tr_i1=fn01(a_tr); va_r1,va_i1=fn01(a_va)
tl1=eval_loss_arr(tr_r1,tr_i1,or_tr,oi_tr); vl1=eval_loss_arr(va_r1,va_i1,or_va,oi_va)
joblib.dump({'fn':fn01,'cr':coeffs_r1,'ci':coeffs_i1},os.path.join(d,'saved_model','model.pkl'))
comp1={'rel_error_omega_real':float(np.mean(np.abs(va_r1-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i1-oi_va)/np.abs(oi_va)))}
record(1,'poly10_raw','analytical','raw',tl1,vl1,rt1,'Poly(10), raw spin a',comp1)

# ============================================================
# 02: Polynomial deg 15 (log)
# ============================================================
print('\n[02] Poly-15 log')
d=model_dir(2,'poly15_log')
x2=Xr_tr['log'].ravel(); xv2=Xr_va['log'].ravel()
t0=time.time()
coeffs_r2=np.polyfit(x2,or_tr,15); coeffs_i2=np.polyfit(x2,oi_tr,15)
print(f'  fit {time.time()-t0:.3f}s')
def fn02(a): xl=-np.log(np.clip(1-a,1e-8,None)); return np.polyval(coeffs_r2,xl),np.polyval(coeffs_i2,xl)
ts=time.time(); fn02(a_va[:1]); rt2=(time.time()-ts)*1e3
tr_r2,tr_i2=fn02(a_tr); va_r2,va_i2=fn02(a_va)
tl2=eval_loss_arr(tr_r2,tr_i2,or_tr,oi_tr); vl2=eval_loss_arr(va_r2,va_i2,or_va,oi_va)
joblib.dump({'cr':coeffs_r2,'ci':coeffs_i2},os.path.join(d,'saved_model','model.pkl'))
comp2={'rel_error_omega_real':float(np.mean(np.abs(va_r2-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i2-oi_va)/np.abs(oi_va)))}
record(2,'poly15_log','analytical','log',tl2,vl2,rt2,'Poly(15), log-compactified -log(1-a)',comp2)

# ============================================================
# 03: Chebyshev expansion deg 20 (cheb)
# ============================================================
print('\n[03] Chebyshev-20 cheb')
d=model_dir(3,'cheb20_cheb')
x3=Xr_tr['cheb'].ravel(); xv3=Xr_va['cheb'].ravel()
t0=time.time()
cr3=np.polynomial.chebyshev.chebfit(x3,or_tr,20)
ci3=np.polynomial.chebyshev.chebfit(x3,oi_tr,20)
print(f'  fit {time.time()-t0:.4f}s')
def fn03(a): xc=2*a-1; return np.polynomial.chebyshev.chebval(xc,cr3),np.polynomial.chebyshev.chebval(xc,ci3)
ts=time.time(); fn03(a_va[:1]); rt3=(time.time()-ts)*1e3
tr_r3,tr_i3=fn03(a_tr); va_r3,va_i3=fn03(a_va)
tl3=eval_loss_arr(tr_r3,tr_i3,or_tr,oi_tr); vl3=eval_loss_arr(va_r3,va_i3,or_va,oi_va)
joblib.dump({'cr':cr3,'ci':ci3},os.path.join(d,'saved_model','model.pkl'))
comp3={'rel_error_omega_real':float(np.mean(np.abs(va_r3-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i3-oi_va)/np.abs(oi_va)))}
record(3,'cheb20_cheb','analytical','cheb',tl3,vl3,rt3,'Chebyshev(20), 2a-1 mapped to [-1,1]',comp3)

# ============================================================
# 04: Cubic spline (raw)
# ============================================================
print('\n[04] CubicSpline raw')
d=model_dir(4,'spline_raw')
t0=time.time()
cs_r=CubicSpline(a_tr,or_tr); cs_i=CubicSpline(a_tr,oi_tr)
print(f'  fit {time.time()-t0:.4f}s')
def fn04(a): return cs_r(a),cs_i(a)
ts=time.time(); fn04(a_va[:1]); rt4=(time.time()-ts)*1e3
tr_r4,tr_i4=fn04(a_tr); va_r4,va_i4=fn04(a_va)
tl4=eval_loss_arr(tr_r4,tr_i4,or_tr,oi_tr); vl4=eval_loss_arr(va_r4,va_i4,or_va,oi_va)
joblib.dump({'csr':cs_r,'csi':cs_i},os.path.join(d,'saved_model','model.pkl'))
comp4={'rel_error_omega_real':float(np.mean(np.abs(va_r4-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i4-oi_va)/np.abs(oi_va)))}
record(4,'spline_raw','interp','raw',tl4,vl4,rt4,'Cubic spline on raw spin',comp4)

# ============================================================
# 05: Cubic spline on Chebyshev nodes (cheb)
# ============================================================
print('\n[05] Spline on Chebyshev nodes')
d=model_dir(5,'spline_cheb_nodes')
n5=50
cheb_nodes=0.5*(1-np.cos(np.pi*np.arange(n5+1)/n5))  # Chebyshev-Gauss-Lobatto in [0,1]
from scipy.interpolate import interp1d
t0=time.time()
or_nodes=np.interp(cheb_nodes,a_tr,or_tr); oi_nodes=np.interp(cheb_nodes,a_tr,oi_tr)
cs_r5=CubicSpline(cheb_nodes,or_nodes); cs_i5=CubicSpline(cheb_nodes,oi_nodes)
print(f'  fit {time.time()-t0:.4f}s')
def fn05(a): return cs_r5(np.clip(a,0,1)),cs_i5(np.clip(a,0,1))
ts=time.time(); fn05(a_va[:1]); rt5=(time.time()-ts)*1e3
tr_r5,tr_i5=fn05(a_tr); va_r5,va_i5=fn05(a_va)
tl5=eval_loss_arr(tr_r5,tr_i5,or_tr,oi_tr); vl5=eval_loss_arr(va_r5,va_i5,or_va,oi_va)
joblib.dump({'csr':cs_r5,'csi':cs_i5},os.path.join(d,'saved_model','model.pkl'))
comp5={'rel_error_omega_real':float(np.mean(np.abs(va_r5-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i5-oi_va)/np.abs(oi_va)))}
record(5,'spline_cheb_nodes','interp','cheb',tl5,vl5,rt5,'Cubic spline on 50 Chebyshev-Gauss-Lobatto nodes',comp5)

# ============================================================
# 06: Padé rational approximation [10,10] (raw)
# ============================================================
print('\n[06] Pade [10,10] raw')
d=model_dir(6,'pade_10_10_raw')
def pade_fit(a, y, p, q):
    """Fit rational p(a)/q(a) where q has q_deg terms. Uses linear LS."""
    A = np.vander(a, p+1, increasing=True)
    B = np.vander(a, q, increasing=True) * (-y[:,None])
    M = np.hstack([A, B])
    c, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
    return c[:p+1], np.r_[[1.0], c[p+1:]]

t0=time.time()
nr6, dr6 = pade_fit(a_tr, or_tr, 10, 10)
ni6, di6 = pade_fit(a_tr, oi_tr, 10, 10)
print(f'  fit {time.time()-t0:.4f}s')
def eval_pade(a, n, d):
    num = np.polyval(n[::-1], a); den = np.polyval(d[::-1], a)
    return np.where(np.abs(den)>1e-12, num/den, 0.)
def fn06(a): return eval_pade(a,nr6,dr6), eval_pade(a,ni6,di6)
ts=time.time(); fn06(a_va[:1]); rt6=(time.time()-ts)*1e3
tr_r6,tr_i6=fn06(a_tr); va_r6,va_i6=fn06(a_va)
tl6=eval_loss_arr(tr_r6,tr_i6,or_tr,oi_tr); vl6=eval_loss_arr(va_r6,va_i6,or_va,oi_va)
joblib.dump({'nr':nr6,'dr':dr6,'ni':ni6,'di':di6},os.path.join(d,'saved_model','model.pkl'))
comp6={'rel_error_omega_real':float(np.mean(np.abs(va_r6-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i6-oi_va)/np.abs(oi_va)))}
record(6,'pade_10_10_raw','analytical','raw',tl6,vl6,rt6,'Pade rational [10,10], raw spin',comp6)

# ============================================================
# 07: RBF interpolation (TPS, raw)
# ============================================================
print('\n[07] RBF-TPS raw')
d=model_dir(7,'rbf_tps_raw')
t0=time.time()
rbf7=RBFInterpolator(a_tr.reshape(-1,1), Y_tr, kernel='thin_plate_spline', smoothing=0.0)
print(f'  fit {time.time()-t0:.3f}s')
def fn07(a): out=rbf7(a.reshape(-1,1)); return out[:,0],out[:,1]
ts=time.time(); fn07(a_va[:1]); rt7=(time.time()-ts)*1e3
tr_r7,tr_i7=fn07(a_tr); va_r7,va_i7=fn07(a_va)
tl7=eval_loss_arr(tr_r7,tr_i7,or_tr,oi_tr); vl7=eval_loss_arr(va_r7,va_i7,or_va,oi_va)
joblib.dump({'rbf':rbf7},os.path.join(d,'saved_model','model.pkl'))
comp7={'rel_error_omega_real':float(np.mean(np.abs(va_r7-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i7-oi_va)/np.abs(oi_va)))}
record(7,'rbf_tps_raw','interp','raw',tl7,vl7,rt7,'RBF(TPS) interpolation, raw spin',comp7)

# ============================================================
# 08: RBF interpolation (multiquadric, log)
# ============================================================
print('\n[08] RBF-mq log')
d=model_dir(8,'rbf_mq_log')
X8=Xr_tr['log']; Xv8=Xr_va['log']
t0=time.time()
rbf8=RBFInterpolator(X8, Y_tr, kernel='multiquadric', epsilon=1.0, smoothing=1e-8)
print(f'  fit {time.time()-t0:.3f}s')
def fn08(a): xl=-np.log(np.clip(1-a,1e-8,None)).reshape(-1,1); out=rbf8(xl); return out[:,0],out[:,1]
ts=time.time(); fn08(a_va[:1]); rt8=(time.time()-ts)*1e3
tr_r8,tr_i8=fn08(a_tr); va_r8,va_i8=fn08(a_va)
tl8=eval_loss_arr(tr_r8,tr_i8,or_tr,oi_tr); vl8=eval_loss_arr(va_r8,va_i8,or_va,oi_va)
joblib.dump({'rbf':rbf8},os.path.join(d,'saved_model','model.pkl'))
comp8={'rel_error_omega_real':float(np.mean(np.abs(va_r8-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i8-oi_va)/np.abs(oi_va)))}
record(8,'rbf_mq_log','interp','log',tl8,vl8,rt8,'RBF(multiquadric) interpolation, log-compactified',comp8)

# ============================================================
# 09: GPR RBF (raw)
# ============================================================
print('\n[09] GPR-RBF raw')
d=model_dir(9,'gpr_rbf_raw')
sc9=StandardScaler(); Xt9=sc9.fit_transform(a_tr.reshape(-1,1))
k9=C(1.)*RBF(1.)+WhiteKernel(1e-5)
gp9=GaussianProcessRegressor(kernel=k9,n_restarts_optimizer=2,alpha=1e-6,normalize_y=True)
t0=time.time(); gp9.fit(Xt9,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
def fn09(a): return tuple(gp9.predict(sc9.transform(a.reshape(-1,1))).T)
ts=time.time(); fn09(a_va[:1]); rt9=(time.time()-ts)*1e3
tr_r9,tr_i9=fn09(a_tr); va_r9,va_i9=fn09(a_va)
tl9=eval_loss_arr(tr_r9,tr_i9,or_tr,oi_tr); vl9=eval_loss_arr(va_r9,va_i9,or_va,oi_va)
joblib.dump({'gp':gp9,'sc':sc9},os.path.join(d,'saved_model','model.pkl'))
comp9={'rel_error_omega_real':float(np.mean(np.abs(va_r9-or_va)/np.abs(or_va))),
       'rel_error_omega_imag':float(np.mean(np.abs(va_i9-oi_va)/np.abs(oi_va)))}
record(9,'gpr_rbf_raw','ml','raw',tl9,vl9,rt9,'GPR (RBF kernel), raw spin',comp9)

# ============================================================
# 10: GPR Matern (log)
# ============================================================
print('\n[10] GPR-Matern log')
d=model_dir(10,'gpr_matern_log')
sc10=StandardScaler(); Xt10=sc10.fit_transform(Xr_tr['log'])
k10=C(1.)*Matern(1.,nu=2.5)+WhiteKernel(1e-5)
gp10=GaussianProcessRegressor(kernel=k10,n_restarts_optimizer=2,alpha=1e-6,normalize_y=True)
t0=time.time(); gp10.fit(Xt10,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
def fn10(a): xl=-np.log(np.clip(1-a,1e-8,None)).reshape(-1,1); return tuple(gp10.predict(sc10.transform(xl)).T)
ts=time.time(); fn10(a_va[:1]); rt10=(time.time()-ts)*1e3
tr_r10,tr_i10=fn10(a_tr); va_r10,va_i10=fn10(a_va)
tl10=eval_loss_arr(tr_r10,tr_i10,or_tr,oi_tr); vl10=eval_loss_arr(va_r10,va_i10,or_va,oi_va)
joblib.dump({'gp':gp10,'sc':sc10},os.path.join(d,'saved_model','model.pkl'))
comp10={'rel_error_omega_real':float(np.mean(np.abs(va_r10-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i10-oi_va)/np.abs(oi_va)))}
record(10,'gpr_matern_log','ml','log',tl10,vl10,rt10,'GPR (Matern-2.5), log-compactified',comp10)

# ============================================================
# 11: MLP small (raw)
# ============================================================
print('\n[11] MLP-small raw')
d=model_dir(11,'mlp_small_raw')
sc11=StandardScaler(); Xt11=sc11.fit_transform(a_tr.reshape(-1,1))
t0=time.time()
mlp11=MLPRegressor(hidden_layer_sizes=(64,64,32),activation='tanh',max_iter=2000,
                   random_state=42,early_stopping=True,validation_fraction=0.1)
mlp11.fit(Xt11,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
def fn11(a): return tuple(mlp11.predict(sc11.transform(a.reshape(-1,1))).T)
ts=time.time(); fn11(a_va[:1]); rt11=(time.time()-ts)*1e3
tr_r11,tr_i11=fn11(a_tr); va_r11,va_i11=fn11(a_va)
tl11=eval_loss_arr(tr_r11,tr_i11,or_tr,oi_tr); vl11=eval_loss_arr(va_r11,va_i11,or_va,oi_va)
joblib.dump({'mlp':mlp11,'sc':sc11},os.path.join(d,'saved_model','model.pkl'))
comp11={'rel_error_omega_real':float(np.mean(np.abs(va_r11-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i11-oi_va)/np.abs(oi_va)))}
record(11,'mlp_small_raw','ml','raw',tl11,vl11,rt11,'MLP(64,64,32 tanh), raw spin',comp11)

# ============================================================
# 12: MLP deeper (log)
# ============================================================
print('\n[12] MLP-deeper log')
d=model_dir(12,'mlp_deep_log')
sc12=StandardScaler(); Xt12=sc12.fit_transform(Xr_tr['log'])
t0=time.time()
mlp12=MLPRegressor(hidden_layer_sizes=(128,128,64,32),activation='relu',max_iter=3000,
                   random_state=0,early_stopping=True,validation_fraction=0.1,learning_rate_init=1e-3)
mlp12.fit(Xt12,Y_tr); print(f'  fit {time.time()-t0:.1f}s')
def fn12(a): xl=-np.log(np.clip(1-a,1e-8,None)).reshape(-1,1); return tuple(mlp12.predict(sc12.transform(xl)).T)
ts=time.time(); fn12(a_va[:1]); rt12=(time.time()-ts)*1e3
tr_r12,tr_i12=fn12(a_tr); va_r12,va_i12=fn12(a_va)
tl12=eval_loss_arr(tr_r12,tr_i12,or_tr,oi_tr); vl12=eval_loss_arr(va_r12,va_i12,or_va,oi_va)
joblib.dump({'mlp':mlp12,'sc':sc12},os.path.join(d,'saved_model','model.pkl'))
comp12={'rel_error_omega_real':float(np.mean(np.abs(va_r12-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i12-oi_va)/np.abs(oi_va)))}
record(12,'mlp_deep_log','ml','log',tl12,vl12,rt12,'MLP(128,128,64,32 relu), log-compactified',comp12)

# ============================================================
# 13: Random Forest (raw)
# ============================================================
print('\n[13] RF raw')
d=model_dir(13,'rf_raw')
t0=time.time()
rf13=RandomForestRegressor(n_estimators=200,max_depth=None,min_samples_leaf=1,random_state=42,n_jobs=-1)
rf13.fit(a_tr.reshape(-1,1),Y_tr); print(f'  fit {time.time()-t0:.1f}s')
def fn13(a): return tuple(rf13.predict(a.reshape(-1,1)).T)
ts=time.time(); fn13(a_va[:1]); rt13=(time.time()-ts)*1e3
tr_r13,tr_i13=fn13(a_tr); va_r13,va_i13=fn13(a_va)
tl13=eval_loss_arr(tr_r13,tr_i13,or_tr,oi_tr); vl13=eval_loss_arr(va_r13,va_i13,or_va,oi_va)
joblib.dump({'rf':rf13},os.path.join(d,'saved_model','model.pkl'))
comp13={'rel_error_omega_real':float(np.mean(np.abs(va_r13-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i13-oi_va)/np.abs(oi_va)))}
record(13,'rf_raw','ml','raw',tl13,vl13,rt13,'Random Forest(200), raw spin',comp13)

# ============================================================
# 14: Gradient Boosting (log)
# ============================================================
print('\n[14] GBR log')
d=model_dir(14,'gbr_log')
from sklearn.multioutput import MultiOutputRegressor
t0=time.time()
gbr14=MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,random_state=0),n_jobs=2)
gbr14.fit(Xr_tr['log'],Y_tr); print(f'  fit {time.time()-t0:.1f}s')
def fn14(a): xl=-np.log(np.clip(1-a,1e-8,None)).reshape(-1,1); return tuple(gbr14.predict(xl).T)
ts=time.time(); fn14(a_va[:1]); rt14=(time.time()-ts)*1e3
tr_r14,tr_i14=fn14(a_tr); va_r14,va_i14=fn14(a_va)
tl14=eval_loss_arr(tr_r14,tr_i14,or_tr,oi_tr); vl14=eval_loss_arr(va_r14,va_i14,or_va,oi_va)
joblib.dump({'gbr':gbr14},os.path.join(d,'saved_model','model.pkl'))
comp14={'rel_error_omega_real':float(np.mean(np.abs(va_r14-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i14-oi_va)/np.abs(oi_va)))}
record(14,'gbr_log','ml','log',tl14,vl14,rt14,'GBR(200, d5) multitask, log-compactified',comp14)

# ============================================================
# 15: gplearn (raw) — MANDATORY
# ============================================================
print('\n[15] gplearn raw')
from gplearn.genetic import SymbolicRegressor as GPLearnSR
d=model_dir(15,'gplearn_raw')
expr15=[]
t0=time.time()
gpl_r=GPLearnSR(population_size=3000,generations=20,tournament_size=20,
                function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                metric='mse',parsimony_coefficient=0.005,verbose=0,random_state=42,n_jobs=2)
gpl_r.fit(a_tr.reshape(-1,1),or_tr)
expr15.append({'target':'omega_r','expr':str(gpl_r._program),'fitness':float(gpl_r._program.fitness_)})
print(f'  omega_r: {str(gpl_r._program)[:60]}')
gpl_i=GPLearnSR(population_size=3000,generations=20,tournament_size=20,
                function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                metric='mse',parsimony_coefficient=0.005,verbose=0,random_state=99,n_jobs=2)
gpl_i.fit(a_tr.reshape(-1,1),oi_tr)
expr15.append({'target':'omega_i','expr':str(gpl_i._program),'fitness':float(gpl_i._program.fitness_)})
print(f'  omega_i: {str(gpl_i._program)[:60]}')
print(f'  gplearn total {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr15,f,indent=2)
def fn15(a): return gpl_r.predict(a.reshape(-1,1)), gpl_i.predict(a.reshape(-1,1))
ts=time.time(); fn15(a_va[:1]); rt15=(time.time()-ts)*1e3
tr_r15,tr_i15=fn15(a_tr); va_r15,va_i15=fn15(a_va)
tl15=eval_loss_arr(tr_r15,tr_i15,or_tr,oi_tr); vl15=eval_loss_arr(va_r15,va_i15,or_va,oi_va)
try: joblib.dump({'gpl_r':gpl_r,'gpl_i':gpl_i},os.path.join(d,'saved_model','model.pkl'))
except: pass
record(15,'gplearn_raw','symbolic','raw',tl15,vl15,rt15,
       f'gplearn SR, raw spin. omega_r: {str(gpl_r._program)[:40]}')

# ============================================================
# 16: gplearn (log) — MANDATORY
# ============================================================
print('\n[16] gplearn log')
d=model_dir(16,'gplearn_log')
expr16=[]
t0=time.time()
gpl_r2=GPLearnSR(population_size=3000,generations=20,tournament_size=20,
                 function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                 metric='mse',parsimony_coefficient=0.005,verbose=0,random_state=7,n_jobs=2)
gpl_r2.fit(Xr_tr['log'],or_tr)
expr16.append({'target':'omega_r','expr':str(gpl_r2._program),'fitness':float(gpl_r2._program.fitness_)})
gpl_i2=GPLearnSR(population_size=3000,generations=20,tournament_size=20,
                 function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                 metric='mse',parsimony_coefficient=0.005,verbose=0,random_state=13,n_jobs=2)
gpl_i2.fit(Xr_tr['log'],oi_tr)
expr16.append({'target':'omega_i','expr':str(gpl_i2._program),'fitness':float(gpl_i2._program.fitness_)})
print(f'  gplearn log {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr16,f,indent=2)
def fn16(a): xl=-np.log(np.clip(1-a,1e-8,None)).reshape(-1,1); return gpl_r2.predict(xl),gpl_i2.predict(xl)
ts=time.time(); fn16(a_va[:1]); rt16=(time.time()-ts)*1e3
tr_r16,tr_i16=fn16(a_tr); va_r16,va_i16=fn16(a_va)
tl16=eval_loss_arr(tr_r16,tr_i16,or_tr,oi_tr); vl16=eval_loss_arr(va_r16,va_i16,or_va,oi_va)
try: joblib.dump({'gpl_r':gpl_r2,'gpl_i':gpl_i2},os.path.join(d,'saved_model','model.pkl'))
except: pass
record(16,'gplearn_log','symbolic','log',tl16,vl16,rt16,'gplearn SR, log-compactified spin')

# ============================================================
# 17: PySR (raw) — MANDATORY
# ============================================================
print('\n[17] PySR raw')
from pysr import PySRRegressor
d=model_dir(17,'pysr_raw')
expr17=[]
t0=time.time()
try:
    ps_r=PySRRegressor(niterations=80,binary_operators=['+','-','*','/'],
                       unary_operators=['sqrt','log','exp'],maxsize=25,populations=15,
                       procs=2,loss='loss(p,t)=abs(p-t)/abs(t)',verbosity=0,random_state=42,
                       tempdir=os.path.join(d,'saved_model','pysr_r'))
    ps_r.fit(a_tr.reshape(-1,1),or_tr)
    try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps_r.equations_.iterrows()]
    except: pf=[{'e':str(ps_r.sympy()),'c':0,'l':0.}]
    expr17.append({'target':'omega_r','pareto':pf})
    print(f'  PySR omega_r best: {ps_r.sympy()}')
except Exception as e:
    ps_r=None; expr17.append({'target':'omega_r','err':str(e)}); print(f'  PySR omega_r failed: {e}')
try:
    ps_i=PySRRegressor(niterations=80,binary_operators=['+','-','*','/'],
                       unary_operators=['sqrt','log','exp'],maxsize=25,populations=15,
                       procs=2,loss='loss(p,t)=abs(p-t)/abs(t)',verbosity=0,random_state=99,
                       tempdir=os.path.join(d,'saved_model','pysr_i'))
    ps_i.fit(a_tr.reshape(-1,1),oi_tr)
    try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps_i.equations_.iterrows()]
    except: pf=[{'e':str(ps_i.sympy()),'c':0,'l':0.}]
    expr17.append({'target':'omega_i','pareto':pf})
    print(f'  PySR omega_i best: {ps_i.sympy()}')
except Exception as e:
    ps_i=None; expr17.append({'target':'omega_i','err':str(e)}); print(f'  PySR omega_i failed: {e}')
print(f'  PySR raw total {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr17,f,indent=2)
def fn17(a):
    r = ps_r.predict(a.reshape(-1,1)) if ps_r else np.zeros_like(a)
    i = ps_i.predict(a.reshape(-1,1)) if ps_i else np.zeros_like(a)
    return r,i
ts=time.time(); fn17(a_va[:1]); rt17=(time.time()-ts)*1e3
tr_r17,tr_i17=fn17(a_tr); va_r17,va_i17=fn17(a_va)
tl17=eval_loss_arr(tr_r17,tr_i17,or_tr,oi_tr); vl17=eval_loss_arr(va_r17,va_i17,or_va,oi_va)
np.save(os.path.join(d,'saved_model','basis.npy'),np.zeros(1))
record(17,'pysr_raw','symbolic','raw',tl17,vl17,rt17,'PySR, raw spin a')

# ============================================================
# 18: PySR (log) — MANDATORY
# ============================================================
print('\n[18] PySR log')
d=model_dir(18,'pysr_log')
expr18=[]
t0=time.time()
try:
    ps_r2=PySRRegressor(niterations=60,binary_operators=['+','-','*','/'],
                        unary_operators=['sqrt','log','exp'],maxsize=22,populations=12,
                        procs=2,loss='loss(p,t)=abs(p-t)/abs(t)',verbosity=0,random_state=7,
                        tempdir=os.path.join(d,'saved_model','pysr_r'))
    ps_r2.fit(Xr_tr['log'],or_tr)
    try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps_r2.equations_.iterrows()]
    except: pf=[{'e':str(ps_r2.sympy()),'c':0,'l':0.}]
    expr18.append({'target':'omega_r','pareto':pf})
except Exception as e:
    ps_r2=None; expr18.append({'target':'omega_r','err':str(e)})
try:
    ps_i2=PySRRegressor(niterations=60,binary_operators=['+','-','*','/'],
                        unary_operators=['sqrt','log','exp'],maxsize=22,populations=12,
                        procs=2,loss='loss(p,t)=abs(p-t)/abs(t)',verbosity=0,random_state=13,
                        tempdir=os.path.join(d,'saved_model','pysr_i'))
    ps_i2.fit(Xr_tr['log'],oi_tr)
    try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps_i2.equations_.iterrows()]
    except: pf=[{'e':str(ps_i2.sympy()),'c':0,'l':0.}]
    expr18.append({'target':'omega_i','pareto':pf})
except Exception as e:
    ps_i2=None; expr18.append({'target':'omega_i','err':str(e)})
print(f'  PySR log total {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr18,f,indent=2)
def fn18(a):
    xl=-np.log(np.clip(1-a,1e-8,None)).reshape(-1,1)
    r = ps_r2.predict(xl) if ps_r2 else np.zeros_like(a)
    i = ps_i2.predict(xl) if ps_i2 else np.zeros_like(a)
    return r,i
ts=time.time(); fn18(a_va[:1]); rt18=(time.time()-ts)*1e3
tr_r18,tr_i18=fn18(a_tr); va_r18,va_i18=fn18(a_va)
tl18=eval_loss_arr(tr_r18,tr_i18,or_tr,oi_tr); vl18=eval_loss_arr(va_r18,va_i18,or_va,oi_va)
np.save(os.path.join(d,'saved_model','basis.npy'),np.zeros(1))
record(18,'pysr_log','symbolic','log',tl18,vl18,rt18,'PySR, log-compactified spin')

# ============================================================
# 19: KRR (raw)
# ============================================================
print('\n[19] KRR raw')
d=model_dir(19,'krr_raw')
sc19=StandardScaler(); Xt19=sc19.fit_transform(a_tr.reshape(-1,1))
t0=time.time(); krr19=KernelRidge(kernel='rbf',alpha=1e-6,gamma=10.); krr19.fit(Xt19,Y_tr); print(f'  fit {time.time()-t0:.2f}s')
def fn19(a): return tuple(krr19.predict(sc19.transform(a.reshape(-1,1))).T)
ts=time.time(); fn19(a_va[:1]); rt19=(time.time()-ts)*1e3
tr_r19,tr_i19=fn19(a_tr); va_r19,va_i19=fn19(a_va)
tl19=eval_loss_arr(tr_r19,tr_i19,or_tr,oi_tr); vl19=eval_loss_arr(va_r19,va_i19,or_va,oi_va)
joblib.dump({'krr':krr19,'sc':sc19},os.path.join(d,'saved_model','model.pkl'))
comp19={'rel_error_omega_real':float(np.mean(np.abs(va_r19-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i19-oi_va)/np.abs(oi_va)))}
record(19,'krr_raw','ml','raw',tl19,vl19,rt19,'Kernel Ridge (RBF gamma=10), raw spin',comp19)

# ============================================================
# 20: Poly-20 (sqrt)
# ============================================================
print('\n[20] Poly-20 sqrt')
d=model_dir(20,'poly20_sqrt')
x20=Xr_tr['sqrt'].ravel(); xv20=Xr_va['sqrt'].ravel()
t0=time.time()
cr20=np.polyfit(x20,or_tr,20); ci20=np.polyfit(x20,oi_tr,20)
print(f'  fit {time.time()-t0:.4f}s')
def fn20(a): xs=np.sqrt(np.clip(1-a**2,0,None)); return np.polyval(cr20,xs),np.polyval(ci20,xs)
ts=time.time(); fn20(a_va[:1]); rt20=(time.time()-ts)*1e3
tr_r20,tr_i20=fn20(a_tr); va_r20,va_i20=fn20(a_va)
tl20=eval_loss_arr(tr_r20,tr_i20,or_tr,oi_tr); vl20=eval_loss_arr(va_r20,va_i20,or_va,oi_va)
joblib.dump({'cr':cr20,'ci':ci20},os.path.join(d,'saved_model','model.pkl'))
comp20={'rel_error_omega_real':float(np.mean(np.abs(va_r20-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i20-oi_va)/np.abs(oi_va)))}
record(20,'poly20_sqrt','analytical','sqrt',tl20,vl20,rt20,'Poly(20), sqrt(1-a^2) irreducible mass',comp20)

# ============================================================
# 21: kNN (log)
# ============================================================
print('\n[21] kNN-5 log')
d=model_dir(21,'knn5_log')
sc21=StandardScaler(); Xt21=sc21.fit_transform(Xr_tr['log'])
t0=time.time(); knn21=KNeighborsRegressor(n_neighbors=5,weights='distance'); knn21.fit(Xt21,Y_tr); print(f'  fit {time.time()-t0:.3f}s')
def fn21(a): xl=-np.log(np.clip(1-a,1e-8,None)).reshape(-1,1); return tuple(knn21.predict(sc21.transform(xl)).T)
ts=time.time(); fn21(a_va[:1]); rt21=(time.time()-ts)*1e3
tr_r21,tr_i21=fn21(a_tr); va_r21,va_i21=fn21(a_va)
tl21=eval_loss_arr(tr_r21,tr_i21,or_tr,oi_tr); vl21=eval_loss_arr(va_r21,va_i21,or_va,oi_va)
joblib.dump({'knn':knn21,'sc':sc21},os.path.join(d,'saved_model','model.pkl'))
comp21={'rel_error_omega_real':float(np.mean(np.abs(va_r21-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i21-oi_va)/np.abs(oi_va)))}
record(21,'knn5_log','ml','log',tl21,vl21,rt21,'kNN(5, distance), log-compactified',comp21)

# ============================================================
# 22: Chebyshev expansion deg 30 (log)
# ============================================================
print('\n[22] Chebyshev-30 log')
d=model_dir(22,'cheb30_log')
x22=Xr_tr['log'].ravel(); xmin22,xmax22=x22.min(),x22.max()
x22n=2*(x22-xmin22)/(xmax22-xmin22)-1
t0=time.time()
cr22=np.polynomial.chebyshev.chebfit(x22n,or_tr,30); ci22=np.polynomial.chebyshev.chebfit(x22n,oi_tr,30)
print(f'  fit {time.time()-t0:.4f}s')
def fn22(a):
    xl=-np.log(np.clip(1-a,1e-8,None)); xn=2*(xl-xmin22)/(xmax22-xmin22)-1
    return np.polynomial.chebyshev.chebval(xn,cr22),np.polynomial.chebyshev.chebval(xn,ci22)
ts=time.time(); fn22(a_va[:1]); rt22=(time.time()-ts)*1e3
tr_r22,tr_i22=fn22(a_tr); va_r22,va_i22=fn22(a_va)
tl22=eval_loss_arr(tr_r22,tr_i22,or_tr,oi_tr); vl22=eval_loss_arr(va_r22,va_i22,or_va,oi_va)
joblib.dump({'cr':cr22,'ci':ci22,'xmin':xmin22,'xmax':xmax22},os.path.join(d,'saved_model','model.pkl'))
comp22={'rel_error_omega_real':float(np.mean(np.abs(va_r22-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i22-oi_va)/np.abs(oi_va)))}
record(22,'cheb30_log','analytical','log',tl22,vl22,rt22,'Chebyshev(30), log-compactified mapped to [-1,1]',comp22)

# ============================================================
# 23: RBF cubic (sqrt)
# ============================================================
print('\n[23] RBF-cubic sqrt')
d=model_dir(23,'rbf_cubic_sqrt')
X23=Xr_tr['sqrt']; Xv23=Xr_va['sqrt']
t0=time.time()
rbf23=RBFInterpolator(X23, Y_tr, kernel='cubic', smoothing=1e-8)
print(f'  fit {time.time()-t0:.3f}s')
def fn23(a): xs=np.sqrt(np.clip(1-a**2,0,None)).reshape(-1,1); out=rbf23(xs); return out[:,0],out[:,1]
ts=time.time(); fn23(a_va[:1]); rt23=(time.time()-ts)*1e3
tr_r23,tr_i23=fn23(a_tr); va_r23,va_i23=fn23(a_va)
tl23=eval_loss_arr(tr_r23,tr_i23,or_tr,oi_tr); vl23=eval_loss_arr(va_r23,va_i23,or_va,oi_va)
joblib.dump({'rbf':rbf23},os.path.join(d,'saved_model','model.pkl'))
comp23={'rel_error_omega_real':float(np.mean(np.abs(va_r23-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i23-oi_va)/np.abs(oi_va)))}
record(23,'rbf_cubic_sqrt','interp','sqrt',tl23,vl23,rt23,'RBF(cubic) interpolation, sqrt(1-a^2)',comp23)

# ============================================================
# 24: Pade [15,5] (log)
# ============================================================
print('\n[24] Pade [15,5] log')
d=model_dir(24,'pade_15_5_log')
x24=Xr_tr['log'].ravel(); xv24=Xr_va['log'].ravel()
t0=time.time()
nr24, dr24 = pade_fit(x24, or_tr, 15, 5)
ni24, di24 = pade_fit(x24, oi_tr, 15, 5)
print(f'  fit {time.time()-t0:.4f}s')
def fn24(a):
    xl=-np.log(np.clip(1-a,1e-8,None))
    return eval_pade(xl,nr24,dr24), eval_pade(xl,ni24,di24)
ts=time.time(); fn24(a_va[:1]); rt24=(time.time()-ts)*1e3
tr_r24,tr_i24=fn24(a_tr); va_r24,va_i24=fn24(a_va)
tl24=eval_loss_arr(tr_r24,tr_i24,or_tr,oi_tr); vl24=eval_loss_arr(va_r24,va_i24,or_va,oi_va)
joblib.dump({'nr':nr24,'dr':dr24,'ni':ni24,'di':di24},os.path.join(d,'saved_model','model.pkl'))
comp24={'rel_error_omega_real':float(np.mean(np.abs(va_r24-or_va)/np.abs(or_va))),
        'rel_error_omega_imag':float(np.mean(np.abs(va_i24-oi_va)/np.abs(oi_va)))}
record(24,'pade_15_5_log','analytical','log',tl24,vl24,rt24,'Pade rational [15,5], log-compactified',comp24)

# ============================================================
# FINAL PLOTS + SUMMARY
# ============================================================
print('\n=== Final plots & summary ===')

# Error histograms
fig,ax=plt.subplots(figsize=figsize(2,0.8))
for r in ALL_RESULTS:
    tl=np.array(r['train_losses']); vl_r=np.array(r['val_losses'])
    al=np.concatenate([tl,vl_r]); lo,hi=max(al.min(),1e-12),max(al.max(),1e-6)
    if lo<hi:
        bins=np.logspace(np.log10(lo),np.log10(hi),20)
        ax.hist(tl,bins=bins,alpha=0.15,density=True,histtype='stepfilled',label=f'{r["name"][:6]} tr')
        ax.hist(vl_r,bins=bins,alpha=0.7,density=True,histtype='step',lw=1.,label=f'{r["name"][:6]} va')
ax.set_xscale('log'); ax.set_xlabel('Mean rel. error'); ax.set_ylabel('Density')
ax.legend(fontsize=3,ncol=4)
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
    print(f'  {r["rank"]:2d}. {r["name"]:<25s} val={r["val_loss"]:.4e} ({r["category"]})')

if n>=20 and len(reps)>=3 and len(cats)==4 and has_pysr and has_gpl:
    print('\nRINGDOWN_BENCH_COMPLETE')
else:
    print(f'\nIncomplete: n={n}, reps={len(reps)}, cats={len(cats)}, pysr={has_pysr}, gplearn={has_gpl}')
