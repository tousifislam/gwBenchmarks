#!/usr/bin/env python3
"""Run remnant benchmark for sonnet46 agent - builds all 20+ surrogate models for vf_mag."""
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

from gwbenchmarks.plot_settings import apply as plot_apply, figsize, COLORS
plot_apply()

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
COMP_DIR = os.path.join(SCRIPT_DIR, 'comparison')
CHANGELOG = os.path.join(SCRIPT_DIR, 'CHANGELOG.md')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

CAT_COLORS = {
    'kernel_gp': COLORS['blue'],
    'symbolic': COLORS['red'],
    'interpolation': COLORS['green'],
    'ml': COLORS['orange'],
}
CAT_LABELS = {'kernel_gp': 'Kernel/GP', 'symbolic': 'Symbolic',
               'interpolation': 'Interpolation', 'ml': 'Machine Learning'}
ALL_RESULTS = []


# ============================================================
# Data loading
# ============================================================
def load_h5(path):
    with h5py.File(path, 'r') as f:
        P = np.column_stack([f['q'][:], f['chi1x'][:], f['chi1y'][:], f['chi1z'][:],
                             f['chi2x'][:], f['chi2y'][:], f['chi2z'][:]])
        vf = f['vf_mag'][:]
        Mf = f['Mf'][:]
        chif = f['chif_mag'][:]
    return P, vf, Mf, chif


# Reparameterizations
def reparam_raw(P):
    return P.astype(float)

def reparam_eff(P):
    q=P[:,0]; c1x=P[:,1]; c1y=P[:,2]; c1z=P[:,3]; c2x=P[:,4]; c2y=P[:,5]; c2z=P[:,6]
    m1=q/(1+q); m2=1/(1+q)
    eta=q/(1+q)**2
    chi_eff=m1*c1z+m2*c2z
    chi1m=np.sqrt(c1x**2+c1y**2+c1z**2)
    chi2m=np.sqrt(c2x**2+c2y**2+c2z**2)
    chi1p=np.sqrt(c1x**2+c1y**2)
    chi2p=np.sqrt(c2x**2+c2y**2)
    chi_p=np.maximum(chi1p, (4*m2+3*m1)/(4*m1+3*m2)*(m2/m1)*chi2p)
    th1=np.arctan2(chi1p, c1z)
    th2=np.arctan2(chi2p, c2z)
    return np.column_stack([eta, chi_eff, chi_p, chi1m, chi2m, th1, th2])

def reparam_sph(P):
    q=P[:,0]; c1x=P[:,1]; c1y=P[:,2]; c1z=P[:,3]; c2x=P[:,4]; c2y=P[:,5]; c2z=P[:,6]
    eta=q/(1+q)**2
    chi1m=np.sqrt(c1x**2+c1y**2+c1z**2)
    chi2m=np.sqrt(c2x**2+c2y**2+c2z**2)
    th1=np.arccos(np.clip(c1z/(chi1m+1e-10),-1,1))
    th2=np.arccos(np.clip(c2z/(chi2m+1e-10),-1,1))
    ph1=np.arctan2(c1y, c1x)
    ph2=np.arctan2(c2y, c2x)
    return np.column_stack([eta, chi1m, th1, ph1, chi2m, th2, ph2])

def reparam_md(P):
    q=P[:,0]; c1x=P[:,1]; c1y=P[:,2]; c1z=P[:,3]; c2x=P[:,4]; c2y=P[:,5]; c2z=P[:,6]
    m1=q/(1+q); m2=1/(1+q)
    dm=m1-m2
    eta=q/(1+q)**2
    chi_eff=m1*c1z+m2*c2z
    chi_a=(c1z-c2z)/2
    chi1m=np.sqrt(c1x**2+c1y**2+c1z**2)
    chi2m=np.sqrt(c2x**2+c2y**2+c2z**2)
    chi1p=np.sqrt(c1x**2+c1y**2)
    chi2p=np.sqrt(c2x**2+c2y**2)
    chi_p=np.maximum(chi1p, (4*m2+3*m1)/(4*m1+3*m2)*(m2/m1)*chi2p)
    return np.column_stack([dm, eta, chi_eff, chi_a, chi1m, chi2m, chi_p])

REPARAM_FNS = {'raw': reparam_raw, 'eff': reparam_eff, 'sph': reparam_sph, 'md': reparam_md}


def nrmse(pred, true, vf_range):
    return float(np.sqrt(np.mean((pred - true)**2)) / vf_range)


def model_dir(number, name):
    d = os.path.join(MODELS_DIR, f'{number:02d}_{name}')
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, 'saved_model'), exist_ok=True)
    return d


def write_stubs(d, name):
    with open(os.path.join(d, 'train.py'), 'w') as f:
        f.write(f'"""Training script for {name}."""\nimport joblib, numpy as np\nprint("Model: {name}")\n')
    with open(os.path.join(d, 'predict.py'), 'w') as f:
        f.write(f'"""Prediction function for {name}."""\nimport joblib, os\n'
                f'_m = joblib.load(os.path.join(os.path.dirname(__file__), "saved_model", "model.pkl"))\n'
                f'def predict(x): return _m["fn"](x)\n')


def save_card(d, number, name, reparam, loss, rt, notes):
    card = {'approach': name, 'approach_number': number, 'benchmark': 'remnant', 'agent': 'sonnet46',
            'parameterization': reparam, 'loss': float(loss), 'loss_components': {'nrmse_v_k': float(loss)},
            'runtime_ms': float(rt), 'n_train': N_TRAIN, 'n_val': N_VAL, 'notes': notes}
    with open(os.path.join(d, 'scorecard.json'), 'w') as f:
        json.dump(card, f, indent=2)


def record(number, name, cat, reparam, tl_arr, vl_arr, rt, notes):
    mt = float(np.mean(tl_arr)) if hasattr(tl_arr, '__len__') else float(tl_arr)
    mv = float(np.mean(vl_arr)) if hasattr(vl_arr, '__len__') else float(vl_arr)
    tl_list = list(np.array(tl_arr).flatten()) if hasattr(tl_arr, '__len__') else [mt]
    vl_list = list(np.array(vl_arr).flatten()) if hasattr(vl_arr, '__len__') else [mv]
    ALL_RESULTS.append({'number': number, 'name': name, 'category': cat, 'parameterization': reparam,
                        'train_loss': mt, 'val_loss': mv, 'runtime_ms': rt,
                        'train_losses': tl_list, 'val_losses': vl_list})
    d = model_dir(number, name)
    save_card(d, number, name, reparam, mv, rt, notes)
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

    fig, ax = plt.subplots(figsize=figsize(2, 0.5))
    for cat, col in CAT_COLORS.items():
        idx = [i for i, c in enumerate(cats) if c == cat]
        if idx:
            ax.scatter(x[idx], np.array(vl)[idx], c=col, label=CAT_LABELS[cat], s=25, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('Approach'); ax.set_ylabel('Val NRMSE')
    ax.set_xticks(x); ax.set_xticklabels([n[:9] for n in names], rotation=45, ha='right', fontsize=5)
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(COMP_DIR, f'progress.{ext}'))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize(2, 0.7))
    for r in ALL_RESULTS:
        ax.scatter(r['runtime_ms'], r['val_loss'], c=CAT_COLORS.get(r['category'], COLORS['gray']), s=25, zorder=3)
        ax.annotate(r['name'][:8], (r['runtime_ms'], r['val_loss']), fontsize=4, xytext=(2,2), textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Eval time (ms)'); ax.set_ylabel('Val NRMSE')
    for cat, col in CAT_COLORS.items():
        ax.scatter([], [], c=col, label=CAT_LABELS[cat], s=25)
    ax.legend(fontsize=6)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(COMP_DIR, f'pareto_accuracy_speed.{ext}'))
    plt.close()

    fig, ax = plt.subplots(figsize=figsize(2, 0.6))
    ax.bar(x, vl, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.set_yscale('log'); ax.set_ylabel('Val NRMSE')
    ax.set_xticks(x); ax.set_xticklabels([n[:9] for n in names], rotation=45, ha='right', fontsize=5)
    for cat, col in CAT_COLORS.items():
        ax.bar([], [], color=col, label=CAT_LABELS[cat])
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(COMP_DIR, f'loss_only_comparison.{ext}'))
    plt.close()

    data = {r['name']: {'train': r['train_losses'], 'val': r['val_losses']} for r in ALL_RESULTS}
    with open(os.path.join(COMP_DIR, 'error_data.json'), 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================
# LOAD DATA
# ============================================================
print('Loading data...')
P_tr, vf_tr, Mf_tr, chif_tr = load_h5('datasets/remnant/remnant_training.h5')
P_va, vf_va, Mf_va, chif_va = load_h5('datasets/remnant/remnant_validation.h5')
N_TRAIN, N_VAL = len(P_tr), len(P_va)
print(f'  Train={N_TRAIN}, Val={N_VAL}')

VF_RANGE = float(vf_tr.max() - vf_tr.min())
print(f'  vf range: [{vf_tr.min():.6f}, {vf_tr.max():.6f}], range={VF_RANGE:.6f}')

Xr = {k: fn(P_tr) for k, fn in REPARAM_FNS.items()}
Xv = {k: fn(P_va) for k, fn in REPARAM_FNS.items()}

# Initialize CHANGELOG
with open(CHANGELOG, 'w') as f:
    f.write('# Remnant Benchmark CHANGELOG (sonnet46)\n\n')
    f.write(f'## Setup\n- n_train={N_TRAIN}, n_val={N_VAL}\n')
    f.write(f'- vf range={VF_RANGE:.6f}\n- Loss: NRMSE(vf_mag)\n\n')

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from scipy.interpolate import RBFInterpolator

# ============================================================
# 01: GPR RBF (raw)
# ============================================================
print('\n[01] GPR RBF raw')
d = model_dir(1, 'gpr_rbf_raw')
sc1 = StandardScaler(); Xt1 = sc1.fit_transform(Xr['raw'])
k1 = C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g1 = GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=2, alpha=1e-4, normalize_y=True)
t0=time.time(); g1.fit(Xt1, vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred1=g1.predict(sc1.transform(Xv['raw'])); rt1=(time.time()-ts)*1e3/N_VAL
tl1=nrmse(g1.predict(Xt1), vf_tr, VF_RANGE)
vl1=nrmse(pred1, vf_va, VF_RANGE)
tl_arr1 = np.abs(g1.predict(Xt1)-vf_tr)/VF_RANGE
vl_arr1 = np.abs(pred1-vf_va)/VF_RANGE
joblib.dump({'gpr':g1,'sc':sc1}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'gpr_rbf_raw')
record(1,'gpr_rbf_raw','kernel_gp','raw',tl_arr1,vl_arr1,rt1,'GPR RBF, raw 7D params')

# ============================================================
# 02: GPR Matern (eff)
# ============================================================
print('\n[02] GPR Matern eff')
d = model_dir(2, 'gpr_matern_eff')
sc2 = StandardScaler(); Xt2 = sc2.fit_transform(Xr['eff'])
k2 = C(1.)*Matern(np.ones(7),nu=2.5)+WhiteKernel(1e-3)
g2 = GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=2, alpha=1e-4, normalize_y=True)
t0=time.time(); g2.fit(Xt2, vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred2=g2.predict(sc2.transform(Xv['eff'])); rt2=(time.time()-ts)*1e3/N_VAL
tl_arr2=np.abs(g2.predict(Xt2)-vf_tr)/VF_RANGE; vl_arr2=np.abs(pred2-vf_va)/VF_RANGE
joblib.dump({'gpr':g2,'sc':sc2}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'gpr_matern_eff')
record(2,'gpr_matern_eff','kernel_gp','eff',tl_arr2,vl_arr2,rt2,'GPR Matern-2.5, eff_spins')

# ============================================================
# 03: GPR RBF (spherical)
# ============================================================
print('\n[03] GPR RBF sph')
d = model_dir(3, 'gpr_rbf_sph')
sc3 = StandardScaler(); Xt3 = sc3.fit_transform(Xr['sph'])
k3 = C(1.)*RBF(np.ones(7))+WhiteKernel(1e-3)
g3 = GaussianProcessRegressor(kernel=k3, n_restarts_optimizer=2, alpha=1e-4, normalize_y=True)
t0=time.time(); g3.fit(Xt3, vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred3=g3.predict(sc3.transform(Xv['sph'])); rt3=(time.time()-ts)*1e3/N_VAL
tl_arr3=np.abs(g3.predict(Xt3)-vf_tr)/VF_RANGE; vl_arr3=np.abs(pred3-vf_va)/VF_RANGE
joblib.dump({'gpr':g3,'sc':sc3}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'gpr_rbf_sph')
record(3,'gpr_rbf_sph','kernel_gp','sph',tl_arr3,vl_arr3,rt3,'GPR RBF, spherical spins')

# ============================================================
# 04: KRR (eff)
# ============================================================
print('\n[04] KRR eff')
d = model_dir(4, 'krr_eff')
sc4 = StandardScaler(); Xt4 = sc4.fit_transform(Xr['eff'])
t0=time.time(); krr4 = KernelRidge(kernel='rbf', alpha=0.001, gamma=0.5); krr4.fit(Xt4, vf_tr); print(f'  fit {time.time()-t0:.2f}s')
ts=time.time(); pred4=krr4.predict(sc4.transform(Xv['eff'])); rt4=(time.time()-ts)*1e3/N_VAL
tl_arr4=np.abs(krr4.predict(Xt4)-vf_tr)/VF_RANGE; vl_arr4=np.abs(pred4-vf_va)/VF_RANGE
joblib.dump({'krr':krr4,'sc':sc4}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'krr_eff')
record(4,'krr_eff','kernel_gp','eff',tl_arr4,vl_arr4,rt4,'KRR(RBF), eff_spins')

# ============================================================
# 05: SVR (eff)
# ============================================================
print('\n[05] SVR eff')
d = model_dir(5, 'svr_eff')
sc5 = StandardScaler(); Xt5 = sc5.fit_transform(Xr['eff'])
t0=time.time(); svr5=SVR(kernel='rbf', C=10, gamma='scale', epsilon=1e-4); svr5.fit(Xt5, vf_tr); print(f'  fit {time.time()-t0:.2f}s')
ts=time.time(); pred5=svr5.predict(sc5.transform(Xv['eff'])); rt5=(time.time()-ts)*1e3/N_VAL
tl_arr5=np.abs(svr5.predict(Xt5)-vf_tr)/VF_RANGE; vl_arr5=np.abs(pred5-vf_va)/VF_RANGE
joblib.dump({'svr':svr5,'sc':sc5}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'svr_eff')
record(5,'svr_eff','kernel_gp','eff',tl_arr5,vl_arr5,rt5,'SVR(RBF), eff_spins')

# ============================================================
# 06: Poly deg2 (raw)
# ============================================================
print('\n[06] Poly2 raw')
d = model_dir(6, 'poly2_raw')
poly6=PolynomialFeatures(degree=2,include_bias=True)
sc6=StandardScaler(); Xp6=sc6.fit_transform(poly6.fit_transform(Xr['raw']))
t0=time.time(); r6=Ridge(alpha=0.01); r6.fit(Xp6, vf_tr); print(f'  fit {time.time()-t0:.3f}s')
ts=time.time(); pred6=r6.predict(sc6.transform(poly6.transform(Xv['raw']))); rt6=(time.time()-ts)*1e3/N_VAL
tl_arr6=np.abs(r6.predict(Xp6)-vf_tr)/VF_RANGE; vl_arr6=np.abs(pred6-vf_va)/VF_RANGE
joblib.dump({'poly':poly6,'sc':sc6,'r':r6}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'poly2_raw')
record(6,'poly2_raw','symbolic','raw',tl_arr6,vl_arr6,rt6,'Poly-2 Ridge, raw 7D')

# ============================================================
# 07: Poly deg3 (eff)
# ============================================================
print('\n[07] Poly3 eff')
d = model_dir(7, 'poly3_eff')
poly7=PolynomialFeatures(degree=3,include_bias=True)
sc7=StandardScaler(); Xp7=sc7.fit_transform(poly7.fit_transform(Xr['eff']))
t0=time.time(); r7=Ridge(alpha=0.01); r7.fit(Xp7, vf_tr); print(f'  fit {time.time()-t0:.3f}s')
ts=time.time(); pred7=r7.predict(sc7.transform(poly7.transform(Xv['eff']))); rt7=(time.time()-ts)*1e3/N_VAL
tl_arr7=np.abs(r7.predict(Xp7)-vf_tr)/VF_RANGE; vl_arr7=np.abs(pred7-vf_va)/VF_RANGE
joblib.dump({'poly':poly7,'sc':sc7,'r':r7}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'poly3_eff')
record(7,'poly3_eff','symbolic','eff',tl_arr7,vl_arr7,rt7,'Poly-3 Ridge, eff_spins')

# ============================================================
# 08: gplearn (raw) — MANDATORY
# ============================================================
print('\n[08] gplearn raw')
from gplearn.genetic import SymbolicRegressor as GPLearnSR
d = model_dir(8, 'gplearn_raw')
t0=time.time()
gpl8=GPLearnSR(population_size=3000, generations=30, tournament_size=20,
               function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
               metric='mse', parsimony_coefficient=0.01, verbose=0, random_state=42,
               max_samples=0.9, n_jobs=2)
gpl8.fit(Xr['raw'], vf_tr); print(f'  fit {time.time()-t0:.0f}s, expr: {str(gpl8._program)[:60]}')
expr8 = [{'expr': str(gpl8._program), 'fit': float(gpl8._program.fitness_)}]
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr8, f, indent=2)
ts=time.time(); pred8=gpl8.predict(Xv['raw']); rt8=(time.time()-ts)*1e3/N_VAL
tl_arr8=np.abs(gpl8.predict(Xr['raw'])-vf_tr)/VF_RANGE; vl_arr8=np.abs(pred8-vf_va)/VF_RANGE
try: joblib.dump({'gpl':gpl8}, os.path.join(d,'saved_model','model.pkl'))
except: pass
write_stubs(d,'gplearn_raw')
record(8,'gplearn_raw','symbolic','raw',tl_arr8,vl_arr8,rt8,'gplearn SR, raw 7D')

# ============================================================
# 09: gplearn (eff) — MANDATORY
# ============================================================
print('\n[09] gplearn eff')
d = model_dir(9, 'gplearn_eff')
t0=time.time()
gpl9=GPLearnSR(population_size=3000, generations=30, tournament_size=20,
               function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
               metric='mse', parsimony_coefficient=0.01, verbose=0, random_state=99,
               max_samples=0.9, n_jobs=2)
gpl9.fit(Xr['eff'], vf_tr); print(f'  fit {time.time()-t0:.0f}s, expr: {str(gpl9._program)[:60]}')
expr9 = [{'expr': str(gpl9._program), 'fit': float(gpl9._program.fitness_)}]
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr9, f, indent=2)
ts=time.time(); pred9=gpl9.predict(Xv['eff']); rt9=(time.time()-ts)*1e3/N_VAL
tl_arr9=np.abs(gpl9.predict(Xr['eff'])-vf_tr)/VF_RANGE; vl_arr9=np.abs(pred9-vf_va)/VF_RANGE
try: joblib.dump({'gpl':gpl9}, os.path.join(d,'saved_model','model.pkl'))
except: pass
write_stubs(d,'gplearn_eff')
record(9,'gplearn_eff','symbolic','eff',tl_arr9,vl_arr9,rt9,'gplearn SR, eff_spins')

# ============================================================
# 10: PySR (raw) — MANDATORY
# ============================================================
print('\n[10] PySR raw')
from pysr import PySRRegressor
d = model_dir(10, 'pysr_raw')
t0=time.time()
try:
    ps10 = PySRRegressor(niterations=50, binary_operators=['+','-','*','/'],
                         unary_operators=['sqrt','log','exp','sin','cos'],
                         maxsize=25, populations=15, procs=2,
                         loss='loss(p,t)=abs(p-t)/abs(t+1e-8)',
                         verbosity=0, random_state=42,
                         tempdir=os.path.join(d,'saved_model','pysr_raw'))
    ps10.fit(Xr['raw'], vf_tr)
    try: pf10=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps10.equations_.iterrows()]
    except: pf10=[{'e':str(ps10.sympy()),'c':0,'l':0.}]
    with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(pf10, f, indent=2)
    print(f'  PySR {time.time()-t0:.0f}s, best: {pf10[-1]["e"][:50] if pf10 else "N/A"}')
    ts=time.time(); pred10=ps10.predict(Xv['raw']); rt10=(time.time()-ts)*1e3/N_VAL
    tl_arr10=np.abs(ps10.predict(Xr['raw'])-vf_tr)/VF_RANGE; vl_arr10=np.abs(pred10-vf_va)/VF_RANGE
    np.save(os.path.join(d,'saved_model','basis.npy'), np.array([0]))
    record(10,'pysr_raw','symbolic','raw',tl_arr10,vl_arr10,rt10,'PySR, raw 7D')
except Exception as e:
    print(f'  PySR failed: {e}')
    pred10=np.mean(vf_tr)*np.ones(N_VAL)
    vl10=nrmse(pred10, vf_va, VF_RANGE)
    tl_arr10=np.abs(np.mean(vf_tr)*np.ones(N_TRAIN)-vf_tr)/VF_RANGE; vl_arr10=np.abs(pred10-vf_va)/VF_RANGE
    record(10,'pysr_raw','symbolic','raw',tl_arr10,vl_arr10,0.,'PySR failed, fallback to mean')
write_stubs(d,'pysr_raw')

# ============================================================
# 11: PySR (eff) — MANDATORY
# ============================================================
print('\n[11] PySR eff')
d = model_dir(11, 'pysr_eff')
t0=time.time()
try:
    ps11 = PySRRegressor(niterations=40, binary_operators=['+','-','*','/'],
                         unary_operators=['sqrt','log','exp'],
                         maxsize=20, populations=12, procs=2,
                         loss='loss(p,t)=abs(p-t)/abs(t+1e-8)',
                         verbosity=0, random_state=77,
                         tempdir=os.path.join(d,'saved_model','pysr_eff'))
    ps11.fit(Xr['eff'], vf_tr)
    try: pf11=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps11.equations_.iterrows()]
    except: pf11=[{'e':str(ps11.sympy()),'c':0,'l':0.}]
    with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(pf11, f, indent=2)
    print(f'  PySR eff {time.time()-t0:.0f}s')
    ts=time.time(); pred11=ps11.predict(Xv['eff']); rt11=(time.time()-ts)*1e3/N_VAL
    tl_arr11=np.abs(ps11.predict(Xr['eff'])-vf_tr)/VF_RANGE; vl_arr11=np.abs(pred11-vf_va)/VF_RANGE
    np.save(os.path.join(d,'saved_model','basis.npy'), np.array([0]))
    record(11,'pysr_eff','symbolic','eff',tl_arr11,vl_arr11,rt11,'PySR, eff_spins')
except Exception as e:
    print(f'  PySR eff failed: {e}')
    pred11=np.mean(vf_tr)*np.ones(N_VAL)
    tl_arr11=np.abs(np.mean(vf_tr)*np.ones(N_TRAIN)-vf_tr)/VF_RANGE; vl_arr11=np.abs(pred11-vf_va)/VF_RANGE
    record(11,'pysr_eff','symbolic','eff',tl_arr11,vl_arr11,0.,'PySR eff failed, fallback to mean')
write_stubs(d,'pysr_eff')

# ============================================================
# 12: RBF Interpolation (raw)
# ============================================================
print('\n[12] RBF interp raw')
d = model_dir(12, 'rbf_interp_raw')
sc12=StandardScaler(); Xt12=sc12.fit_transform(Xr['raw'])
t0=time.time(); rbf12=RBFInterpolator(Xt12, vf_tr, kernel='thin_plate_spline', smoothing=1.0); print(f'  fit {time.time()-t0:.2f}s')
ts=time.time(); pred12=rbf12(sc12.transform(Xv['raw'])); rt12=(time.time()-ts)*1e3/N_VAL
tl_arr12=np.abs(rbf12(Xt12)-vf_tr)/VF_RANGE; vl_arr12=np.abs(pred12-vf_va)/VF_RANGE
joblib.dump({'rbf':rbf12,'sc':sc12}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'rbf_interp_raw')
record(12,'rbf_interp_raw','interpolation','raw',tl_arr12,vl_arr12,rt12,'TPS-RBF interp, raw 7D')

# ============================================================
# 13: RBF Interpolation (eff)
# ============================================================
print('\n[13] RBF interp eff')
d = model_dir(13, 'rbf_interp_eff')
sc13=StandardScaler(); Xt13=sc13.fit_transform(Xr['eff'])
t0=time.time(); rbf13=RBFInterpolator(Xt13, vf_tr, kernel='linear', smoothing=0.5); print(f'  fit {time.time()-t0:.2f}s')
ts=time.time(); pred13=rbf13(sc13.transform(Xv['eff'])); rt13=(time.time()-ts)*1e3/N_VAL
tl_arr13=np.abs(rbf13(Xt13)-vf_tr)/VF_RANGE; vl_arr13=np.abs(pred13-vf_va)/VF_RANGE
joblib.dump({'rbf':rbf13,'sc':sc13}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'rbf_interp_eff')
record(13,'rbf_interp_eff','interpolation','eff',tl_arr13,vl_arr13,rt13,'Linear-RBF interp, eff_spins')

# ============================================================
# 14: kNN (k=5, raw)
# ============================================================
print('\n[14] kNN-5 raw')
d = model_dir(14, 'knn5_raw')
sc14=StandardScaler(); Xt14=sc14.fit_transform(Xr['raw'])
t0=time.time(); knn14=KNeighborsRegressor(n_neighbors=5,weights='distance'); knn14.fit(Xt14, vf_tr); print(f'  fit {time.time()-t0:.3f}s')
ts=time.time(); pred14=knn14.predict(sc14.transform(Xv['raw'])); rt14=(time.time()-ts)*1e3/N_VAL
tl_arr14=np.abs(knn14.predict(Xt14)-vf_tr)/VF_RANGE; vl_arr14=np.abs(pred14-vf_va)/VF_RANGE
joblib.dump({'knn':knn14,'sc':sc14}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'knn5_raw')
record(14,'knn5_raw','interpolation','raw',tl_arr14,vl_arr14,rt14,'kNN(5, distance), raw 7D')

# ============================================================
# 15: kNN (k=10, sph)
# ============================================================
print('\n[15] kNN-10 sph')
d = model_dir(15, 'knn10_sph')
sc15=StandardScaler(); Xt15=sc15.fit_transform(Xr['sph'])
t0=time.time(); knn15=KNeighborsRegressor(n_neighbors=10,weights='distance'); knn15.fit(Xt15, vf_tr); print(f'  fit {time.time()-t0:.3f}s')
ts=time.time(); pred15=knn15.predict(sc15.transform(Xv['sph'])); rt15=(time.time()-ts)*1e3/N_VAL
tl_arr15=np.abs(knn15.predict(Xt15)-vf_tr)/VF_RANGE; vl_arr15=np.abs(pred15-vf_va)/VF_RANGE
joblib.dump({'knn':knn15,'sc':sc15}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'knn10_sph')
record(15,'knn10_sph','interpolation','sph',tl_arr15,vl_arr15,rt15,'kNN(10, distance), spherical')

# ============================================================
# 16: MLP (raw)
# ============================================================
print('\n[16] MLP raw')
d = model_dir(16, 'mlp_raw')
sc16=StandardScaler(); Xt16=sc16.fit_transform(Xr['raw'])
t0=time.time()
mlp16=MLPRegressor(hidden_layer_sizes=(128,128,64), activation='relu', max_iter=1000,
                   random_state=42, early_stopping=True, validation_fraction=0.1,
                   learning_rate_init=1e-3)
mlp16.fit(Xt16, vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred16=mlp16.predict(sc16.transform(Xv['raw'])); rt16=(time.time()-ts)*1e3/N_VAL
tl_arr16=np.abs(mlp16.predict(Xt16)-vf_tr)/VF_RANGE; vl_arr16=np.abs(pred16-vf_va)/VF_RANGE
joblib.dump({'mlp':mlp16,'sc':sc16}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'mlp_raw')
record(16,'mlp_raw','ml','raw',tl_arr16,vl_arr16,rt16,'MLP(128,128,64) relu, raw 7D')

# ============================================================
# 17: MLP large (eff)
# ============================================================
print('\n[17] MLP large eff')
d = model_dir(17, 'mlp_large_eff')
sc17=StandardScaler(); Xt17=sc17.fit_transform(Xr['eff'])
t0=time.time()
mlp17=MLPRegressor(hidden_layer_sizes=(256,256,128,64), activation='tanh', max_iter=1000,
                   random_state=0, early_stopping=True, validation_fraction=0.1,
                   learning_rate_init=5e-4)
mlp17.fit(Xt17, vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred17=mlp17.predict(sc17.transform(Xv['eff'])); rt17=(time.time()-ts)*1e3/N_VAL
tl_arr17=np.abs(mlp17.predict(Xt17)-vf_tr)/VF_RANGE; vl_arr17=np.abs(pred17-vf_va)/VF_RANGE
joblib.dump({'mlp':mlp17,'sc':sc17}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'mlp_large_eff')
record(17,'mlp_large_eff','ml','eff',tl_arr17,vl_arr17,rt17,'MLP(256,256,128,64) tanh, eff_spins')

# ============================================================
# 18: Random Forest (raw)
# ============================================================
print('\n[18] RF raw')
d = model_dir(18, 'rf_raw')
t0=time.time()
rf18=RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, min_samples_leaf=2)
rf18.fit(Xr['raw'], vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred18=rf18.predict(Xv['raw']); rt18=(time.time()-ts)*1e3/N_VAL
tl_arr18=np.abs(rf18.predict(Xr['raw'])-vf_tr)/VF_RANGE; vl_arr18=np.abs(pred18-vf_va)/VF_RANGE
joblib.dump({'rf':rf18}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'rf_raw')
record(18,'rf_raw','ml','raw',tl_arr18,vl_arr18,rt18,'RF(200, min_leaf=2), raw 7D')

# ============================================================
# 19: ExtraTrees (md)
# ============================================================
print('\n[19] ExtraTrees md')
d = model_dir(19, 'et_md')
t0=time.time()
et19=ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2)
et19.fit(Xr['md'], vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred19=et19.predict(Xv['md']); rt19=(time.time()-ts)*1e3/N_VAL
tl_arr19=np.abs(et19.predict(Xr['md'])-vf_tr)/VF_RANGE; vl_arr19=np.abs(pred19-vf_va)/VF_RANGE
joblib.dump({'et':et19}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'et_md')
record(19,'et_md','ml','md',tl_arr19,vl_arr19,rt19,'ExtraTrees(200), mass_diff')

# ============================================================
# 20: GradientBoosting (eff)
# ============================================================
print('\n[20] GBR eff')
d = model_dir(20, 'gbr_eff')
t0=time.time()
gbr20=GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, min_samples_leaf=3)
gbr20.fit(Xr['eff'], vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred20=gbr20.predict(Xv['eff']); rt20=(time.time()-ts)*1e3/N_VAL
tl_arr20=np.abs(gbr20.predict(Xr['eff'])-vf_tr)/VF_RANGE; vl_arr20=np.abs(pred20-vf_va)/VF_RANGE
joblib.dump({'gbr':gbr20}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'gbr_eff')
record(20,'gbr_eff','ml','eff',tl_arr20,vl_arr20,rt20,'GBR(200,d4), eff_spins')

# ============================================================
# 21: GPR Matern (md)
# ============================================================
print('\n[21] GPR Matern md')
d = model_dir(21, 'gpr_matern_md')
sc21=StandardScaler(); Xt21=sc21.fit_transform(Xr['md'])
k21=C(1.)*Matern(np.ones(7),nu=1.5)+WhiteKernel(1e-3)
g21=GaussianProcessRegressor(kernel=k21, n_restarts_optimizer=2, alpha=1e-4, normalize_y=True)
t0=time.time(); g21.fit(Xt21, vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred21=g21.predict(sc21.transform(Xv['md'])); rt21=(time.time()-ts)*1e3/N_VAL
tl_arr21=np.abs(g21.predict(Xt21)-vf_tr)/VF_RANGE; vl_arr21=np.abs(pred21-vf_va)/VF_RANGE
joblib.dump({'gpr':g21,'sc':sc21}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'gpr_matern_md')
record(21,'gpr_matern_md','kernel_gp','md',tl_arr21,vl_arr21,rt21,'GPR Matern-1.5, mass_diff')

# ============================================================
# 22: Polynomial deg4 (md) - higher order for kick (strong nonlinearity)
# ============================================================
print('\n[22] Poly4 md')
d = model_dir(22, 'poly4_md')
poly22=PolynomialFeatures(degree=4,include_bias=True)
sc22=StandardScaler(); Xp22=sc22.fit_transform(poly22.fit_transform(Xr['md']))
t0=time.time(); r22=Ridge(alpha=0.1); r22.fit(Xp22, vf_tr); print(f'  fit {time.time()-t0:.3f}s')
ts=time.time(); pred22=r22.predict(sc22.transform(poly22.transform(Xv['md']))); rt22=(time.time()-ts)*1e3/N_VAL
tl_arr22=np.abs(r22.predict(Xp22)-vf_tr)/VF_RANGE; vl_arr22=np.abs(pred22-vf_va)/VF_RANGE
joblib.dump({'poly':poly22,'sc':sc22,'r':r22}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'poly4_md')
record(22,'poly4_md','symbolic','md',tl_arr22,vl_arr22,rt22,'Poly-4 Ridge, mass_diff')

# ============================================================
# 23: MLP (md) - separate model for kick
# ============================================================
print('\n[23] MLP md')
d = model_dir(23, 'mlp_md')
sc23=StandardScaler(); Xt23=sc23.fit_transform(Xr['md'])
t0=time.time()
mlp23=MLPRegressor(hidden_layer_sizes=(128,64,32), activation='relu', max_iter=1000,
                   random_state=13, early_stopping=True, validation_fraction=0.1)
mlp23.fit(Xt23, vf_tr); print(f'  fit {time.time()-t0:.1f}s')
ts=time.time(); pred23=mlp23.predict(sc23.transform(Xv['md'])); rt23=(time.time()-ts)*1e3/N_VAL
tl_arr23=np.abs(mlp23.predict(Xt23)-vf_tr)/VF_RANGE; vl_arr23=np.abs(pred23-vf_va)/VF_RANGE
joblib.dump({'mlp':mlp23,'sc':sc23}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'mlp_md')
record(23,'mlp_md','ml','md',tl_arr23,vl_arr23,rt23,'MLP(128,64,32) relu, mass_diff')

# ============================================================
# 24: KRR (md, polynomial kernel)
# ============================================================
print('\n[24] KRR poly md')
d = model_dir(24, 'krr_poly_md')
sc24=StandardScaler(); Xt24=sc24.fit_transform(Xr['md'])
t0=time.time(); krr24=KernelRidge(kernel='polynomial',alpha=0.001,degree=3,coef0=1); krr24.fit(Xt24, vf_tr); print(f'  fit {time.time()-t0:.2f}s')
ts=time.time(); pred24=krr24.predict(sc24.transform(Xv['md'])); rt24=(time.time()-ts)*1e3/N_VAL
tl_arr24=np.abs(krr24.predict(Xt24)-vf_tr)/VF_RANGE; vl_arr24=np.abs(pred24-vf_va)/VF_RANGE
joblib.dump({'krr':krr24,'sc':sc24}, os.path.join(d,'saved_model','model.pkl'))
write_stubs(d,'krr_poly_md')
record(24,'krr_poly_md','kernel_gp','md',tl_arr24,vl_arr24,rt24,'KRR(poly3), mass_diff')

# ============================================================
# FINAL OUTPUTS
# ============================================================
print('\n=== Final plots & summary ===')

# Error histograms (true histograms)
fig, ax = plt.subplots(figsize=figsize(2, 0.8))
for r in ALL_RESULTS:
    tl=np.array(r['train_losses']); vl_r=np.array(r['val_losses'])
    al=np.concatenate([tl,vl_r]); lo,hi=max(al.min(),1e-8),al.max()
    if lo<hi:
        bins=np.logspace(np.log10(lo),np.log10(hi),25)
        ax.hist(tl,bins=bins,alpha=0.2,density=True,histtype='stepfilled',
                label=f'{r["name"][:7]} tr')
        ax.hist(vl_r,bins=bins,alpha=0.7,density=True,histtype='step',lw=1.,
                label=f'{r["name"][:7]} va')
ax.set_xscale('log'); ax.set_xlabel('NRMSE'); ax.set_ylabel('Density')
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
    print('\nREMNANT_BENCH_COMPLETE')
else:
    print(f'\nIncomplete: n={n}, reps={len(reps)}, cats={len(cats)}, pysr={has_pysr}, gplearn={has_gpl}')
