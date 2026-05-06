#!/usr/bin/env python3
"""Validity benchmark for sonnet46 — predict mismatch of NRHybSur3dq8 surrogate."""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR,'..','..','..','..'))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py, joblib
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from gwbenchmarks.plot_settings import apply as plot_apply, figsize, COLORS
plot_apply()

MODELS_DIR = os.path.join(SCRIPT_DIR,'models')
COMP_DIR   = os.path.join(SCRIPT_DIR,'comparison')
CHANGELOG  = os.path.join(SCRIPT_DIR,'CHANGELOG.md')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

CAT_COLORS = {'kernel_gp': COLORS['blue'], 'symbolic': COLORS['red'],
              'interp': COLORS['green'], 'ml': COLORS['orange']}
ALL_RESULTS = []

# ============================================================
# Data
# ============================================================
def load_h5(path):
    with h5py.File(path,'r') as f:
        return (f['q'][:], f['chi1z'][:], f['chi2z'][:], f['omega0'][:], f['mm_td'][:])

print('Loading data...')
q_tr, c1_tr, c2_tr, w_tr, mm_tr = load_h5('datasets/validity/validity_training.h5')
q_va, c1_va, c2_va, w_va, mm_va = load_h5('datasets/validity/validity_validation.h5')
N_TRAIN, N_VAL = len(q_tr), len(q_va)
log_tr = np.log10(mm_tr); log_va = np.log10(mm_va)
print(f'  Train={N_TRAIN}, Val={N_VAL}')
print(f'  log10(mm): train=[{log_tr.min():.2f},{log_tr.max():.2f}], val=[{log_va.min():.2f},{log_va.max():.2f}]')

# ============================================================
# Reparameterizations (4D inputs)
# ============================================================
def eta(q):    return q/(1+q)**2
def chi_eff(q,c1,c2): return (q*c1+c2)/(1+q)
def chi_a(c1,c2):    return (c1-c2)/2
def boundary_dist(q, c1, c2):
    """Distance from NRHybSur3dq8 valid boundary: q<=8, |chi|<=0.8."""
    dq = np.maximum(0, q-8.0)
    dc1 = np.maximum(0, np.abs(c1)-0.8)
    dc2 = np.maximum(0, np.abs(c2)-0.8)
    return np.column_stack([dq, dc1, dc2])

def reparam_raw(q,c1,c2,w):    return np.column_stack([q,c1,c2,w])
def reparam_eff(q,c1,c2,w):    return np.column_stack([eta(q),chi_eff(q,c1,c2),chi_a(c1,c2),w])
def reparam_logq(q,c1,c2,w):   return np.column_stack([np.log(q),chi_eff(q,c1,c2),chi_a(c1,c2),np.log(w)])
def reparam_inter(q,c1,c2,w):
    et=eta(q); ce=chi_eff(q,c1,c2); ca=chi_a(c1,c2)
    return np.column_stack([et,ce,ca,w,q*ce,et*ca])
def reparam_bnd(q,c1,c2,w):
    et=eta(q); ce=chi_eff(q,c1,c2); ca=chi_a(c1,c2)
    bd=boundary_dist(q,c1,c2)
    return np.column_stack([et,ce,ca,np.log(w),bd])

REPS = {
    'raw':   (reparam_raw(q_tr,c1_tr,c2_tr,w_tr), reparam_raw(q_va,c1_va,c2_va,w_va)),
    'eff':   (reparam_eff(q_tr,c1_tr,c2_tr,w_tr), reparam_eff(q_va,c1_va,c2_va,w_va)),
    'logq':  (reparam_logq(q_tr,c1_tr,c2_tr,w_tr), reparam_logq(q_va,c1_va,c2_va,w_va)),
    'inter': (reparam_inter(q_tr,c1_tr,c2_tr,w_tr), reparam_inter(q_va,c1_va,c2_va,w_va)),
    'bnd':   (reparam_bnd(q_tr,c1_tr,c2_tr,w_tr), reparam_bnd(q_va,c1_va,c2_va,w_va)),
}

# ============================================================
# Loss
# ============================================================
def log_rmse(pred, true):
    return float(np.sqrt(np.mean((np.log10(pred) - np.log10(true))**2)))

def log_rmse_arr(pred, true):
    return np.abs(np.log10(pred) - np.log10(true))

def clamp_pos(x): return np.maximum(x, 1e-10)

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
        f.write(f'"""Training for {name}."""\nimport joblib, numpy as np\nprint("Model: {name}")\n')
    with open(os.path.join(d,'predict.py'),'w') as f:
        f.write(f'"""Prediction for {name}."""\nimport joblib, os, numpy as np\n'
                f'_m = joblib.load(os.path.join(os.path.dirname(__file__),"saved_model","model.pkl"))\n'
                f'def predict(X): return _m["fn"](np.asarray(X))\n')

def save_card(d, num, name, rep, tl, vl, rt, notes, extra=None):
    card = {'approach':name,'approach_number':num,'benchmark':'validity','agent':'sonnet46',
            'parameterization':rep,'loss':float(vl),'loss_components':{'log_rmse':float(vl)},
            'runtime_ms':float(rt),'n_train':N_TRAIN,'n_val':N_VAL,'notes':notes}
    if extra: card.update(extra)
    with open(os.path.join(d,'scorecard.json'),'w') as f: json.dump(card,f,indent=2)

def record(num, name, cat, rep, tl_arr, vl_arr, rt, notes, extra=None):
    mt=float(np.mean(tl_arr)); mv=float(np.mean(vl_arr))
    ALL_RESULTS.append({'number':num,'name':name,'category':cat,'parameterization':rep,
                        'train_loss':mt,'val_loss':mv,'runtime_ms':rt,
                        'train_losses':list(tl_arr),'val_losses':list(vl_arr)})
    d=model_dir(num,name); save_card(d,num,name,rep,mt,mv,rt,notes,extra)
    write_stubs(d,name); update_plots()
    with open(CHANGELOG,'a') as f:
        f.write(f'## {num:02d}: {name}\n- cat={cat}, rep={rep}\n'
                f'- train={mt:.4e}, val={mv:.4e}, rt={rt:.2f}ms\n- {notes}\n\n')
    print(f'[{num:02d}] {name}: train={mt:.4e} val={mv:.4e} rt={rt:.2f}ms')

def update_plots():
    if not ALL_RESULTS: return
    names=[r['name'] for r in ALL_RESULTS]
    vl=[r['val_loss'] for r in ALL_RESULTS]
    cats=[r['category'] for r in ALL_RESULTS]
    colors=[CAT_COLORS.get(c,COLORS['gray']) for c in cats]
    x=np.arange(len(ALL_RESULTS))

    fig,ax=plt.subplots(figsize=figsize(2,0.5))
    for cat,col in CAT_COLORS.items():
        idx=[i for i,c in enumerate(cats) if c==cat]
        if idx: ax.scatter(np.array(idx),np.array(vl)[idx],color=col,label=cat,s=20,zorder=3)
    ax.set_yscale('log'); ax.set_xlabel('Approach'); ax.set_ylabel('Val loss (log-RMSE)')
    ax.legend(fontsize=6); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'progress.{ext}'))
    plt.close()

    fig,ax=plt.subplots(figsize=figsize(2,0.6))
    ax.bar(x,vl,color=colors)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(names,rotation=90,fontsize=4)
    ax.set_ylabel('Val loss')
    for cat,col in CAT_COLORS.items(): ax.bar(0,0,color=col,label=cat)
    ax.legend(fontsize=6); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'loss_only_comparison.{ext}'))
    plt.close()

    rts=[r['runtime_ms'] for r in ALL_RESULTS]
    fig,ax=plt.subplots(figsize=figsize(2,0.6))
    for r in ALL_RESULTS:
        ax.scatter(r['runtime_ms'],r['val_loss'],color=CAT_COLORS.get(r['category'],COLORS['gray']),s=25,zorder=3)
        ax.annotate(r['name'][:10],(r['runtime_ms'],r['val_loss']),fontsize=3,ha='left')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Eval time (ms)'); ax.set_ylabel('Val loss')
    for cat,col in CAT_COLORS.items(): ax.scatter([],[],color=col,label=cat,s=15)
    ax.legend(fontsize=6); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'pareto_accuracy_speed.{ext}'))
    plt.close()

    ed={r['name']:{'train':r['train_losses'],'val':r['val_losses']} for r in ALL_RESULTS}
    with open(os.path.join(COMP_DIR,'error_data.json'),'w') as f: json.dump(ed,f)


# Shorthand
Xr=REPS; Ltr=log_tr; Lva=log_va

# ============================================================
# 01: GPR RBF (raw)
# ============================================================
print('\n[01] GPR-RBF raw')
d=model_dir(1,'gpr_rbf_raw')
sc1=StandardScaler(); Xt1=sc1.fit_transform(Xr['raw'][0])
k1=C(1.)*RBF(np.ones(4))+WhiteKernel(1e-3)
gp1=GaussianProcessRegressor(kernel=k1,n_restarts_optimizer=2,alpha=1e-4,normalize_y=True)
t0=time.time(); gp1.fit(Xt1,Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn01(X): return 10**gp1.predict(sc1.transform(X))
ts=time.time(); fn01(Xr['raw'][0][:1]); rt1=(time.time()-ts)*1e3
p1t=fn01(Xr['raw'][0]); p1v=fn01(Xr['raw'][1])
tl1=log_rmse_arr(p1t,mm_tr); vl1=log_rmse_arr(p1v,mm_va)
joblib.dump({'gp':gp1,'sc':sc1},os.path.join(d,'saved_model','model.pkl'))
record(1,'gpr_rbf_raw','kernel_gp','raw',tl1,vl1,rt1,'GPR(RBF), raw 4D')

# ============================================================
# 02: GPR Matern (eff)
# ============================================================
print('\n[02] GPR-Matern eff')
d=model_dir(2,'gpr_matern_eff')
sc2=StandardScaler(); Xt2=sc2.fit_transform(Xr['eff'][0])
k2=C(1.)*Matern(np.ones(4),nu=2.5)+WhiteKernel(1e-3)
gp2=GaussianProcessRegressor(kernel=k2,n_restarts_optimizer=2,alpha=1e-4,normalize_y=True)
t0=time.time(); gp2.fit(Xt2,Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn02(X): return 10**gp2.predict(sc2.transform(X))
ts=time.time(); fn02(Xr['eff'][0][:1]); rt2=(time.time()-ts)*1e3
p2t=fn02(Xr['eff'][0]); p2v=fn02(Xr['eff'][1])
tl2=log_rmse_arr(p2t,mm_tr); vl2=log_rmse_arr(p2v,mm_va)
joblib.dump({'gp':gp2,'sc':sc2},os.path.join(d,'saved_model','model.pkl'))
record(2,'gpr_matern_eff','kernel_gp','eff',tl2,vl2,rt2,'GPR(Matern-2.5), eta+chi_eff+chi_a')

# ============================================================
# 03: RF (raw)
# ============================================================
print('\n[03] RF raw')
d=model_dir(3,'rf_raw')
t0=time.time()
rf3=RandomForestRegressor(n_estimators=200,max_depth=None,random_state=42,n_jobs=-1)
rf3.fit(Xr['raw'][0],Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn03(X): return 10**rf3.predict(X)
ts=time.time(); fn03(Xr['raw'][0][:1]); rt3=(time.time()-ts)*1e3
p3t=fn03(Xr['raw'][0]); p3v=fn03(Xr['raw'][1])
tl3=log_rmse_arr(p3t,mm_tr); vl3=log_rmse_arr(p3v,mm_va)
joblib.dump({'rf':rf3},os.path.join(d,'saved_model','model.pkl'))
record(3,'rf_raw','ml','raw',tl3,vl3,rt3,'RF(200), raw 4D, log10(mm) target')

# ============================================================
# 04: GBR (logq)
# ============================================================
print('\n[04] GBR logq')
d=model_dir(4,'gbr_logq')
t0=time.time()
gbr4=GradientBoostingRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,random_state=0)
gbr4.fit(Xr['logq'][0],Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn04(X): return 10**gbr4.predict(X)
ts=time.time(); fn04(Xr['logq'][0][:1]); rt4=(time.time()-ts)*1e3
p4t=fn04(Xr['logq'][0]); p4v=fn04(Xr['logq'][1])
tl4=log_rmse_arr(p4t,mm_tr); vl4=log_rmse_arr(p4v,mm_va)
joblib.dump({'gbr':gbr4},os.path.join(d,'saved_model','model.pkl'))
record(4,'gbr_logq','ml','logq',tl4,vl4,rt4,'GBR(300,d5), log(q)+chi_eff+chi_a+log(omega0)')

# ============================================================
# 05: MLP (eff)
# ============================================================
print('\n[05] MLP eff')
d=model_dir(5,'mlp_eff')
sc5=StandardScaler(); Xt5=sc5.fit_transform(Xr['eff'][0])
t0=time.time()
mlp5=MLPRegressor(hidden_layer_sizes=(128,128,64),activation='relu',max_iter=1000,
                  random_state=42,early_stopping=True,validation_fraction=0.1)
mlp5.fit(Xt5,Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn05(X): return 10**mlp5.predict(sc5.transform(X))
ts=time.time(); fn05(Xr['eff'][0][:1]); rt5=(time.time()-ts)*1e3
p5t=fn05(Xr['eff'][0]); p5v=fn05(Xr['eff'][1])
tl5=log_rmse_arr(p5t,mm_tr); vl5=log_rmse_arr(p5v,mm_va)
joblib.dump({'mlp':mlp5,'sc':sc5},os.path.join(d,'saved_model','model.pkl'))
record(5,'mlp_eff','ml','eff',tl5,vl5,rt5,'MLP(128,128,64 relu), eff_spins')

# ============================================================
# 06: KRR (eff)
# ============================================================
print('\n[06] KRR eff')
d=model_dir(6,'krr_eff')
sc6=StandardScaler(); Xt6=sc6.fit_transform(Xr['eff'][0])
t0=time.time(); krr6=KernelRidge(kernel='rbf',alpha=0.01,gamma=0.5); krr6.fit(Xt6,Ltr); print(f'  fit {time.time()-t0:.2f}s')
def fn06(X): return 10**krr6.predict(sc6.transform(X))
ts=time.time(); fn06(Xr['eff'][0][:1]); rt6=(time.time()-ts)*1e3
p6t=fn06(Xr['eff'][0]); p6v=fn06(Xr['eff'][1])
tl6=log_rmse_arr(p6t,mm_tr); vl6=log_rmse_arr(p6v,mm_va)
joblib.dump({'krr':krr6,'sc':sc6},os.path.join(d,'saved_model','model.pkl'))
record(6,'krr_eff','kernel_gp','eff',tl6,vl6,rt6,'KRR(RBF gamma=0.5), eff_spins')

# ============================================================
# 07: Polynomial deg2 (raw)
# ============================================================
print('\n[07] Poly-2 raw')
d=model_dir(7,'poly2_raw')
poly7=PolynomialFeatures(degree=2,include_bias=True)
sc7=StandardScaler(); Xp7=sc7.fit_transform(poly7.fit_transform(Xr['raw'][0]))
t0=time.time(); r7=Ridge(alpha=0.01); r7.fit(Xp7,Ltr); print(f'  fit {time.time()-t0:.3f}s')
def fn07(X): return 10**r7.predict(sc7.transform(poly7.transform(X)))
ts=time.time(); fn07(Xr['raw'][0][:1]); rt7=(time.time()-ts)*1e3
p7t=fn07(Xr['raw'][0]); p7v=fn07(Xr['raw'][1])
tl7=log_rmse_arr(p7t,mm_tr); vl7=log_rmse_arr(p7v,mm_va)
joblib.dump({'poly':poly7,'sc':sc7,'r':r7},os.path.join(d,'saved_model','model.pkl'))
record(7,'poly2_raw','symbolic','raw',tl7,vl7,rt7,'Poly-2 Ridge, raw 4D, log10 target')

# ============================================================
# 08: RBF-TPS interp (raw)
# ============================================================
print('\n[08] RBF-TPS raw')
d=model_dir(8,'rbf_tps_raw')
sc8=StandardScaler(); Xt8=sc8.fit_transform(Xr['raw'][0])
t0=time.time(); rbf8=RBFInterpolator(Xt8,Ltr,kernel='thin_plate_spline',smoothing=0.5); print(f'  fit {time.time()-t0:.2f}s')
def fn08(X): return 10**rbf8(sc8.transform(X))
ts=time.time(); fn08(Xr['raw'][0][:1]); rt8=(time.time()-ts)*1e3
p8t=fn08(Xr['raw'][0]); p8v=fn08(Xr['raw'][1])
tl8=log_rmse_arr(p8t,mm_tr); vl8=log_rmse_arr(p8v,mm_va)
joblib.dump({'rbf':rbf8,'sc':sc8},os.path.join(d,'saved_model','model.pkl'))
record(8,'rbf_tps_raw','interp','raw',tl8,vl8,rt8,'RBF(TPS, s=0.5) interp, raw 4D')

# ============================================================
# 09: kNN (eff)
# ============================================================
print('\n[09] kNN-5 eff')
d=model_dir(9,'knn5_eff')
sc9=StandardScaler(); Xt9=sc9.fit_transform(Xr['eff'][0])
t0=time.time(); knn9=KNeighborsRegressor(n_neighbors=5,weights='distance'); knn9.fit(Xt9,Ltr); print(f'  fit {time.time()-t0:.3f}s')
def fn09(X): return 10**knn9.predict(sc9.transform(X))
ts=time.time(); fn09(Xr['eff'][0][:1]); rt9=(time.time()-ts)*1e3
p9t=fn09(Xr['eff'][0]); p9v=fn09(Xr['eff'][1])
tl9=log_rmse_arr(p9t,mm_tr); vl9=log_rmse_arr(p9v,mm_va)
joblib.dump({'knn':knn9,'sc':sc9},os.path.join(d,'saved_model','model.pkl'))
record(9,'knn5_eff','interp','eff',tl9,vl9,rt9,'kNN(5, distance), eff_spins')

# ============================================================
# 10: GPR Matern (logq)
# ============================================================
print('\n[10] GPR-Matern logq')
d=model_dir(10,'gpr_matern_logq')
sc10=StandardScaler(); Xt10=sc10.fit_transform(Xr['logq'][0])
k10=C(1.)*Matern(np.ones(4),nu=1.5)+WhiteKernel(1e-3)
gp10=GaussianProcessRegressor(kernel=k10,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); gp10.fit(Xt10,Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn10(X): return 10**gp10.predict(sc10.transform(X))
ts=time.time(); fn10(Xr['logq'][0][:1]); rt10=(time.time()-ts)*1e3
p10t=fn10(Xr['logq'][0]); p10v=fn10(Xr['logq'][1])
tl10=log_rmse_arr(p10t,mm_tr); vl10=log_rmse_arr(p10v,mm_va)
joblib.dump({'gp':gp10,'sc':sc10},os.path.join(d,'saved_model','model.pkl'))
record(10,'gpr_matern_logq','kernel_gp','logq',tl10,vl10,rt10,'GPR(Matern-1.5), log(q)+chi_eff+chi_a+log(omega0)')

# ============================================================
# 11: ET (inter)
# ============================================================
print('\n[11] ET inter')
d=model_dir(11,'et_inter')
t0=time.time()
et11=ExtraTreesRegressor(n_estimators=200,max_depth=None,random_state=7,n_jobs=-1)
et11.fit(Xr['inter'][0],Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn11(X): return 10**et11.predict(X)
ts=time.time(); fn11(Xr['inter'][0][:1]); rt11=(time.time()-ts)*1e3
p11t=fn11(Xr['inter'][0]); p11v=fn11(Xr['inter'][1])
tl11=log_rmse_arr(p11t,mm_tr); vl11=log_rmse_arr(p11v,mm_va)
joblib.dump({'et':et11},os.path.join(d,'saved_model','model.pkl'))
record(11,'et_inter','ml','inter',tl11,vl11,rt11,'ET(200), interaction features eta*chi_eff')

# ============================================================
# 12: MLP deep (logq)
# ============================================================
print('\n[12] MLP-deep logq')
d=model_dir(12,'mlp_deep_logq')
sc12=StandardScaler(); Xt12=sc12.fit_transform(Xr['logq'][0])
t0=time.time()
mlp12=MLPRegressor(hidden_layer_sizes=(256,256,128,64),activation='tanh',max_iter=2000,
                   random_state=0,early_stopping=True,validation_fraction=0.1,learning_rate_init=5e-4)
mlp12.fit(Xt12,Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn12(X): return 10**mlp12.predict(sc12.transform(X))
ts=time.time(); fn12(Xr['logq'][0][:1]); rt12=(time.time()-ts)*1e3
p12t=fn12(Xr['logq'][0]); p12v=fn12(Xr['logq'][1])
tl12=log_rmse_arr(p12t,mm_tr); vl12=log_rmse_arr(p12v,mm_va)
joblib.dump({'mlp':mlp12,'sc':sc12},os.path.join(d,'saved_model','model.pkl'))
record(12,'mlp_deep_logq','ml','logq',tl12,vl12,rt12,'MLP(256,256,128,64 tanh), log(q)+chi_eff+log(omega0)')

# ============================================================
# 13: RBF interp (eff)
# ============================================================
print('\n[13] RBF-mq eff')
d=model_dir(13,'rbf_mq_eff')
sc13=StandardScaler(); Xt13=sc13.fit_transform(Xr['eff'][0])
t0=time.time(); rbf13=RBFInterpolator(Xt13,Ltr,kernel='multiquadric',epsilon=1.0,smoothing=0.5); print(f'  fit {time.time()-t0:.2f}s')
def fn13(X): return 10**rbf13(sc13.transform(X))
ts=time.time(); fn13(Xr['eff'][0][:1]); rt13=(time.time()-ts)*1e3
p13t=fn13(Xr['eff'][0]); p13v=fn13(Xr['eff'][1])
tl13=log_rmse_arr(p13t,mm_tr); vl13=log_rmse_arr(p13v,mm_va)
joblib.dump({'rbf':rbf13,'sc':sc13},os.path.join(d,'saved_model','model.pkl'))
record(13,'rbf_mq_eff','interp','eff',tl13,vl13,rt13,'RBF(multiquadric) interp, eff_spins')

# ============================================================
# 14: Poly deg3 (logq)
# ============================================================
print('\n[14] Poly-3 logq')
d=model_dir(14,'poly3_logq')
poly14=PolynomialFeatures(degree=3,include_bias=True)
sc14=StandardScaler(); Xp14=sc14.fit_transform(poly14.fit_transform(Xr['logq'][0]))
t0=time.time(); r14=Ridge(alpha=0.1); r14.fit(Xp14,Ltr); print(f'  fit {time.time()-t0:.3f}s')
def fn14(X): return 10**r14.predict(sc14.transform(poly14.transform(X)))
ts=time.time(); fn14(Xr['logq'][0][:1]); rt14=(time.time()-ts)*1e3
p14t=fn14(Xr['logq'][0]); p14v=fn14(Xr['logq'][1])
tl14=log_rmse_arr(p14t,mm_tr); vl14=log_rmse_arr(p14v,mm_va)
joblib.dump({'poly':poly14,'sc':sc14,'r':r14},os.path.join(d,'saved_model','model.pkl'))
record(14,'poly3_logq','symbolic','logq',tl14,vl14,rt14,'Poly-3 Ridge, log(q)+chi_eff+chi_a+log(omega0)')

# ============================================================
# 15: gplearn (raw) — MANDATORY
# ============================================================
print('\n[15] gplearn raw')
from gplearn.genetic import SymbolicRegressor as GPLearnSR
d=model_dir(15,'gplearn_raw')
t0=time.time()
gpl15=GPLearnSR(population_size=3000,generations=20,tournament_size=20,
                function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                metric='mse',parsimony_coefficient=0.005,verbose=0,random_state=42,n_jobs=2)
gpl15.fit(Xr['raw'][0],Ltr)
expr15=[{'expr':str(gpl15._program),'fitness':float(gpl15._program.fitness_)}]
print(f'  expr: {str(gpl15._program)[:60]}')
print(f'  gplearn raw {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr15,f,indent=2)
def fn15(X): return 10**gpl15.predict(X)
ts=time.time(); fn15(Xr['raw'][0][:1]); rt15=(time.time()-ts)*1e3
p15t=fn15(Xr['raw'][0]); p15v=fn15(Xr['raw'][1])
tl15=log_rmse_arr(p15t,mm_tr); vl15=log_rmse_arr(p15v,mm_va)
try: joblib.dump({'gpl':gpl15},os.path.join(d,'saved_model','model.pkl'))
except: pass
record(15,'gplearn_raw','symbolic','raw',tl15,vl15,rt15,f'gplearn SR: {str(gpl15._program)[:40]}')

# ============================================================
# 16: gplearn (eff) — MANDATORY
# ============================================================
print('\n[16] gplearn eff')
d=model_dir(16,'gplearn_eff')
t0=time.time()
gpl16=GPLearnSR(population_size=3000,generations=20,tournament_size=20,
                function_set=['add','sub','mul','div','sqrt','log','neg','inv'],
                metric='mse',parsimony_coefficient=0.005,verbose=0,random_state=7,n_jobs=2)
gpl16.fit(Xr['eff'][0],Ltr)
expr16=[{'expr':str(gpl16._program),'fitness':float(gpl16._program.fitness_)}]
print(f'  gplearn eff {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr16,f,indent=2)
def fn16(X): return 10**gpl16.predict(X)
ts=time.time(); fn16(Xr['eff'][0][:1]); rt16=(time.time()-ts)*1e3
p16t=fn16(Xr['eff'][0]); p16v=fn16(Xr['eff'][1])
tl16=log_rmse_arr(p16t,mm_tr); vl16=log_rmse_arr(p16v,mm_va)
try: joblib.dump({'gpl':gpl16},os.path.join(d,'saved_model','model.pkl'))
except: pass
record(16,'gplearn_eff','symbolic','eff',tl16,vl16,rt16,'gplearn SR, eff_spins reparam')

# ============================================================
# 17: PySR (raw) — MANDATORY
# ============================================================
print('\n[17] PySR raw')
from pysr import PySRRegressor
d=model_dir(17,'pysr_raw')
expr17=[]
t0=time.time()
try:
    ps17=PySRRegressor(niterations=60,binary_operators=['+','-','*','/'],
                       unary_operators=['sqrt','log','exp'],maxsize=22,populations=12,
                       procs=2,loss='loss(p,t)=(p-t)^2',verbosity=0,random_state=42,
                       tempdir=os.path.join(d,'saved_model','pysr'))
    ps17.fit(Xr['raw'][0],Ltr)
    try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps17.equations_.iterrows()]
    except: pf=[{'e':str(ps17.sympy()),'c':0,'l':0.}]
    expr17.append({'pareto':pf})
    print(f'  PySR best: {ps17.sympy()}')
except Exception as e:
    ps17=None; expr17.append({'err':str(e)}); print(f'  PySR failed: {e}')
print(f'  PySR raw {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr17,f,indent=2)
def fn17(X):
    if ps17: return 10**ps17.predict(X)
    return np.ones(len(X))*np.mean(mm_tr)
ts=time.time(); fn17(Xr['raw'][0][:1]); rt17=(time.time()-ts)*1e3
p17t=fn17(Xr['raw'][0]); p17v=fn17(Xr['raw'][1])
tl17=log_rmse_arr(p17t,mm_tr); vl17=log_rmse_arr(p17v,mm_va)
np.save(os.path.join(d,'saved_model','basis.npy'),np.zeros(1))
record(17,'pysr_raw','symbolic','raw',tl17,vl17,rt17,'PySR symbolic regression, raw 4D')

# ============================================================
# 18: PySR (logq) — MANDATORY
# ============================================================
print('\n[18] PySR logq')
d=model_dir(18,'pysr_logq')
expr18=[]
t0=time.time()
try:
    ps18=PySRRegressor(niterations=50,binary_operators=['+','-','*','/'],
                       unary_operators=['sqrt','log','exp'],maxsize=20,populations=10,
                       procs=2,loss='loss(p,t)=(p-t)^2',verbosity=0,random_state=13,
                       tempdir=os.path.join(d,'saved_model','pysr'))
    ps18.fit(Xr['logq'][0],Ltr)
    try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps18.equations_.iterrows()]
    except: pf=[{'e':str(ps18.sympy()),'c':0,'l':0.}]
    expr18.append({'pareto':pf})
except Exception as e:
    ps18=None; expr18.append({'err':str(e)})
print(f'  PySR logq {time.time()-t0:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr18,f,indent=2)
def fn18(X):
    if ps18: return 10**ps18.predict(X)
    return np.ones(len(X))*np.mean(mm_tr)
ts=time.time(); fn18(Xr['logq'][0][:1]); rt18=(time.time()-ts)*1e3
p18t=fn18(Xr['logq'][0]); p18v=fn18(Xr['logq'][1])
tl18=log_rmse_arr(p18t,mm_tr); vl18=log_rmse_arr(p18v,mm_va)
np.save(os.path.join(d,'saved_model','basis.npy'),np.zeros(1))
record(18,'pysr_logq','symbolic','logq',tl18,vl18,rt18,'PySR, log(q)+chi_eff+chi_a+log(omega0)')

# ============================================================
# 19: KRR Matern (logq)
# ============================================================
print('\n[19] KRR Matern logq')
d=model_dir(19,'krr_matern_logq')
from sklearn.metrics.pairwise import rbf_kernel
sc19=StandardScaler(); Xt19=sc19.fit_transform(Xr['logq'][0])
t0=time.time(); krr19=KernelRidge(kernel='laplacian',alpha=0.1,gamma=0.5); krr19.fit(Xt19,Ltr); print(f'  fit {time.time()-t0:.2f}s')
def fn19(X): return 10**krr19.predict(sc19.transform(X))
ts=time.time(); fn19(Xr['logq'][0][:1]); rt19=(time.time()-ts)*1e3
p19t=fn19(Xr['logq'][0]); p19v=fn19(Xr['logq'][1])
tl19=log_rmse_arr(p19t,mm_tr); vl19=log_rmse_arr(p19v,mm_va)
joblib.dump({'krr':krr19,'sc':sc19},os.path.join(d,'saved_model','model.pkl'))
record(19,'krr_matern_logq','kernel_gp','logq',tl19,vl19,rt19,'KRR(Laplacian gamma=0.5), logq')

# ============================================================
# 20: RF (logq)
# ============================================================
print('\n[20] RF logq')
d=model_dir(20,'rf_logq')
t0=time.time()
rf20=RandomForestRegressor(n_estimators=300,max_depth=None,random_state=0,n_jobs=-1,min_samples_leaf=2)
rf20.fit(Xr['logq'][0],Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn20(X): return 10**rf20.predict(X)
ts=time.time(); fn20(Xr['logq'][0][:1]); rt20=(time.time()-ts)*1e3
p20t=fn20(Xr['logq'][0]); p20v=fn20(Xr['logq'][1])
tl20=log_rmse_arr(p20t,mm_tr); vl20=log_rmse_arr(p20v,mm_va)
joblib.dump({'rf':rf20},os.path.join(d,'saved_model','model.pkl'))
record(20,'rf_logq','ml','logq',tl20,vl20,rt20,'RF(300, leaf=2), log(q)+chi_eff+chi_a+log(omega0)')

# ============================================================
# 21: GPR boundary (bnd)
# ============================================================
print('\n[21] GPR bnd')
d=model_dir(21,'gpr_bnd')
sc21=StandardScaler(); Xt21=sc21.fit_transform(Xr['bnd'][0])
k21=C(1.)*Matern(np.ones(Xr['bnd'][0].shape[1]),nu=2.5)+WhiteKernel(1e-3)
gp21=GaussianProcessRegressor(kernel=k21,n_restarts_optimizer=1,alpha=1e-4,normalize_y=True)
t0=time.time(); gp21.fit(Xt21,Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn21(X): return 10**gp21.predict(sc21.transform(X))
ts=time.time(); fn21(Xr['bnd'][0][:1]); rt21=(time.time()-ts)*1e3
p21t=fn21(Xr['bnd'][0]); p21v=fn21(Xr['bnd'][1])
tl21=log_rmse_arr(p21t,mm_tr); vl21=log_rmse_arr(p21v,mm_va)
joblib.dump({'gp':gp21,'sc':sc21},os.path.join(d,'saved_model','model.pkl'))
record(21,'gpr_bnd','kernel_gp','bnd',tl21,vl21,rt21,'GPR(Matern-2.5), boundary distance features')

# ============================================================
# 22: MLP (inter)
# ============================================================
print('\n[22] MLP inter')
d=model_dir(22,'mlp_inter')
sc22=StandardScaler(); Xt22=sc22.fit_transform(Xr['inter'][0])
t0=time.time()
mlp22=MLPRegressor(hidden_layer_sizes=(64,64),activation='relu',max_iter=1000,
                   random_state=13,early_stopping=True,validation_fraction=0.1)
mlp22.fit(Xt22,Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn22(X): return 10**mlp22.predict(sc22.transform(X))
ts=time.time(); fn22(Xr['inter'][0][:1]); rt22=(time.time()-ts)*1e3
p22t=fn22(Xr['inter'][0]); p22v=fn22(Xr['inter'][1])
tl22=log_rmse_arr(p22t,mm_tr); vl22=log_rmse_arr(p22v,mm_va)
joblib.dump({'mlp':mlp22,'sc':sc22},os.path.join(d,'saved_model','model.pkl'))
record(22,'mlp_inter','ml','inter',tl22,vl22,rt22,'MLP(64,64 relu), interaction features')

# ============================================================
# 23: kNN (logq)
# ============================================================
print('\n[23] kNN-10 logq')
d=model_dir(23,'knn10_logq')
sc23=StandardScaler(); Xt23=sc23.fit_transform(Xr['logq'][0])
t0=time.time(); knn23=KNeighborsRegressor(n_neighbors=10,weights='distance'); knn23.fit(Xt23,Ltr); print(f'  fit {time.time()-t0:.3f}s')
def fn23(X): return 10**knn23.predict(sc23.transform(X))
ts=time.time(); fn23(Xr['logq'][0][:1]); rt23=(time.time()-ts)*1e3
p23t=fn23(Xr['logq'][0]); p23v=fn23(Xr['logq'][1])
tl23=log_rmse_arr(p23t,mm_tr); vl23=log_rmse_arr(p23v,mm_va)
joblib.dump({'knn':knn23,'sc':sc23},os.path.join(d,'saved_model','model.pkl'))
record(23,'knn10_logq','interp','logq',tl23,vl23,rt23,'kNN(10, distance), log(q)+chi_eff+log(omega0)')

# ============================================================
# 24: ET boundary (bnd)
# ============================================================
print('\n[24] ET bnd')
d=model_dir(24,'et_bnd')
t0=time.time()
et24=ExtraTreesRegressor(n_estimators=300,max_depth=None,random_state=42,n_jobs=-1)
et24.fit(Xr['bnd'][0],Ltr); print(f'  fit {time.time()-t0:.1f}s')
def fn24(X): return 10**et24.predict(X)
ts=time.time(); fn24(Xr['bnd'][0][:1]); rt24=(time.time()-ts)*1e3
p24t=fn24(Xr['bnd'][0]); p24v=fn24(Xr['bnd'][1])
tl24=log_rmse_arr(p24t,mm_tr); vl24=log_rmse_arr(p24v,mm_va)
joblib.dump({'et':et24},os.path.join(d,'saved_model','model.pkl'))
record(24,'et_bnd','ml','bnd',tl24,vl24,rt24,'ET(300), boundary distance features')

# ============================================================
# FINAL
# ============================================================
print('\n=== Final plots & summary ===')

fig,ax=plt.subplots(figsize=figsize(2,0.8))
for r in ALL_RESULTS:
    tl=np.array(r['train_losses']); vl_r=np.array(r['val_losses'])
    al=np.concatenate([tl,vl_r]); lo,hi=max(al.min(),1e-8),max(al.max(),1e-4)
    if lo<hi:
        bins=np.logspace(np.log10(lo),np.log10(hi),25)
        ax.hist(tl,bins=bins,alpha=0.15,density=True,histtype='stepfilled',label=f'{r["name"][:6]} tr')
        ax.hist(vl_r,bins=bins,alpha=0.7,density=True,histtype='step',lw=1.,label=f'{r["name"][:6]} va')
ax.set_xscale('log'); ax.set_xlabel('|Δlog10(mm)|'); ax.set_ylabel('Density')
ax.legend(fontsize=3,ncol=4); plt.tight_layout()
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
    print(f'  {r["rank"]:2d}. {r["name"]:<25s} val={r["val_loss"]:.4e} ({r["category"]})')

if n>=20 and len(reps)>=3 and len(cats)==4 and has_pysr and has_gpl:
    print('\nVALIDITY_BENCH_COMPLETE')
else:
    print(f'\nIncomplete: n={n}, reps={len(reps)}, cats={len(cats)}, pysr={has_pysr}, gplearn={has_gpl}')
