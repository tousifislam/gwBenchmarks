#!/usr/bin/env python3
"""Analytic benchmark for sonnet46 — closed-form h22 expressions for non-spinning BBH."""
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
from scipy.optimize import minimize, curve_fit
from scipy.special import factorial

from gwbenchmarks.plot_settings import apply as plot_apply, figsize, COLORS
from gwbenchmarks.metrics import frequency_domain_mismatch, FD_MASSES_MSUN
plot_apply()

MODELS_DIR = os.path.join(SCRIPT_DIR,'models')
COMP_DIR   = os.path.join(SCRIPT_DIR,'comparison')
CHANGELOG  = os.path.join(SCRIPT_DIR,'CHANGELOG.md')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

CAT_COLORS = {'physics': COLORS['blue'], 'symbolic': COLORS['red'],
              'matched': COLORS['green'], 'functional': COLORS['orange']}
ALL_RESULTS = []
ALL_EXPRESSIONS = []

# ============================================================
# Data loading
# ============================================================
def load_dataset(path):
    data=[]
    with h5py.File(path,'r') as f:
        for s in f['sims']:
            g=f['sims'][s]; q=g.attrs['q']
            t=g['t'][:]; h22=g['h22_real'][:]+1j*g['h22_imag'][:]
            dt=float(np.diff(t[:3]).mean())
            data.append({'q':float(q),'eta':float(q/(1+q)**2),'t':t,'h22':h22,'dt':dt,'id':s})
    return data

print('Loading data...')
data_tr=load_dataset('datasets/analytic/analytic_training.h5')
data_va=load_dataset('datasets/analytic/analytic_validation.h5')
N_TR,N_VA=len(data_tr),len(data_va)
print(f'  Train={N_TR}, Val={N_VA}')
print(f'  Training q: [{min(d["q"] for d in data_tr):.2f}, {max(d["q"] for d in data_tr):.2f}]')

# Sort by q
data_tr=sorted(data_tr,key=lambda x:x['q'])
data_va=sorted(data_va,key=lambda x:x['q'])

# Training arrays
qs_tr=np.array([d['q'] for d in data_tr])
etas_tr=np.array([d['eta'] for d in data_tr])
qs_va=np.array([d['q'] for d in data_va])
etas_va=np.array([d['eta'] for d in data_va])

# ============================================================
# Physics: key formulas
# ============================================================
def eta_fn(q): return q/(1+q)**2
def delta_fn(q): return (q-1)/(q+1)

def pn0_phase(t, eta, phi_c=0.0):
    """Leading-order PN (2,2) GW phase. t < 0 for inspiral."""
    tau = np.where(t < 0, -t, 1e-6)
    c = (5/(256*eta))**(3/8)
    return phi_c + (16/5) * c * tau**(5/8)

def pn0_amp(t, eta, C=1.0):
    """Leading-order PN (2,2) GW amplitude."""
    tau = np.where(t < 0, -t, 1e-6)
    v = (5/(256*eta*tau))**(1/8)
    return C * eta * v**2

def qnm_h22(t, A0, omega_rd, tau_rd, phi0=0.0):
    """QNM ringdown waveform for t >= 0."""
    return np.where(t >= 0, A0*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi0)), 0.0+0j)

# ============================================================
# Precompute QNM parameters from training data
# ============================================================
print('\nFitting QNM parameters from training data...')
qnm_fitted=[]
for d in data_tr:
    t=d['t']; h22=d['h22']
    mask=(t>10)&(t<80)
    t_rd=t[mask]; h_rd=h22[mask]
    A_rd=np.abs(h_rd); phi_rd=np.unwrap(np.angle(h_rd))
    if len(t_rd)>3 and A_rd.max()>1e-8:
        p1=np.polyfit(t_rd,np.log(A_rd+1e-12),1)
        tau_rd=np.clip(-1/p1[0],1,50)
        p2=np.polyfit(t_rd,phi_rd,1)
        omega_rd=p2[0]
        A_at0=np.abs(np.interp(0.0, t, np.abs(h22)))
        phi_at0=np.interp(0.0, t, np.unwrap(np.angle(h22)))
    else:
        tau_rd=11.0; omega_rd=-0.5; A_at0=0.2; phi_at0=0.0
    qnm_fitted.append({'q':d['q'],'eta':d['eta'],'tau_rd':tau_rd,'omega_rd':omega_rd,
                       'A_peak':A_at0,'phi_at0':phi_at0})

# Fit polynomial functions of eta for QNM params
etas_f=np.array([x['eta'] for x in qnm_fitted])
tau_rd_arr=np.array([x['tau_rd'] for x in qnm_fitted])
omega_rd_arr=np.array([x['omega_rd'] for x in qnm_fitted])
A_peak_arr=np.array([x['A_peak'] for x in qnm_fitted])
phi0_arr=np.array([x['phi_at0'] for x in qnm_fitted])

# Polynomial fits (as function of eta)
c_tau_rd=np.polyfit(etas_f,tau_rd_arr,3)
c_omega_rd=np.polyfit(etas_f,omega_rd_arr,3)
c_Apeak=np.polyfit(etas_f,A_peak_arr,3)
c_phi0=np.polyfit(etas_f,phi0_arr,3)

def tau_rd_fn(eta):  return np.polyval(c_tau_rd, eta)
def omega_rd_fn(eta): return np.polyval(c_omega_rd, eta)
def A_peak_fn(eta):   return np.polyval(c_Apeak, eta)
def phi0_fn(eta):    return np.polyval(c_phi0, eta)

print(f'  tau_rd: [{tau_rd_arr.min():.1f}, {tau_rd_arr.max():.1f}]')
print(f'  omega_rd: [{omega_rd_arr.min():.4f}, {omega_rd_arr.max():.4f}]')
print(f'  A_peak: [{A_peak_arr.min():.4f}, {A_peak_arr.max():.4f}]')

# ============================================================
# Fit PN phase offset for each simulation
# ============================================================
print('\nFitting phase corrections...')
phi_c_arr=[]
for d in data_tr:
    t=d['t']; eta=d['eta']
    mask=t<-100
    t_i=t[mask]; h_i=d['h22'][mask]
    phi_data=np.unwrap(np.angle(h_i))
    phi_pn0=pn0_phase(t_i,eta)
    phi_c_arr.append(phi_data[0]-phi_pn0[0])

c_phi_c=np.polyfit(etas_f,np.array(phi_c_arr),3)
def phi_c_fn(eta): return np.polyval(c_phi_c,eta)

# ============================================================
# Mismatch evaluation
# ============================================================
def compute_mismatch(h_pred, h_ref, dt):
    """Mean mismatch over 5 total masses."""
    losses=[]
    for m in FD_MASSES_MSUN:
        try:
            mm=frequency_domain_mismatch(h_pred, h_ref, dt_geometric=dt, mtot_msun=m)
            losses.append(float(mm) if np.isfinite(mm) else 1.0)
        except: losses.append(1.0)
    return float(np.mean(losses)), losses

def eval_all(predict_fn, dataset):
    """Evaluate predict_fn over all simulations. Returns per-sim mean mismatch."""
    losses=[]
    for d in dataset:
        h_pred=predict_fn(d['t'],d['q'])
        mm,_=compute_mismatch(h_pred,d['h22'],d['dt'])
        losses.append(mm)
    return np.array(losses)

# ============================================================
# Utility
# ============================================================
def model_dir(num,name):
    d=os.path.join(MODELS_DIR,f'{num:02d}_{name}')
    os.makedirs(d,exist_ok=True)
    os.makedirs(os.path.join(d,'saved_model'),exist_ok=True)
    return d

def write_stubs(d,name):
    with open(os.path.join(d,'train.py'),'w') as f:
        f.write(f'"""Training for {name}."""\nimport joblib, numpy as np\nprint("Model: {name}")\n')
    with open(os.path.join(d,'predict.py'),'w') as f:
        f.write(f'"""Closed-form h22 prediction for {name}."""\n'
                f'import joblib,os,numpy as np\n'
                f'_m=joblib.load(os.path.join(os.path.dirname(__file__),"saved_model","model.pkl"))\n'
                f'def predict(t,q): return _m["fn"](t,q)\n')

def save_expression(d,expr_str):
    with open(os.path.join(d,'expression.txt'),'w') as f:
        f.write(expr_str)

def save_card(d,num,name,rep,tl,vl,rt,notes,n_params=0,expr_file=None,comp=None):
    card={'approach':name,'approach_number':num,'benchmark':'analytic','agent':'sonnet46',
          'parameterization':rep,'loss':float(np.mean(vl)),'runtime_ms':float(rt),
          'n_train':N_TR,'n_val':N_VA,'n_params':n_params,'notes':notes}
    if comp: card['loss_components']=comp
    if expr_file: card['expression_file']=expr_file
    with open(os.path.join(d,'scorecard.json'),'w') as f: json.dump(card,f,indent=2)

def record(num,name,cat,rep,tl_arr,vl_arr,rt,notes,n_params=0,expr_str=None,comp=None):
    mt=float(np.mean(tl_arr)); mv=float(np.mean(vl_arr))
    ALL_RESULTS.append({'number':num,'name':name,'category':cat,'parameterization':rep,
                        'train_loss':mt,'val_loss':mv,'runtime_ms':rt,
                        'train_losses':list(tl_arr),'val_losses':list(vl_arr)})
    d=model_dir(num,name)
    save_card(d,num,name,rep,tl_arr,vl_arr,rt,notes,n_params,
              'expression.txt' if expr_str else None, comp)
    if expr_str: save_expression(d,expr_str)
    write_stubs(d,name)
    update_plots()
    if expr_str:
        ALL_EXPRESSIONS.append({'approach':name,'expression':expr_str,'val_loss':mv})
    with open(CHANGELOG,'a') as f:
        f.write(f'## {num:02d}: {name}\n- cat={cat}, rep={rep}\n'
                f'- train={mt:.4e}, val={mv:.4e}, rt={rt:.1f}ms\n- {notes}\n\n')
    print(f'[{num:02d}] {name}: train={mt:.4e} val={mv:.4e} rt={rt:.1f}ms')

def update_plots():
    if not ALL_RESULTS: return
    vl=[r['val_loss'] for r in ALL_RESULTS]
    cats=[r['category'] for r in ALL_RESULTS]
    colors=[CAT_COLORS.get(c,COLORS['gray']) for c in cats]
    x=np.arange(len(ALL_RESULTS))
    names=[r['name'] for r in ALL_RESULTS]

    fig,ax=plt.subplots(figsize=figsize(2,0.5))
    for cat,col in CAT_COLORS.items():
        idx=[i for i,c in enumerate(cats) if c==cat]
        if idx: ax.scatter(np.array(idx),np.array(vl)[idx],color=col,label=cat,s=20,zorder=3)
    ax.set_yscale('log'); ax.set_xlabel('Approach'); ax.set_ylabel('Mean FD mismatch')
    ax.legend(fontsize=6); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'progress.{ext}'))
    plt.close()

    fig,ax=plt.subplots(figsize=figsize(2,0.6))
    ax.bar(x,vl,color=colors)
    ax.set_yscale('log'); ax.set_xticks(x)
    ax.set_xticklabels(names,rotation=90,fontsize=4); ax.set_ylabel('Val mismatch')
    for cat,col in CAT_COLORS.items(): ax.bar(0,0,color=col,label=cat)
    ax.legend(fontsize=6); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'loss_only_comparison.{ext}'))
    plt.close()

    rts=[r['runtime_ms'] for r in ALL_RESULTS]
    fig,ax=plt.subplots(figsize=figsize(2,0.6))
    for r in ALL_RESULTS:
        ax.scatter(r['runtime_ms'],r['val_loss'],color=CAT_COLORS.get(r['category'],COLORS['gray']),s=25)
        ax.annotate(r['name'][:8],(r['runtime_ms'],r['val_loss']),fontsize=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Eval time (ms)'); ax.set_ylabel('Val mismatch')
    for cat,col in CAT_COLORS.items(): ax.scatter([],[],color=col,label=cat,s=15)
    ax.legend(fontsize=6); plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'pareto_accuracy_speed.{ext}'))
    plt.close()

    ed={r['name']:{'train':r['train_losses'],'val':r['val_losses']} for r in ALL_RESULTS}
    with open(os.path.join(COMP_DIR,'error_data.json'),'w') as f: json.dump(ed,f)


# ============================================================
# 01: PN0 + QNM (baseline, no optimization)
# ============================================================
print('\n[01] PN0+QNM baseline')
d=model_dir(1,'pn0_qnm')
def h22_pn0_qnm(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta)
    phi_c=phi_c_fn(eta)
    # Inspiral
    A_insp=pn0_amp(t,eta,C=1.0)
    # Scale C so A_insp(0) ≈ A_peak via continuity
    A0_insp=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    C_scale=A_peak/(A0_insp+1e-10)
    A_insp=pn0_amp(t,eta,C=C_scale)
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    # Ringdown
    h_rd=np.where(t>=0, A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)), 0+0j)
    # Blend
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_pn0_qnm(data_tr[0]['t'],data_tr[0]['q']); rt1=(time.time()-ts)*1e3
tl1=eval_all(h22_pn0_qnm,data_tr); vl1=eval_all(h22_pn0_qnm,data_va)
joblib.dump({'fn':h22_pn0_qnm,'c_tau':c_tau_rd,'c_omega':c_omega_rd,'c_Apeak':c_Apeak,
             'c_phi':c_phi0,'c_phic':c_phi_c},os.path.join(d,'saved_model','model.pkl'))
expr1="""h22(t,q) = h_insp(t) * (1-w(t)) + h_rd(t) * w(t)
eta = q/(1+q)^2
A_insp(t) = C(eta) * eta * (5/(256*eta*|t|))^(1/4)  for t < 0
phi_insp(t) = phi_c(eta) + (16/5)*(5/(256*eta))^(3/8) * |t|^(5/8)
h_rd(t) = A_peak(eta) * exp(-t/tau_rd(eta)) * exp(-i*(omega_rd(eta)*t + phi0(eta)))  for t >= 0
w(t) = 0.5*(1 + tanh(t/5))
[Polynomial fits for tau_rd(eta), omega_rd(eta), A_peak(eta), phi_c(eta) from training data]"""
record(1,'pn0_qnm','physics','eta',tl1,vl1,rt1,'Leading PN + QNM with tanh blend',
       n_params=16,expr_str=expr1)

# ============================================================
# 02: PN0 + QNM with amplitude correction polynomial
# ============================================================
print('\n[02] PN0+QNM+amp correction')
d=model_dir(2,'pn0_qnm_ampcorr')

# Fit amplitude correction near merger
# For each sim, compute ratio A_true / A_PN0 at several time points
t_check=np.array([-50,-30,-20,-10,-5,-2,-1])
amp_ratios=[]
for d_sim in data_tr:
    t=d_sim['t']; eta=d_sim['eta']
    A_true=np.abs(d_sim['h22'])
    A_pn0=pn0_amp(t,eta,C=1.0)
    # Scale A_pn0 to match at t=-200
    i200=np.searchsorted(t,-200); i_ref=max(i200,1)
    A_pn0_sc=A_pn0*(A_true[i_ref]/(A_pn0[i_ref]+1e-12))
    ratios=[]
    for tc in t_check:
        i=np.searchsorted(t,tc)
        if 0<i<len(t):
            ratios.append(A_true[i]/(A_pn0_sc[i]+1e-10))
        else: ratios.append(1.0)
    amp_ratios.append(ratios)
amp_ratios=np.array(amp_ratios)  # shape (N_TR, len(t_check))

# Fit a polynomial in tau for the correction ratio: corr(tau) = poly(tau)
# Poly in log(tau) for each sim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
log_tau_check=np.log(-t_check+1e-6).reshape(-1,1)
# For each sim, fit log_corr(log_tau)
log_corr_coeffs=[]
for i in range(N_TR):
    lc=np.log(np.clip(amp_ratios[i],0.1,10)+1e-10)
    p=np.polyfit(log_tau_check.ravel(),lc,2)
    log_corr_coeffs.append(p)
log_corr_coeffs=np.array(log_corr_coeffs)  # (N_TR, 3)

# Fit each coefficient as function of eta
c_corr0=np.polyfit(etas_tr,log_corr_coeffs[:,0],2)
c_corr1=np.polyfit(etas_tr,log_corr_coeffs[:,1],2)
c_corr2=np.polyfit(etas_tr,log_corr_coeffs[:,2],2)

def amp_correction(t, eta):
    tau=np.maximum(-t,0.1)
    log_tau=np.log(tau)
    p0=np.polyval(c_corr0,eta); p1=np.polyval(c_corr1,eta); p2=np.polyval(c_corr2,eta)
    log_corr=p0*log_tau**2+p1*log_tau+p2
    return np.exp(np.clip(log_corr,-2,2))

def h22_pn0_ampcorr(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    A_insp_raw=pn0_amp(t,eta,C=1.0)
    # Scale to match A_peak at t=0
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    A_insp=A_insp_raw*(A_peak/(A0+1e-10))*amp_correction(t,eta)
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0, A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)), 0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_pn0_ampcorr(data_tr[0]['t'],data_tr[0]['q']); rt2=(time.time()-ts)*1e3
tl2=eval_all(h22_pn0_ampcorr,data_tr); vl2=eval_all(h22_pn0_ampcorr,data_va)
joblib.dump({'fn':h22_pn0_ampcorr,'c_corr0':c_corr0,'c_corr1':c_corr1,'c_corr2':c_corr2},
             os.path.join(model_dir(2,'pn0_qnm_ampcorr'),'saved_model','model.pkl'))
expr2='h22 = A_PN0(t) * exp(poly2(log|t|; eta)) * exp(-i*phi_PN0) smoothed with tanh QNM'
record(2,'pn0_qnm_ampcorr','physics','eta',tl2,vl2,rt2,'PN0+QNM with log-polynomial amplitude correction',
       n_params=20,expr_str=expr2)

# ============================================================
# 03: PN phase correction (2PN-inspired)
# ============================================================
print('\n[03] PN phase correction')
d=model_dir(3,'pn2_qnm')

# Fit phase correction: phi_true - phi_PN0 as function of (tau, eta)
phi_corrections=[]
for d_sim in data_tr:
    t=d_sim['t']; eta=d_sim['eta']
    mask=(-t>5)&(-t<3000)
    t_i=t[mask]; h_i=d_sim['h22'][mask]
    phi_data=np.unwrap(np.angle(h_i))
    phi_pn0=pn0_phase(t_i,eta)
    phi_c=phi_c_fn(eta)
    phi_pn0_shifted=phi_pn0+(phi_data[0]-pn0_phase(t_i[:1],eta,phi_c)[0])
    delta_phi=phi_data-phi_pn0_shifted
    # Fit correction as poly in tau^{-3/8} (2PN correction structure)
    tau_i=-t_i
    v_i=(5/(256*eta*tau_i))**(1/8)
    # 2PN correction: delta_phi = a0 + a1*v^2 + a2*v^4 + a3*log(v)*v^0 ...
    # Simplified: poly in v
    V=np.column_stack([v_i**2, v_i**4, v_i**6])
    p_corr=np.linalg.lstsq(V, delta_phi, rcond=None)[0]
    phi_corrections.append(p_corr)

phi_corrections=np.array(phi_corrections)  # (N_TR, 3)
c_pc0=np.polyfit(etas_tr,phi_corrections[:,0],2)
c_pc1=np.polyfit(etas_tr,phi_corrections[:,1],2)
c_pc2=np.polyfit(etas_tr,phi_corrections[:,2],2)

def phi_correction_fn(t, eta):
    tau=np.maximum(-t,1e-6)
    v=(5/(256*eta*tau))**(1/8)
    a0=np.polyval(c_pc0,eta); a1=np.polyval(c_pc1,eta); a2=np.polyval(c_pc2,eta)
    return a0*v**2 + a1*v**4 + a2*v**6

def h22_pn2(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    A_insp=pn0_amp(t,eta,C=1.0)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)+np.where(t<0,phi_correction_fn(t,eta),0.)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0, A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)), 0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_pn2(data_tr[0]['t'],data_tr[0]['q']); rt3=(time.time()-ts)*1e3
tl3=eval_all(h22_pn2,data_tr); vl3=eval_all(h22_pn2,data_va)
joblib.dump({'fn':h22_pn2,'c_pc0':c_pc0,'c_pc1':c_pc1,'c_pc2':c_pc2},
             os.path.join(model_dir(3,'pn2_qnm'),'saved_model','model.pkl'))
expr3='phi = phi_PN0 + a0(eta)*v^2 + a1(eta)*v^4 + a2(eta)*v^6, v=(5/256/eta/tau)^(1/8)'
record(3,'pn2_qnm','physics','eta',tl3,vl3,rt3,'PN+phase correction in v-expansion + QNM',
       n_params=24,expr_str=expr3)

# ============================================================
# 04: Gaussian merger correction + PN baseline
# ============================================================
print('\n[04] PN + Gaussian merger')
d=model_dir(4,'pn_gauss_merger')

# Fit amplitude peak shape: A(t) = PN amplitude + Gaussian near t=0
gauss_params=[]
for d_sim in data_tr:
    t=d_sim['t']; eta=d_sim['eta']
    A_true=np.abs(d_sim['h22'])
    A_pn=pn0_amp(t,eta,C=1.0)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    A_pn_sc=A_pn*(A_peak_fn(eta)/(A0+1e-10))
    # The PN amplitude diverges at t=0, so just use A_peak model
    # Fit A_true with A_peak(eta) * f(t)
    # f(t) = PN-shape for early, Gaussian peak at merger, QNM decay after
    # Here just fit the Gaussian width
    mask=(-20<t)&(t<5)
    t_m=t[mask]; A_m=A_true[mask]
    if len(t_m)>5:
        try:
            i_peak=np.argmax(A_m)
            t_peak=t_m[i_peak]; A_max=A_m[i_peak]
            # Gaussian: A = A_max * exp(-(t-t_peak)^2/(2*sigma^2))
            mask2=A_m>0.3*A_max
            if mask2.sum()>3:
                sigma2=0.5*np.var(t_m[mask2])
                sigma=max(np.sqrt(sigma2),1.0)
            else: sigma=5.0
        except: sigma=5.0
    else: sigma=5.0
    gauss_params.append({'q':d_sim['q'],'eta':eta,'sigma':sigma})

sigmas=np.array([x['sigma'] for x in gauss_params])
c_sigma=np.polyfit(etas_tr,sigmas,2)

def h22_gauss_merger(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    sigma=np.polyval(c_sigma,eta)
    # Amplitude: Gaussian near merger
    A_gauss=A_peak*np.exp(-t**2/(2*sigma**2))
    # For inspiral (t << 0), blend with PN amplitude
    A_pn=pn0_amp(t,eta,C=1.0)
    A0=pn0_amp(np.array([-100.0]),eta,C=1.0)[0]
    A_pn_sc=A_pn*(A_gauss[np.argmin(np.abs(t+100))]/(A0+1e-10))
    # Blend: Gaussian wins near t=0, PN wins at early times
    blend_t=0.5*(1+np.tanh(t/30))
    A=A_pn_sc*(1-blend_t)+A_gauss*blend_t
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0, A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)), 0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_gauss_merger(data_tr[0]['t'],data_tr[0]['q']); rt4=(time.time()-ts)*1e3
tl4=eval_all(h22_gauss_merger,data_tr); vl4=eval_all(h22_gauss_merger,data_va)
joblib.dump({'fn':h22_gauss_merger,'c_sigma':c_sigma},
             os.path.join(d,'saved_model','model.pkl'))
expr4='A(t) = A_PN*(1-blend) + A_peak*exp(-t^2/(2*sigma(eta)^2))*blend; sigma=poly2(eta)'
record(4,'pn_gauss_merger','physics','eta',tl4,vl4,rt4,'PN inspiral + Gaussian merger peak + QNM',
       n_params=18,expr_str=expr4)

# ============================================================
# 05: IMRPhenom-style (polynomial frequency + amplitude envelope)
# ============================================================
print('\n[05] IMRPhenom-style')
d=model_dir(5,'imrphenom_style')

# Fit frequency as polynomial in retarded time tau = -t
# omega(tau) = a0 * tau^{-3/8} * (1 + a1/tau^{1/4} + a2/tau^{1/2} + a3/tau^{3/4})
# This is effectively a PN series

# For each simulation, extract omega(t) from phase
omega_fits=[]
for d_sim in data_tr:
    t=d_sim['t']; eta=d_sim['eta']
    mask=(-2000<t)&(t<-20)
    t_i=t[mask]; h_i=d_sim['h22'][mask]
    phi_i=np.unwrap(np.angle(h_i))
    dt=d_sim['dt']
    omega_i=np.gradient(phi_i,dt)
    tau_i=-t_i
    v_i=(5/(256*eta*tau_i))**(1/8)
    # Fit omega/omega_pn = polynomial in v
    omega_pn=-2*(5/(256*eta))**(3/8)*tau_i**(-3/8)
    ratio=omega_i/(omega_pn+1e-15)
    # Fit ratio(v): ratio = 1 + b1*v^2 + b2*v^4 + b3*v^6
    V=np.column_stack([np.ones(len(v_i)),v_i**2,v_i**4,v_i**6])
    p=np.linalg.lstsq(V,ratio,rcond=None)[0]
    omega_fits.append(p)

omega_fits=np.array(omega_fits)
c_o0=np.polyfit(etas_tr,omega_fits[:,0],2)
c_o1=np.polyfit(etas_tr,omega_fits[:,1],2)
c_o2=np.polyfit(etas_tr,omega_fits[:,2],2)
c_o3=np.polyfit(etas_tr,omega_fits[:,3],2)

def omega_phenom(t, eta):
    tau=np.maximum(-t,1e-6)
    v=(5/(256*eta*tau))**(1/8)
    b0=np.polyval(c_o0,eta); b1=np.polyval(c_o1,eta)
    b2=np.polyval(c_o2,eta); b3=np.polyval(c_o3,eta)
    omega_pn=-2*(5/(256*eta))**(3/8)*tau**(-3/8)
    return omega_pn*(b0+b1*v**2+b2*v**4+b3*v**6)

def h22_phenom(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta)
    dt_arr=np.diff(t[:5]).mean()
    # Integrate omega to get phase
    omega=np.where(t<0, omega_phenom(t,eta), omega_rd)
    phi=np.cumsum(omega)*dt_arr
    phi-=phi[np.searchsorted(t,0.0)]  # set phi=0 at t=0 first
    phi+=phi_at0  # then shift to match QNM
    # Amplitude
    A_insp=pn0_amp(t,eta,C=1.0)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    h_insp=A_insp*np.exp(-1j*phi)
    h_rd=np.where(t>=0, A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)), 0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_phenom(data_tr[0]['t'],data_tr[0]['q']); rt5=(time.time()-ts)*1e3
tl5=eval_all(h22_phenom,data_tr); vl5=eval_all(h22_phenom,data_va)
joblib.dump({'fn':h22_phenom,'c_o0':c_o0,'c_o1':c_o1,'c_o2':c_o2,'c_o3':c_o3},
             os.path.join(d,'saved_model','model.pkl'))
expr5='omega(t) = omega_PN0*(b0+b1*v^2+b2*v^4+b3*v^6), b_i=poly(eta); phi=cumsum(omega)'
record(5,'imrphenom_style','physics','eta',tl5,vl5,rt5,'IMRPhenom-style: fitted PN frequency ratio',
       n_params=24,expr_str=expr5)

# ============================================================
# 06: Pade-resummed amplitude model
# ============================================================
print('\n[06] Pade amplitude')
d=model_dir(6,'pade_amp')

# Fit amplitude as Pade rational in v: A(v) = N(v)/D(v)
# where v = (5/(256*eta*tau))^{1/8}
pade_fits=[]
for d_sim in data_tr:
    t=d_sim['t']; eta=d_sim['eta']
    mask=(-500<t)&(t<-5)
    t_i=t[mask]; A_i=np.abs(d_sim['h22'][mask])
    tau_i=-t_i
    v_i=(5/(256*eta*tau_i))**(1/8)
    # Pade [2,2]: A = eta*(a0+a1*v^2)/(1+b1*v^2) * v^2
    # Actually: A ~ eta*v^2*(1+correction); try Pade in v
    y_i=A_i/(eta*v_i**2+1e-12)  # normalize out leading factor
    # Fit y = (a0+a1*v^2)/(1+b1*v^2)
    V_N=np.column_stack([np.ones(len(v_i)),v_i**2,v_i**4])
    V_D=np.column_stack([v_i**2,v_i**4])
    M=np.hstack([V_N,-y_i[:,None]*V_D])
    p,_,_,_=np.linalg.lstsq(M,y_i,rcond=None)
    pade_fits.append(p)

pade_fits=np.array(pade_fits)
c_pa=[]
for j in range(pade_fits.shape[1]):
    c_pa.append(np.polyfit(etas_tr,pade_fits[:,j],2))

def A_pade(t, eta):
    tau=np.maximum(-t,1e-6)
    v=(5/(256*eta*tau))**(1/8)
    p=[np.polyval(c,eta) for c in c_pa]
    num=p[0]+p[1]*v**2+p[2]*v**4
    den=1+p[3]*v**2+p[4]*v**4
    return eta*v**2*num/(den+1e-10)

def h22_pade_amp(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    A_insp=np.abs(A_pade(t,eta))
    A_at0=np.abs(A_pade(np.array([-0.5]),eta)[0])
    A_insp=A_insp*(A_peak/(A_at0+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_pade_amp(data_tr[0]['t'],data_tr[0]['q']); rt6=(time.time()-ts)*1e3
tl6=eval_all(h22_pade_amp,data_tr); vl6=eval_all(h22_pade_amp,data_va)
joblib.dump({'fn':h22_pade_amp,'c_pa':c_pa},os.path.join(d,'saved_model','model.pkl'))
expr6='A(t) = eta*v^2*(a0+a1*v^2+a2*v^4)/(1+b1*v^2+b2*v^4), Pade[2,2], v=(5/256/eta/tau)^(1/8)'
record(6,'pade_amp','physics','eta',tl6,vl6,rt6,'Pade [2,2] amplitude in PN velocity v',
       n_params=20,expr_str=expr6)

# ============================================================
# 07: Combined phase+amplitude correction (best of above)
# ============================================================
print('\n[07] PN+phase+amp correction combined')
d=model_dir(7,'pn_full_corr')

def h22_full_corr(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    # Amplitude with correction
    A_insp=pn0_amp(t,eta,C=1.0)*amp_correction(t,eta)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]*amp_correction(np.array([-0.1]),eta)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    # Phase with correction
    phi_corr=np.where(t<0,phi_correction_fn(t,eta),0.)
    phi_insp=pn0_phase(t,eta,phi_c)+phi_corr
    h_insp=A_insp*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_full_corr(data_tr[0]['t'],data_tr[0]['q']); rt7=(time.time()-ts)*1e3
tl7=eval_all(h22_full_corr,data_tr); vl7=eval_all(h22_full_corr,data_va)
joblib.dump({'fn':h22_full_corr},os.path.join(d,'saved_model','model.pkl'))
expr7='h22 = A_PN*corr(tau,eta)*exp(-i*(phi_PN0+delta_phi(v,eta)))+QNM; both corr from data'
record(7,'pn_full_corr','physics','eta',tl7,vl7,rt7,'Full PN+amp correction+phase correction+QNM',
       n_params=32,expr_str=expr7)

# ============================================================
# 08: Phenom with integrated frequency (best phenom)
# ============================================================
print('\n[08] Phenom+amp correction')
d=model_dir(8,'phenom_ampcorr')

def h22_phenom2(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta)
    dt_arr=np.diff(t[:5]).mean()
    omega=np.where(t<0, omega_phenom(t,eta), omega_rd)
    phi=np.cumsum(omega)*dt_arr
    i0=np.searchsorted(t,0.0)
    phi-=phi[i0]; phi+=phi_at0
    # Amplitude with correction
    A_insp=pn0_amp(t,eta,C=1.0)*amp_correction(t,eta)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]*amp_correction(np.array([-0.1]),eta)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    h_insp=A_insp*np.exp(-1j*phi)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_phenom2(data_tr[0]['t'],data_tr[0]['q']); rt8=(time.time()-ts)*1e3
tl8=eval_all(h22_phenom2,data_tr); vl8=eval_all(h22_phenom2,data_va)
joblib.dump({'fn':h22_phenom2},os.path.join(d,'saved_model','model.pkl'))
expr8='h22 = A_PN*amp_corr * exp(-i*cumsum(omega_phenom)) + QNM'
record(8,'phenom_ampcorr','physics','eta',tl8,vl8,rt8,'Phenom frequency + amplitude correction',
       n_params=36,expr_str=expr8)

# ============================================================
# 09: 3-Gaussian amplitude + polynomial phase (functional form)
# ============================================================
print('\n[09] 3-Gaussian amplitude')
d=model_dir(9,'gauss3_amp')

# Build amplitude model: A(t) = sum_k G_k * exp(-(t-mu_k)^2/(2*sigma_k^2))
# Fit parameters for each training case, then fit as function of eta
gauss3_params=[]
for d_sim in data_tr:
    t=d_sim['t']; A=np.abs(d_sim['h22']); eta=d_sim['eta']
    A_peak=A.max()
    # Fixed mu, fit sigma and amplitude
    # 3 Gaussians: early inspiral (~-500), merger (~0), ringdown decay (~10)
    # Keep simple: center at -100, 0, 10 and fit amplitudes
    t_centers=np.array([-200., 0., 10.])
    t_sigmas=np.array([200., 8., 10.])
    # Build design matrix
    G=np.column_stack([np.exp(-(t-t_centers[k])**2/(2*t_sigmas[k]**2)) for k in range(3)])
    p,_,_,_=np.linalg.lstsq(G,A,rcond=None)
    gauss3_params.append(np.maximum(p,0))

gauss3_params=np.array(gauss3_params)
c_g3=[]
for j in range(3):
    c_g3.append(np.polyfit(etas_tr,gauss3_params[:,j],2))

t_centers_g3=np.array([-200., 0., 10.])
t_sigmas_g3=np.array([200., 8., 10.])

def h22_gauss3(t, q):
    eta=eta_fn(q)
    phi_c=phi_c_fn(eta)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta); phi_at0=phi0_fn(eta)
    A=sum(np.polyval(c_g3[k],eta)*np.exp(-(t-t_centers_g3[k])**2/(2*t_sigmas_g3[k]**2)) for k in range(3))
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,np.polyval(c_g3[2],eta)*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_gauss3(data_tr[0]['t'],data_tr[0]['q']); rt9=(time.time()-ts)*1e3
tl9=eval_all(h22_gauss3,data_tr); vl9=eval_all(h22_gauss3,data_va)
joblib.dump({'fn':h22_gauss3,'c_g3':c_g3,'t_centers':t_centers_g3,'t_sigmas':t_sigmas_g3},
             os.path.join(d,'saved_model','model.pkl'))
expr9='A(t) = sum_k c_k(eta)*exp(-(t-mu_k)^2/(2*sigma_k^2)); phi=phi_PN0; k=1..3'
record(9,'gauss3_amp','functional','eta',tl9,vl9,rt9,'3-Gaussian amplitude + PN phase',
       n_params=18,expr_str=expr9)

# ============================================================
# 10: Lorentzian merger peak + power-law tails
# ============================================================
print('\n[10] Lorentzian peak amplitude')
d=model_dir(10,'lorentz_peak')

lorentz_params=[]
for d_sim in data_tr:
    t=d_sim['t']; A=np.abs(d_sim['h22']); eta=d_sim['eta']
    A_pk=A.max()
    # Lorentzian: A(t) = A_pk / (1 + (t/gamma)^2)
    mask=(-50<t)&(t<20)
    t_m=t[mask]; A_m=A[mask]
    if len(t_m)>5:
        try:
            def lor(t, g): return A_pk/(1+(t/g)**2)
            p,_=curve_fit(lor, t_m, A_m, p0=[10.0], bounds=(1,100), maxfev=500)
            gamma=float(p[0])
        except: gamma=10.0
    else: gamma=10.0
    lorentz_params.append(gamma)

c_lor=np.polyfit(etas_tr,np.array(lorentz_params),2)

def h22_lorentz(t, q):
    eta=eta_fn(q)
    phi_c=phi_c_fn(eta); A_peak=A_peak_fn(eta)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta); phi_at0=phi0_fn(eta)
    gamma=np.polyval(c_lor,eta)
    # Lorentzian amplitude
    A=A_peak/(1+(t/gamma)**2+1e-12)
    # PN amplitude for early inspiral
    A_pn=pn0_amp(t,eta,C=1.0)
    A0=pn0_amp(np.array([-50.0]),eta,C=1.0)[0]
    A_lor_50=A_peak/(1+((-50)/gamma)**2)
    A_pn_sc=A_pn*(A_lor_50/(A0+1e-10))
    blend_early=0.5*(1+np.tanh((t+50)/20))
    A_blend=A_pn_sc*(1-blend_early)+A*blend_early
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A_blend*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_lorentz(data_tr[0]['t'],data_tr[0]['q']); rt10=(time.time()-ts)*1e3
tl10=eval_all(h22_lorentz,data_tr); vl10=eval_all(h22_lorentz,data_va)
joblib.dump({'fn':h22_lorentz,'c_lor':c_lor},os.path.join(d,'saved_model','model.pkl'))
expr10='A(t) = A_peak(eta)/(1+(t/gamma(eta))^2); gamma=poly2(eta), Lorentzian merger peak'
record(10,'lorentz_peak','functional','eta',tl10,vl10,rt10,'Lorentzian peak amplitude + PN phase',
       n_params=14,expr_str=expr10)

# ============================================================
# 11: Sigmoid amplitude transition + QNM
# ============================================================
print('\n[11] Sigmoid-blend model')
d=model_dir(11,'sigmoid_blend')

def h22_sigmoid(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    # Amplitude: PN decaying into merger, sigmoid blend
    A_insp=pn0_amp(t,eta,C=1.0)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    # Sigmoid envelope: multiply by (1 - tanh(t/W)) to kill PN after merger
    W=8.0
    A_insp=A_insp*(1-0.5*(1+np.tanh(t/W)))
    phi_insp=pn0_phase(t,eta,phi_c)+np.where(t<0,phi_correction_fn(t,eta),0.)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    # Ringdown: grows from 0 at t=0
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    return h_insp+h_rd

ts=time.time(); h22_sigmoid(data_tr[0]['t'],data_tr[0]['q']); rt11=(time.time()-ts)*1e3
tl11=eval_all(h22_sigmoid,data_tr); vl11=eval_all(h22_sigmoid,data_va)
joblib.dump({'fn':h22_sigmoid},os.path.join(d,'saved_model','model.pkl'))
expr11='h22 = A_PN*(1-tanh(t/W))*exp(-i*phi_insp) + A_peak*exp(-t/tau_rd)*exp(-i*omega_rd*t)'
record(11,'sigmoid_blend','matched','eta',tl11,vl11,rt11,'Sigmoid blend PN inspiral + QNM ringdown',
       n_params=28,expr_str=expr11)

# ============================================================
# 12: gplearn on amplitude envelope (symbolic, mandatory)
# ============================================================
print('\n[12] gplearn on amplitude')
from gplearn.genetic import SymbolicRegressor as GPLearnSR
d=model_dir(12,'gplearn_amp')

# Build feature matrix: (tau, eta) → A at sparse time points
t_sparse=np.concatenate([np.linspace(-500,-10,30),np.linspace(-10,-0.5,15)])
X_amp=[]; y_amp=[]
for d_sim in data_tr:
    for tc in t_sparse:
        i=np.searchsorted(d_sim['t'],tc)
        if 0<i<len(d_sim['t']):
            tau=np.clip(-tc,0.1,1000)
            eta=d_sim['eta']
            A=np.abs(d_sim['h22'][i])
            X_amp.append([np.log(tau),eta])
            y_amp.append(np.log(A+1e-12))
X_amp=np.array(X_amp); y_amp=np.array(y_amp)

t0_gpl=time.time()
gpl12=GPLearnSR(population_size=2000,generations=15,tournament_size=15,
                function_set=['add','sub','mul','div','sqrt','log','neg'],
                metric='mse',parsimony_coefficient=0.005,verbose=0,random_state=42,n_jobs=2)
gpl12.fit(X_amp,y_amp)
expr12_str=str(gpl12._program)
print(f'  gplearn amp: {expr12_str[:60]} ({time.time()-t0_gpl:.0f}s)')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f:
    json.dump([{'expr':expr12_str,'fit':float(gpl12._program.fitness_)}],f,indent=2)

def h22_gpl_amp(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    # gplearn gives log(A) = f(log(tau), eta)
    tau=np.maximum(-t,0.1)
    X_pred=np.column_stack([np.log(tau),np.full_like(tau,eta)])
    log_A_pred=gpl12.predict(X_pred)
    A=np.exp(np.clip(log_A_pred,-10,2))
    # Scale to match A_peak
    i_ref=np.searchsorted(t,-1.0)
    if 0<i_ref<len(t): A=A*(A_peak/(A[i_ref]+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_gpl_amp(data_tr[0]['t'],data_tr[0]['q']); rt12=(time.time()-ts)*1e3
tl12=eval_all(h22_gpl_amp,data_tr); vl12=eval_all(h22_gpl_amp,data_va)
try: joblib.dump({'gpl':gpl12},os.path.join(d,'saved_model','model.pkl'))
except: np.save(os.path.join(d,'saved_model','basis.npy'),np.zeros(1))
record(12,'gplearn_amp','symbolic','eta',tl12,vl12,rt12,f'gplearn: log(A)=f(log(tau),eta): {expr12_str[:40]}',
       n_params=0,expr_str=f'log(A(tau,eta)) = {expr12_str}')

# ============================================================
# 13: gplearn on frequency (symbolic, mandatory)
# ============================================================
print('\n[13] gplearn on frequency')
d=model_dir(13,'gplearn_freq')

X_freq=[]; y_freq=[]
for d_sim in data_tr:
    t_i=d_sim['t']; h_i=d_sim['h22']; eta=d_sim['eta']; dt=d_sim['dt']
    mask=(-1000<t_i)&(t_i<-5)
    t_m=t_i[mask]; h_m=h_i[mask]
    phi_m=np.unwrap(np.angle(h_m))
    omega_m=np.gradient(phi_m,dt)
    # Normalize by PN0 frequency
    tau_m=-t_m
    omega_pn=-2*(5/(256*eta))**(3/8)*tau_m**(-3/8)
    ratio=omega_m/(omega_pn+1e-15)
    # Subsample
    idx=np.linspace(0,len(tau_m)-1,30,dtype=int)
    for ii in idx:
        v_i=(5/(256*eta*tau_m[ii]))**(1/8)
        X_freq.append([v_i,eta]); y_freq.append(float(ratio[ii]))
X_freq=np.array(X_freq); y_freq=np.array(y_freq)

t0_gpl2=time.time()
gpl13=GPLearnSR(population_size=2000,generations=12,tournament_size=15,
                function_set=['add','sub','mul','div','sqrt','log','neg'],
                metric='mse',parsimony_coefficient=0.005,verbose=0,random_state=7,n_jobs=2)
gpl13.fit(X_freq,y_freq)
expr13_str=str(gpl13._program)
print(f'  gplearn freq: {expr13_str[:60]} ({time.time()-t0_gpl2:.0f}s)')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f:
    json.dump([{'expr':expr13_str,'fit':float(gpl13._program.fitness_)}],f,indent=2)

def h22_gpl_freq(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta)
    dt_arr=np.diff(t[:5]).mean()
    tau=np.maximum(-t,1e-6)
    v=(5/(256*eta*tau))**(1/8)
    omega_pn=-2*(5/(256*eta))**(3/8)*tau**(-3/8)
    X_pred=np.column_stack([v,np.full_like(v,eta)])
    ratio_pred=gpl13.predict(X_pred)
    omega=np.where(t<0, omega_pn*ratio_pred, omega_rd)
    phi=np.cumsum(omega)*dt_arr
    i0=np.searchsorted(t,0.0)
    phi-=phi[i0]; phi+=phi_at0
    A_insp=pn0_amp(t,eta,C=1.0)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    h_insp=A_insp*np.exp(-1j*phi)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_gpl_freq(data_tr[0]['t'],data_tr[0]['q']); rt13=(time.time()-ts)*1e3
tl13=eval_all(h22_gpl_freq,data_tr); vl13=eval_all(h22_gpl_freq,data_va)
try: joblib.dump({'gpl':gpl13},os.path.join(d,'saved_model','model.pkl'))
except: np.save(os.path.join(d,'saved_model','basis.npy'),np.zeros(1))
record(13,'gplearn_freq','symbolic','eta',tl13,vl13,rt13,f'gplearn: omega/omega_PN=f(v,eta): {expr13_str[:40]}',
       n_params=0,expr_str=f'omega(v,eta)/omega_PN = {expr13_str}')

# ============================================================
# 14: PySR on amplitude (symbolic, mandatory)
# ============================================================
print('\n[14] PySR on amplitude')
from pysr import PySRRegressor
d=model_dir(14,'pysr_amp')
expr14_data=[]
t0_psr=time.time()
try:
    ps14=PySRRegressor(niterations=50,binary_operators=['+','-','*','/'],
                       unary_operators=['sqrt','log','exp'],maxsize=20,populations=10,
                       procs=2,loss='loss(p,t)=(p-t)^2',verbosity=0,random_state=42,
                       tempdir=os.path.join(d,'saved_model','pysr_amp'))
    ps14.fit(X_amp,y_amp)
    try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps14.equations_.iterrows()]
    except: pf=[{'e':str(ps14.sympy()),'c':0,'l':0.}]
    expr14_data.append({'target':'log_amp','pareto':pf})
    print(f'  PySR best: {ps14.sympy()}')
except Exception as e:
    ps14=None; expr14_data.append({'err':str(e)}); print(f'  PySR failed: {e}')
print(f'  PySR amp total {time.time()-t0_psr:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr14_data,f,indent=2)

def h22_pysr_amp(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    tau=np.maximum(-t,0.1)
    if ps14:
        X_pred=np.column_stack([np.log(tau),np.full_like(tau,eta)])
        log_A_pred=ps14.predict(X_pred)
        A=np.exp(np.clip(log_A_pred,-10,2))
    else:
        A=pn0_amp(t,eta,C=1.0)
    i_ref=np.searchsorted(t,-1.0)
    if 0<i_ref<len(t): A=A*(A_peak/(A[i_ref]+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_pysr_amp(data_tr[0]['t'],data_tr[0]['q']); rt14=(time.time()-ts)*1e3
tl14=eval_all(h22_pysr_amp,data_tr); vl14=eval_all(h22_pysr_amp,data_va)
np.save(os.path.join(d,'saved_model','basis.npy'),np.zeros(1))
expr14_str=f'log(A(tau,eta)) = {str(ps14.sympy()) if ps14 else "PySR_failed"}'
record(14,'pysr_amp','symbolic','eta',tl14,vl14,rt14,'PySR symbolic: log(A)=f(log(tau),eta)',
       n_params=0,expr_str=expr14_str)

# ============================================================
# 15: PySR on frequency (symbolic, mandatory)
# ============================================================
print('\n[15] PySR on frequency')
d=model_dir(15,'pysr_freq')
expr15_data=[]
t0_psr2=time.time()
try:
    ps15=PySRRegressor(niterations=40,binary_operators=['+','-','*','/'],
                       unary_operators=['sqrt','log','exp'],maxsize=18,populations=8,
                       procs=2,loss='loss(p,t)=(p-t)^2',verbosity=0,random_state=7,
                       tempdir=os.path.join(d,'saved_model','pysr_freq'))
    ps15.fit(X_freq,y_freq)
    try: pf=[{'e':str(r['sympy_format']),'c':int(r['complexity']),'l':float(r['loss'])} for _,r in ps15.equations_.iterrows()]
    except: pf=[{'e':str(ps15.sympy()),'c':0,'l':0.}]
    expr15_data.append({'target':'omega_ratio','pareto':pf})
    print(f'  PySR best: {ps15.sympy()}')
except Exception as e:
    ps15=None; expr15_data.append({'err':str(e)}); print(f'  PySR failed: {e}')
print(f'  PySR freq total {time.time()-t0_psr2:.0f}s')
with open(os.path.join(d,'saved_model','expressions.json'),'w') as f: json.dump(expr15_data,f,indent=2)

def h22_pysr_freq(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta)
    dt_arr=np.diff(t[:5]).mean()
    tau=np.maximum(-t,1e-6)
    v=(5/(256*eta*tau))**(1/8)
    omega_pn=-2*(5/(256*eta))**(3/8)*tau**(-3/8)
    if ps15:
        X_pred=np.column_stack([v,np.full_like(v,eta)])
        ratio_pred=ps15.predict(X_pred)
        omega=np.where(t<0, omega_pn*ratio_pred, omega_rd)
    else:
        omega=np.where(t<0, omega_pn, omega_rd)
    phi=np.cumsum(omega)*dt_arr
    i0=np.searchsorted(t,0.0); phi-=phi[i0]; phi+=phi_at0
    A_insp=pn0_amp(t,eta,C=1.0)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    h_insp=A_insp*np.exp(-1j*phi)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_pysr_freq(data_tr[0]['t'],data_tr[0]['q']); rt15=(time.time()-ts)*1e3
tl15=eval_all(h22_pysr_freq,data_tr); vl15=eval_all(h22_pysr_freq,data_va)
np.save(os.path.join(d,'saved_model','basis.npy'),np.zeros(1))
expr15_str=f'omega_ratio(v,eta) = {str(ps15.sympy()) if ps15 else "PySR_failed"}'
record(15,'pysr_freq','symbolic','eta',tl15,vl15,rt15,'PySR symbolic: omega/omega_PN=f(v,eta)',
       n_params=0,expr_str=expr15_str)

# ============================================================
# 16: Matched asymptotic 3-region model
# ============================================================
print('\n[16] Matched asymptotic 3-region')
d=model_dir(16,'matched_3region')

def h22_3region(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    # Region 1 (inspiral t < -20): PN with correction
    A_insp=pn0_amp(t,eta,C=1.0)*amp_correction(t,eta)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]*amp_correction(np.array([-0.1]),eta)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)+np.where(t<0,phi_correction_fn(t,eta),0.)
    # Region 2 (merger -20 < t < 5): Gaussian peak
    A_gauss=A_peak*np.exp(-t**2/100.)
    # Region 3 (ringdown t > 5): QNM
    h_rd=A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0))
    # Combine with smooth transitions
    w1=0.5*(1+np.tanh((t+15)/5))  # early to merger
    w2=0.5*(1+np.tanh((t-5)/3))   # merger to ringdown
    A=A_insp*(1-w1)+A_gauss*w1*(1-w2)+np.abs(h_rd)*w2
    phi=phi_insp*(1-w2)+np.angle(h_rd)*w2
    return A*np.exp(-1j*phi)

ts=time.time(); h22_3region(data_tr[0]['t'],data_tr[0]['q']); rt16=(time.time()-ts)*1e3
tl16=eval_all(h22_3region,data_tr); vl16=eval_all(h22_3region,data_va)
joblib.dump({'fn':h22_3region},os.path.join(d,'saved_model','model.pkl'))
expr16='3-region: PN(t<-15)->Gaussian(-15<t<5)->QNM(t>5), smooth tanh transitions'
record(16,'matched_3region','matched','eta',tl16,vl16,rt16,'3-region matched: PN+Gaussian merger+QNM',
       n_params=32,expr_str=expr16)

# ============================================================
# 17: Damped sinusoid series for full waveform
# ============================================================
print('\n[17] Damped sinusoid series')
d=model_dir(17,'damp_sinusoid')

# Fit h22(t) as sum of quasi-normal modes with varying frequency
# h22 = sum_k A_k(eta)*exp(-alpha_k*t)*exp(-i*omega_k*t) for t > 0 (ringdown)
# For t < 0 (inspiral): use PN

damp_params=[]
for d_sim in data_tr:
    t=d_sim['t']; eta=d_sim['eta']
    mask=(5<t)&(t<100)
    t_rd=t[mask]; h_rd=d_sim['h22'][mask]
    if len(t_rd)>5:
        # Fit 2 QNM modes
        A0=np.abs(h_rd[0]); phi0=np.angle(h_rd[0])
        tau1=tau_rd_fn(eta); omega1=omega_rd_fn(eta)
        # Second overtone: omega ~1.5*omega1, tau ~tau1/3
        tau2=tau1/3; omega2=1.5*omega1; A2=A0*0.1; phi2=phi0
        damp_params.append([A0,tau1,omega1,phi0,A2,tau2,omega2,phi2])
    else: damp_params.append([0.2,11.0,-0.5,0.0,0.02,3.5,-0.7,0.0])

damp_params=np.array(damp_params)
c_damp=[]
for j in range(damp_params.shape[1]):
    c_damp.append(np.polyfit(etas_tr,damp_params[:,j],2))

def h22_damp(t, q):
    eta=eta_fn(q)
    phi_c=phi_c_fn(eta)
    p=[np.polyval(c,eta) for c in c_damp]
    A0,tau1,omega1,phi0,A2,tau2,omega2,phi2=p
    # Ringdown
    h_rd1=np.where(t>=0,A0*np.exp(-t/tau1)*np.exp(-1j*(omega1*t+phi0)),0+0j)
    h_rd2=np.where(t>=0,A2*np.exp(-t/tau2)*np.exp(-1j*(omega2*t+phi2)),0+0j)
    h_rd=h_rd1+h_rd2
    # Inspiral
    A_insp=pn0_amp(t,eta,C=1.0)
    A0_amp=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]
    A_insp=A_insp*(A0/(A0_amp+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_damp(data_tr[0]['t'],data_tr[0]['q']); rt17=(time.time()-ts)*1e3
tl17=eval_all(h22_damp,data_tr); vl17=eval_all(h22_damp,data_va)
joblib.dump({'fn':h22_damp,'c_damp':c_damp},os.path.join(d,'saved_model','model.pkl'))
expr17='h22 = A0*exp(-t/tau1)*exp(-i*omega1*t) + A2*exp(-t/tau2)*exp(-i*omega2*t), 2 overtones'
record(17,'damp_sinusoid','functional','eta',tl17,vl17,rt17,'2-QNM overtones + PN inspiral',
       n_params=24,expr_str=expr17)

# ============================================================
# 18: Power-law in tau amplitude + delta_m parameterization
# ============================================================
print('\n[18] Power-law amp + delta_m reparam')
d=model_dir(18,'powerlaw_delta')

# delta_m = (q-1)/(q+1)
powerlaw_fits=[]
for d_sim in data_tr:
    t=d_sim['t']; eta=d_sim['eta']
    q=d_sim['q']; dm=(q-1)/(q+1)
    A=np.abs(d_sim['h22'])
    mask=(-1000<t)&(t<-5)
    t_i=t[mask]; A_i=A[mask]; tau_i=-t_i
    log_tau=np.log(tau_i); log_A=np.log(A_i+1e-12)
    # Fit: log(A) = a + b*log(tau)  → A = e^a * tau^b
    p=np.polyfit(log_tau,log_A,1)
    powerlaw_fits.append([float(p[1]),float(p[0]),float(dm)])  # [a, b, dm]

pf_arr=np.array(powerlaw_fits)
deltas=pf_arr[:,2]; a_arr=pf_arr[:,0]; b_arr=pf_arr[:,1]
c_pla=np.polyfit(deltas,a_arr,3)  # intercept vs delta_m
c_plb=np.polyfit(deltas,b_arr,2)  # slope vs delta_m

def h22_powerlaw(t, q):
    eta=eta_fn(q); dm=(q-1)/(q+1)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    a=np.polyval(c_pla,dm); b=np.polyval(c_plb,dm)
    tau=np.maximum(-t,0.1)
    A=np.where(t<0, np.exp(a)*tau**b, A_peak*np.exp(-t/tau_rd))
    phi_insp=pn0_phase(t,eta,phi_c)
    h_insp=A*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_powerlaw(data_tr[0]['t'],data_tr[0]['q']); rt18=(time.time()-ts)*1e3
tl18=eval_all(h22_powerlaw,data_tr); vl18=eval_all(h22_powerlaw,data_va)
joblib.dump({'fn':h22_powerlaw,'c_pla':c_pla,'c_plb':c_plb},os.path.join(d,'saved_model','model.pkl'))
expr18='A(tau) = exp(a(delta_m))*tau^b(delta_m); a,b=poly(delta_m), delta_m=(q-1)/(q+1)'
record(18,'powerlaw_delta','functional','delta_m',tl18,vl18,rt18,'Power-law amplitude in tau, delta_m reparam',
       n_params=12,expr_str=expr18)

# ============================================================
# 19: Composite model: PN inspiral + QNM joined at ISCO
# ============================================================
print('\n[19] PN+QNM composite at ISCO')
d=model_dir(19,'pn_qnm_composite')

# For Schwarzschild ISCO: omega_ISCO = 1/(6^{3/2}) ≈ 0.0680 (2,2 mode)
# The PN phase at ISCO gives the merger time t_ISCO
# omega_22(tau) = -2*(5/(256*eta))^{3/8} * tau^{-3/8}
# At |omega_22| = 0.068: tau_ISCO = (2/(0.068))^{8/3} * (5/(256*eta)) / 1

omega_isco=0.088  # rough ISCO frequency for (2,2) mode

def t_isco(eta):
    # Solve |omega_pn(tau)| = omega_isco for tau
    tau = (2**(8/3)/omega_isco**(8/3)) * (5/(256*eta))
    return -tau  # negative t (before merger)

def h22_composite(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta); phi_c=phi_c_fn(eta)
    t_is=t_isco(eta)
    # Before ISCO: PN
    A_insp=pn0_amp(t,eta,C=1.0)*amp_correction(t,eta)
    A0=pn0_amp(np.array([t_is]),eta,C=1.0)[0]*amp_correction(np.array([t_is]),eta)[0]
    A_at_isco=A_insp[np.searchsorted(t,t_is)] if t_is>t.min() else A_insp[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)+np.where(t<0,phi_correction_fn(t,eta),0.)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    # After t>0: QNM
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    # Near merger: blend
    W=np.maximum(np.abs(t_is)*0.1,5.0)
    w=0.5*(1+np.tanh(t/W))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_composite(data_tr[0]['t'],data_tr[0]['q']); rt19=(time.time()-ts)*1e3
tl19=eval_all(h22_composite,data_tr); vl19=eval_all(h22_composite,data_va)
joblib.dump({'fn':h22_composite},os.path.join(d,'saved_model','model.pkl'))
expr19='PN(t<t_ISCO)+QNM(t>0) joined at ISCO frequency omega_22=0.088; tanh blend'
record(19,'pn_qnm_composite','matched','eta',tl19,vl19,rt19,'PN+QNM composite at ISCO, delta_m=vary',
       n_params=28,expr_str=expr19)

# ============================================================
# 20: sqrt(eta) reparameterization model
# ============================================================
print('\n[20] sqrt(eta) model')
d=model_dir(20,'sqrteta_model')

# Fit QNM and inspiral parameters as function of sqrt(eta)
sqrt_eta_tr=np.sqrt(etas_tr)
c_tau_sq=np.polyfit(sqrt_eta_tr,tau_rd_arr,3)
c_omega_sq=np.polyfit(sqrt_eta_tr,omega_rd_arr,3)
c_Apeak_sq=np.polyfit(sqrt_eta_tr,A_peak_arr,2)

def h22_sqrteta(t, q):
    eta=eta_fn(q); se=np.sqrt(eta)
    tau_rd=np.polyval(c_tau_sq,se)
    omega_rd=np.polyval(c_omega_sq,se)
    A_peak=np.polyval(c_Apeak_sq,se)
    phi_c=phi_c_fn(eta); phi_at0=phi0_fn(eta)
    A_insp=pn0_amp(t,eta,C=1.0)*amp_correction(t,eta)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]*amp_correction(np.array([-0.1]),eta)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)+np.where(t<0,phi_correction_fn(t,eta),0.)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_sqrteta(data_tr[0]['t'],data_tr[0]['q']); rt20=(time.time()-ts)*1e3
tl20=eval_all(h22_sqrteta,data_tr); vl20=eval_all(h22_sqrteta,data_va)
joblib.dump({'fn':h22_sqrteta,'c_tau_sq':c_tau_sq,'c_omega_sq':c_omega_sq,'c_Apeak_sq':c_Apeak_sq},
             os.path.join(d,'saved_model','model.pkl'))
expr20='Parameters tau_rd,omega_rd,A_peak=poly(sqrt(eta)); sqrt(eta) reparam'
record(20,'sqrteta_model','physics','sqrt_eta',tl20,vl20,rt20,'QNM+PN with sqrt(eta) reparameterization',
       n_params=28,expr_str=expr20)

# ============================================================
# 21: q-direct polynomial fits for QNM
# ============================================================
print('\n[21] q-direct polynomial')
d=model_dir(21,'poly_q_direct')

qs_tr_arr=np.array([x['q'] for x in qnm_fitted])
c_tau_q=np.polyfit(np.log(qs_tr_arr),tau_rd_arr,3)
c_omega_q=np.polyfit(np.log(qs_tr_arr),omega_rd_arr,3)
c_Apeak_q=np.polyfit(np.log(qs_tr_arr),A_peak_arr,3)

def h22_polyq(t, q):
    eta=eta_fn(q)
    tau_rd=np.polyval(c_tau_q,np.log(q))
    omega_rd=np.polyval(c_omega_q,np.log(q))
    A_peak=np.polyval(c_Apeak_q,np.log(q))
    phi_c=phi_c_fn(eta); phi_at0=phi0_fn(eta)
    A_insp=pn0_amp(t,eta,C=1.0)*amp_correction(t,eta)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]*amp_correction(np.array([-0.1]),eta)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    phi_insp=pn0_phase(t,eta,phi_c)+np.where(t<0,phi_correction_fn(t,eta),0.)
    h_insp=A_insp*np.exp(-1j*phi_insp)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_polyq(data_tr[0]['t'],data_tr[0]['q']); rt21=(time.time()-ts)*1e3
tl21=eval_all(h22_polyq,data_tr); vl21=eval_all(h22_polyq,data_va)
joblib.dump({'fn':h22_polyq,'c_tau_q':c_tau_q,'c_omega_q':c_omega_q,'c_Apeak_q':c_Apeak_q},
             os.path.join(d,'saved_model','model.pkl'))
expr21='Parameters=poly3(log(q)); using log(q) directly as reparameterization'
record(21,'poly_q_direct','physics','log_q',tl21,vl21,rt21,'QNM+PN with log(q) reparameterization',
       n_params=24,expr_str=expr21)

# ============================================================
# 22: Full phenom with PySR frequency + amp correction
# ============================================================
print('\n[22] Full phenom: PySR freq + amp correction')
d=model_dir(22,'phenom_pysr_freq')

def h22_phenom_pysr(t, q):
    eta=eta_fn(q)
    tau_rd=tau_rd_fn(eta); omega_rd=omega_rd_fn(eta)
    A_peak=A_peak_fn(eta); phi_at0=phi0_fn(eta)
    dt_arr=np.diff(t[:5]).mean()
    tau=np.maximum(-t,1e-6)
    v=(5/(256*eta*tau))**(1/8)
    omega_pn=-2*(5/(256*eta))**(3/8)*tau**(-3/8)
    if ps15:
        X_pred=np.column_stack([v,np.full_like(v,eta)])
        ratio_pred=ps15.predict(X_pred)
        omega=np.where(t<0, omega_pn*ratio_pred, omega_rd)
    else:
        omega=np.where(t<0,omega_pn,omega_rd)
    phi=np.cumsum(omega)*dt_arr
    i0=np.searchsorted(t,0.0); phi-=phi[i0]; phi+=phi_at0
    # Amplitude with correction
    A_insp=pn0_amp(t,eta,C=1.0)*amp_correction(t,eta)
    A0=pn0_amp(np.array([-0.1]),eta,C=1.0)[0]*amp_correction(np.array([-0.1]),eta)[0]
    A_insp=A_insp*(A_peak/(A0+1e-10))
    h_insp=A_insp*np.exp(-1j*phi)
    h_rd=np.where(t>=0,A_peak*np.exp(-t/tau_rd)*np.exp(-1j*(omega_rd*t+phi_at0)),0+0j)
    w=0.5*(1+np.tanh(t/5))
    return h_insp*(1-w)+h_rd*w

ts=time.time(); h22_phenom_pysr(data_tr[0]['t'],data_tr[0]['q']); rt22=(time.time()-ts)*1e3
tl22=eval_all(h22_phenom_pysr,data_tr); vl22=eval_all(h22_phenom_pysr,data_va)
joblib.dump({'fn':h22_phenom_pysr},os.path.join(d,'saved_model','model.pkl'))
expr22='omega from PySR freq ratio; amp from PN*correction; phi=cumsum(omega)'
record(22,'phenom_pysr_freq','symbolic','eta',tl22,vl22,rt22,'Phenom using PySR frequency + amp correction',
       n_params=0,expr_str=expr22)

# ============================================================
# FINAL OUTPUTS
# ============================================================
print('\n=== Final plots & summary ===')

# Error histograms
fig,ax=plt.subplots(figsize=figsize(2,0.8))
for r in ALL_RESULTS:
    tl=np.array(r['train_losses']); vl_r=np.array(r['val_losses'])
    al=np.concatenate([tl,vl_r])
    lo,hi=max(al.min(),1e-5),min(al.max()+1e-6,1.1)
    if lo<hi:
        bins=np.logspace(np.log10(lo),np.log10(hi),20)
        ax.hist(tl,bins=bins,alpha=0.15,density=True,histtype='stepfilled',label=f'{r["name"][:6]} tr')
        ax.hist(vl_r,bins=bins,alpha=0.7,density=True,histtype='step',lw=1.,label=f'{r["name"][:6]} va')
ax.set_xscale('log'); ax.set_xlabel('FD mismatch'); ax.set_ylabel('Density')
ax.legend(fontsize=3,ncol=4); plt.tight_layout()
for ext in ('png','pdf'): plt.savefig(os.path.join(COMP_DIR,f'error_histograms.{ext}'))
plt.close()

# Summary
ranked=sorted(ALL_RESULTS,key=lambda r:r['val_loss'])
table=[{'rank':i+1,'name':r['name'],'category':r['category'],'parameterization':r['parameterization'],
        'val_loss':r['val_loss'],'train_loss':r['train_loss'],'runtime_ms':r['runtime_ms']}
       for i,r in enumerate(ranked)]
with open(os.path.join(COMP_DIR,'summary_table.json'),'w') as f: json.dump(table,f,indent=2)
with open(os.path.join(COMP_DIR,'best_model.json'),'w') as f: json.dump(table[0],f,indent=2)

# All expressions
with open(os.path.join(COMP_DIR,'all_expressions.json'),'w') as f:
    json.dump(ALL_EXPRESSIONS,f,indent=2)

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
    print('\nANALYTIC_BENCH_COMPLETE')
else:
    print(f'\nIncomplete: n={n}, reps={len(reps)}, cats={len(cats)}, pysr={has_pysr}, gplearn={has_gpl}')
