#!/usr/bin/env python3
"""Build all 26 ringdown models: predict omega_real and omega_imag from spin.

Target mode: l2/m+2/n0
Loss: L = (mean(|pred_R - true_R|/|true_R|) + mean(|pred_I - true_I|/|true_I|)) / 2
"""

import sys, os, numpy as np, json, time, warnings, joblib
warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(WORK_DIR, "../../../.."))
sys.path.insert(0, ROOT)
import h5py
import gwbenchmarks.plot_settings as pset
pset.apply()

# ─── Load data ───────────────────────────────────────────────────────────────
def load_ringdown(path):
    with h5py.File(path, "r") as f:
        g = f["l2/m+2/n0"]
        spin = g["spin"][:]
        omega_real = g["omega_real"][:]
        omega_imag = g["omega_imag"][:]
    return spin, omega_real, omega_imag

print("Loading data...")
a_tr, wr_tr, wi_tr = load_ringdown(os.path.join(ROOT, "datasets/ringdown/ringdown_training.h5"))
a_va, wr_va, wi_va = load_ringdown(os.path.join(ROOT, "datasets/ringdown/ringdown_validation.h5"))
N_TR, N_VA = len(a_tr), len(a_va)
print(f"Train: {N_TR}, Val: {N_VA}")
print(f"  spin  [{a_tr.min():.6f}, {a_tr.max():.6f}]")
print(f"  wR    [{wr_tr.min():.6f}, {wr_tr.max():.6f}]")
print(f"  wI    [{wi_tr.min():.8f}, {wi_tr.max():.8f}]")

# Sort training data by spin for interpolation methods
sort_tr = np.argsort(a_tr)
a_tr_s = a_tr[sort_tr]; wr_tr_s = wr_tr[sort_tr]; wi_tr_s = wi_tr[sort_tr]

# ─── Loss function ───────────────────────────────────────────────────────────
def ringdown_loss(pred_R, true_R, pred_I, true_I):
    """L = (mean(|pred_R - true_R|/|true_R|) + mean(|pred_I - true_I|/|true_I|)) / 2"""
    mre_R = np.mean(np.abs(pred_R - true_R) / np.abs(true_R))
    mre_I = np.mean(np.abs(pred_I - true_I) / np.abs(true_I))
    return float((mre_R + mre_I) / 2.0)

# ─── Reparameterizations ────────────────────────────────────────────────────
def reparam(a, scheme="raw"):
    if scheme == "raw":
        return a.copy()
    elif scheme == "neglog":
        return -np.log(1.0 - a + 1e-15)
    elif scheme == "sqrt1ma2":
        return np.sqrt(1.0 - a**2)
    elif scheme == "linear_scaled":
        return 2.0 * a - 1.0
    else:
        return a.copy()

SCHEMES = ["raw", "neglog", "sqrt1ma2", "linear_scaled"]
ps = {k: reparam(a_tr, k) for k in SCHEMES}
pv = {k: reparam(a_va, k) for k in SCHEMES}
# Sorted versions for interpolation
ps_s = {k: reparam(a_tr_s, k) for k in SCHEMES}

# ─── Helpers ─────────────────────────────────────────────────────────────────
results = []; err_data = {}; cl = []

def md(n, name):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    os.makedirs(os.path.join(d, "saved_model"), exist_ok=True)
    return d

def rec(n, name, sch, loss_val, pred_R_val, pred_I_val, pred_R_tr, pred_I_tr, rt, np_, notes):
    d = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    loss_tr = ringdown_loss(pred_R_tr, wr_tr, pred_I_tr, wi_tr)
    mre_R_val = float(np.mean(np.abs(pred_R_val - wr_va) / np.abs(wr_va)))
    mre_I_val = float(np.mean(np.abs(pred_I_val - wi_va) / np.abs(wi_va)))
    sc = {"approach": name, "approach_number": n, "benchmark": "ringdown",
          "agent": "opus46", "mode": "l2/m+2/n0",
          "parameterization": sch, "loss": float(loss_val),
          "loss_components": {"mre_omega_real": mre_R_val, "mre_omega_imag": mre_I_val},
          "runtime_ms": rt, "n_train": N_TR, "n_val": N_VA, "n_params": np_, "notes": notes}
    with open(os.path.join(d, "scorecard.json"), "w") as f:
        json.dump(sc, f, indent=2)
    results.append(sc)
    # Per-sample relative errors for histograms
    err_data[name] = {
        "val_rel_err_R": [float(x) for x in np.abs(pred_R_val - wr_va) / np.abs(wr_va)],
        "val_rel_err_I": [float(x) for x in np.abs(pred_I_val - wi_va) / np.abs(wi_va)],
        "train_rel_err_R": [float(x) for x in np.abs(pred_R_tr - wr_tr) / np.abs(wr_tr)],
        "train_rel_err_I": [float(x) for x in np.abs(pred_I_tr - wi_tr) / np.abs(wi_tr)],
    }
    cl.append(f"## {n}: {name}\n- Param: {sch}, Loss: {loss_val:.8f}, RT: {rt:.2f}ms\n- {notes}\n")
    print(f"  [{n:02d}] {name}: Loss={loss_val:.8f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: ANALYTICAL (6 models)
# ═══════════════════════════════════════════════════════════════════════════════

# --- 1. Polynomial degree 10 (raw) ---
print("\n=== 1: Poly deg 10 (raw) ===")
d = md(1, "poly10_raw")
coeffR = np.polyfit(a_tr, wr_tr, 10)
coeffI = np.polyfit(a_tr, wi_tr, 10)
t0 = time.time()
pR = np.polyval(coeffR, a_va); pI = np.polyval(coeffI, a_va)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = np.polyval(coeffR, a_tr); pI_tr = np.polyval(coeffI, a_tr)
joblib.dump({"coeffR": coeffR, "coeffI": coeffI}, os.path.join(d, "saved_model/model.joblib"))
rec(1, "poly10_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 22, "Polynomial degree 10, raw spin")

# --- 2. Polynomial degree 15 (raw) ---
print("\n=== 2: Poly deg 15 (raw) ===")
d = md(2, "poly15_raw")
coeffR = np.polyfit(a_tr, wr_tr, 15)
coeffI = np.polyfit(a_tr, wi_tr, 15)
t0 = time.time()
pR = np.polyval(coeffR, a_va); pI = np.polyval(coeffI, a_va)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = np.polyval(coeffR, a_tr); pI_tr = np.polyval(coeffI, a_tr)
joblib.dump({"coeffR": coeffR, "coeffI": coeffI}, os.path.join(d, "saved_model/model.joblib"))
rec(2, "poly15_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 32, "Polynomial degree 15, raw spin")

# --- 3. Polynomial degree 20 (raw) ---
print("\n=== 3: Poly deg 20 (raw) ===")
d = md(3, "poly20_raw")
coeffR = np.polyfit(a_tr, wr_tr, 20)
coeffI = np.polyfit(a_tr, wi_tr, 20)
t0 = time.time()
pR = np.polyval(coeffR, a_va); pI = np.polyval(coeffI, a_va)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = np.polyval(coeffR, a_tr); pI_tr = np.polyval(coeffI, a_tr)
joblib.dump({"coeffR": coeffR, "coeffI": coeffI}, os.path.join(d, "saved_model/model.joblib"))
rec(3, "poly20_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 42, "Polynomial degree 20, raw spin")

# --- 4. Chebyshev polynomial (neglog) ---
print("\n=== 4: Chebyshev poly (neglog) ===")
d = md(4, "chebyshev_neglog")
from numpy.polynomial import chebyshev
x_tr_nl = ps["neglog"]
x_va_nl = pv["neglog"]
# Fit Chebyshev polynomials of degree 20
chebR = chebyshev.Chebyshev.fit(x_tr_nl, wr_tr, 20)
chebI = chebyshev.Chebyshev.fit(x_tr_nl, wi_tr, 20)
t0 = time.time()
pR = chebR(x_va_nl); pI = chebI(x_va_nl)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = chebR(x_tr_nl); pI_tr = chebI(x_tr_nl)
joblib.dump({"chebR_coef": chebR.coef.tolist(), "chebR_domain": chebR.domain.tolist(),
             "chebR_window": chebR.window.tolist(),
             "chebI_coef": chebI.coef.tolist(), "chebI_domain": chebI.domain.tolist(),
             "chebI_window": chebI.window.tolist()},
            os.path.join(d, "saved_model/model.joblib"))
rec(4, "chebyshev_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, 42,
    "Chebyshev deg 20, neglog reparameterization")

# --- 5. Pade / rational approximation (raw) ---
print("\n=== 5: Pade rational (raw) ===")
d = md(5, "pade_rational_raw")
from scipy.optimize import least_squares

def rational_func(x, num_coeffs, den_coeffs):
    """Evaluate P(x)/Q(x) where Q has leading coefficient 1."""
    num = np.polyval(num_coeffs, x)
    den = np.polyval(np.concatenate([[1.0], den_coeffs]), x)
    return num / den

def rational_residuals(params, x, y, n_num, n_den):
    num_c = params[:n_num]
    den_c = params[n_num:n_num + n_den]
    pred = rational_func(x, num_c, den_c)
    return pred - y

# [7,6] rational for omega_R, [7,6] for omega_I
n_num_R, n_den_R = 8, 7
p0_R = np.zeros(n_num_R + n_den_R)
p0_R[:n_num_R] = np.polyfit(a_tr, wr_tr, n_num_R - 1)
res_R = least_squares(rational_residuals, p0_R, args=(a_tr, wr_tr, n_num_R, n_den_R),
                      method='lm', max_nfev=10000)

n_num_I, n_den_I = 8, 7
p0_I = np.zeros(n_num_I + n_den_I)
p0_I[:n_num_I] = np.polyfit(a_tr, wi_tr, n_num_I - 1)
res_I = least_squares(rational_residuals, p0_I, args=(a_tr, wi_tr, n_num_I, n_den_I),
                      method='lm', max_nfev=10000)

t0 = time.time()
pR = rational_func(a_va, res_R.x[:n_num_R], res_R.x[n_num_R:])
pI = rational_func(a_va, res_I.x[:n_num_I], res_I.x[n_num_I:])
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = rational_func(a_tr, res_R.x[:n_num_R], res_R.x[n_num_R:])
pI_tr = rational_func(a_tr, res_I.x[:n_num_I], res_I.x[n_num_I:])
joblib.dump({"params_R": res_R.x.tolist(), "n_num_R": n_num_R, "n_den_R": n_den_R,
             "params_I": res_I.x.tolist(), "n_num_I": n_num_I, "n_den_I": n_den_I},
            os.path.join(d, "saved_model/model.joblib"))
rec(5, "pade_rational_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 30,
    "Pade/rational approx [7,6]/[7,6], raw spin")

# --- 6. Ridge polynomial (neglog) ---
print("\n=== 6: Ridge poly (neglog) ===")
d = md(6, "ridge_poly_neglog")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

x_nl_tr = ps["neglog"].reshape(-1, 1)
x_nl_va = pv["neglog"].reshape(-1, 1)
poly6 = PolynomialFeatures(degree=20)
X6_tr = poly6.fit_transform(x_nl_tr)
X6_va = poly6.transform(x_nl_va)
sc6 = StandardScaler().fit(X6_tr)
X6_tr_s = sc6.transform(X6_tr)
X6_va_s = sc6.transform(X6_va)

ridgeR = Ridge(alpha=1e-6).fit(X6_tr_s, wr_tr)
ridgeI = Ridge(alpha=1e-6).fit(X6_tr_s, wi_tr)
t0 = time.time()
pR = ridgeR.predict(X6_va_s); pI = ridgeI.predict(X6_va_s)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = ridgeR.predict(X6_tr_s); pI_tr = ridgeI.predict(X6_tr_s)
joblib.dump({"ridgeR": ridgeR, "ridgeI": ridgeI, "poly": poly6, "scaler": sc6},
            os.path.join(d, "saved_model/model.joblib"))
rec(6, "ridge_poly_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, 42,
    "Ridge poly deg 20 + neglog reparam")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: SYMBOLIC REGRESSION (4 models)
# ═══════════════════════════════════════════════════════════════════════════════

from pysr import PySRRegressor
from gplearn.genetic import SymbolicRegressor

# --- 7. PySR raw ---
print("\n=== 7: PySR raw ===")
d = md(7, "pysr_raw")
# Fit omega_R
m7R = PySRRegressor(niterations=40, maxsize=20, populations=10, procs=1,
                    verbosity=0, progress=False, random_state=42, temp_equation_file=True,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "exp", "sin", "cos"])
m7R.fit(a_tr.reshape(-1, 1), wr_tr)
# Fit omega_I
m7I = PySRRegressor(niterations=40, maxsize=20, populations=10, procs=1,
                    verbosity=0, progress=False, random_state=42, temp_equation_file=True,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "exp", "sin", "cos"])
m7I.fit(a_tr.reshape(-1, 1), wi_tr)

t0 = time.time()
pR = m7R.predict(a_va.reshape(-1, 1)); pI = m7I.predict(a_va.reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = m7R.predict(a_tr.reshape(-1, 1)); pI_tr = m7I.predict(a_tr.reshape(-1, 1))
try:
    exprR = str(m7R.sympy()); exprI = str(m7I.sympy())
    eqsR = [{"expression": str(r.get("equation", "")), "complexity": int(r.get("complexity", 0)),
             "loss": float(r.get("loss", 0))} for _, r in m7R.equations_.iterrows()]
    eqsI = [{"expression": str(r.get("equation", "")), "complexity": int(r.get("complexity", 0)),
             "loss": float(r.get("loss", 0))} for _, r in m7I.equations_.iterrows()]
except Exception:
    exprR = "error"; exprI = "error"; eqsR = []; eqsI = []
with open(os.path.join(d, "saved_model/expressions.json"), "w") as f:
    json.dump({"best_R": exprR, "best_I": exprI,
               "pareto_front_R": eqsR, "pareto_front_I": eqsI}, f, indent=2, default=str)
joblib.dump({"modelR": m7R, "modelI": m7I}, os.path.join(d, "saved_model/model.joblib"))
rec(7, "pysr_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 40,
    "PySR symbolic regression, raw spin, separate R/I fits")

# --- 8. PySR neglog ---
print("\n=== 8: PySR neglog ===")
d = md(8, "pysr_neglog")
x_nl_1d_tr = ps["neglog"].reshape(-1, 1)
x_nl_1d_va = pv["neglog"].reshape(-1, 1)
m8R = PySRRegressor(niterations=40, maxsize=20, populations=10, procs=1,
                    verbosity=0, progress=False, random_state=42, temp_equation_file=True,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "exp", "sin", "cos"])
m8R.fit(x_nl_1d_tr, wr_tr)
m8I = PySRRegressor(niterations=40, maxsize=20, populations=10, procs=1,
                    verbosity=0, progress=False, random_state=42, temp_equation_file=True,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "exp", "sin", "cos"])
m8I.fit(x_nl_1d_tr, wi_tr)

t0 = time.time()
pR = m8R.predict(x_nl_1d_va); pI = m8I.predict(x_nl_1d_va)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = m8R.predict(x_nl_1d_tr); pI_tr = m8I.predict(x_nl_1d_tr)
try:
    exprR = str(m8R.sympy()); exprI = str(m8I.sympy())
    eqsR = [{"expression": str(r.get("equation", "")), "complexity": int(r.get("complexity", 0)),
             "loss": float(r.get("loss", 0))} for _, r in m8R.equations_.iterrows()]
    eqsI = [{"expression": str(r.get("equation", "")), "complexity": int(r.get("complexity", 0)),
             "loss": float(r.get("loss", 0))} for _, r in m8I.equations_.iterrows()]
except Exception:
    exprR = "error"; exprI = "error"; eqsR = []; eqsI = []
with open(os.path.join(d, "saved_model/expressions.json"), "w") as f:
    json.dump({"best_R": exprR, "best_I": exprI,
               "pareto_front_R": eqsR, "pareto_front_I": eqsI}, f, indent=2, default=str)
joblib.dump({"modelR": m8R, "modelI": m8I}, os.path.join(d, "saved_model/model.joblib"))
rec(8, "pysr_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, 40,
    "PySR symbolic regression, neglog reparam, separate R/I fits")

# --- 9. gplearn raw ---
print("\n=== 9: gplearn raw ===")
d = md(9, "gplearn_raw")
gp9R = SymbolicRegressor(population_size=2000, generations=30, verbose=0, random_state=42)
gp9R.fit(a_tr.reshape(-1, 1), wr_tr)
gp9I = SymbolicRegressor(population_size=2000, generations=30, verbose=0, random_state=42)
gp9I.fit(a_tr.reshape(-1, 1), wi_tr)

t0 = time.time()
pR = gp9R.predict(a_va.reshape(-1, 1)); pI = gp9I.predict(a_va.reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = gp9R.predict(a_tr.reshape(-1, 1)); pI_tr = gp9I.predict(a_tr.reshape(-1, 1))
with open(os.path.join(d, "saved_model/expressions.json"), "w") as f:
    json.dump({"expression_R": str(gp9R._program), "expression_I": str(gp9I._program),
               "complexity_R": gp9R._program.length_, "complexity_I": gp9I._program.length_,
               "fitness_R": float(gp9R._program.fitness_), "fitness_I": float(gp9I._program.fitness_)},
              f, indent=2, default=str)
joblib.dump({"modelR": gp9R, "modelI": gp9I}, os.path.join(d, "saved_model/model.joblib"))
rec(9, "gplearn_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 60,
    "gplearn symbolic regression, raw spin, separate R/I fits")

# --- 10. gplearn neglog ---
print("\n=== 10: gplearn neglog ===")
d = md(10, "gplearn_neglog")
gp10R = SymbolicRegressor(population_size=2000, generations=30, verbose=0, random_state=42)
gp10R.fit(x_nl_1d_tr, wr_tr)
gp10I = SymbolicRegressor(population_size=2000, generations=30, verbose=0, random_state=42)
gp10I.fit(x_nl_1d_tr, wi_tr)

t0 = time.time()
pR = gp10R.predict(x_nl_1d_va); pI = gp10I.predict(x_nl_1d_va)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = gp10R.predict(x_nl_1d_tr); pI_tr = gp10I.predict(x_nl_1d_tr)
with open(os.path.join(d, "saved_model/expressions.json"), "w") as f:
    json.dump({"expression_R": str(gp10R._program), "expression_I": str(gp10I._program),
               "complexity_R": gp10R._program.length_, "complexity_I": gp10I._program.length_,
               "fitness_R": float(gp10R._program.fitness_), "fitness_I": float(gp10I._program.fitness_)},
              f, indent=2, default=str)
joblib.dump({"modelR": gp10R, "modelI": gp10I}, os.path.join(d, "saved_model/model.joblib"))
rec(10, "gplearn_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, 60,
    "gplearn symbolic regression, neglog reparam, separate R/I fits")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3: INTERPOLATION (4 models)
# ═══════════════════════════════════════════════════════════════════════════════

from scipy.interpolate import CubicSpline, RBFInterpolator
from sklearn.neighbors import KNeighborsRegressor

# --- 11. Cubic spline (raw, sorted) ---
print("\n=== 11: Cubic spline (raw) ===")
d = md(11, "cubic_spline_raw")
csR = CubicSpline(a_tr_s, wr_tr_s)
csI = CubicSpline(a_tr_s, wi_tr_s)
t0 = time.time()
pR = csR(a_va); pI = csI(a_va)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = csR(a_tr); pI_tr = csI(a_tr)
joblib.dump({"csR_c": csR.c.tolist(), "csR_x": csR.x.tolist(),
             "csI_c": csI.c.tolist(), "csI_x": csI.x.tolist()},
            os.path.join(d, "saved_model/model.joblib"))
rec(11, "cubic_spline_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, N_TR * 4,
    "Cubic spline interpolation, raw spin (sorted)")

# --- 12. RBF thin_plate_spline (raw) ---
print("\n=== 12: RBF thin_plate_spline (raw) ===")
d = md(12, "rbf_tps_raw")
rbfR12 = RBFInterpolator(a_tr.reshape(-1, 1), wr_tr, kernel='thin_plate_spline', smoothing=1e-4)
rbfI12 = RBFInterpolator(a_tr.reshape(-1, 1), wi_tr, kernel='thin_plate_spline', smoothing=1e-4)
t0 = time.time()
pR = rbfR12(a_va.reshape(-1, 1)); pI = rbfI12(a_va.reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = rbfR12(a_tr.reshape(-1, 1)); pI_tr = rbfI12(a_tr.reshape(-1, 1))
joblib.dump({"rbfR": rbfR12, "rbfI": rbfI12}, os.path.join(d, "saved_model/model.joblib"))
rec(12, "rbf_tps_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, N_TR,
    "RBF thin_plate_spline interpolation, raw spin")

# --- 13. RBF linear (neglog) ---
print("\n=== 13: RBF linear (neglog) ===")
d = md(13, "rbf_linear_neglog")
rbfR13 = RBFInterpolator(ps["neglog"].reshape(-1, 1), wr_tr, kernel='linear', smoothing=1e-4)
rbfI13 = RBFInterpolator(ps["neglog"].reshape(-1, 1), wi_tr, kernel='linear', smoothing=1e-4)
t0 = time.time()
pR = rbfR13(pv["neglog"].reshape(-1, 1)); pI = rbfI13(pv["neglog"].reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = rbfR13(ps["neglog"].reshape(-1, 1)); pI_tr = rbfI13(ps["neglog"].reshape(-1, 1))
joblib.dump({"rbfR": rbfR13, "rbfI": rbfI13}, os.path.join(d, "saved_model/model.joblib"))
rec(13, "rbf_linear_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, N_TR,
    "RBF linear interpolation, neglog reparam")

# --- 14. KNN (raw) ---
print("\n=== 14: KNN (raw) ===")
d = md(14, "knn_raw")
knnR14 = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(a_tr.reshape(-1, 1), wr_tr)
knnI14 = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(a_tr.reshape(-1, 1), wi_tr)
t0 = time.time()
pR = knnR14.predict(a_va.reshape(-1, 1)); pI = knnI14.predict(a_va.reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = knnR14.predict(a_tr.reshape(-1, 1)); pI_tr = knnI14.predict(a_tr.reshape(-1, 1))
joblib.dump({"knnR": knnR14, "knnI": knnI14}, os.path.join(d, "saved_model/model.joblib"))
rec(14, "knn_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 0,
    "5-NN distance-weighted, raw spin")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4: ML (12 models)
# ═══════════════════════════════════════════════════════════════════════════════

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPR_RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

# Best parameterization choice for ML: scale the input
sc_raw = StandardScaler().fit(a_tr.reshape(-1, 1))
a_tr_sc = sc_raw.transform(a_tr.reshape(-1, 1))
a_va_sc = sc_raw.transform(a_va.reshape(-1, 1))

sc_nl = StandardScaler().fit(ps["neglog"].reshape(-1, 1))
a_tr_nl_sc = sc_nl.transform(ps["neglog"].reshape(-1, 1))
a_va_nl_sc = sc_nl.transform(pv["neglog"].reshape(-1, 1))

sc_sq = StandardScaler().fit(ps["sqrt1ma2"].reshape(-1, 1))
a_tr_sq_sc = sc_sq.transform(ps["sqrt1ma2"].reshape(-1, 1))
a_va_sq_sc = sc_sq.transform(pv["sqrt1ma2"].reshape(-1, 1))

sc_ls = StandardScaler().fit(ps["linear_scaled"].reshape(-1, 1))
a_tr_ls_sc = sc_ls.transform(ps["linear_scaled"].reshape(-1, 1))
a_va_ls_sc = sc_ls.transform(pv["linear_scaled"].reshape(-1, 1))

# --- 15. GPR RBF (neglog) ---
print("\n=== 15: GPR RBF (neglog) ===")
d = md(15, "gpr_rbf_neglog")
kR15 = ConstantKernel(1.0) * GPR_RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
gprR15 = GaussianProcessRegressor(kernel=kR15, n_restarts_optimizer=5, alpha=1e-8, normalize_y=True)
gprR15.fit(a_tr_nl_sc, wr_tr)
kI15 = ConstantKernel(1.0) * GPR_RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
gprI15 = GaussianProcessRegressor(kernel=kI15, n_restarts_optimizer=5, alpha=1e-8, normalize_y=True)
gprI15.fit(a_tr_nl_sc, wi_tr)
t0 = time.time()
pR = gprR15.predict(a_va_nl_sc); pI = gprI15.predict(a_va_nl_sc)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = gprR15.predict(a_tr_nl_sc); pI_tr = gprI15.predict(a_tr_nl_sc)
joblib.dump({"gprR": gprR15, "gprI": gprI15, "scaler": sc_nl}, os.path.join(d, "saved_model/model.joblib"))
rec(15, "gpr_rbf_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, N_TR,
    "GPR RBF kernel, neglog reparam")

# --- 16. GPR Matern (neglog) ---
print("\n=== 16: GPR Matern (neglog) ===")
d = md(16, "gpr_matern_neglog")
kR16 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
gprR16 = GaussianProcessRegressor(kernel=kR16, n_restarts_optimizer=5, alpha=1e-8, normalize_y=True)
gprR16.fit(a_tr_nl_sc, wr_tr)
kI16 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
gprI16 = GaussianProcessRegressor(kernel=kI16, n_restarts_optimizer=5, alpha=1e-8, normalize_y=True)
gprI16.fit(a_tr_nl_sc, wi_tr)
t0 = time.time()
pR = gprR16.predict(a_va_nl_sc); pI = gprI16.predict(a_va_nl_sc)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = gprR16.predict(a_tr_nl_sc); pI_tr = gprI16.predict(a_tr_nl_sc)
joblib.dump({"gprR": gprR16, "gprI": gprI16, "scaler": sc_nl}, os.path.join(d, "saved_model/model.joblib"))
rec(16, "gpr_matern_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, N_TR,
    "GPR Matern-5/2 kernel, neglog reparam")

# --- 17. KRR (neglog) ---
print("\n=== 17: KRR (neglog) ===")
d = md(17, "krr_neglog")
krrR17 = KernelRidge(kernel='rbf', alpha=1e-6, gamma=0.5).fit(a_tr_nl_sc, wr_tr)
krrI17 = KernelRidge(kernel='rbf', alpha=1e-6, gamma=0.5).fit(a_tr_nl_sc, wi_tr)
t0 = time.time()
pR = krrR17.predict(a_va_nl_sc); pI = krrI17.predict(a_va_nl_sc)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = krrR17.predict(a_tr_nl_sc); pI_tr = krrI17.predict(a_tr_nl_sc)
joblib.dump({"krrR": krrR17, "krrI": krrI17, "scaler": sc_nl}, os.path.join(d, "saved_model/model.joblib"))
rec(17, "krr_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, N_TR,
    "KRR RBF kernel, neglog reparam")

# --- 18. MLP (neglog) ---
print("\n=== 18: MLP (neglog) ===")
d = md(18, "mlp_neglog")
# Train on 2D output: [omega_R, omega_I]
y_tr_2d = np.column_stack([wr_tr, wi_tr])
sy18 = StandardScaler().fit(y_tr_2d)
y_tr_2d_s = sy18.transform(y_tr_2d)
mlp18 = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=5000,
                     early_stopping=True, validation_fraction=0.15,
                     random_state=42, learning_rate_init=0.001)
mlp18.fit(a_tr_nl_sc, y_tr_2d_s)
t0 = time.time()
pred_2d_s = mlp18.predict(a_va_nl_sc)
pred_2d = sy18.inverse_transform(pred_2d_s)
rt = (time.time() - t0) / N_VA * 1000
pR = pred_2d[:, 0]; pI = pred_2d[:, 1]
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pred_tr_2d = sy18.inverse_transform(mlp18.predict(a_tr_nl_sc))
pR_tr = pred_tr_2d[:, 0]; pI_tr = pred_tr_2d[:, 1]
joblib.dump({"mlp": mlp18, "scaler_X": sc_nl, "scaler_y": sy18}, os.path.join(d, "saved_model/model.joblib"))
rec(18, "mlp_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt,
    sum(c.size for c in mlp18.coefs_), "MLP [256,128,64], neglog reparam, 2D output")

# --- 19. MLP (sqrt1ma2) ---
print("\n=== 19: MLP (sqrt1ma2) ===")
d = md(19, "mlp_sqrt1ma2")
mlp19 = MLPRegressor(hidden_layer_sizes=(512, 256, 128), max_iter=5000,
                     early_stopping=True, validation_fraction=0.15,
                     random_state=42, learning_rate_init=0.0005)
mlp19.fit(a_tr_sq_sc, y_tr_2d_s)
t0 = time.time()
pred_2d = sy18.inverse_transform(mlp19.predict(a_va_sq_sc))
rt = (time.time() - t0) / N_VA * 1000
pR = pred_2d[:, 0]; pI = pred_2d[:, 1]
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pred_tr_2d = sy18.inverse_transform(mlp19.predict(a_tr_sq_sc))
pR_tr = pred_tr_2d[:, 0]; pI_tr = pred_tr_2d[:, 1]
joblib.dump({"mlp": mlp19, "scaler_X": sc_sq, "scaler_y": sy18}, os.path.join(d, "saved_model/model.joblib"))
rec(19, "mlp_sqrt1ma2", "sqrt1ma2", loss, pR, pI, pR_tr, pI_tr, rt,
    sum(c.size for c in mlp19.coefs_), "MLP [512,256,128], sqrt(1-a^2) reparam")

# --- 20. RF (raw) ---
print("\n=== 20: RF (raw) ===")
d = md(20, "rf_raw")
rfR20 = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
rfR20.fit(a_tr.reshape(-1, 1), wr_tr)
rfI20 = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
rfI20.fit(a_tr.reshape(-1, 1), wi_tr)
t0 = time.time()
pR = rfR20.predict(a_va.reshape(-1, 1)); pI = rfI20.predict(a_va.reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = rfR20.predict(a_tr.reshape(-1, 1)); pI_tr = rfI20.predict(a_tr.reshape(-1, 1))
joblib.dump({"rfR": rfR20, "rfI": rfI20}, os.path.join(d, "saved_model/model.joblib"))
rec(20, "rf_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 500 * 200 * 2,
    "Random Forest 500 trees, raw spin")

# --- 21. RF (neglog) ---
print("\n=== 21: RF (neglog) ===")
d = md(21, "rf_neglog")
rfR21 = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
rfR21.fit(ps["neglog"].reshape(-1, 1), wr_tr)
rfI21 = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
rfI21.fit(ps["neglog"].reshape(-1, 1), wi_tr)
t0 = time.time()
pR = rfR21.predict(pv["neglog"].reshape(-1, 1)); pI = rfI21.predict(pv["neglog"].reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = rfR21.predict(ps["neglog"].reshape(-1, 1)); pI_tr = rfI21.predict(ps["neglog"].reshape(-1, 1))
joblib.dump({"rfR": rfR21, "rfI": rfI21}, os.path.join(d, "saved_model/model.joblib"))
rec(21, "rf_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, 500 * 200 * 2,
    "Random Forest 500 trees, neglog reparam")

# --- 22. GBR (neglog) ---
print("\n=== 22: GBR (neglog) ===")
d = md(22, "gbr_neglog")
gbrR22 = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, random_state=42)
gbrR22.fit(ps["neglog"].reshape(-1, 1), wr_tr)
gbrI22 = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, random_state=42)
gbrI22.fit(ps["neglog"].reshape(-1, 1), wi_tr)
t0 = time.time()
pR = gbrR22.predict(pv["neglog"].reshape(-1, 1)); pI = gbrI22.predict(pv["neglog"].reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = gbrR22.predict(ps["neglog"].reshape(-1, 1)); pI_tr = gbrI22.predict(ps["neglog"].reshape(-1, 1))
joblib.dump({"gbrR": gbrR22, "gbrI": gbrI22}, os.path.join(d, "saved_model/model.joblib"))
rec(22, "gbr_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, 500 * 2,
    "GBR 500 estimators, neglog reparam")

# --- 23. SVR (neglog) ---
print("\n=== 23: SVR (neglog) ===")
d = md(23, "svr_neglog")
svrR23 = SVR(kernel='rbf', C=100.0, epsilon=0.001).fit(a_tr_nl_sc, wr_tr)
svrI23 = SVR(kernel='rbf', C=100.0, epsilon=0.001).fit(a_tr_nl_sc, wi_tr)
t0 = time.time()
pR = svrR23.predict(a_va_nl_sc); pI = svrI23.predict(a_va_nl_sc)
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = svrR23.predict(a_tr_nl_sc); pI_tr = svrI23.predict(a_tr_nl_sc)
joblib.dump({"svrR": svrR23, "svrI": svrI23, "scaler": sc_nl}, os.path.join(d, "saved_model/model.joblib"))
rec(23, "svr_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, N_TR,
    "SVR RBF C=100, neglog reparam")

# --- 24. ExtraTrees (raw) ---
print("\n=== 24: ExtraTrees (raw) ===")
d = md(24, "extratrees_raw")
etR24 = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                            random_state=42, n_jobs=-1)
etR24.fit(a_tr.reshape(-1, 1), wr_tr)
etI24 = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                            random_state=42, n_jobs=-1)
etI24.fit(a_tr.reshape(-1, 1), wi_tr)
t0 = time.time()
pR = etR24.predict(a_va.reshape(-1, 1)); pI = etI24.predict(a_va.reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = etR24.predict(a_tr.reshape(-1, 1)); pI_tr = etI24.predict(a_tr.reshape(-1, 1))
joblib.dump({"etR": etR24, "etI": etI24}, os.path.join(d, "saved_model/model.joblib"))
rec(24, "extratrees_raw", "raw", loss, pR, pI, pR_tr, pI_tr, rt, 500 * 200 * 2,
    "ExtraTrees 500, raw spin")

# --- 25. ExtraTrees (neglog) ---
print("\n=== 25: ExtraTrees (neglog) ===")
d = md(25, "extratrees_neglog")
etR25 = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                            random_state=42, n_jobs=-1)
etR25.fit(ps["neglog"].reshape(-1, 1), wr_tr)
etI25 = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                            random_state=42, n_jobs=-1)
etI25.fit(ps["neglog"].reshape(-1, 1), wi_tr)
t0 = time.time()
pR = etR25.predict(pv["neglog"].reshape(-1, 1)); pI = etI25.predict(pv["neglog"].reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = etR25.predict(ps["neglog"].reshape(-1, 1)); pI_tr = etI25.predict(ps["neglog"].reshape(-1, 1))
joblib.dump({"etR": etR25, "etI": etI25}, os.path.join(d, "saved_model/model.joblib"))
rec(25, "extratrees_neglog", "neglog", loss, pR, pI, pR_tr, pI_tr, rt, 500 * 200 * 2,
    "ExtraTrees 500, neglog reparam")

# --- 26. GBR (sqrt1ma2) ---
print("\n=== 26: GBR (sqrt1ma2) ===")
d = md(26, "gbr_sqrt1ma2")
gbrR26 = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                                   subsample=0.8, random_state=42)
gbrR26.fit(ps["sqrt1ma2"].reshape(-1, 1), wr_tr)
gbrI26 = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                                   subsample=0.8, random_state=42)
gbrI26.fit(ps["sqrt1ma2"].reshape(-1, 1), wi_tr)
t0 = time.time()
pR = gbrR26.predict(pv["sqrt1ma2"].reshape(-1, 1)); pI = gbrI26.predict(pv["sqrt1ma2"].reshape(-1, 1))
rt = (time.time() - t0) / N_VA * 1000
loss = ringdown_loss(pR, wr_va, pI, wi_va)
pR_tr = gbrR26.predict(ps["sqrt1ma2"].reshape(-1, 1)); pI_tr = gbrI26.predict(ps["sqrt1ma2"].reshape(-1, 1))
joblib.dump({"gbrR": gbrR26, "gbrI": gbrI26}, os.path.join(d, "saved_model/model.joblib"))
rec(26, "gbr_sqrt1ma2", "sqrt1ma2", loss, pR, pI, pR_tr, pI_tr, rt, 500 * 2,
    "GBR 500, sqrt(1-a^2) reparam")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS & GENERATE PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Saving results ===")
os.makedirs(os.path.join(WORK_DIR, "comparison"), exist_ok=True)

with open(os.path.join(WORK_DIR, "comparison/error_data.json"), "w") as f:
    json.dump(err_data, f)

results.sort(key=lambda x: x["loss"])
with open(os.path.join(WORK_DIR, "comparison/summary_table.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(WORK_DIR, "comparison/best_model.json"), "w") as f:
    json.dump(results[0], f, indent=2)
with open(os.path.join(WORK_DIR, "CHANGELOG.md"), "w") as f:
    f.write("# Ringdown Benchmark CHANGELOG -- Opus 4.6\n\n")
    f.write("Mode: l2/m+2/n0 | Loss: mean relative error (omega_R, omega_I)\n\n")
    for c in cl:
        f.write(c + "\n")

# Generate train.py / predict.py for each model
for r in results:
    n = r["approach_number"]; name = r["approach"]; sch = r["parameterization"]
    dpath = os.path.join(WORK_DIR, f"models/{n:02d}_{name}")
    with open(os.path.join(dpath, "train.py"), "w") as f:
        f.write(f'#!/usr/bin/env python3\n"""Training script for {name}."""\n# See build_all.py approach {n}\n')
    with open(os.path.join(dpath, "predict.py"), "w") as f:
        f.write(f'''#!/usr/bin/env python3
"""Prediction function for {name} (ringdown l2/m+2/n0)."""
import os, numpy as np, joblib
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

def reparam(a, scheme="{sch}"):
    if scheme == "raw": return a
    elif scheme == "neglog": return -np.log(1.0 - a + 1e-15)
    elif scheme == "sqrt1ma2": return np.sqrt(1.0 - a**2)
    elif scheme == "linear_scaled": return 2.0 * a - 1.0
    return a

def predict(spin):
    """Predict (omega_real, omega_imag) from spin value(s)."""
    saved = joblib.load(os.path.join(WORK_DIR, "saved_model/model.joblib"))
    a = np.atleast_1d(np.asarray(spin, dtype=float))
    x = reparam(a, "{sch}")
    # Model-specific prediction logic
    if "coeffR" in saved:
        return np.polyval(saved["coeffR"], x), np.polyval(saved["coeffI"], x)
    X = x.reshape(-1, 1)
    if "scaler" in saved:
        X = saved["scaler"].transform(X)
    if "poly" in saved:
        X = saved["poly"].transform(X)
    # Two-output models
    for kR, kI in [("ridgeR","ridgeI"),("gprR","gprI"),("krrR","krrI"),
                   ("svrR","svrI"),("rfR","rfI"),("gbrR","gbrI"),
                   ("etR","etI"),("knnR","knnI"),("rbfR","rbfI"),
                   ("modelR","modelI")]:
        if kR in saved and kI in saved:
            mR, mI = saved[kR], saved[kI]
            yR = mR(X) if callable(mR) else mR.predict(X)
            yI = mI(X) if callable(mI) else mI.predict(X)
            return yR, yI
    # MLP 2D output
    if "mlp" in saved:
        pred = saved["mlp"].predict(X)
        if "scaler_y" in saved:
            pred = saved["scaler_y"].inverse_transform(pred)
        return pred[:, 0], pred[:, 1]
    raise ValueError("Unknown model format")
''')


# ═══════ PLOTS ═══════════════════════════════════════════════════════════════

names = [r["approach"] for r in results]
losses = [r["loss"] for r in results]
rts = [r["runtime_ms"] for r in results]

CAT_COLORS = {
    "analytical": pset.COLORS["blue"],
    "symbolic": pset.COLORS["green"],
    "interpolation": pset.COLORS["orange"],
    "ml": pset.COLORS["red"],
}

def cat(name):
    nl = name.lower()
    if any(x in nl for x in ["poly", "cheby", "pade", "ridge_poly"]):
        return "analytical"
    if any(x in nl for x in ["pysr", "gplearn"]):
        return "symbolic"
    if any(x in nl for x in ["cubic_spline", "rbf_tps", "rbf_linear", "knn"]):
        return "interpolation"
    return "ml"

cats = [cat(n) for n in names]
cols = [CAT_COLORS[c] for c in cats]

# 1. Progress bar chart (horizontal)
fig, ax = plt.subplots(figsize=pset.figsize(2, 0.5))
y_pos = np.arange(len(results))
ax.barh(y_pos, losses, color=cols, height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=6)
ax.set_xlabel("Mean Relative Error")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "comparison/progress.png"))
fig.savefig(os.path.join(WORK_DIR, "comparison/progress.pdf"))
plt.close()

# 2. Pareto accuracy vs speed
fig, ax = plt.subplots(figsize=pset.figsize(1, 0.9))
for c in ["analytical", "symbolic", "interpolation", "ml"]:
    idx = [i for i, x in enumerate(cats) if x == c]
    if idx:
        ax.scatter([losses[i] for i in idx], [rts[i] for i in idx],
                   c=CAT_COLORS[c], label=c.capitalize(), s=30, zorder=5)
        for i in idx:
            ax.annotate(names[i].replace("_", " "), (losses[i], rts[i]),
                        fontsize=4.5, ha='left', rotation=10)
ax.set_xlabel("Mean Relative Error")
ax.set_ylabel("Runtime (ms/sample)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "comparison/pareto_accuracy_speed.png"))
fig.savefig(os.path.join(WORK_DIR, "comparison/pareto_accuracy_speed.pdf"))
plt.close()

# 3. Loss-only comparison (same as progress but clean)
fig, ax = plt.subplots(figsize=pset.figsize(2, 0.5))
ax.barh(y_pos, losses, color=cols, height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=6)
ax.set_xlabel("Mean Relative Error")
ax.set_xscale("log")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "comparison/loss_only_comparison.png"))
fig.savefig(os.path.join(WORK_DIR, "comparison/loss_only_comparison.pdf"))
plt.close()

# 4. Error histograms (top 10)
fig, axes = plt.subplots(1, 2, figsize=pset.figsize(2, 0.7))
sorted_names = sorted(err_data.keys(), key=lambda n: np.mean(err_data[n]["val_rel_err_R"]) + np.mean(err_data[n]["val_rel_err_I"]))[:10]

for ax, key, label in zip(axes, ["val_rel_err_R", "val_rel_err_I"],
                           [r"$\omega_R$ relative error", r"$\omega_I$ relative error"]):
    for name in sorted_names:
        vals = err_data[name][key]
        ax.hist(vals, bins=50, alpha=0.4, label=name.replace("_", " "),
                histtype='stepfilled', linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.legend(fontsize=4, ncol=2)
    ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "comparison/error_histograms.png"))
fig.savefig(os.path.join(WORK_DIR, "comparison/error_histograms.pdf"))
plt.close()


# ═══════ FINAL SUMMARY ══════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RINGDOWN BENCHMARK COMPLETE -- {len(results)} approaches")
print(f"{'='*60}")
for r in results:
    print(f"  {r['approach_number']:02d} {r['approach']:<25s} Loss={r['loss']:.8f}  [{r['parameterization']}]")
print(f"\nBest: {results[0]['approach']} (Loss={results[0]['loss']:.8f})")
print(f"Results in: {WORK_DIR}/comparison/")
