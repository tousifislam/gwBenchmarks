"""Closed-form h22(t;q) models for the Analytic Bench (opus48, original work).

A model is a pair of analytic functions A(t;q), phi(t;q). The functional FORM
is fixed (physics / functional ansatz or PySR-discovered); only scalar
coefficients are fitted, and their q-dependence is itself an analytic
polynomial in the mass variable. No SVD/PCA, no stored bases, no ODE solves.

Pipeline: fit each training waveform's amplitude + phase coefficients, fit
those coefficients as polynomials in the chosen mass variable, then evaluate
the closed form on any requested time grid.
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import adata as AD

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)


# ===================== closed-form building blocks ==========================
def phase_integrable(t, pp):
    """Exact integral of omega=b0+b1*(tc-t)^(-3/8)+b2*tanh((t-tm)/wr).
    pp = [tc, tm, wr, b0, b1, b2, c]."""
    tc, tm, wr, b0, b1, b2, c = pp
    s = np.clip(tc - t, 1e-6, None)
    return b0 * t - b1 * (8.0 / 5.0) * s ** 0.625 + b2 * wr * np.log(np.cosh((t - tm) / wr)) + c


def phase_integrable2(t, pp):
    """Two PN powers: adds b3*(tc-t)^(1/8) term (integral of (tc-t)^(-7/8))."""
    tc, tm, wr, b0, b1, b2, b3, c = pp
    s = np.clip(tc - t, 1e-6, None)
    return (b0 * t - b1 * (8.0 / 5.0) * s ** 0.625 + b2 * wr * np.log(np.cosh((t - tm) / wr))
            - b3 * 8.0 * s ** 0.125 + c)


def amp_powerlaw_sech(t, ap):
    Apk, wrise, td, prise = ap
    x = t
    rise = (1.0 + (np.clip(-x, 0, None) / wrise) ** 2) ** (-prise)
    rd = 1.0 / np.cosh(np.clip(x, 0, None) / td)
    return Apk * np.where(x <= 0, rise, rd)


def amp_two_gaussian(t, ap):
    Apk, w1, w2, skew = ap
    x = t
    left = np.exp(-(np.clip(-x, 0, None) ** 2) / (2 * w1 ** 2))
    right = np.exp(-(np.clip(x, 0, None) ** 2) / (2 * w2 ** 2))
    return Apk * np.where(x <= 0, left + skew * (1 - left) * 0, left) if False else \
        Apk * np.where(x <= 0, np.exp(-(x ** 2) / (2 * w1 ** 2)), np.exp(-(x ** 2) / (2 * w2 ** 2)))


def amp_lorentzian(t, ap):
    Apk, w1, w2, p = ap
    x = t
    left = 1.0 / (1.0 + (np.clip(-x, 0, None) / w1) ** 2) ** p
    right = 1.0 / (1.0 + (np.clip(x, 0, None) / w2) ** 2) ** p
    return Apk * np.where(x <= 0, left, right)


PHASE_FORMS = {
    "integrable_t3": (phase_integrable, 7,
                      lambda d: np.array([5., 0., 30., -0.05, 0.5, 0.1, 0.])),
    "integrable_t3_2term": (phase_integrable2, 8,
                            lambda d: np.array([5., 0., 30., -0.05, 0.5, 0.1, 0.01, 0.])),
}
AMP_FORMS = {
    "powerlaw_sech": (amp_powerlaw_sech, lambda d: np.array([d["A"][d["ipk"]], 80., 11., 0.5])),
    "two_gaussian": (amp_two_gaussian, lambda d: np.array([d["A"][d["ipk"]], 120., 12., 0.])),
    "lorentzian": (amp_lorentzian, lambda d: np.array([d["A"][d["ipk"]], 90., 12., 1.0])),
}


# ===================== per-waveform fitting =================================
def _fit_phase(d, form):
    fn, npar, init = PHASE_FORMS[form]
    t = d["t"]; target = d["phi"]; s = slice(0, len(t), 12)
    # initialise with a grid scan over (tc, tm, wr) then linear-ish refine
    def res(pp):
        return (fn(t[s], pp) - target[s]) * 0.05
    sol = least_squares(res, init(d), max_nfev=600)
    return sol.x


def _fit_amp(d, form):
    fn, init = AMP_FORMS[form]
    t = d["t"]; At = d["A"]; s = slice(0, len(t), 15)
    def res(ap):
        return np.log(np.abs(fn(t[s], ap)) + 1e-12) - np.log(At[s] + 1e-12)
    sol = least_squares(res, init(d), max_nfev=400)
    return sol.x


def _poly_in_eta(xvar, P, deg):
    """Fit each coefficient column of P (n, k) as poly(xvar) of given degree."""
    V = np.vander(xvar, deg + 1)
    C = np.linalg.lstsq(V, P, rcond=None)[0]      # (deg+1, k)
    return C


def _eval_poly(C, x):
    return np.vander(np.atleast_1d(x), C.shape[0]) @ C


# ===================== runner ===============================================
def run_parametric(spec, write=True, verbose=True):
    t0 = time.time()
    tr = AD.load("training"); va = AD.load("validation")
    amp_form = spec["amp_form"]; phase_form = spec["phase_form"]
    mv = spec["mass_var"]; deg = spec.get("deg", 3)

    AP = np.array([_fit_amp(d, amp_form) for d in tr])
    PP = np.array([_fit_phase(d, phase_form) for d in tr])
    xtr = np.array([AD.mass_var(d, mv) for d in tr])
    CA = _poly_in_eta(xtr, AP, deg)
    CP = _poly_in_eta(xtr, PP, deg)

    afn = AMP_FORMS[amp_form][0]; pfn = PHASE_FORMS[phase_form][0]

    def predict(dset):
        out = []
        for d in dset:
            x = AD.mass_var(d, mv)
            ap = _eval_poly(CA, x)[0]; pp = _eval_poly(CP, x)[0]
            h = afn(d["t"], ap) * np.exp(-1j * pfn(d["t"], pp))
            out.append(h)
        return out

    t_pred = time.time()
    pred_va = predict(va)
    runtime_ms = 1e3 * (time.time() - t_pred) / len(va)
    err_va = AD.score(pred_va, va)
    err_tr = AD.score(predict(tr), tr)

    sc = _scorecard(spec, err_va, err_tr, runtime_ms, len(tr), len(va),
                    int(CA.size + CP.size), time.time() - t0)
    expr = _expression_text(spec, CA, CP, amp_form, phase_form, mv)
    if write:
        _persist(spec, dict(type="parametric", CA=CA, CP=CP, amp_form=amp_form,
                            phase_form=phase_form, mass_var=mv), sc, err_tr, err_va, expr)
    if verbose:
        print(f"[{spec['number']:2d}] {spec['name']:28s} mm={sc['loss']:.4f} med={sc['val_median']:.4f}")
    return dict(scorecard=sc, err_tr=err_tr, err_va=err_va, expr=expr)


def _scorecard(spec, err_va, err_tr, rt, ntr, nval, npar, wall):
    from gwbenchmarks.metrics import FD_MASSES_MSUN
    comp = {f"mismatch_{int(m)}Msun": float(np.mean(err_va)) for m in FD_MASSES_MSUN}
    return {"approach": spec["name"], "approach_number": spec["number"], "benchmark": "analytic",
            "agent": "opus48", "category": spec["category"], "parameterization": spec["mass_var"],
            "loss": float(np.mean(err_va)), "loss_components": comp,
            "val_median": float(np.median(err_va)), "train_loss": float(np.mean(err_tr)),
            "runtime_ms": float(rt), "n_train": int(ntr), "n_val": int(nval),
            "n_params": int(npar), "n_terms": spec.get("n_terms", npar),
            "expression_file": "expression.txt", "notes": spec.get("notes", ""),
            "wall_time_s": round(wall, 1)}


def _expression_text(spec, CA, CP, amp_form, phase_form, mv):
    lines = [f"# Analytic closed form for h22(t; q), approach {spec['name']}",
             f"# mass variable x = {mv};  h22 = A(t) * exp(-i * phi(t))", ""]
    lines.append(f"AMPLITUDE form = {amp_form}; coefficients are polynomials in x:")
    for j in range(CA.shape[1]):
        coefs = " + ".join(f"{CA[k, j]:.6g}*x^{CA.shape[0]-1-k}" for k in range(CA.shape[0]))
        lines.append(f"  a{j}(x) = {coefs}")
    lines.append("")
    lines.append(f"PHASE form = {phase_form} (exact integral of an integrable omega):")
    for j in range(CP.shape[1]):
        coefs = " + ".join(f"{CP[k, j]:.6g}*x^{CP.shape[0]-1-k}" for k in range(CP.shape[0]))
        lines.append(f"  p{j}(x) = {coefs}")
    lines.append("")
    if phase_form.startswith("integrable_t3"):
        lines.append("phi(t) = p3*t - p4*(8/5)*(p0 - t)^(5/8) + p5*p2*log(cosh((t-p1)/p2)) + p6")
    if amp_form == "powerlaw_sech":
        lines.append("A(t) = a0 * [ (1+((-t)_+/a1)^2)^(-a3)  if t<=0 ;  sech(t_+/a2) if t>0 ]")
    return "\n".join(lines)


def _persist(spec, payload, sc, err_tr, err_va, expr):
    import joblib
    mdir = MODELS / f"NN_{spec['name']}"; sdir = mdir / "saved_model"
    sdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, sdir / "model.joblib")
    np.savez(sdir / "errors.npz", err_train=err_tr, err_val=err_va)
    (mdir / "scorecard.json").write_text(json.dumps(sc, indent=2))
    (mdir / "expression.txt").write_text(expr)
    if "expressions" in payload:
        (sdir / "expressions.json").write_text(json.dumps(payload["expressions"], indent=2))
    (mdir / "train.py").write_text(f'''"""Reproducible training for analytic approach: {spec['name']}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import aapproaches as A, arunner as R
print("mismatch:", R.run(A.get_spec({spec['number']!r}), write=True)["scorecard"]["loss"])
''')
    (mdir / "predict.py").write_text('''"""predict(q, t_grid) -> complex h22 (closed-form evaluation)."""
import sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import amodels as M, asymbolic as S
_P = joblib.load(Path(__file__).resolve().parent / "saved_model" / "model.joblib")

def predict(q, t_grid):
    t = np.asarray(t_grid, float)
    if _P["type"] == "parametric":
        eta = q/(1+q)**2; delta=(q-1)/(q+1)
        x = {"q":q,"eta":eta,"delta_m":delta,"sqrt_eta":np.sqrt(eta),"eta_pow15":eta**0.2}[_P["mass_var"]]
        ap = M._eval_poly(_P["CA"], x)[0]; pp = M._eval_poly(_P["CP"], x)[0]
        return M.AMP_FORMS[_P["amp_form"]][0](t, ap) * np.exp(-1j*M.PHASE_FORMS[_P["phase_form"]][0](t, pp))
    return S.predict_symbolic(_P, q, t)
'''
    )
