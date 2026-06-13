"""PySR / gplearn closed-form discovery for the Analytic Bench (opus48).

PySR is the primary tool: it discovers a closed-form expression for the
log-amplitude logA(s, x) where s = t/100 (scaled time) and x is the mass
variable. The phase uses the integrable closed-form backbone from amodels
(itself an analytic, PN-inspired formula). The full model stays closed-form.
"""
import json, time
from pathlib import Path
import numpy as np
import amodels as M
import adata as AD


def _pool_amp(tr, mv, per=120):
    S, X, Y = [], [], []
    for d in tr:
        idx = np.linspace(0, len(d["t"]) - 1, per).astype(int)
        s = d["t"][idx] / 100.0
        x = np.full(per, AD.mass_var(d, mv))
        y = np.log(np.clip(d["A"][idx], 1e-6, None))
        S.append(s); X.append(x); Y.append(y)
    return np.column_stack([np.concatenate(S), np.concatenate(X)]), np.concatenate(Y)


def _fit_pysr(Xy):
    X, y = Xy
    from pysr import PySRRegressor
    m = PySRRegressor(niterations=80, binary_operators=["+", "-", "*", "/"],
                      unary_operators=["square", "exp", "sqrt", "tanh", "log"],
                      maxsize=30, populations=20, progress=False, verbosity=0,
                      temp_equation_file=True, random_state=0, deterministic=True,
                      parallelism="serial")
    m.fit(X, y)
    e = []
    try:
        for _, r in m.equations_.iterrows():
            e.append({"equation": str(r["equation"]), "complexity": int(r["complexity"]),
                      "loss": float(r["loss"])})
    except Exception:
        pass
    return m, e


def _fit_gplearn(Xy):
    X, y = Xy
    from gplearn.genetic import SymbolicRegressor
    m = SymbolicRegressor(population_size=3000, generations=30, tournament_size=20,
                          function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
                          metric="mse", parsimony_coefficient=0.001, random_state=42, verbose=0)
    m.fit(X, y)
    e = [{"equation": str(m._program), "complexity": int(m._program.length_),
          "loss": float(m.run_details_["best_fitness"][-1])}]
    return m, e


def run_symbolic(spec, write=True, verbose=True):
    t0 = time.time()
    tr = AD.load("training"); va = AD.load("validation")
    mv = spec["mass_var"]; phase_form = spec.get("phase_form", "integrable_t3")
    deg = spec.get("deg", 3)

    # phase backbone (closed-form, fitted)
    PP = np.array([M._fit_phase(d, phase_form) for d in tr])
    xtr = np.array([AD.mass_var(d, mv) for d in tr])
    CP = M._poly_in_eta(xtr, PP, deg)
    pfn = M.PHASE_FORMS[phase_form][0]

    # symbolic amplitude
    Xy = _pool_amp(tr, mv)
    amp_model, exprs = (_fit_pysr if spec["backend"] == "pysr" else _fit_gplearn)(Xy)

    def predict(dset):
        out = []
        for d in dset:
            x = AD.mass_var(d, mv)
            s = d["t"] / 100.0
            feat = np.column_stack([s, np.full(len(s), x)])
            A = np.exp(np.asarray(amp_model.predict(feat)).ravel())
            pp = M._eval_poly(CP, x)[0]
            out.append(A * np.exp(-1j * pfn(d["t"], pp)))
        return out

    tp = time.time()
    pred_va = predict(va); runtime_ms = 1e3 * (time.time() - tp) / len(va)
    err_va = AD.score(pred_va, va); err_tr = AD.score(predict(tr), tr)

    sc = M._scorecard(spec, err_va, err_tr, runtime_ms, len(tr), len(va), 0, time.time() - t0)
    best = exprs[-1]["equation"] if exprs else ""
    expr_txt = (f"# Analytic closed form (symbolic {spec['backend']}), approach {spec['name']}\n"
                f"# x = {mv}; s = t/100; h22 = A(s,x)*exp(-i*phi(t))\n\n"
                f"log A(s, x) = {best}\n\n"
                f"phi(t): integrable closed-form backbone {phase_form} with x-polynomial coeffs.\n")
    payload = dict(type="symbolic", amp_model=amp_model, CP=CP, phase_form=phase_form,
                   mass_var=mv, expressions={"logA": exprs})
    if write:
        M._persist(spec, payload, sc, err_tr, err_va, expr_txt)
    if verbose:
        print(f"[{spec['number']:2d}] {spec['name']:28s} mm={sc['loss']:.4f} med={sc['val_median']:.4f}")
    return dict(scorecard=sc, err_tr=err_tr, err_va=err_va, expr=expr_txt)


def predict_symbolic(payload, q, t):
    eta = q / (1 + q) ** 2; delta = (q - 1) / (q + 1)
    x = {"q": q, "eta": eta, "delta_m": delta, "sqrt_eta": np.sqrt(eta),
         "eta_pow15": eta ** 0.2}[payload["mass_var"]]
    s = t / 100.0
    feat = np.column_stack([s, np.full(len(s), x)])
    A = np.exp(np.asarray(payload["amp_model"].predict(feat)).ravel())
    pp = M._eval_poly(payload["CP"], x)[0]
    return A * np.exp(-1j * M.PHASE_FORMS[payload["phase_form"]][0](t, pp))
