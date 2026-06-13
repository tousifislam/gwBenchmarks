"""Reusable surrogate machinery for the Waveform Bench (opus48, original work).

A waveform surrogate here = SVD bases for logA(tau) and phi(tau) + a pair of
multi-output coefficient regressors (params -> SVD coeffs). Every modelling
"approach" is just a different choice of (reparameterisation, SVD rank,
coefficient regressor). This module provides:

  * SVDBasis            - mean-subtracted truncated SVD with project/reconstruct
  * load_data           - cached features + tau matrices + endpoints
  * predict_waves       - coeffs -> complex waveforms on requested t grids
  * run_approach        - train, predict, score, and persist one approach

run_approach writes a self-contained model directory (train.py, predict.py,
saved_model/, scorecard.json) plus returns a summary used for the comparison
plots.
"""
import os, sys, json, time, importlib
import numpy as np
import joblib
from pathlib import Path

import common as C

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)
COMP = HERE / "comparison"
COMP.mkdir(exist_ok=True)


class SVDBasis:
    """Mean-subtracted truncated SVD over rows (one row per waveform)."""

    def __init__(self, M=None, rank=None):
        if M is None:
            return
        self.mu = M.mean(0)
        U, s, Vt = np.linalg.svd(M - self.mu, full_matrices=False)
        self.rank = rank
        self.s = s[:rank]
        self.V = Vt[:rank]              # (rank, N_TAU)

    def project(self, M):
        return (M - self.mu) @ self.V.T

    def reconstruct(self, coeffs):
        coeffs = np.atleast_2d(coeffs)
        return self.mu + coeffs @ self.V

    def to_dict(self):
        return {"type": "svd", "mu": self.mu, "V": self.V,
                "s": self.s, "rank": self.rank}

    @classmethod
    def from_dict(cls, d):
        o = cls()
        o.mu = d["mu"]; o.V = d["V"]; o.s = d["s"]; o.rank = int(d["rank"])
        return o


class EIMBasis:
    """Empirical Interpolation Method basis built greedily from an SVD basis.

    Coefficients here are the waveform VALUES at empirically-selected tau
    nodes (more interpretable / physical than abstract SVD coeffs).
    """

    def __init__(self, M=None, rank=None):
        if M is None:
            return
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        B = Vt[:rank].copy()              # (rank, N_TAU)
        nodes = [int(np.argmax(np.abs(B[0])))]
        for i in range(1, rank):
            A = B[:i][:, nodes].T          # (i, i)
            c = np.linalg.solve(A, B[i][nodes])
            r = B[i] - c @ B[:i]
            nodes.append(int(np.argmax(np.abs(r))))
        self.rank = rank
        self.B = B
        self.nodes = np.array(nodes)
        self.Binv = np.linalg.inv(B[:, self.nodes].T)   # node values -> coeffs

    def project(self, M):
        return M[:, self.nodes]            # values at empirical nodes

    def reconstruct(self, nodevals):
        nodevals = np.atleast_2d(nodevals)
        coeffs = nodevals @ self.Binv.T
        return coeffs @ self.B

    def to_dict(self):
        return {"type": "eim", "B": self.B, "nodes": self.nodes,
                "Binv": self.Binv, "rank": self.rank}

    @classmethod
    def from_dict(cls, d):
        o = cls()
        o.B = d["B"]; o.nodes = d["nodes"]; o.Binv = d["Binv"]
        o.rank = int(d["rank"])
        return o


def basis_from_dict(d):
    if d.get("type") == "eim":
        return EIMBasis.from_dict(d)
    return SVDBasis.from_dict(d)


def _make_basis(M, rank, kind):
    return EIMBasis(M, rank) if kind == "eim" else SVDBasis(M, rank)


def load_data(reparam_kind, rankA, rankP, basis_kind="svd"):
    """Return everything an approach needs, with bases built on training."""
    Ptr = C.load_split("training")[0]
    Pva = C.load_split("validation")[0]
    logA_tr, phi_tr, ends_tr = C.build_tau_matrices("training")
    logA_va, phi_va, ends_va = C.build_tau_matrices("validation")

    Xtr = C.reparam(Ptr, reparam_kind)
    Xva = C.reparam(Pva, reparam_kind)
    Xtr_s, mean, std = C.standardize(Xtr)
    Xva_s, _, _ = C.standardize(Xva, mean, std)

    basisA = _make_basis(logA_tr, rankA, basis_kind)
    basisP = _make_basis(phi_tr, rankP, basis_kind)
    cA_tr = basisA.project(logA_tr)
    cP_tr = basisP.project(phi_tr)

    return dict(Xtr=Xtr_s, Xva=Xva_s, mean=mean, std=std,
                basisA=basisA, basisP=basisP, cA_tr=cA_tr, cP_tr=cP_tr,
                ends_tr=ends_tr, ends_va=ends_va, reparam_kind=reparam_kind)


def predict_waves(modelA, modelP, basisA, basisP, X, ends, t_grids):
    """Reconstruct complex waveforms for feature rows X on given t grids."""
    cA = np.atleast_2d(modelA.predict(X))
    cP = np.atleast_2d(modelP.predict(X))
    waves = []
    for i in range(len(X)):
        logA_tau = basisA.reconstruct(cA[i])[0]
        phi_tau = basisP.reconstruct(cP[i])[0]
        t = t_grids[i]
        h = C.tau_to_wave(logA_tau, phi_tau, t, ends[i, 0], ends[i, 1])
        waves.append((t, h))
    return waves


def _score_subset(waves_split, pred_waves, idx, masses=None):
    true = [waves_split[i] for i in idx]
    pred = [pred_waves[j] for j in range(len(idx))]
    return C.score_waveforms(pred, true, masses=masses)


def run_approach(spec, n_val=None, n_train_eval=60, write_model_dir=True,
                 verbose=True):
    """Train + evaluate + persist one approach.

    spec keys:
      name (str), category (str), reparam (str), rankA (int), rankP (int),
      make_modelA (callable -> estimator), make_modelP (callable -> estimator),
      parameterization (str label), notes (str), number (int),
      extra (dict, optional, written into scorecard).
    """
    t_start = time.time()
    D = load_data(spec["reparam"], spec["rankA"], spec["rankP"],
                  basis_kind=spec.get("basis", "svd"))

    modelA = spec["make_modelA"]()
    modelP = spec["make_modelP"]()
    modelA.fit(D["Xtr"], D["cA_tr"])
    modelP.fit(D["Xtr"], D["cP_tr"])

    expr = {}
    if hasattr(modelA, "expressions_"):
        expr["logA_coeffs"] = modelA.expressions_
    if hasattr(modelP, "expressions_"):
        expr["phi_coeffs"] = modelP.expressions_
    if expr:
        spec.setdefault("extra", {})["expressions"] = expr

    # validation waves + grids
    Pva, waves_va = C.load_split("validation")
    Ptr, waves_tr = C.load_split("training")
    nval = len(waves_va) if n_val is None else min(n_val, len(waves_va))
    val_idx = np.arange(nval)
    t_grids_va = [waves_va[i][0] for i in val_idx]

    # measure runtime per waveform (predict only)
    t0 = time.time()
    pred_va = predict_waves(modelA, modelP, D["basisA"], D["basisP"],
                            D["Xva"][val_idx], D["ends_va"][val_idx], t_grids_va)
    runtime_ms = 1e3 * (time.time() - t0) / max(1, nval)

    err_va = _score_subset(waves_va, pred_va, val_idx)

    # train-error subset
    tr_idx = np.linspace(0, len(waves_tr) - 1, min(n_train_eval, len(waves_tr))).astype(int)
    t_grids_tr = [waves_tr[i][0] for i in tr_idx]
    pred_tr = predict_waves(modelA, modelP, D["basisA"], D["basisP"],
                            D["Xtr"][tr_idx], D["ends_tr"][tr_idx], t_grids_tr)
    err_tr = _score_subset(waves_tr, pred_tr, tr_idx)

    loss = float(np.mean(err_va))
    # per-mass components on val subset (smaller subset for speed)
    comp_idx = val_idx[:: max(1, nval // 40)]
    from gwbenchmarks.metrics import FD_MASSES_MSUN
    comp = {}
    for m in FD_MASSES_MSUN:
        e = C.score_waveforms([pred_va[i] for i in comp_idx],
                              [waves_va[i] for i in comp_idx], masses=[m])
        comp[f"mismatch_{int(m)}Msun"] = float(np.mean(e))

    n_params = _count_params(modelA) + _count_params(modelP)

    scorecard = {
        "approach": spec["name"],
        "approach_number": spec["number"],
        "benchmark": "waveform",
        "agent": "opus48",
        "category": spec["category"],
        "parameterization": spec.get("parameterization", spec["reparam"]),
        "time_convention": "tau_normalized_peak_aligned",
        "rankA": spec["rankA"], "rankP": spec["rankP"],
        "loss": loss,
        "loss_components": comp,
        "val_median": float(np.median(err_va)),
        "val_max": float(np.max(err_va)),
        "train_loss": float(np.mean(err_tr)),
        "train_median": float(np.median(err_tr)),
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(waves_tr)),
        "n_val": int(nval),
        "n_params": int(n_params),
        "notes": spec.get("notes", ""),
        "wall_time_s": round(time.time() - t_start, 1),
    }
    scorecard.update(spec.get("extra", {}))

    if write_model_dir:
        _persist(spec, modelA, modelP, D, scorecard, err_tr, err_va, tr_idx, val_idx)

    if verbose:
        print(f"[{spec['number']:2d}] {spec['name']:28s} "
              f"loss={loss:.3e} med={scorecard['val_median']:.3e} "
              f"train={scorecard['train_loss']:.3e} "
              f"rt={runtime_ms:.1f}ms ({scorecard['wall_time_s']}s)")

    return dict(scorecard=scorecard, err_tr=err_tr, err_va=err_va,
                tr_idx=tr_idx, val_idx=val_idx)


def _count_params(model):
    """Best-effort parameter count for the scorecard."""
    total = 0
    try:
        est = model
        # unwrap multi-output wrappers
        ests = getattr(est, "estimators_", [est])
        for e in ests:
            for attr in ["coef_", "coefs_", "dual_coef_"]:
                v = getattr(e, attr, None)
                if v is not None:
                    if isinstance(v, (list, tuple)):
                        total += int(sum(np.size(x) for x in v))
                    else:
                        total += int(np.size(v))
            if hasattr(e, "n_features_in_") and total == 0:
                total += int(getattr(e, "n_features_in_"))
    except Exception:
        pass
    return total


def _persist(spec, modelA, modelP, D, scorecard, err_tr, err_va, tr_idx, val_idx):
    mdir = MODELS / f"NN_{spec['name']}"
    sdir = mdir / "saved_model"
    sdir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"modelA": modelA, "modelP": modelP,
                 "basisA": D["basisA"].to_dict(),
                 "basisP": D["basisP"].to_dict(),
                 "mean": D["mean"], "std": D["std"],
                 "reparam": spec["reparam"]}, sdir / "model.joblib")
    np.savez(sdir / "errors.npz", err_train=err_tr, err_val=err_va,
             train_idx=tr_idx, val_idx=val_idx)

    with open(mdir / "scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2)

    _write_train_py(mdir, spec)
    _write_predict_py(mdir)
    if "expressions" in spec.get("extra", {}):
        with open(sdir / "expressions.json", "w") as f:
            json.dump(spec["extra"]["expressions"], f, indent=2)


def _write_train_py(mdir, spec):
    code = f'''"""Self-contained training for approach: {spec['name']}.

Run from the waveform/ work dir:  python models/NN_{spec['name']}/train.py
Rebuilds the SVD surrogate and overwrites saved_model/model.joblib.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # waveform/ dir
import surrogate as S
import approaches  # the registry of approach specs

spec = approaches.get_spec({spec['number']!r})
res = S.run_approach(spec, write_model_dir=True, verbose=True)
print("scorecard:", res["scorecard"]["loss"])
'''
    (mdir / "train.py").write_text(code)


def _write_predict_py(mdir):
    code = '''"""Importable prediction for this approach.

predict(params10, t_grid) -> complex h22 on t_grid.
params10 = [q,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z,chi_eff,chi_p,omega0].
"""
import sys, json
from pathlib import Path
import numpy as np
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as C
import surrogate as S

_SAVED = Path(__file__).resolve().parent / "saved_model" / "model.joblib"
_M = joblib.load(_SAVED)
_basisA = S.basis_from_dict(_M["basisA"])
_basisP = S.basis_from_dict(_M["basisP"])


def predict(params10, t_grid):
    P = np.atleast_2d(np.asarray(params10, float))
    X = C.reparam(P, _M["reparam"])
    X = (X - _M["mean"]) / _M["std"]
    cA = np.atleast_2d(_M["modelA"].predict(X))[0]
    cP = np.atleast_2d(_M["modelP"].predict(X))[0]
    logA = _basisA.reconstruct(cA)[0]
    phi = _basisP.reconstruct(cP)[0]
    t = np.asarray(t_grid, float)
    return C.tau_to_wave(logA, phi, t, t[0], t[-1])
'''
    (mdir / "predict.py").write_text(code)
