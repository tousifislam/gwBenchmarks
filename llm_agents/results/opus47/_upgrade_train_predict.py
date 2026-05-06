"""Upgrade per-model train.py / predict.py to be self-contained and reproducible.

Each train.py becomes: imports build_models, calls the right function.
Each predict.py becomes: loads the saved_model artifacts and exposes predict().
"""
from __future__ import annotations
import os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BENCHMARKS = ["waveform", "remnant", "dynamics", "ringdown", "validity", "analytic"]


def _train_template(approach_dir_name: str, fn_name: str) -> str:
    return f'''"""Self-contained training script for approach {approach_dir_name}.

Re-runs the training and re-generates saved_model artifacts.
Usage: python train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import build_models  # imports + initialises shared SVD/data globals

if __name__ == "__main__":
    fn = getattr(build_models, "{fn_name}")
    print(f"[train.py] Re-running {{fn.__name__}}...")
    fn()
    print(f"[train.py] Done. Artifacts saved in {{Path(__file__).parent / 'saved_model'}}")
'''


def _predict_template(approach_dir_name: str) -> str:
    """Generic predict.py: tries to load model.pkl / state.pkl and applies the saved model.

    The exact reconstruction logic varies per approach, so we expose a `predict` that
    routes to the correct apply pattern based on the saved model dict's keys.
    """
    return '''"""Prediction module for approach {approach}.

Loads serialised artifacts from saved_model/ and exposes a `predict(X)` function.

The dispatch is generic: it inspects the saved-model dict to figure out the right
inverse-transform / reconstruction. For tougher approaches (PySR / cubic-splines /
EIM), an approach-specific .pkl is loaded and applied.
"""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_HERE = Path(__file__).parent
_state = None


def _load():
    global _state
    if _state is not None:
        return _state
    cand = _HERE / "saved_model" / "model.pkl"
    if not cand.exists():
        cand = _HERE / "saved_model" / "state.pkl"
    if not cand.exists():
        raise FileNotFoundError(f"No model.pkl or state.pkl in {_HERE / 'saved_model'}")
    with open(cand, "rb") as f:
        _state = pickle.load(f)
    return _state


def predict(X):
    """Apply the saved model.

    X: (n, n_features) array (parameters in the right reparameterization for this model).

    For approaches that include a basis V (SVD/EIM), X is mapped to coefficients then
    reconstructed by `coeffs @ V`. For direct regressors, the model is applied directly.
    """
    s = _load()
    X = np.asarray(X)
    # Try common patterns
    if "reg" in s:
        coeffs = s["reg"].predict(X)
    elif "p" in s:
        coeffs = s["p"].predict(X)
    elif "m" in s:
        coeffs = s["m"].predict(s.get("sc", _Identity()).transform(X) if "sc" in s else X)
    elif "rf" in s:
        coeffs = s["rf"].predict(X)
    elif "et" in s:
        coeffs = s["et"].predict(X)
    elif "gbm" in s:
        coeffs = s["gbm"].predict(X)
    elif "knn" in s:
        coeffs = s["knn"].predict(X)
    elif "kr" in s:
        coeffs = s["kr"].predict(X)
    elif "g" in s:
        coeffs = s["g"].predict(X)
    elif "gprs" in s:
        coeffs = np.column_stack([g.predict(X) for g in s["gprs"]])
    elif "g1" in s and "g2" in s:
        return s["g1"].predict(X) + s["g2"].predict(X)
    elif "mlps" in s:
        sc = s.get("sc")
        Xs = sc.transform(X) if sc is not None else X
        return np.mean([m.predict(Xs) for m in s["mlps"]], axis=0)
    elif "rest_reg" in s:
        coeffs = s["rest_reg"].predict(X)
    else:
        raise ValueError(f"predict.py does not know how to apply state with keys {list(s.keys())}")
    # Reconstruct via SVD basis if present
    if "V_K" in s:
        V = s["V_K"]
        N_T = V.shape[1] // 2
        X_full = coeffs @ V
        return X_full[:, :N_T] + 1j * X_full[:, N_T:]
    if "V_K_l" in s:
        V = s["V_K_l"]
        return np.exp(coeffs @ V)
    if "V_K_ap" in s:
        V = s["V_K_ap"]
        N_T = V.shape[1] // 2
        X_full = coeffs @ V
        amp = np.maximum(X_full[:, :N_T], 0)
        phase = X_full[:, N_T:]
        return amp * np.exp(1j * phase)
    # Otherwise return raw predictions
    return coeffs


class _Identity:
    def transform(self, X): return X
'''.replace("{approach}", approach_dir_name)


def upgrade_one(model_dir: Path):
    name = model_dir.name
    # Map model dir name to function in build_models.py
    # Convention: approach number -> appNN function name
    num = int(name.split("_")[0])
    base = name[len(name.split("_")[0]) + 1:]
    # Find the right fn name in build_models.py for this benchmark
    bench_dir = model_dir.parent.parent
    bm_path = bench_dir / "build_models.py"
    if not bm_path.exists():
        return
    src = bm_path.read_text()
    # Find function name matching app{num}* and the second underscore-separated piece
    # Try: appNN, appNN_*
    pat = re.compile(rf"def (app{num:02d}\w*|app{num}\w*)\(")
    m = pat.search(src)
    fn_name = m.group(1) if m else f"app{num:02d}"
    # Confirm the fn exists in src
    if f"def {fn_name}(" not in src:
        # Look for any "def app<num>" loosely
        loose = re.search(rf"def (app{num}[a-zA-Z0-9_]*)\(", src)
        fn_name = loose.group(1) if loose else fn_name
    train_path = model_dir / "train.py"
    predict_path = model_dir / "predict.py"
    train_path.write_text(_train_template(name, fn_name))
    predict_path.write_text(_predict_template(name))


def main():
    for bench in BENCHMARKS:
        models_dir = ROOT / bench / "models"
        if not models_dir.exists():
            print(f"[skip] {bench}: no models dir")
            continue
        n = 0
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir():
                continue
            try:
                upgrade_one(d)
                n += 1
            except Exception as e:
                print(f"  [warn] {bench}/{d.name}: {e}")
        print(f"[{bench}] upgraded {n} model dirs")


if __name__ == "__main__":
    main()
