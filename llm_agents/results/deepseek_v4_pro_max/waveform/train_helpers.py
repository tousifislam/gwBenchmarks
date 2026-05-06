import sys, os, time, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_data, raw_params, eta_chieff_params, spherical_params, mass_diff_params,
    eval_mismatch, save_json, load_json
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
COMPARISON_DIR = ROOT / "comparison"
SVD_CACHE = ROOT / "svd_basis.npz"

def load_all(force_reload=False):
    """Load SVD basis, data, etc."""
    d = np.load(SVD_CACHE, allow_pickle=True)
    X_mean = d["X_mean"]; Vt = d["Vt"]; S = d["S"]; n_modes = int(d["n_modes"]); max_len = int(d["max_len"])
    U_train = np.load(ROOT / "U_train.npy")
    with open(ROOT / "all_data.pkl", "rb") as f:
        data = pickle.load(f)
    return X_mean, Vt, S, n_modes, max_len, U_train, data

def create_model_dir(approach_num, approach_name):
    d = MODELS_DIR / f"NN_{approach_num:02d}_{approach_name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "saved_model").mkdir(exist_ok=True)
    return d

def write_scorecard(model_dir, approach, approach_number, param_name, loss_train,
                    loss_val, runtime_ms, loss_components_val, n_train, n_val, category, notes=""):
    sc = {
        "approach": approach, "approach_number": approach_number,
        "benchmark": "waveform", "agent": "deepseek_v4_pro_max",
        "parameterization": param_name, "time_convention": "t0_at_peak",
        "loss": float(loss_val), "train_loss": float(loss_train),
        "loss_components": {f"mismatch_{int(m)}Msun": float(v) for m, v in
                           zip([40,80,120,160,200], loss_components_val)},
        "runtime_ms": float(runtime_ms), "n_train": n_train, "n_val": n_val,
        "category": category, "notes": notes
    }
    save_json(sc, model_dir / "scorecard.json")
    return sc

def append_changelog(entry):
    p = ROOT / "CHANGELOG.md"
    if not p.exists():
        p.write_text("# Waveform Bench Changelog — DeepSeek V4 Pro Max\n\n")
    with open(p, "a") as f:
        f.write(entry + "\n\n")

def evaluate_model(model, X_train, X_val, h_train_padded, h_val_padded, dt, max_len):
    n_pts = max_len  # each h has shape (max_len, 2)
    train_losses = []
    val_losses = []
    val_components = None
    t0 = time.time()
    for i, h_true in enumerate(h_train_padded):
        try:
            h_pred = model.predict(X_train[i:i+1])[0]
            # Pad/truncate to same length
            min_n = min(len(h_pred), h_true.shape[0])
            loss, _ = eval_mismatch(h_pred[:min_n], h_true[:min_n, 0] + 1j*h_true[:min_n, 1], dt)
            train_losses.append(loss)
        except Exception as e:
            train_losses.append(1.0)
    for i, h_true in enumerate(h_val_padded):
        try:
            h_pred = model.predict(X_val[i:i+1])[0]
            min_n = min(len(h_pred), h_true.shape[0])
            loss, mm_list = eval_mismatch(h_pred[:min_n], h_true[:min_n, 0] + 1j*h_true[:min_n, 1], dt)
            val_losses.append(loss)
            if val_components is None:
                val_components = mm_list
        except Exception as e:
            val_losses.append(1.0)
    eval_time = (time.time() - t0) * 1000
    if val_components is None:
        val_components = [np.mean(val_losses)] * 5
    return np.array(train_losses), np.array(val_losses), val_components, eval_time

def update_progress_plot(all_results):
    import matplotlib
    matplotlib.use("Agg")
    from gwbenchmarks.plot_settings import apply, COLOR_CYCLE, SINGLE_COL
    import matplotlib.pyplot as plt
    apply()
    names = [r["approach"] for r in all_results]
    train_l = [r.get("train_loss", 1.0) for r in all_results]
    val_l = [r["val_loss"] for r in all_results]
    x = range(len(all_results))
    fig, ax = plt.subplots(figsize=(SINGLE_COL*2, SINGLE_COL*0.8))
    ax.bar([i-0.2 for i in x], train_l, 0.35, label="Train", color=COLOR_CYCLE[0])
    ax.bar([i+0.2 for i in x], val_l, 0.35, label="Validation", color=COLOR_CYCLE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Mean FD Mismatch")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "progress.png", dpi=300)
    fig.savefig(COMPARISON_DIR / "progress.pdf", dpi=300)
    plt.close(fig)

print("train_helpers loaded.")
