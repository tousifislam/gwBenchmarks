"""
Recompute metrics for remnant, validity, and ringdown benchmarks.

Remnant:  4 agents stored NMAE (mean |pred-true|/range) instead of canonical
          NRMSE (sqrt(mean((pred-true)^2))/range) for kick velocity v_k.
Validity: 3 agents stored MAE instead of canonical RMSE on log10(mismatch).
Ringdown: 2 agents evaluated all QNM modes; restrict to (2,2,0) for fair comparison.
"""
import json, sys, os, pickle, warnings
from pathlib import Path
import numpy as np
import h5py
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "llm_agents" / "results"
DATASETS = ROOT / "datasets"


# ─── Dataset loaders ───────────────────────────────────────────────

def load_remnant_val():
    with h5py.File(DATASETS / "remnant" / "remnant_validation.h5") as f:
        q = f["q"][:]
        chi1x, chi1y, chi1z = f["chi1x"][:], f["chi1y"][:], f["chi1z"][:]
        chi2x, chi2y, chi2z = f["chi2x"][:], f["chi2y"][:], f["chi2z"][:]
        vf = f["vf_mag"][:]
        Mf = f["Mf"][:]
        chif = f["chif_mag"][:]
    X_raw = np.column_stack([q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z])
    return X_raw, vf, Mf, chif


def load_validity_val():
    with h5py.File(DATASETS / "validity" / "validity_validation.h5") as f:
        q = f["q"][:]
        chi1z = f["chi1z"][:]
        chi2z = f["chi2z"][:]
        omega0 = f["omega0"][:]
        mm_td = f["mm_td"][:]
    X_raw = np.column_stack([q, chi1z, chi2z, omega0])
    y_true = np.log10(np.maximum(mm_td, 1e-30))
    return X_raw, y_true


def load_ringdown_220_val():
    with h5py.File(DATASETS / "ringdown" / "ringdown_validation.h5") as f:
        g = f["l2"]["m+2"]["n0"]
        spin = g["spin"][:]
        omega_real = g["omega_real"][:]
        omega_imag = g["omega_imag"][:]
    X = np.column_stack([spin, np.full(len(spin), 2), np.full(len(spin), 2), np.zeros(len(spin))])
    Y = np.column_stack([omega_real, omega_imag])
    return X, Y


# ─── Parameterization helpers ──────────────────────────────────────

def eta(q):
    return q / (1 + q) ** 2


# Sonnet remnant "md" parameterization
def reparam_remnant_md(X_raw):
    q = X_raw[:, 0]
    c1x, c1y, c1z = X_raw[:, 1], X_raw[:, 2], X_raw[:, 3]
    c2x, c2y, c2z = X_raw[:, 4], X_raw[:, 5], X_raw[:, 6]
    m1 = q / (1 + q); m2 = 1 / (1 + q)
    dm = m1 - m2
    et = eta(q)
    chi_eff = m1 * c1z + m2 * c2z
    chi_a = (c1z - c2z) / 2
    chi1m = np.sqrt(c1x**2 + c1y**2 + c1z**2)
    chi2m = np.sqrt(c2x**2 + c2y**2 + c2z**2)
    chi1p = np.sqrt(c1x**2 + c1y**2)
    chi2p = np.sqrt(c2x**2 + c2y**2)
    chi_p = np.maximum(chi1p, (4*m2+3*m1)/(4*m1+3*m2)*(m2/m1)*chi2p)
    return np.column_stack([dm, et, chi_eff, chi_a, chi1m, chi2m, chi_p])


# GPT-5.5 / GPT-5.3 remnant "effective_spins" parameterization
def reparam_remnant_effective_spins(X_raw):
    q = X_raw[:, 0]
    c1 = X_raw[:, 1:4]; c2 = X_raw[:, 4:7]
    m1 = q / (1 + q); m2 = 1 / (1 + q)
    chi_eff = m1 * c1[:, 2] + m2 * c2[:, 2]
    chi_p = np.maximum(np.linalg.norm(c1[:, :2], axis=1), np.linalg.norm(c2[:, :2], axis=1))
    n1 = np.linalg.norm(c1, axis=1); n2 = np.linalg.norm(c2, axis=1)
    th1 = np.arccos(np.clip(c1[:, 2] / np.maximum(n1, 1e-12), -1, 1))
    th2 = np.arccos(np.clip(c2[:, 2] / np.maximum(n2, 1e-12), -1, 1))
    return np.column_stack([eta(q), chi_eff, chi_p, n1, n2, th1, th2])


# Gemini remnant "spherical" parameterization
def reparam_remnant_spherical(X_raw):
    q = X_raw[:, 0]
    c1x, c1y, c1z = X_raw[:, 1], X_raw[:, 2], X_raw[:, 3]
    c2x, c2y, c2z = X_raw[:, 4], X_raw[:, 5], X_raw[:, 6]
    et = eta(q)
    c1 = np.sqrt(c1x**2 + c1y**2 + c1z**2)
    c2 = np.sqrt(c2x**2 + c2y**2 + c2z**2)
    th1 = np.arccos(np.clip(c1z / np.maximum(c1, 1e-12), -1, 1))
    ph1 = np.arctan2(c1y, c1x)
    th2 = np.arccos(np.clip(c2z / np.maximum(c2, 1e-12), -1, 1))
    ph2 = np.arctan2(c2y, c2x)
    return np.column_stack([et, c1, th1, ph1, c2, th2, ph2])


# Sonnet validity "logq" parameterization
def reparam_validity_logq(X_raw):
    q, c1, c2, w = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2], X_raw[:, 3]
    chi_eff = (q * c1 + c2) / (1 + q)
    chi_a = (c1 - c2) / 2
    return np.column_stack([np.log(q), chi_eff, chi_a, np.log(w)])


# GPT-5.5 / GPT-5.3 validity "effective_spins" parameterization
def reparam_validity_effective_spins(X_raw):
    q, c1, c2, w = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2], X_raw[:, 3]
    chi_eff = (q * c1 + c2) / (1 + q)
    chi_a = (c1 - c2) / 2
    return np.column_stack([eta(q), chi_eff, chi_a, w])


# ─── Metric functions ──────────────────────────────────────────────

def nrmse(pred, true):
    """NRMSE = sqrt(mean((pred-true)^2)) / ptp(true)"""
    return float(np.sqrt(np.mean((pred - true) ** 2)) / np.ptp(true))


def rmse(pred, true):
    """RMSE = sqrt(mean((pred-true)^2))"""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def nmae(pred, true):
    """NMAE = mean(|pred-true|) / ptp(true)"""
    return float(np.mean(np.abs(pred - true)) / np.ptp(true))


def mae(pred, true):
    """MAE = mean(|pred-true|)"""
    return float(np.mean(np.abs(pred - true)))


def ringdown_loss(pred, true):
    """Mean relative error on QNM frequencies, averaged over Re and Im."""
    rel_re = np.abs(pred[:, 0] - true[:, 0]) / (np.abs(true[:, 0]) + 1e-12)
    rel_im = np.abs(pred[:, 1] - true[:, 1]) / (np.abs(true[:, 1]) + 1e-12)
    return float((rel_re.mean() + rel_im.mean()) / 2), {
        "rel_error_omega_real": float(rel_re.mean()),
        "rel_error_omega_imag": float(rel_im.mean()),
    }


def per_sample_nrmse_components(pred, true):
    """Per-sample (pred-true)^2 / ptp(true)^2, so NRMSE = sqrt(mean(this))."""
    return ((pred - true) ** 2) / (np.ptp(true) ** 2)


# ─── Model loading helpers ─────────────────────────────────────────

def load_sonnet_remnant_model():
    path = RESULTS / "sonnet46" / "remnant" / "models" / "19_et_md" / "saved_model" / "model.pkl"
    d = joblib.load(path)
    return d["et"]


def load_gpt55_remnant_model():
    path = RESULTS / "gpt55_high" / "remnant" / "models" / "17_randomforest" / "saved_model" / "model.joblib"
    return joblib.load(path)


def load_gpt53_remnant_model():
    path = RESULTS / "gpt53_codex_high" / "remnant" / "models" / "17_randomforest" / "saved_model" / "model.joblib"
    return joblib.load(path)


class GenericModel:
    """Stub for unpickling Gemini's wrapped models."""
    def __init__(self, regressor=None):
        self.regressor = regressor
    def predict(self, X):
        if hasattr(self, "models"):
            return np.column_stack([m.predict(X) for m in self.models])
        return self.regressor.predict(X)


def load_gemini_remnant_model():
    path = RESULTS / "gemini31_pro_preview" / "remnant" / "models" / "10_rf_sph" / "saved_model" / "model.pkl"
    import importlib
    # Register GenericModel in __main__ so pickle can find it
    import __main__
    __main__.GenericModel = GenericModel
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sonnet_validity_model():
    path = RESULTS / "sonnet46" / "validity" / "models" / "20_rf_logq" / "saved_model" / "model.pkl"
    d = joblib.load(path)
    return d["rf"]


def load_gpt55_validity_model():
    path = RESULTS / "gpt55_high" / "validity" / "models" / "17_randomforest" / "saved_model" / "model.joblib"
    return joblib.load(path)


def load_gpt53_validity_model():
    path = RESULTS / "gpt53_codex_high" / "validity" / "models" / "17_randomforest" / "saved_model" / "model.joblib"
    return joblib.load(path)


# ─── Recomputation ─────────────────────────────────────────────────

def recompute_remnant():
    print("=" * 60)
    print("REMNANT BENCHMARK: NMAE → NRMSE recomputation")
    print("=" * 60)

    X_raw, vf_true, Mf_true, chif_true = load_remnant_val()
    results = {}

    # Sonnet 4.6
    m = load_sonnet_remnant_model()
    X = reparam_remnant_md(X_raw)
    pred = m.predict(X).ravel()
    old_nmae = nmae(pred, vf_true)
    new_nrmse = nrmse(pred, vf_true)
    per_sample = ((pred - vf_true) ** 2)
    results["sonnet46"] = {
        "old_loss_nmae": old_nmae, "new_loss_nrmse": new_nrmse,
        "per_sample_sq_errors": per_sample,
        "model": "et_md",
    }
    print(f"  Sonnet 4.6:      NMAE={old_nmae:.6f} → NRMSE={new_nrmse:.6f}")

    # GPT-5.5
    m = load_gpt55_remnant_model()
    X = reparam_remnant_effective_spins(X_raw)
    pred = m.predict(X).ravel()
    old_nmae = nmae(pred, vf_true)
    new_nrmse = nrmse(pred, vf_true)
    per_sample = ((pred - vf_true) ** 2)
    results["gpt55_high"] = {
        "old_loss_nmae": old_nmae, "new_loss_nrmse": new_nrmse,
        "per_sample_sq_errors": per_sample,
        "model": "RandomForest (effective_spins)",
    }
    print(f"  GPT-5.5:         NMAE={old_nmae:.6f} → NRMSE={new_nrmse:.6f}")

    # GPT-5.3 Codex
    m = load_gpt53_remnant_model()
    X = reparam_remnant_effective_spins(X_raw)
    pred = m.predict(X).ravel()
    old_nmae = nmae(pred, vf_true)
    new_nrmse = nrmse(pred, vf_true)
    per_sample = ((pred - vf_true) ** 2)
    results["gpt53_codex_high"] = {
        "old_loss_nmae": old_nmae, "new_loss_nrmse": new_nrmse,
        "per_sample_sq_errors": per_sample,
        "model": "RandomForest (effective_spins)",
    }
    print(f"  GPT-5.3 Codex:   NMAE={old_nmae:.6f} → NRMSE={new_nrmse:.6f}")

    # Gemini 3.1 Pro (predicts 3 outputs: Mf, chif, vf_mag)
    m = load_gemini_remnant_model()
    X = reparam_remnant_spherical(X_raw)
    pred_all = m.predict(X)  # shape (1000, 3)
    pred_vk = pred_all[:, 2]  # v_k is index 2
    old_nmae = nmae(pred_vk, vf_true)
    new_nrmse = nrmse(pred_vk, vf_true)
    per_sample = ((pred_vk - vf_true) ** 2)
    results["gemini31_pro_preview"] = {
        "old_loss_nmae": old_nmae, "new_loss_nrmse": new_nrmse,
        "per_sample_sq_errors": per_sample,
        "model": "10_rf_sph",
    }
    print(f"  Gemini 3.1 Pro:  NMAE={old_nmae:.6f} → NRMSE={new_nrmse:.6f}")

    return results, vf_true


def recompute_validity():
    print("\n" + "=" * 60)
    print("VALIDITY BENCHMARK: MAE → RMSE recomputation")
    print("=" * 60)

    X_raw, y_true = load_validity_val()
    results = {}

    # Sonnet 4.6
    m = load_sonnet_validity_model()
    X = reparam_validity_logq(X_raw)
    pred = m.predict(np.asarray(X)).ravel()
    old_mae = mae(pred, y_true)
    new_rmse = rmse(pred, y_true)
    per_sample = ((pred - y_true) ** 2)
    results["sonnet46"] = {
        "old_loss_mae": old_mae, "new_loss_rmse": new_rmse,
        "per_sample_sq_errors": per_sample,
        "model": "rf_logq",
    }
    print(f"  Sonnet 4.6:      MAE={old_mae:.6f} → RMSE={new_rmse:.6f}")

    # GPT-5.5
    m = load_gpt55_validity_model()
    X = reparam_validity_effective_spins(X_raw)
    pred = m.predict(X).ravel()
    old_mae = mae(pred, y_true)
    new_rmse = rmse(pred, y_true)
    per_sample = ((pred - y_true) ** 2)
    results["gpt55_high"] = {
        "old_loss_mae": old_mae, "new_loss_rmse": new_rmse,
        "per_sample_sq_errors": per_sample,
        "model": "RandomForest (effective_spins)",
    }
    print(f"  GPT-5.5:         MAE={old_mae:.6f} → RMSE={new_rmse:.6f}")

    # GPT-5.3 Codex
    m = load_gpt53_validity_model()
    X = reparam_validity_effective_spins(X_raw)
    pred = m.predict(X).ravel()
    old_mae = mae(pred, y_true)
    new_rmse = rmse(pred, y_true)
    per_sample = ((pred - y_true) ** 2)
    results["gpt53_codex_high"] = {
        "old_loss_mae": old_mae, "new_loss_rmse": new_rmse,
        "per_sample_sq_errors": per_sample,
        "model": "RandomForest (effective_spins)",
    }
    print(f"  GPT-5.3 Codex:   MAE={old_mae:.6f} → RMSE={new_rmse:.6f}")

    return results, y_true


def recompute_ringdown():
    from scipy.interpolate import CubicSpline

    print("\n" + "=" * 60)
    print("RINGDOWN BENCHMARK: all-modes → (2,2,0) recomputation")
    print("=" * 60)

    X_220, Y_220 = load_ringdown_220_val()
    results = {}

    for agent in ["opus47", "gpt54_mini"]:
        model_dir = RESULTS / agent / "ringdown" / "models" / "04_cubic_spline_per_mode"
        saved = model_dir / "saved_model"

        # Rebuild (2,2,0) spline from training data
        data_path = saved / "data.npz"
        if data_path.exists():
            d = np.load(data_path)
            Xt, Yt = d["Xt"], d["Yt"]
        else:
            # Fallback: load from HDF5
            with h5py.File(DATASETS / "ringdown" / "ringdown_training.h5") as f:
                g = f["l2"]["m+2"]["n0"]
                spin_t = g["spin"][:]
                wr_t = g["omega_real"][:]
                wi_t = g["omega_imag"][:]
            Xt = np.column_stack([spin_t, np.full(len(spin_t), 2),
                                  np.full(len(spin_t), 2), np.zeros(len(spin_t))])
            Yt = np.column_stack([wr_t, wi_t])

        # Filter to (2,2,0) training samples
        mask = (Xt[:, 1] == 2) & (Xt[:, 2] == 2) & (Xt[:, 3] == 0)
        a_train = Xt[mask, 0]
        idx = np.argsort(a_train)
        a_train = a_train[idx]
        Y_re_train = Yt[mask][idx, 0]
        Y_im_train = Yt[mask][idx, 1]

        cs_re = CubicSpline(a_train, Y_re_train, extrapolate=True)
        cs_im = CubicSpline(a_train, Y_im_train, extrapolate=True)

        # Predict on (2,2,0) validation
        a_val = X_220[:, 0]
        pred_220 = np.column_stack([cs_re(a_val), cs_im(a_val)])

        loss_220, comp_220 = ringdown_loss(pred_220, Y_220)

        # Per-sample relative errors
        rel_re = np.abs(pred_220[:, 0] - Y_220[:, 0]) / (np.abs(Y_220[:, 0]) + 1e-12)
        rel_im = np.abs(pred_220[:, 1] - Y_220[:, 1]) / (np.abs(Y_220[:, 1]) + 1e-12)
        per_sample = (rel_re + rel_im) / 2

        # Read old loss
        bm_path = RESULTS / agent / "ringdown" / "comparison" / "best_model.json"
        with open(bm_path) as f:
            old_bm = json.load(f)
        old_loss = old_bm["loss"]

        results[agent] = {
            "old_loss_all_modes": old_loss,
            "new_loss_220": loss_220,
            "loss_components": comp_220,
            "per_sample_loss": per_sample,
            "model": "cubic_spline_per_mode",
        }
        label = "Opus 4.7" if agent == "opus47" else "GPT-5.4 Mini"
        print(f"  {label:18s} all-modes={old_loss:.6e} → (2,2,0)={loss_220:.6e}")

    return results


# ─── Update files ──────────────────────────────────────────────────

def update_best_model_json(agent, bench, new_loss, per_sample, old_loss_key, old_loss_val):
    path = RESULTS / agent / bench / "comparison" / "best_model.json"
    with open(path) as f:
        d = json.load(f)

    d[old_loss_key] = old_loss_val
    d["loss"] = new_loss
    d["per_sample_loss"] = per_sample.tolist()

    if bench == "remnant":
        d["loss_components"] = {"nrmse_v_k": new_loss}
        d["metric_recomputed"] = "NMAE→NRMSE on v_k (2026-05-02)"
    elif bench == "validity":
        d["loss_components"] = {"rmse_log10_mismatch": new_loss}
        d["metric_recomputed"] = "MAE→RMSE on log10(mismatch) (2026-05-02)"
    elif bench == "ringdown":
        d["metric_recomputed"] = "all-modes→(2,2,0) only (2026-05-02)"
        d["n_val"] = len(per_sample)

    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"    Updated {path.relative_to(ROOT)}")


def update_scorecard(agent, bench, model_subdir, new_loss, component_dict):
    path = RESULTS / agent / bench / "models" / model_subdir / "scorecard.json"
    if not path.exists():
        return
    with open(path) as f:
        d = json.load(f)
    d["loss"] = new_loss
    d["loss_components"] = component_dict
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"    Updated {path.relative_to(ROOT)}")


def write_recompute_note(agent, bench, note_text):
    path = RESULTS / agent / bench / "RECOMPUTE_NOTE.md"
    with open(path, "w") as f:
        f.write(note_text)
    print(f"    Wrote {path.relative_to(ROOT)}")


# ─── Main ──────────────────────────────────────────────────────────

def main():
    # ── Remnant ──
    remnant_results, vf_true = recompute_remnant()

    print("\nUpdating remnant files...")
    model_subdirs = {
        "sonnet46": "19_et_md",
        "gpt55_high": "17_randomforest",
        "gpt53_codex_high": "17_randomforest",
        "gemini31_pro_preview": "10_rf_sph",
    }
    agent_labels = {
        "sonnet46": "Sonnet 4.6",
        "gpt55_high": "GPT-5.5 High",
        "gpt53_codex_high": "GPT-5.3 Codex High",
        "gemini31_pro_preview": "Gemini 3.1 Pro Preview",
    }
    for agent, r in remnant_results.items():
        vk_range = float(np.ptp(vf_true))
        per_sample_nrmse = np.sqrt(r["per_sample_sq_errors"]) / vk_range
        update_best_model_json(agent, "remnant", r["new_loss_nrmse"],
                               per_sample_nrmse, "loss_old_nmae", r["old_loss_nmae"])
        update_scorecard(agent, "remnant", model_subdirs[agent],
                         r["new_loss_nrmse"], {"nrmse_v_k": r["new_loss_nrmse"]})
        write_recompute_note(agent, "remnant",
            f"# Remnant Metric Recomputation (2026-05-02)\n\n"
            f"The original evaluation for {agent_labels[agent]} used NMAE "
            f"(mean absolute error / range) as the loss metric.\n"
            f"The canonical metric is NRMSE (root mean squared error / range).\n\n"
            f"- Old loss (NMAE): {r['old_loss_nmae']:.8f}\n"
            f"- New loss (NRMSE): {r['new_loss_nrmse']:.8f}\n\n"
            f"NRMSE >= NMAE by Jensen's inequality. The recomputation loads the\n"
            f"saved model, re-predicts on the 1000-sample validation set, and\n"
            f"computes sqrt(mean((pred - true)^2)) / ptp(true) for kick velocity v_k.\n"
            f"\nScript: llm_agents/recompute_remnant_validity_ringdown.py\n")

    # ── Validity ──
    validity_results, y_true = recompute_validity()

    print("\nUpdating validity files...")
    model_subdirs_v = {
        "sonnet46": "20_rf_logq",
        "gpt55_high": "17_randomforest",
        "gpt53_codex_high": "17_randomforest",
    }
    for agent, r in validity_results.items():
        per_sample_abs = np.sqrt(r["per_sample_sq_errors"])
        update_best_model_json(agent, "validity", r["new_loss_rmse"],
                               per_sample_abs, "loss_old_mae", r["old_loss_mae"])
        update_scorecard(agent, "validity", model_subdirs_v[agent],
                         r["new_loss_rmse"], {"rmse_log10_mismatch": r["new_loss_rmse"]})
        write_recompute_note(agent, "validity",
            f"# Validity Metric Recomputation (2026-05-02)\n\n"
            f"The original evaluation for {agent_labels[agent]} used MAE on\n"
            f"log10(mismatch) as the loss metric.\n"
            f"The canonical metric is RMSE on log10(mismatch).\n\n"
            f"- Old loss (MAE): {r['old_loss_mae']:.8f}\n"
            f"- New loss (RMSE): {r['new_loss_rmse']:.8f}\n\n"
            f"RMSE >= MAE for any non-degenerate distribution. The recomputation\n"
            f"loads the saved model, re-predicts on the 393-sample validation set,\n"
            f"and computes sqrt(mean((pred - true)^2)) in log10-space.\n"
            f"\nScript: llm_agents/recompute_remnant_validity_ringdown.py\n")

    # ── Ringdown ──
    ringdown_results = recompute_ringdown()

    print("\nUpdating ringdown files...")
    for agent, r in ringdown_results.items():
        label = "Opus 4.7" if agent == "opus47" else "GPT-5.4 Mini"
        bm_path = RESULTS / agent / "ringdown" / "comparison" / "best_model.json"
        with open(bm_path) as f:
            d = json.load(f)
        d["loss_old_all_modes"] = r["old_loss_all_modes"]
        d["loss"] = r["new_loss_220"]
        d["loss_components"] = r["loss_components"]
        d["per_sample_loss"] = r["per_sample_loss"].tolist()
        d["n_val"] = 531
        d["metric_recomputed"] = "all-modes→(2,2,0) only (2026-05-02)"
        with open(bm_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"    Updated {bm_path.relative_to(ROOT)}")

        # Scorecard
        sc_path = RESULTS / agent / "ringdown" / "models" / "04_cubic_spline_per_mode" / "scorecard.json"
        if sc_path.exists():
            with open(sc_path) as f:
                sc = json.load(f)
            sc["loss_old_all_modes"] = r["old_loss_all_modes"]
            sc["loss"] = r["new_loss_220"]
            sc["loss_components"] = r["loss_components"]
            sc["n_val"] = 531
            with open(sc_path, "w") as f:
                json.dump(sc, f, indent=2)
            print(f"    Updated {sc_path.relative_to(ROOT)}")

        write_recompute_note(agent, "ringdown",
            f"# Ringdown Scope Recomputation (2026-05-02)\n\n"
            f"The original evaluation for {label} evaluated all QNM modes\n"
            f"(51 modes, ~1.2M validation samples). For fair comparison with\n"
            f"other agents, the loss was recomputed on the (2,2,0) mode only\n"
            f"(531 validation samples).\n\n"
            f"- Old loss (all modes, n=1,204,578): {r['old_loss_all_modes']:.6e}\n"
            f"- New loss ((2,2,0) only, n=531):    {r['new_loss_220']:.6e}\n"
            f"  - Re(omega) relative error: {r['loss_components']['rel_error_omega_real']:.6e}\n"
            f"  - Im(omega) relative error: {r['loss_components']['rel_error_omega_imag']:.6e}\n\n"
            f"Script: llm_agents/recompute_remnant_validity_ringdown.py\n")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY OF RECOMPUTED LOSSES")
    print("=" * 60)
    print(f"\n{'Agent':<22} {'Bench':<10} {'Old':>12} {'New':>12} {'Change'}")
    print("-" * 72)
    for agent, r in remnant_results.items():
        print(f"{agent:<22} {'remnant':<10} {r['old_loss_nmae']:>12.6f} {r['new_loss_nrmse']:>12.6f} NMAE→NRMSE")
    for agent, r in validity_results.items():
        print(f"{agent:<22} {'validity':<10} {r['old_loss_mae']:>12.6f} {r['new_loss_rmse']:>12.6f} MAE→RMSE")
    for agent, r in ringdown_results.items():
        print(f"{agent:<22} {'ringdown':<10} {r['old_loss_all_modes']:>12.6e} {r['new_loss_220']:>12.6e} all→(2,2,0)")

    return remnant_results, validity_results, ringdown_results


if __name__ == "__main__":
    remnant_results, validity_results, ringdown_results = main()
