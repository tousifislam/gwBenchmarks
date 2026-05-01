"""Curate the Waveform Bench dataset from SXS catalog.

Selects quasi-circular BBH simulations (including precessing), preprocesses
coprecessing-frame h22 waveforms, computes NR resolution errors via
frequency-domain mismatch at multiple total masses, and splits into
250 training + 250 validation.

Source: SXS Gravitational Waveform Database (v3.0.0)

Usage:
    python curate_waveform_dataset.py
    python curate_waveform_dataset.py --skip-nr-errors
    python curate_waveform_dataset.py --skip-preprocessing
"""

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import h5py
import numpy as np
import sxs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gwbenchmarks.metrics import frequency_domain_mismatch

# ── Configuration ──────────────────────────────────────────────────────────

N_TRAIN = 250
N_VAL = 250

DT_GEOMETRIC = 0.1
EXTRA_BUFFER_M = 100.0
T_END_AFTER_PEAK_M = 100.0
MAX_LENGTH_M = 10000.0

ECC_THRESHOLD = 1e-3
Q_MIN, Q_MAX = 1.0, 8.0
CHI_MAX = 0.8
N_ORBITS_MIN, N_ORBITS_MAX = 5, 40

MTOT_VALUES = [40, 80, 120, 160, 200]

OUTPUT_DIR = Path(__file__).parent.parent

# ── Helpers ───────────────────────────────────────────────────────────────


def parse_eccentricity(ecc):
    if ecc is None:
        return float("nan")
    if isinstance(ecc, str):
        ecc = ecc.strip().lstrip("<>~")
        try:
            return float(ecc)
        except ValueError:
            return float("nan")
    return float(ecc)


def chi_eff(q, chi1z, chi2z):
    return (q * chi1z + chi2z) / (1.0 + q)


def chi_p(q, chi1_perp, chi2_perp):
    m1 = q / (1.0 + q)
    m2 = 1.0 / (1.0 + q)
    A1 = 2.0 + 3.0 * m2 / (2.0 * m1)
    A2 = 2.0 + 3.0 * m1 / (2.0 * m2)
    return max(A1 * m1**2 * chi1_perp, A2 * m2**2 * chi2_perp) / (A1 * m1**2)


def find_peak_time_h22(t, h22):
    """Find peak time of |h22| using quadratic refinement."""
    amp2 = np.abs(h22) ** 2
    i_max = np.argmax(amp2)
    if i_max == 0 or i_max == len(amp2) - 1:
        return t[i_max]
    t_m1, t_0, t_p1 = t[i_max - 1], t[i_max], t[i_max + 1]
    a_m1, a_0, a_p1 = amp2[i_max - 1], amp2[i_max], amp2[i_max + 1]
    dt_half = (t_p1 - t_m1) / 2.0
    numer = (a_m1 - a_p1) * dt_half
    denom = 2.0 * (a_m1 - 2 * a_0 + a_p1)
    if abs(denom) < 1e-30:
        return t_0
    return t_0 + numer / denom


# ── Catalog selection ────────────────────────────────────────────────────


def filter_catalog():
    print("Loading SXS catalog...")
    simulations = sxs.Simulations.load()
    print(f"Total simulations in catalog: {len(simulations)}")

    candidates = []
    skipped = {"not_bbh": 0, "missing_q": 0, "q_range": 0, "chi_range": 0,
               "ecc": 0, "no_remnant": 0, "short": 0, "too_long": 0,
               "missing_ecc": 0, "missing_data": 0}

    for sid in sorted(simulations.keys()):
        if not sid.startswith("SXS:BBH:"):
            skipped["not_bbh"] += 1
            continue

        s = simulations[sid]
        if s.get("object_types") != "BHBH":
            skipped["not_bbh"] += 1
            continue

        q = s.get("reference_mass_ratio")
        if q is None:
            skipped["missing_q"] += 1
            continue

        chi1 = s.get("reference_dimensionless_spin1")
        chi2 = s.get("reference_dimensionless_spin2")
        if chi1 is None or chi2 is None:
            skipped["missing_data"] += 1
            continue

        chi1_vec = np.array(chi1, dtype=float)
        chi2_vec = np.array(chi2, dtype=float)
        ecc_raw = s.get("reference_eccentricity")
        ecc = parse_eccentricity(ecc_raw)
        n_orbits = s.get("number_of_orbits")
        chi1_perp_val = s.get("reference_chi1_perp", 0.0)
        chi2_perp_val = s.get("reference_chi2_perp", 0.0)
        if not isinstance(chi1_perp_val, (int, float)):
            chi1_perp_val = 0.0
        if not isinstance(chi2_perp_val, (int, float)):
            chi2_perp_val = 0.0

        remnant_mass = s.get("remnant_mass")
        remnant_spin = s.get("remnant_dimensionless_spin")
        lev_numbers = s.get("lev_numbers", [])

        if q < Q_MIN or q > Q_MAX:
            skipped["q_range"] += 1
            continue

        chi1_mag = float(np.linalg.norm(chi1_vec))
        chi2_mag = float(np.linalg.norm(chi2_vec))
        if chi1_mag > CHI_MAX or chi2_mag > CHI_MAX:
            skipped["chi_range"] += 1
            continue

        if np.isnan(ecc):
            skipped["missing_ecc"] += 1
            continue
        if ecc > ECC_THRESHOLD:
            skipped["ecc"] += 1
            continue

        if remnant_mass is None or remnant_spin is None:
            skipped["no_remnant"] += 1
            continue

        if n_orbits is None or (isinstance(n_orbits, (int, float)) and n_orbits < N_ORBITS_MIN):
            skipped["short"] += 1
            continue
        if isinstance(n_orbits, (int, float)) and n_orbits >= N_ORBITS_MAX:
            skipped["too_long"] += 1
            continue

        is_non_spinning = (chi1_mag < 1e-3) and (chi2_mag < 1e-3)
        is_precessing = (float(chi1_perp_val) > 1e-3) or (float(chi2_perp_val) > 1e-3)

        sxs_num = int(sid.split(":")[-1])
        chi1_perp_mag = np.sqrt(chi1_vec[0]**2 + chi1_vec[1]**2)
        chi2_perp_mag = np.sqrt(chi2_vec[0]**2 + chi2_vec[1]**2)

        candidates.append({
            "sxs_id": sid,
            "sxs_num": sxs_num,
            "is_non_spinning": is_non_spinning,
            "is_precessing": is_precessing,
            "q": float(q),
            "chi1x": float(chi1_vec[0]),
            "chi1y": float(chi1_vec[1]),
            "chi1z": float(chi1_vec[2]),
            "chi2x": float(chi2_vec[0]),
            "chi2y": float(chi2_vec[1]),
            "chi2z": float(chi2_vec[2]),
            "chi1_mag": chi1_mag,
            "chi2_mag": chi2_mag,
            "chi_eff": chi_eff(float(q), float(chi1_vec[2]), float(chi2_vec[2])),
            "chi_p": chi_p(float(q), float(chi1_perp_mag), float(chi2_perp_mag)),
            "ecc": ecc,
            "n_orbits": float(n_orbits) if isinstance(n_orbits, (int, float)) else 0.0,
            "lev_numbers": sorted(lev_numbers),
            "n_levs": len(lev_numbers),
            "max_lev": max(lev_numbers) if lev_numbers else -1,
        })

    print(f"\nFiltering results:")
    print(f"  Candidates: {len(candidates)}")
    for k, v in skipped.items():
        if v > 0:
            print(f"  Skipped ({k}): {v}")
    print(f"  With >=2 Levs: {sum(1 for c in candidates if c['n_levs'] >= 2)}")

    return candidates


# ── Greedy selection ────────────────────────────────────────────────────


def greedy_select(candidates, n_select, params_cols, w_recency=0.1, w_length=0.1):
    n_cand = len(candidates)
    if n_cand == 0 or n_select == 0:
        return []
    n_select = min(n_select, n_cand)

    params = np.array([[c[p] for p in params_cols] for c in candidates])
    mins = params.min(axis=0)
    maxs = params.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    params_norm = (params - mins) / ranges

    sxs_nums = np.array([c["sxs_num"] for c in candidates], dtype=float)
    recency = (sxs_nums - sxs_nums.min()) / max(sxs_nums.max() - sxs_nums.min(), 1)

    orbits = np.array([c["n_orbits"] for c in candidates], dtype=float)
    length = (orbits - orbits.min()) / max(orbits.max() - orbits.min(), 1)

    selected_idx = []
    remaining = set(range(n_cand))

    center = np.full(params_norm.shape[1], 0.5)
    dists_to_center = np.linalg.norm(params_norm - center, axis=1)
    first = np.argmin(dists_to_center)
    selected_idx.append(first)
    remaining.remove(first)

    for i in range(1, n_select):
        if not remaining:
            break
        remaining_list = list(remaining)
        selected_params = params_norm[selected_idx]
        dists = np.array([
            np.min(np.linalg.norm(selected_params - params_norm[j], axis=1))
            for j in remaining_list
        ])
        coverage = dists / max(dists.max(), 1e-10)
        scores = coverage + w_recency * recency[remaining_list] + w_length * length[remaining_list]
        best = remaining_list[np.argmax(scores)]
        selected_idx.append(best)
        remaining.remove(best)
        if (i + 1) % 50 == 0:
            print(f"    Selected {i + 1}/{n_select}...")

    return selected_idx


def tiered_select_split(pool_ns, pool_np, pool_pr,
                         n_ns, n_np, n_pr,
                         w_recency, w_length):
    selected = []
    if pool_ns and n_ns > 0:
        idx = greedy_select(pool_ns, min(n_ns, len(pool_ns)),
                            ["q"], w_recency=w_recency, w_length=w_length)
        selected.extend([pool_ns[i] for i in idx])
    if pool_np and n_np > 0:
        idx = greedy_select(pool_np, min(n_np, len(pool_np)),
                            ["q", "chi1z", "chi2z"],
                            w_recency=w_recency, w_length=w_length)
        selected.extend([pool_np[i] for i in idx])
    if pool_pr and n_pr > 0:
        idx = greedy_select(pool_pr, min(n_pr, len(pool_pr)),
                            ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z"],
                            w_recency=w_recency, w_length=w_length)
        selected.extend([pool_pr[i] for i in idx])
    return selected


def select_train_val(candidates):
    non_spin = [c for c in candidates if c["is_non_spinning"]]
    non_prec = [c for c in candidates if not c["is_non_spinning"] and not c["is_precessing"]]
    prec = [c for c in candidates if c["is_precessing"]]

    print(f"\nTiers:")
    print(f"  Non-spinning:   {len(non_spin)}")
    print(f"  Non-precessing: {len(non_prec)}")
    print(f"  Precessing:     {len(prec)}")

    N_NS, N_NP, N_PR = 10, 40, 200

    print(f"\nSelecting training: {N_NS} non-spin + {N_NP} non-prec + {N_PR} prec = {N_TRAIN}")
    train = tiered_select_split(non_spin, non_prec, prec,
                                 N_NS, N_NP, N_PR,
                                 w_recency=0.1, w_length=0.1)
    train_ids = {s["sxs_id"] for s in train}

    rem_ns = [c for c in non_spin if c["sxs_id"] not in train_ids]
    rem_np = [c for c in non_prec if c["sxs_id"] not in train_ids]
    rem_pr = [c for c in prec if c["sxs_id"] not in train_ids]

    print(f"Selecting validation: {N_NS} non-spin + {N_NP} non-prec + {N_PR} prec = {N_VAL}")
    val = tiered_select_split(rem_ns, rem_np, rem_pr,
                               N_NS, N_NP, N_PR,
                               w_recency=0.0, w_length=0.0)

    return train, val


# ── Waveform preprocessing ──────────────────────────────────────────────


def preprocess_sim(sxs_id, max_lev):
    """Load SXS sim, transform to coprecessing frame, extract h22,
    resample to uniform grid, peak-align, apply phase convention."""
    sim = sxs.load(f"{sxs_id}/Lev{max_lev}", ignore_deprecation=True)
    strain_inertial = sim.strain
    strain_copr = strain_inertial.to_coprecessing_frame()

    t = np.asarray(strain_copr.t, dtype=float)
    h22 = np.asarray(strain_copr.data[:, strain_copr.index(2, 2)], dtype=np.complex128)

    t_ref = float(sim.metadata.reference_time)
    keep = t >= t_ref + EXTRA_BUFFER_M
    t = t[keep]
    h22 = h22[keep]

    t_peak = find_peak_time_h22(t, h22)
    t_shift = t - t_peak

    t_max = float(np.floor(min(t_shift[-1], T_END_AFTER_PEAK_M) / DT_GEOMETRIC) * DT_GEOMETRIC)
    t_min_inspiral = -(MAX_LENGTH_M - t_max)
    t_min = float(np.ceil(max(t_shift[0], t_min_inspiral) / DT_GEOMETRIC) * DT_GEOMETRIC)
    n = int(round((t_max - t_min) / DT_GEOMETRIC)) + 1
    t_uniform = t_min + np.arange(n) * DT_GEOMETRIC

    h22_uniform = (np.interp(t_uniform, t_shift, np.real(h22))
                   + 1j * np.interp(t_uniform, t_shift, np.imag(h22)))

    i0 = int(np.argmin(np.abs(t_uniform)))
    phi0 = float(np.angle(h22_uniform[i0]))
    h22_uniform *= np.exp(-1j * phi0)

    ref_omega_vec = np.array(sim.metadata.reference_orbital_frequency, dtype=float)
    omega0 = float(np.linalg.norm(ref_omega_vec))

    return {
        "t": t_uniform.astype(np.float64),
        "h22_real": np.real(h22_uniform).astype(np.float64),
        "h22_imag": np.imag(h22_uniform).astype(np.float64),
        "n_pts": n,
        "omega0": omega0,
        "lev_used": max_lev,
        "wf_length": float(t_uniform[-1] - t_uniform[0]),
    }


# ── NR error computation ────────────────────────────────────────────────


def load_copr_h22_peak_aligned(sxs_id, lev):
    """Load coprecessing h22, drop junk, peak-align, return (t_shifted, h22)."""
    sim = sxs.load(f"{sxs_id}/Lev{lev}", ignore_deprecation=True)
    strain = sim.strain
    strain_copr = strain.to_coprecessing_frame()

    t = np.asarray(strain_copr.t, dtype=float)
    h22 = np.asarray(strain_copr.data[:, strain_copr.index(2, 2)], dtype=np.complex128)

    t_ref = float(sim.metadata.reference_time)
    keep = t >= t_ref + EXTRA_BUFFER_M
    t = t[keep]
    h22 = h22[keep]

    t_peak = find_peak_time_h22(t, h22)
    return t - t_peak, h22


def compute_nr_errors_sim(sxs_id, lev_numbers):
    """Compute FD mismatch between highest and second-highest Lev."""
    levs = sorted(lev_numbers)
    high_lev = levs[-1]
    low_lev = levs[-2]

    t_h, h_h = load_copr_h22_peak_aligned(sxs_id, high_lev)
    t_l, h_l = load_copr_h22_peak_aligned(sxs_id, low_lev)

    t_start = max(t_h[0], t_l[0])
    t_end = min(t_h[-1], t_l[-1], T_END_AFTER_PEAK_M)
    t_max = float(np.floor(t_end / DT_GEOMETRIC) * DT_GEOMETRIC)
    t_min_cap = max(t_start, -(MAX_LENGTH_M - t_max))
    t_min = float(np.ceil(t_min_cap / DT_GEOMETRIC) * DT_GEOMETRIC)
    n = int(round((t_max - t_min) / DT_GEOMETRIC)) + 1
    t_common = t_min + np.arange(n) * DT_GEOMETRIC

    h_h_r = (np.interp(t_common, t_h, np.real(h_h))
             + 1j * np.interp(t_common, t_h, np.imag(h_h)))
    h_l_r = (np.interp(t_common, t_l, np.real(h_l))
             + 1j * np.interp(t_common, t_l, np.imag(h_l)))

    result = {"sxs_id": sxs_id, "high_lev": high_lev, "low_lev": low_lev}
    for mtot in MTOT_VALUES:
        try:
            mm = frequency_domain_mismatch(h_l_r, h_h_r, DT_GEOMETRIC, mtot, f_low=20.0)
            result[f"fd_mismatch_M{mtot}"] = float(mm) if not np.isnan(mm) else None
        except Exception:
            result[f"fd_mismatch_M{mtot}"] = None

    vals = [result[f"fd_mismatch_M{m}"] for m in MTOT_VALUES
            if result.get(f"fd_mismatch_M{m}") is not None]
    result["fd_mismatch_combined"] = max(vals) if vals else None

    return result


# ── HDF5 output ──────────────────────────────────────────────────────────


def write_split(selected, wf_data, nr_errors, output_path, split_name):
    """Write one split (training or validation) to HDF5."""
    gz = dict(compression="gzip", compression_opts=4)

    processed = [(s, wf_data[s["sxs_id"]]) for s in selected if s["sxs_id"] in wf_data]

    with h5py.File(output_path, "w") as h5:
        h5.attrs["description"] = (
            f"Waveform Bench ({split_name}): coprecessing-frame h22 from SXS NR simulations."
        )
        h5.attrs["source"] = "SXS Gravitational Waveform Database v3.0.0"
        h5.attrs["split"] = split_name
        h5.attrs["n_simulations"] = len(processed)
        h5.attrs["dt_geometric"] = DT_GEOMETRIC
        h5.attrs["t_align"] = "h22_copr_peak"
        h5.attrs["phase_convention"] = "phase(h22_copr)=0 at t=0"

        for i, (sim_info, wf) in enumerate(processed):
            grp = h5.create_group(f"sim_{i:04d}")
            grp.attrs["sxs_id"] = sim_info["sxs_id"]
            grp.attrs["q"] = sim_info["q"]
            grp.attrs["chi1x"] = sim_info["chi1x"]
            grp.attrs["chi1y"] = sim_info["chi1y"]
            grp.attrs["chi1z"] = sim_info["chi1z"]
            grp.attrs["chi2x"] = sim_info["chi2x"]
            grp.attrs["chi2y"] = sim_info["chi2y"]
            grp.attrs["chi2z"] = sim_info["chi2z"]
            grp.attrs["omega0"] = wf["omega0"]
            grp.attrs["n_pts"] = wf["n_pts"]
            grp.attrs["lev_used"] = wf["lev_used"]
            grp.attrs["wf_length"] = wf["wf_length"]

            sid = sim_info["sxs_id"]
            if sid in nr_errors:
                nr = nr_errors[sid]
                for mtot in MTOT_VALUES:
                    v = nr.get(f"fd_mismatch_M{mtot}")
                    if v is not None:
                        grp.attrs[f"nr_fd_mm_M{mtot}"] = v
                if nr.get("fd_mismatch_combined") is not None:
                    grp.attrs["nr_fd_mm_combined"] = nr["fd_mismatch_combined"]

            grp.create_dataset("t", data=wf["t"], **gz)
            grp.create_dataset("h22_real", data=wf["h22_real"], **gz)
            grp.create_dataset("h22_imag", data=wf["h22_imag"], **gz)

        meta = h5.create_group("metadata")
        meta.create_dataset("sxs_id",
                            data=np.array([s["sxs_id"] for s, _ in processed], dtype="S"))
        for key in ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z"]:
            meta.create_dataset(key, data=np.array([s[key] for s, _ in processed]))
        meta.create_dataset("omega0", data=np.array([w["omega0"] for _, w in processed]))
        meta.create_dataset("chi_eff",
                            data=np.array([s["chi_eff"] for s, _ in processed]))
        meta.create_dataset("chi_p",
                            data=np.array([s["chi_p"] for s, _ in processed]))
        meta.create_dataset("wf_length",
                            data=np.array([w["wf_length"] for _, w in processed]))

        nr_grp = h5.create_group("nr_errors")
        nr_sims = [(s, nr_errors[s["sxs_id"]]) for s, _ in processed
                    if s["sxs_id"] in nr_errors]
        if nr_sims:
            nr_grp.create_dataset("sxs_id",
                                  data=np.array([s["sxs_id"] for s, _ in nr_sims], dtype="S"))
            for mtot in MTOT_VALUES:
                key = f"fd_mismatch_M{mtot}"
                nr_grp.create_dataset(key,
                    data=np.array([nr.get(key, np.nan) or np.nan for _, nr in nr_sims]))
            nr_grp.create_dataset("fd_mismatch_combined",
                data=np.array([nr.get("fd_mismatch_combined", np.nan) or np.nan
                               for _, nr in nr_sims]))

    print(f"  {split_name}: {len(processed)} sims -> {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description="Curate Waveform Bench dataset from SXS catalog"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-nr-errors", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.output_dir / "checkpoint.json"

    # ── Step 1: Select simulations ──────────────────────────────────
    candidates = filter_catalog()
    train_sims, val_sims = select_train_val(candidates)
    print(f"\nSelected: {len(train_sims)} training + {len(val_sims)} validation")

    # Save selection metadata
    lev_map = {c["sxs_id"]: c["lev_numbers"] for c in candidates}

    # ── Step 2: Preprocess waveforms ────────────────────────────────
    wf_data = {}
    if not args.skip_preprocessing:
        all_sims = train_sims + val_sims
        n_total = len(all_sims)
        n_failed = 0
        t0 = time.time()

        # Load checkpoint
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            wf_data = ckpt.get("wf_data", {})
            # Convert lists back to numpy arrays
            for sid, wf in wf_data.items():
                for key in ["t", "h22_real", "h22_imag"]:
                    wf[key] = np.array(wf[key])
            print(f"Loaded checkpoint: {len(wf_data)} preprocessed sims")

        print(f"\nPreprocessing {n_total} waveforms (dt={DT_GEOMETRIC}M, no taper)...")
        for i, sim_info in enumerate(all_sims):
            sid = sim_info["sxs_id"]
            if sid in wf_data:
                continue

            levs = sim_info["lev_numbers"]
            if not levs:
                print(f"  [SKIP] {sid}: no Lev numbers")
                n_failed += 1
                continue
            max_lev = max(levs)

            for attempt in range(3):
                try:
                    t_start = time.time()
                    result = preprocess_sim(sid, max_lev)
                    elapsed = time.time() - t_start

                    wf_data[sid] = result
                    n_done = len(wf_data)
                    rate = n_done / (time.time() - t0) if (time.time() - t0) > 0 else 0
                    eta = (n_total - i - 1) / rate / 60 if rate > 0 else 0

                    if (i + 1) % 10 == 0 or i == 0:
                        print(
                            f"  [{i+1}/{n_total}] {sid} Lev{max_lev} "
                            f"n={result['n_pts']} len={result['wf_length']:.0f}M "
                            f"({elapsed:.1f}s, ETA {eta:.0f}min)"
                        )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait = 30 * (2 ** attempt)
                        print(f"  [RATE LIMITED] {sid}: retry in {wait}s")
                        time.sleep(wait)
                        continue
                    n_failed += 1
                    print(f"  [FAIL] {sid}: {e}")
                    traceback.print_exc()
                    break

            if (i + 1) % 50 == 0:
                ckpt_data = {}
                for s, w in wf_data.items():
                    ckpt_data[s] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                    for k, v in w.items()}
                with open(checkpoint_path, "w") as f:
                    json.dump({"wf_data": ckpt_data}, f)

        elapsed_total = (time.time() - t0) / 60
        print(f"Preprocessing done: {len(wf_data)} succeeded, {n_failed} failed, "
              f"{elapsed_total:.1f}min")

    # ── Step 3: Compute NR errors ───────────────────────────────────
    nr_errors = {}
    if not args.skip_nr_errors:
        all_sims = train_sims + val_sims
        multi_lev = [(s, s["lev_numbers"]) for s in all_sims
                     if len(s.get("lev_numbers", [])) >= 2]
        print(f"\nComputing NR errors for {len(multi_lev)} sims with >=2 Levs...")

        n_done = 0
        n_failed_nr = 0
        t0 = time.time()

        for i, (sim_info, levs) in enumerate(multi_lev):
            sid = sim_info["sxs_id"]
            for attempt in range(3):
                try:
                    t_start = time.time()
                    result = compute_nr_errors_sim(sid, levs)
                    elapsed = time.time() - t_start
                    nr_errors[sid] = result
                    n_done += 1

                    if (n_done) % 10 == 0 or n_done == 1:
                        rate = n_done / (time.time() - t0) if (time.time() - t0) > 0 else 0
                        eta = (len(multi_lev) - i - 1) / rate / 60 if rate > 0 else 0
                        comb = result.get("fd_mismatch_combined", "?")
                        if isinstance(comb, float):
                            comb = f"{comb:.2e}"
                        print(
                            f"  [{n_done}/{len(multi_lev)}] {sid} "
                            f"Lev{levs[-2]}->{levs[-1]} "
                            f"combined={comb} ({elapsed:.1f}s, ETA {eta:.0f}min)"
                        )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait = 30 * (2 ** attempt)
                        print(f"  [RATE LIMITED] {sid}: retry in {wait}s")
                        time.sleep(wait)
                        continue
                    n_failed_nr += 1
                    print(f"  [NR FAIL] {sid}: {e}")
                    traceback.print_exc()
                    break

        elapsed_total = (time.time() - t0) / 60
        print(f"NR errors done: {len(nr_errors)} computed, {n_failed_nr} failed, "
              f"{elapsed_total:.1f}min")

        if nr_errors:
            vals = [v["fd_mismatch_combined"] for v in nr_errors.values()
                    if v.get("fd_mismatch_combined") is not None]
            if vals:
                print(f"  Combined FD mismatch: median={np.median(vals):.2e}, "
                      f"95th={np.percentile(vals, 95):.2e}, max={np.max(vals):.2e}")

    # ── Step 4: Write HDF5 ──────────────────────────────────────────
    print("\nWriting HDF5 files...")
    write_split(train_sims, wf_data, nr_errors,
                args.output_dir / "waveform_training.h5", "training")
    write_split(val_sims, wf_data, nr_errors,
                args.output_dir / "waveform_validation.h5", "validation")

    # ── Summary ─────────────────────────────────────────────────────
    all_processed = [s for s in train_sims + val_sims if s["sxs_id"] in wf_data]
    if all_processed:
        q_all = np.array([s["q"] for s in all_processed])
        ce_all = np.array([s["chi_eff"] for s in all_processed])
        cp_all = np.array([s["chi_p"] for s in all_processed])
        lens = np.array([wf_data[s["sxs_id"]]["wf_length"] for s in all_processed])
        print(f"\nSummary:")
        print(f"  q: [{q_all.min():.2f}, {q_all.max():.2f}]")
        print(f"  chi_eff: [{ce_all.min():.3f}, {ce_all.max():.3f}]")
        print(f"  chi_p: [{cp_all.min():.4f}, {cp_all.max():.3f}]")
        print(f"  wf_length: [{lens.min():.0f}, {lens.max():.0f}] M, median={np.median(lens):.0f} M")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\nDone.")


if __name__ == "__main__":
    main()
