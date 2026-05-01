"""Curate the Remnant Bench dataset from SXS catalog metadata.

Selects quasi-circular BBH simulations (including precessing) from the SXS
catalog, uses greedy coverage-optimized selection to pick 300 training +
300 validation simulations with good parameter-space coverage, and stores
binary parameters + remnant properties + NR error estimates.

Source: SXS Gravitational Waveform Database (v3.0.0)

Usage:
    python curate_remnant_dataset.py
    python curate_remnant_dataset.py --skip-nr-errors   # fast, no Lev downloads
"""

import argparse
import re
import time
from pathlib import Path

import h5py
import numpy as np
import sxs

# ── Configuration ──────────────────────────────────────────────────────────

ECC_THRESHOLD = 0.01
N_TRAIN = 1000
N_VAL = 1000

# ── Helpers ───────────────────────────────────────────────────────────────


def parse_eccentricity(ecc):
    """Parse eccentricity from SXS metadata, handling strings like '<9.5e-05'."""
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
    """Effective spin parameter."""
    return (q * chi1z + chi2z) / (1.0 + q)


def chi_p(q, chi1_perp, chi2_perp):
    """Precessing spin parameter (single-spin approximation)."""
    m1 = q / (1.0 + q)
    m2 = 1.0 / (1.0 + q)
    A1 = 2.0 + 3.0 * m2 / (2.0 * m1)
    A2 = 2.0 + 3.0 * m1 / (2.0 * m2)
    return max(A1 * m1 * m1 * chi1_perp, A2 * m2 * m2 * chi2_perp) / (A1 * m1 * m1)


# ── Catalog selection ────────────────────────────────────────────────────


def select_all_candidates():
    """Select all quasi-circular BBH simulations with remnant data."""
    print("Loading SXS catalog...")
    simulations = sxs.Simulations.load()

    candidates = []
    skipped = {"no_bbh": 0, "missing_data": 0, "eccentric": 0, "no_remnant": 0}

    for sim_id in simulations:
        if not sim_id.startswith("SXS:BBH:"):
            skipped["no_bbh"] += 1
            continue

        sim = simulations[sim_id]
        try:
            q = sim.get("reference_mass_ratio", None)
            chi1 = sim.get("reference_dimensionless_spin1", None)
            chi2 = sim.get("reference_dimensionless_spin2", None)
            if q is None or chi1 is None or chi2 is None:
                skipped["missing_data"] += 1
                continue

            chi1 = np.array(chi1, dtype=float)
            chi2 = np.array(chi2, dtype=float)

            ecc_raw = sim.get("reference_eccentricity", None)
            ecc_val = parse_eccentricity(ecc_raw)
            if np.isnan(ecc_val) or ecc_val > ECC_THRESHOLD:
                skipped["eccentric"] += 1
                continue

            rm = sim.get("remnant_mass", None)
            rs = sim.get("remnant_dimensionless_spin", None)
            rv = sim.get("remnant_velocity", None)
            if rm is None or rs is None or rv is None:
                skipped["no_remnant"] += 1
                continue

            rs = np.array(rs, dtype=float)
            rv = np.array(rv, dtype=float)

            ref_omega_vec = sim.get("reference_orbital_frequency", None)
            omega0 = float(np.linalg.norm(ref_omega_vec)) if ref_omega_vec is not None else float("nan")

            chi1_perp = np.sqrt(chi1[0] ** 2 + chi1[1] ** 2)
            chi2_perp = np.sqrt(chi2[0] ** 2 + chi2[1] ** 2)
            is_non_spinning = float(np.linalg.norm(chi1)) < 1e-3 and float(np.linalg.norm(chi2)) < 1e-3
            is_precessing = chi1_perp > 1e-3 or chi2_perp > 1e-3

            num = int(sim_id.split(":")[-1])
            candidates.append({
                "sxs_id": sim_id,
                "num": num,
                "q": float(q),
                "chi1x": float(chi1[0]),
                "chi1y": float(chi1[1]),
                "chi1z": float(chi1[2]),
                "chi2x": float(chi2[0]),
                "chi2y": float(chi2[1]),
                "chi2z": float(chi2[2]),
                "ecc": ecc_val,
                "omega0": omega0,
                "Mf": float(rm),
                "chif_mag": float(np.linalg.norm(rs)),
                "vf_mag": float(np.linalg.norm(rv)),
                "is_non_spinning": is_non_spinning,
                "is_precessing": is_precessing,
            })

        except Exception:
            skipped["missing_data"] += 1
            continue

    candidates.sort(key=lambda x: (x["q"], x["num"]))
    print(f"Total candidates: {len(candidates)}")
    print(f"Skipped: {skipped}")

    n_ns = sum(1 for c in candidates if c["is_non_spinning"])
    n_as = sum(1 for c in candidates if not c["is_non_spinning"] and not c["is_precessing"])
    n_pr = sum(1 for c in candidates if c["is_precessing"])
    print(f"Tiers: {n_ns} non-spinning, {n_as} aligned-spin, {n_pr} precessing")
    return candidates


# ── Greedy coverage-optimized selection ──────────────────────────────────


def greedy_select(candidates, n_select, param_cols, w_recency=0.1):
    """Greedy selection maximizing parameter-space coverage with recency bonus.

    At each step, pick the candidate whose minimum distance to already-selected
    points (in normalized parameter space) is largest, with a small bonus for
    newer simulations (higher SXS number).
    """
    n_cand = len(candidates)
    if n_select >= n_cand:
        return list(range(n_cand))

    params = np.array([[c[p] for p in param_cols] for c in candidates])
    mins = params.min(axis=0)
    maxs = params.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    params_norm = (params - mins) / ranges

    sxs_nums = np.array([c["num"] for c in candidates], dtype=float)
    recency = (sxs_nums - sxs_nums.min()) / max(sxs_nums.max() - sxs_nums.min(), 1)

    selected_idx = []
    remaining = set(range(n_cand))

    center = np.full(params_norm.shape[1], 0.5)
    dists_to_center = np.linalg.norm(params_norm - center, axis=1)
    first = int(np.argmin(dists_to_center))
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
        scores = coverage + w_recency * recency[remaining_list]
        best = remaining_list[int(np.argmax(scores))]
        selected_idx.append(best)
        remaining.remove(best)

        if (i + 1) % 50 == 0:
            print(f"    Selected {i + 1}/{n_select}...")

    return selected_idx


def select_train_val(candidates, n_train=N_TRAIN, n_val=N_VAL):
    """Select training and validation sets using greedy coverage.

    Uses tiered selection: non-spinning, aligned-spin, precessing — each
    tier gets a proportional allocation to ensure boundary coverage.
    """
    non_spin = [c for c in candidates if c["is_non_spinning"]]
    aligned = [c for c in candidates if not c["is_non_spinning"] and not c["is_precessing"]]
    prec = [c for c in candidates if c["is_precessing"]]

    total = len(non_spin) + len(aligned) + len(prec)
    f_ns = len(non_spin) / total
    f_as = len(aligned) / total
    f_pr = len(prec) / total

    n_ns = max(5, int(round(n_train * f_ns)))
    n_as = max(10, int(round(n_train * f_as)))
    n_pr = n_train - n_ns - n_as

    print(f"\nTraining allocation: {n_ns} non-spinning + {n_as} aligned + {n_pr} precessing = {n_train}")
    print(f"  Selecting training set...")

    train = []
    if non_spin and n_ns > 0:
        idx = greedy_select(non_spin, min(n_ns, len(non_spin)), ["q"], w_recency=0.1)
        train.extend([non_spin[i] for i in idx])
    if aligned and n_as > 0:
        idx = greedy_select(aligned, min(n_as, len(aligned)),
                            ["q", "chi1z", "chi2z"], w_recency=0.1)
        train.extend([aligned[i] for i in idx])
    if prec and n_pr > 0:
        idx = greedy_select(prec, min(n_pr, len(prec)),
                            ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z"],
                            w_recency=0.1)
        train.extend([prec[i] for i in idx])

    train_ids = {c["sxs_id"] for c in train}

    rem_ns = [c for c in non_spin if c["sxs_id"] not in train_ids]
    rem_as = [c for c in aligned if c["sxs_id"] not in train_ids]
    rem_pr = [c for c in prec if c["sxs_id"] not in train_ids]

    n_ns_v = max(5, int(round(n_val * f_ns)))
    n_as_v = max(10, int(round(n_val * f_as)))
    n_pr_v = n_val - n_ns_v - n_as_v

    print(f"\nValidation allocation: {n_ns_v} non-spinning + {n_as_v} aligned + {n_pr_v} precessing = {n_val}")
    print(f"  Selecting validation set...")

    val = []
    if rem_ns and n_ns_v > 0:
        idx = greedy_select(rem_ns, min(n_ns_v, len(rem_ns)), ["q"], w_recency=0.0)
        val.extend([rem_ns[i] for i in idx])
    if rem_as and n_as_v > 0:
        idx = greedy_select(rem_as, min(n_as_v, len(rem_as)),
                            ["q", "chi1z", "chi2z"], w_recency=0.0)
        val.extend([rem_as[i] for i in idx])
    if rem_pr and n_pr_v > 0:
        idx = greedy_select(rem_pr, min(n_pr_v, len(rem_pr)),
                            ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z"],
                            w_recency=0.0)
        val.extend([rem_pr[i] for i in idx])

    print(f"\nFinal: {len(train)} training, {len(val)} validation")
    return train, val


# ── NR error estimation ──────────────────────────────────────────────────


def get_available_levs(sxs_id, simulations_catalog):
    """Get list of available resolution levels from the SXS catalog metadata."""
    sim = simulations_catalog.get(sxs_id, {})
    levs_available = sim.get("files", {})
    levs = []
    for key in levs_available:
        m = re.search(r"Lev(\d+)", str(key))
        if m:
            levs.append(int(m.group(1)))
    return sorted(set(levs))


def get_remnant_from_lev(sxs_id, lev):
    """Load a specific Lev and extract remnant properties from its metadata."""
    sim_data = sxs.load(f"{sxs_id}/Lev{lev}", progress=False,
                        ignore_deprecation=True)
    meta = sim_data.metadata
    rm = float(meta.remnant_mass)
    rs = np.array(meta.remnant_dimensionless_spin, dtype=float)
    rv = np.array(meta.remnant_velocity, dtype=float)
    return rm, float(np.linalg.norm(rs)), float(np.linalg.norm(rv))


def compute_nr_errors(records):
    """For each simulation, compute NR error as |highest_lev - second_highest_lev|."""
    print("\nComputing NR errors (highest vs second-highest Lev)...")
    simulations = sxs.Simulations.load()

    n_total = len(records)
    n_done = 0
    n_with_errors = 0
    t0 = time.time()

    for rec in records:
        sid = rec["sxs_id"]
        n_done += 1
        rec["delta_Mf"] = float("nan")
        rec["delta_chif"] = float("nan")
        rec["delta_vf"] = float("nan")
        rec["lev_high"] = -1
        rec["lev_low"] = -1

        try:
            levs = get_available_levs(sid, simulations)
            if len(levs) < 2:
                continue

            lev_high = levs[-1]
            lev_low = levs[-2]

            Mf_high, chif_high, vf_high = get_remnant_from_lev(sid, lev_high)
            Mf_low, chif_low, vf_low = get_remnant_from_lev(sid, lev_low)

            rec["delta_Mf"] = abs(Mf_high - Mf_low)
            rec["delta_chif"] = abs(chif_high - chif_low)
            rec["delta_vf"] = abs(vf_high - vf_low)
            rec["lev_high"] = lev_high
            rec["lev_low"] = lev_low
            n_with_errors += 1

            if n_done % 50 == 0 or n_done == n_total:
                rate = n_done / (time.time() - t0)
                eta = (n_total - n_done) / rate / 60 if rate > 0 else 0
                print(f"  [{n_done}/{n_total}] {sid} Lev{lev_high}-Lev{lev_low} "
                      f"dMf={rec['delta_Mf']:.2e} dchif={rec['delta_chif']:.2e} "
                      f"dvf={rec['delta_vf']:.2e} (ETA {eta:.0f}min)")

        except Exception as e:
            if n_done % 100 == 0:
                print(f"  [{n_done}/{n_total}] {sid}: NR error failed: {e}")

    elapsed = (time.time() - t0) / 60
    print(f"NR errors: {n_with_errors}/{n_total} computed in {elapsed:.1f}min")


# ── HDF5 output ──────────────────────────────────────────────────────────


def write_split(records, output, split_name):
    """Write one split to HDF5."""
    with h5py.File(output, "w") as h5:
        h5.attrs["description"] = (
            f"Remnant Bench ({split_name}): remnant properties (Mf, chif, vf) "
            f"for quasi-circular BBH from SXS catalog."
        )
        h5.attrs["source"] = "SXS Gravitational Waveform Database v3.0.0"
        h5.attrs["split"] = split_name
        h5.attrs["n_simulations"] = len(records)

        fields = {
            "q": "q",
            "chi1x": "chi1x", "chi1y": "chi1y", "chi1z": "chi1z",
            "chi2x": "chi2x", "chi2y": "chi2y", "chi2z": "chi2z",
            "omega0": "omega0",
            "Mf": "Mf", "chif_mag": "chif_mag", "vf_mag": "vf_mag",
            "delta_Mf": "delta_Mf", "delta_chif": "delta_chif",
            "delta_vf": "delta_vf",
        }
        for ds_name, key in fields.items():
            h5.create_dataset(ds_name, data=np.array([r[key] for r in records]))

        h5.create_dataset(
            "sxs_id", data=np.array([r["sxs_id"] for r in records], dtype="S")
        )
        h5.create_dataset(
            "lev_high", data=np.array([r["lev_high"] for r in records], dtype=int)
        )
        h5.create_dataset(
            "lev_low", data=np.array([r["lev_low"] for r in records], dtype=int)
        )

    print(f"  {split_name}: {len(records)} simulations -> {output}")


# ── Plots ────────────────────────────────────────────────────────────────


def make_plots(train, val, plot_dir):
    """Generate Nature-style reference plots for remnant properties."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from gwbenchmarks import plot_settings
    import matplotlib.pyplot as plt

    plot_settings.apply()

    def get_arrays(records):
        q = np.array([r["q"] for r in records])
        chi1_perp = np.sqrt(
            np.array([r["chi1x"] for r in records]) ** 2
            + np.array([r["chi1y"] for r in records]) ** 2
        )
        chi2_perp = np.sqrt(
            np.array([r["chi2x"] for r in records]) ** 2
            + np.array([r["chi2y"] for r in records]) ** 2
        )
        ce = np.array([chi_eff(r["q"], r["chi1z"], r["chi2z"]) for r in records])
        cp = np.array([chi_p(r["q"], cp1, cp2) for r, cp1, cp2
                        in zip(records, chi1_perp, chi2_perp)])
        Mf = np.array([r["Mf"] for r in records])
        chif = np.array([r["chif_mag"] for r in records])
        vf = np.array([r["vf_mag"] for r in records])
        return q, ce, cp, Mf, chif, vf

    q_tr, ce_tr, cp_tr, Mf_tr, chif_tr, vf_tr = get_arrays(train)
    q_va, ce_va, cp_va, Mf_va, chif_va, vf_va = get_arrays(val)

    remnant_labels = [
        ("Mf", r"$M_f / M$", Mf_tr, Mf_va),
        ("chif", r"$|\chi_f|$", chif_tr, chif_va),
        ("vf", r"$|v_f|$", vf_tr, vf_va),
    ]
    param_labels = [
        ("q", r"$q$", q_tr, q_va),
        ("chieff", r"$\chi_{\mathrm{eff}}$", ce_tr, ce_va),
        ("chip", r"$\chi_p$", cp_tr, cp_va),
    ]

    for rname, rlabel, r_tr, r_va in remnant_labels:
        for pname, plabel, p_tr, p_va in param_labels:
            fig, ax = plt.subplots(
                figsize=plot_settings.figsize(cols=1, aspect=0.75)
            )
            ax.scatter(p_tr, r_tr, s=8, alpha=0.5,
                       color=plot_settings.COLORS["blue"], label="Training",
                       rasterized=True)
            ax.scatter(p_va, r_va, s=8, alpha=0.5, marker="s",
                       color=plot_settings.COLORS["red"], label="Validation",
                       rasterized=True)
            ax.set_xlabel(plabel)
            ax.set_ylabel(rlabel)
            ax.legend(markerscale=1.5)
            for ext in ("png", "pdf"):
                fig.savefig(plot_dir / f"{rname}_vs_{pname}.{ext}")
            plt.close(fig)

    print(f"  Remnant property plots saved to {plot_dir}")

    # NR error histograms
    dMf_all = np.concatenate([
        np.array([r["delta_Mf"] for r in train]),
        np.array([r["delta_Mf"] for r in val]),
    ])
    dchif_all = np.concatenate([
        np.array([r["delta_chif"] for r in train]),
        np.array([r["delta_chif"] for r in val]),
    ])
    dvf_all = np.concatenate([
        np.array([r["delta_vf"] for r in train]),
        np.array([r["delta_vf"] for r in val]),
    ])

    mask = ~np.isnan(dMf_all) & (dMf_all > 0)
    if mask.sum() > 5:
        error_data = [
            (r"$\Delta M_f$", dMf_all[mask]),
            (r"$\Delta|\chi_f|$", dchif_all[mask]),
            (r"$\Delta|v_f|$", dvf_all[mask]),
        ]
        fig, axes = plt.subplots(
            1, 3, figsize=plot_settings.figsize(cols=2, aspect=0.4)
        )
        fig.subplots_adjust(wspace=0.45)
        for ax, (elabel, edata) in zip(axes, error_data):
            edata_pos = edata[edata > 0]
            if len(edata_pos) > 0:
                bins = np.logspace(
                    np.floor(np.log10(edata_pos.min())),
                    np.ceil(np.log10(edata_pos.max())),
                    25,
                )
                ax.hist(edata_pos, bins=bins, alpha=0.7,
                        color=plot_settings.COLORS["blue"])
                ax.set_xscale("log")
            ax.set_xlabel(elabel)
            ax.set_ylabel("Count")
        for ext in ("png", "pdf"):
            fig.savefig(plot_dir / f"nr_errors.{ext}")
        plt.close(fig)
        print(f"  NR error histograms saved to {plot_dir}")

    # Parameter space coverage plots
    fig, axes = plt.subplots(1, 3, figsize=plot_settings.figsize(cols=2, aspect=0.4))
    fig.subplots_adjust(wspace=0.4)
    param_pairs = [
        (q_tr, ce_tr, q_va, ce_va, r"$q$", r"$\chi_{\mathrm{eff}}$"),
        (q_tr, cp_tr, q_va, cp_va, r"$q$", r"$\chi_p$"),
        (ce_tr, cp_tr, ce_va, cp_va, r"$\chi_{\mathrm{eff}}$", r"$\chi_p$"),
    ]
    for ax, (x_tr, y_tr, x_va, y_va, xl, yl) in zip(axes, param_pairs):
        ax.scatter(x_tr, y_tr, s=6, alpha=0.5, color=plot_settings.COLORS["blue"],
                   label="Train", rasterized=True)
        ax.scatter(x_va, y_va, s=6, alpha=0.5, marker="s",
                   color=plot_settings.COLORS["red"], label="Val", rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[-1].legend(fontsize=7, markerscale=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"param_space.{ext}")
    plt.close(fig)
    print(f"  Parameter space plots saved to {plot_dir}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Curate Remnant Bench dataset from SXS catalog"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent,
    )
    parser.add_argument(
        "--skip-nr-errors", action="store_true",
        help="Skip NR error computation (much faster)",
    )
    args = parser.parse_args()

    candidates = select_all_candidates()
    train, val = select_train_val(candidates)

    all_selected = train + val
    if not args.skip_nr_errors:
        compute_nr_errors(all_selected)
    else:
        for rec in all_selected:
            rec["delta_Mf"] = float("nan")
            rec["delta_chif"] = float("nan")
            rec["delta_vf"] = float("nan")
            rec["lev_high"] = -1
            rec["lev_low"] = -1

    train_path = args.output_dir / "remnant_training.h5"
    val_path = args.output_dir / "remnant_validation.h5"

    write_split(train, train_path, "training")
    write_split(val, val_path, "validation")

    # Summary
    Mf_all = np.array([r["Mf"] for r in all_selected])
    chif_all = np.array([r["chif_mag"] for r in all_selected])
    vf_all = np.array([r["vf_mag"] for r in all_selected])
    print(f"\nMf: [{Mf_all.min():.4f}, {Mf_all.max():.4f}]")
    print(f"|chif|: [{chif_all.min():.4f}, {chif_all.max():.4f}]")
    print(f"|vf|: [{vf_all.min():.6f}, {vf_all.max():.6f}]")

    dMf = np.array([r["delta_Mf"] for r in all_selected])
    mask = ~np.isnan(dMf)
    if mask.sum() > 0:
        dMf = dMf[mask]
        dchif = np.array([r["delta_chif"] for r in all_selected])[mask]
        dvf = np.array([r["delta_vf"] for r in all_selected])[mask]
        print(f"\nNR errors ({mask.sum()} sims):")
        print(f"  delta_Mf:   median={np.median(dMf):.2e}, max={np.max(dMf):.2e}")
        print(f"  delta_chif: median={np.median(dchif):.2e}, max={np.max(dchif):.2e}")
        print(f"  delta_vf:   median={np.median(dvf):.2e}, max={np.max(dvf):.2e}")

    # Plots
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    make_plots(train, val, plot_dir)


if __name__ == "__main__":
    main()
