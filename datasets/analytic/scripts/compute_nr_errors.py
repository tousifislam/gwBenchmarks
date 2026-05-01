#!/usr/bin/env python3
"""Compute NR resolution errors for the Analytic Bench dataset.

For each simulation with at least two available resolution levels (Lev),
loads the (2,2) mode at the highest and second-highest Lev, resamples
onto a common uniform grid, and computes the frequency-domain mismatch
at M_tot = {40, 80, 120, 160, 200} Msun using the aLIGO PSD.

Results are written into the existing HDF5 files as an ``nr_errors`` group
and printed to stdout.

Usage:
    conda activate ./envs/gwbench
    python datasets/analytic/scripts/compute_nr_errors.py
"""

import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import sxs

from gwbenchmarks.metrics import frequency_domain_mismatch

DT_GEOMETRIC = 0.1
T_END_AFTER_PEAK_M = 100.0
MAX_LENGTH_M = 10000.0
EXTRA_BUFFER_M = 100.0
MTOT_VALUES = [40, 80, 120, 160, 200]

DATASET_DIR = Path(__file__).parent.parent


def find_available_levs(sxs_id):
    """Return sorted list of available Lev numbers for a simulation."""
    levs = []
    for lev in range(1, 7):
        try:
            sxs.load(f"{sxs_id}/Lev{lev}", progress=False,
                     ignore_deprecation=True)
            levs.append(lev)
        except Exception:
            continue
    return sorted(levs)


def load_h22_peak_aligned(sxs_id, lev):
    """Load h22, drop junk, peak-align, return (t_shifted, h22_complex)."""
    sim = sxs.load(f"{sxs_id}/Lev{lev}", progress=False,
                   ignore_deprecation=True)
    strain = sim.strain

    t = np.asarray(strain.t, dtype=float)
    h22 = np.asarray(strain.data[:, strain.index(2, 2)], dtype=np.complex128)

    t_ref = float(sim.metadata.reference_time)
    keep = t >= (t_ref + EXTRA_BUFFER_M)
    t = t[keep]
    h22 = h22[keep]

    i_peak = np.argmax(np.abs(h22))
    t_peak = t[i_peak]
    return t - t_peak, h22


def compute_nr_error(sxs_id, levs):
    """Compute FD mismatch between highest and second-highest Lev."""
    high_lev = levs[-1]
    low_lev = levs[-2]

    t_h, h_h = load_h22_peak_aligned(sxs_id, high_lev)
    t_l, h_l = load_h22_peak_aligned(sxs_id, low_lev)

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
            mm = frequency_domain_mismatch(h_l_r, h_h_r, DT_GEOMETRIC, mtot,
                                           f_low=15.0)
            result[f"fd_mismatch_M{mtot}"] = float(mm) if not np.isnan(mm) else None
        except Exception:
            result[f"fd_mismatch_M{mtot}"] = None

    vals = [result[f"fd_mismatch_M{m}"] for m in MTOT_VALUES
            if result.get(f"fd_mismatch_M{m}") is not None]
    result["fd_mismatch_combined"] = max(vals) if vals else None
    result["mean_fd_mismatch"] = float(np.mean(vals)) if vals else None
    return result


def process_split(h5_path):
    """Compute NR errors for all sims in one HDF5 file and write them back."""
    print(f"\nProcessing {h5_path.name}")

    with h5py.File(h5_path, "r") as f:
        sxs_ids = list(f["sims"].keys())

    nr_errors = {}
    t0 = time.time()
    for i, sid in enumerate(sxs_ids, 1):
        print(f"  [{i}/{len(sxs_ids)}] {sid} ... ", end="", flush=True)
        try:
            levs = find_available_levs(sid)
            if len(levs) < 2:
                print(f"only {len(levs)} Lev available, skipping")
                continue
            result = compute_nr_error(sid, levs)
            nr_errors[sid] = result
            mm = result.get("mean_fd_mismatch")
            mm_str = f"{mm:.2e}" if mm is not None else "N/A"
            print(f"Lev{result['low_lev']}↔Lev{result['high_lev']}  "
                  f"mean_mm={mm_str}")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n  Computed NR errors for {len(nr_errors)}/{len(sxs_ids)} sims "
          f"in {elapsed/60:.1f} min")

    if not nr_errors:
        print("  No NR errors computed, skipping HDF5 write")
        return nr_errors

    with h5py.File(h5_path, "a") as f:
        if "nr_errors" in f:
            del f["nr_errors"]

        nr_grp = f.create_group("nr_errors")
        ordered = sorted(nr_errors.keys())
        nr_grp.create_dataset("sxs_id",
            data=np.array(ordered, dtype="S"))
        for mtot in MTOT_VALUES:
            key = f"fd_mismatch_M{mtot}"
            nr_grp.create_dataset(key,
                data=np.array([nr_errors[s].get(key, np.nan) or np.nan
                               for s in ordered]))
        nr_grp.create_dataset("fd_mismatch_combined",
            data=np.array([nr_errors[s].get("fd_mismatch_combined", np.nan) or np.nan
                           for s in ordered]))
        nr_grp.create_dataset("mean_fd_mismatch",
            data=np.array([nr_errors[s].get("mean_fd_mismatch", np.nan) or np.nan
                           for s in ordered]))

    print(f"  Written nr_errors group to {h5_path.name}")
    return nr_errors


def main():
    all_errors = {}
    for split in ["training", "validation"]:
        path = DATASET_DIR / f"analytic_{split}.h5"
        if not path.exists():
            print(f"Skipping {path} (not found)")
            continue
        errors = process_split(path)
        all_errors.update(errors)

    if all_errors:
        means = [e["mean_fd_mismatch"] for e in all_errors.values()
                 if e.get("mean_fd_mismatch") is not None]
        if means:
            arr = np.array(means)
            print(f"\n{'='*60}")
            print(f"NR error summary (mean FD mismatch across masses):")
            print(f"  N sims with NR errors: {len(arr)}")
            print(f"  median: {np.median(arr):.2e}")
            print(f"  mean:   {np.mean(arr):.2e}")
            print(f"  min:    {np.min(arr):.2e}")
            print(f"  max:    {np.max(arr):.2e}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
