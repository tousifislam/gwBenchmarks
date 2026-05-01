"""Curate the Validity Bench dataset: NRHybSur3dq8 mismatch against SXS NR.

For each aligned-spin, quasi-circular SXS BBH simulation:
  1. Load h22 from SXS at highest Lev
  2. Generate h22 from NRHybSur3dq8 at the same parameters
  3. Align both waveforms: peak at t=0, initial phase=0, consistent phase sign
  4. Compute time-domain mismatch
  5. Store (q, chi1z, chi2z, omega0, mismatch_td)

Split roughly 50-50 into training and validation.

Source: SXS catalog + gwsurrogate NRHybSur3dq8

Usage:
    python curate_validity_dataset.py
"""

import argparse
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import sxs

# ── Configuration ──────────────────────────────────────────────────────────

DT_GEOMETRIC = 0.1
T_END_AFTER_PEAK_M = 100.0
MAX_LENGTH_M = 10000.0
EXTRA_BUFFER_M = 100.0

PERP_SPIN_THRESHOLD = 0.01
ECC_THRESHOLD = 0.01


# ── Peak finding (from waveform_bench) ─────────────────────────────────────

def get_peak_via_quadratic_fit(t, func):
    """Quadratic fit through 5 points around argmax to refine the peak."""
    t = np.asarray(t, dtype=float)
    func = np.asarray(func, dtype=float)
    index = int(np.argmax(func))
    index = max(2, min(len(t) - 3, index))
    test_t = t[index - 2:index + 3] - t[index]
    test_f = func[index - 2:index + 3]
    x_vecs = np.array([np.ones(5), test_t, test_t ** 2.0])
    inv = np.linalg.inv(
        np.array([[v1.dot(v2) for v1 in x_vecs] for v2 in x_vecs])
    )
    y = np.array([test_f.dot(v) for v in x_vecs])
    coefs = np.array([y.dot(v) for v in inv])
    return (
        t[index] - coefs[1] / (2.0 * coefs[2]),
        coefs[0] - coefs[1] ** 2.0 / (4.0 * coefs[2]),
    )


def find_peak_time_22(t, h22):
    """Refined peak time of the (2,2) mode using |h22|^2 and quadratic fit."""
    amp_sq = np.abs(np.asarray(h22)) ** 2
    tpeak, _ = get_peak_via_quadratic_fit(np.asarray(t), amp_sq)
    return float(tpeak)


# ── Waveform alignment ────────────────────────────────────────────────────

def align_waveform(t, h22):
    """Align a complex h22 waveform: peak at t=0, phase=0 at peak.

    Returns
    -------
    t_aligned, h22_aligned : arrays
        Shifted time and phase-aligned waveform.
    t_peak : float
        Original peak time (for computing omega0).
    """
    t = np.asarray(t, dtype=float)
    h22 = np.asarray(h22, dtype=np.complex128)

    t_peak = find_peak_time_22(t, h22)
    t_aligned = t - t_peak

    # Phase = 0 at peak
    i_peak = int(np.argmin(np.abs(t_aligned)))
    phi_peak = float(np.angle(h22[i_peak]))
    h22_aligned = h22 * np.exp(-1j * phi_peak)

    return t_aligned, h22_aligned, t_peak


def ensure_consistent_phase_sign(h_ref, h_sur, t):
    """Ensure both waveforms have the same phase evolution direction.

    Checks the sign of dphi/dt near the peak and flips h_sur if needed.
    """
    i0 = int(np.argmin(np.abs(t)))
    window = min(50, len(t) - i0 - 1, i0)
    if window < 5:
        return h_sur

    phi_ref = np.unwrap(np.angle(h_ref[i0:i0 + window]))
    phi_sur = np.unwrap(np.angle(h_sur[i0:i0 + window]))

    dphi_ref = phi_ref[-1] - phi_ref[0]
    dphi_sur = phi_sur[-1] - phi_sur[0]

    if dphi_ref * dphi_sur < 0:
        h_sur = np.conj(h_sur)

    return h_sur


# ── Time-domain mismatch ──────────────────────────────────────────────────

def time_domain_mismatch(h1, h2, dt):
    """Compute time-domain mismatch: 1 - <h1|h2>/sqrt(<h1|h1><h2|h2>)."""
    inner_12 = np.sum(np.real(h1 * np.conj(h2))) * dt
    inner_11 = np.sum(np.real(h1 * np.conj(h1))) * dt
    inner_22 = np.sum(np.real(h2 * np.conj(h2))) * dt

    if inner_11 == 0 or inner_22 == 0:
        return 1.0

    overlap = inner_12 / np.sqrt(inner_11 * inner_22)
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


# ── SXS catalog selection ─────────────────────────────────────────────────

def select_simulations():
    """Select all aligned-spin quasi-circular SXS BBH simulations."""
    print("Loading SXS catalog...")
    simulations = sxs.Simulations.load()

    selected = []
    for sim_id in simulations:
        if not sim_id.startswith("SXS:BBH:"):
            continue
        sim = simulations[sim_id]
        try:
            q = sim.get("reference_mass_ratio", None)
            if q is None:
                continue
            chi1 = np.array(sim.get("reference_dimensionless_spin1", None), dtype=float)
            chi2 = np.array(sim.get("reference_dimensionless_spin2", None), dtype=float)
            if chi1 is None or chi2 is None:
                continue

            chi1_perp = np.sqrt(chi1[0] ** 2 + chi1[1] ** 2)
            chi2_perp = np.sqrt(chi2[0] ** 2 + chi2[1] ** 2)
            if chi1_perp > PERP_SPIN_THRESHOLD or chi2_perp > PERP_SPIN_THRESHOLD:
                continue

            ecc = sim.get("reference_eccentricity", None)
            ecc_val = float(ecc) if ecc is not None else 999.0
            if ecc_val > ECC_THRESHOLD:
                continue

            ref_omega_vec = sim.get("reference_orbital_frequency", None)
            if ref_omega_vec is not None:
                omega0 = float(np.linalg.norm(ref_omega_vec))
            else:
                omega0 = float("nan")

            num = int(sim_id.split(":")[-1])
            selected.append({
                "sxs_id": sim_id,
                "q": float(q),
                "chi1z": float(chi1[2]),
                "chi2z": float(chi2[2]),
                "ecc": ecc_val,
                "omega0": omega0,
                "num": num,
            })
        except Exception:
            continue

    selected.sort(key=lambda x: (x["q"], x["chi1z"], x["chi2z"]))
    print(f"Selected {len(selected)} aligned-spin quasi-circular simulations")
    return selected


# ── Surrogate waveform generation ──────────────────────────────────────────

def load_surrogate():
    """Load NRHybSur3dq8 surrogate model."""
    try:
        import gwsurrogate
        sur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
        print("Loaded NRHybSur3dq8 surrogate")
        return sur
    except Exception as e:
        print(f"Failed to load NRHybSur3dq8: {e}")
        print("Trying via gwtools...")
        import gwtools
        sur = gwtools.surrogates.NRHybSur3dq8()
        print("Loaded NRHybSur3dq8 via gwtools")
        return sur


def generate_surrogate_h22(sur, q, chi1z, chi2z, dt, f_low_dimless=None):
    """Generate h22 from NRHybSur3dq8.

    Parameters
    ----------
    sur : surrogate model
    q : float
        Mass ratio (>= 1).
    chi1z, chi2z : float
        z-components of dimensionless spins.
    dt : float
        Time step in geometric units.
    f_low_dimless : float or None
        Starting frequency in geometric units. If None, use surrogate default.

    Returns
    -------
    t, h22 : arrays
    """
    chi1_vec = [0.0, 0.0, chi1z]
    chi2_vec = [0.0, 0.0, chi2z]

    t, h, dyn = sur(q, chi1_vec, chi2_vec, dt=dt, f_low=f_low_dimless)
    # h is a dict of modes; extract (2,2)
    h22 = h[(2, 2)]
    return np.asarray(t, dtype=float), np.asarray(h22, dtype=np.complex128)


# ── Main processing ───────────────────────────────────────────────────────

def find_highest_lev(sxs_id):
    """Determine the highest available Lev for a simulation."""
    for lev in [6, 5, 4, 3, 2]:
        try:
            _ = sxs.load(f"{sxs_id}/Lev{lev}", progress=False,
                         ignore_deprecation=True)
            return lev
        except Exception:
            continue
    return None


def load_sxs_h22(sxs_id, lev):
    """Load h22 from SXS and drop junk radiation."""
    sim = sxs.load(f"{sxs_id}/Lev{lev}", progress=False,
                   ignore_deprecation=True)
    strain = sim.strain
    t_raw = np.asarray(strain.t, dtype=float)
    h22_raw = np.asarray(strain.data[:, strain.index(2, 2)], dtype=np.complex128)

    t_ref = float(sim.metadata.reference_time)
    keep = t_raw >= (t_ref + EXTRA_BUFFER_M)
    return t_raw[keep], h22_raw[keep]


def compute_mismatch_for_sim(sim_info, sur):
    """Compute TD mismatch between SXS NR and NRHybSur3dq8 for one simulation."""
    sid = sim_info["sxs_id"]
    q = sim_info["q"]
    chi1z = sim_info["chi1z"]
    chi2z = sim_info["chi2z"]

    # Load SXS
    lev = find_highest_lev(sid)
    if lev is None:
        raise RuntimeError(f"No Lev found for {sid}")
    t_nr, h22_nr = load_sxs_h22(sid, lev)

    # Align NR: peak at t=0, phase=0 at peak
    t_nr, h22_nr, t_peak_nr = align_waveform(t_nr, h22_nr)

    # Generate surrogate — f_low = omega_orb / pi (GW frequency in cycles/M)
    omega0 = sim_info["omega0"]
    f_low = omega0 / np.pi
    t_sur, h22_sur = generate_surrogate_h22(sur, q, chi1z, chi2z, dt=DT_GEOMETRIC,
                                             f_low_dimless=f_low)

    # Align surrogate: peak at t=0, phase=0 at peak
    t_sur, h22_sur, _ = align_waveform(t_sur, h22_sur)

    # Find common time range
    t_start = max(t_nr[0], t_sur[0])
    t_end = min(t_nr[-1], t_sur[-1], T_END_AFTER_PEAK_M)
    t_start = float(np.ceil(t_start / DT_GEOMETRIC) * DT_GEOMETRIC)
    t_end = float(np.floor(t_end / DT_GEOMETRIC) * DT_GEOMETRIC)

    n_pts = int(round((t_end - t_start) / DT_GEOMETRIC)) + 1
    t_common = t_start + np.arange(n_pts) * DT_GEOMETRIC

    # Interpolate both onto common grid
    h_nr_common = (
        np.interp(t_common, t_nr, np.real(h22_nr))
        + 1j * np.interp(t_common, t_nr, np.imag(h22_nr))
    )
    h_sur_common = (
        np.interp(t_common, t_sur, np.real(h22_sur))
        + 1j * np.interp(t_common, t_sur, np.imag(h22_sur))
    )

    # Ensure consistent phase sign
    h_sur_common = ensure_consistent_phase_sign(h_nr_common, h_sur_common, t_common)

    # Compute mismatch
    mm = time_domain_mismatch(h_nr_common, h_sur_common, DT_GEOMETRIC)

    return {
        "mm_td": mm,
        "lev": lev,
        "n_common": n_pts,
        "t_start": t_start,
        "t_end": t_end,
    }


def split_train_val(records):
    """Split records roughly 50-50. Alternating by sorted index."""
    train = [records[i] for i in range(0, len(records), 2)]
    val = [records[i] for i in range(1, len(records), 2)]
    return train, val


def write_split(records, output, split_name):
    """Write one split to HDF5."""
    with h5py.File(output, "w") as h5:
        h5.attrs["description"] = (
            f"Validity Bench ({split_name}): time-domain mismatch of "
            f"NRHybSur3dq8 against SXS NR for aligned-spin quasi-circular BBH."
        )
        h5.attrs["source_nr"] = "SXS Gravitational Waveform Database"
        h5.attrs["source_surrogate"] = "NRHybSur3dq8"
        h5.attrs["split"] = split_name

        n = len(records)
        q_arr = np.array([r["q"] for r in records])
        chi1z_arr = np.array([r["chi1z"] for r in records])
        chi2z_arr = np.array([r["chi2z"] for r in records])
        omega0_arr = np.array([r["omega0"] for r in records])
        mm_arr = np.array([r["mm_td"] for r in records])
        sxs_ids = [r["sxs_id"] for r in records]

        h5.create_dataset("q", data=q_arr)
        h5.create_dataset("chi1z", data=chi1z_arr)
        h5.create_dataset("chi2z", data=chi2z_arr)
        h5.create_dataset("omega0", data=omega0_arr)
        h5.create_dataset("mm_td", data=mm_arr)
        h5.create_dataset("sxs_id", data=np.array(sxs_ids, dtype="S"))
        h5.attrs["n_simulations"] = n

    print(f"  {split_name}: {n} simulations -> {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Curate Validity Bench dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent,
    )
    args = parser.parse_args()

    selected = select_simulations()
    sur = load_surrogate()
    print()

    # Process all simulations
    n_total = len(selected)
    records = []
    n_failed = 0
    t0 = time.time()

    for i, sim_info in enumerate(selected):
        sid = sim_info["sxs_id"]
        try:
            t_start = time.time()
            result = compute_mismatch_for_sim(sim_info, sur)
            elapsed = time.time() - t_start

            record = {**sim_info, **result}
            records.append(record)

            rate = (i + 1) / (time.time() - t0)
            eta = (n_total - i - 1) / rate / 60 if rate > 0 else 0
            print(
                f"[{i+1}/{n_total}] {sid} q={sim_info['q']:.2f} "
                f"chi1z={sim_info['chi1z']:+.3f} chi2z={sim_info['chi2z']:+.3f} "
                f"mm={result['mm_td']:.2e} ({elapsed:.1f}s, ETA {eta:.0f}min)"
            )
        except Exception as e:
            n_failed += 1
            print(f"[FAIL] {sid}: {e}")
            traceback.print_exc()

    elapsed_total = (time.time() - t0) / 60
    print(f"\nProcessed: {len(records)}, failed: {n_failed}, time: {elapsed_total:.1f}min")

    if not records:
        print("No records to write.")
        return

    # Sort by (q, chi1z, chi2z) and split
    records.sort(key=lambda r: (r["q"], r["chi1z"], r["chi2z"]))
    train, val = split_train_val(records)

    print(f"\nSplit: {len(train)} training, {len(val)} validation")
    write_split(train, args.output_dir / "validity_training.h5", "training")
    write_split(val, args.output_dir / "validity_validation.h5", "validation")

    # Summary stats
    mm_all = np.array([r["mm_td"] for r in records])
    print(f"\nMismatch stats: median={np.median(mm_all):.2e}, "
          f"max={np.max(mm_all):.2e}, min={np.min(mm_all):.2e}")


if __name__ == "__main__":
    main()
