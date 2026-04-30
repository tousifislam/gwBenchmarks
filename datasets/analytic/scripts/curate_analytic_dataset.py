"""Curate the Analytic Bench dataset: h22 mode from non-spinning SXS BBH simulations.

Selects quasi-circular, non-spinning BBH simulations from the SXS catalog
with q in [1, 20], extracts the (2,2) strain mode at the highest available
resolution, resamples onto a uniform time grid aligned to the amplitude peak,
and writes a consolidated benchmark HDF5 file.

For non-spinning systems the coprecessing frame coincides with the inertial
frame and h_{2,-2}(t) = conj(h_{2,2}(t)), so only h22 is stored.

Source: SXS Gravitational Waveform Database (https://data.black-holes.org)

Usage:
    python curate_analytic_dataset.py [--output ../analytic_benchmark.h5]
"""

import argparse
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import sxs

# ── Configuration ──────────────────────────────────────────────────────────

DT_GEOMETRIC = 0.1          # uniform time step in units of M
TAPER_START_M = 200.0       # cosine taper ramp-on width (from start)
TAPER_END_M = 50.0          # cosine taper ramp-off width (to end)
T_END_AFTER_PEAK_M = 100.0  # keep ringdown up to this many M past peak
MAX_LENGTH_M = 10000.0      # maximum total waveform length
EXTRA_BUFFER_M = 100.0      # skip past metadata.reference_time by this much

SPIN_THRESHOLD = 0.01       # max |chi| to count as non-spinning
ECC_THRESHOLD = 0.005       # max eccentricity to count as quasi-circular
Q_MIN, Q_MAX = 1.0, 20.5


def smooth_taper(t, t_start, t_end):
    """Cosine taper: 0 at t <= t_start, 1 at t >= t_end."""
    w = np.ones_like(t)
    mask = (t > t_start) & (t < t_end)
    phase = np.pi * (t[mask] - t_start) / (t_end - t_start)
    w[mask] = 0.5 * (1.0 - np.cos(phase))
    w[t <= t_start] = 0.0
    return w


def select_simulations():
    """Query the SXS catalog and select the best non-spinning quasi-circular
    simulation at each unique mass ratio."""
    print("Loading SXS catalog...")
    simulations = sxs.Simulations.load()

    candidates = []
    for sim_id in simulations:
        if not sim_id.startswith("SXS:BBH:"):
            continue
        sim = simulations[sim_id]
        try:
            q = sim.get("reference_mass_ratio", None)
            if q is None:
                continue
            chi1 = sim.get("reference_dimensionless_spin1", None)
            chi2 = sim.get("reference_dimensionless_spin2", None)
            if chi1 is None or chi2 is None:
                continue
            chi1_mag = float(np.linalg.norm(chi1))
            chi2_mag = float(np.linalg.norm(chi2))
            ecc = sim.get("reference_eccentricity", None)
            ecc_val = float(ecc) if ecc is not None else 999.0

            if (chi1_mag < SPIN_THRESHOLD and chi2_mag < SPIN_THRESHOLD
                    and Q_MIN <= q <= Q_MAX and ecc_val < ECC_THRESHOLD):
                num = int(sim_id.split(":")[-1])
                candidates.append({
                    "sxs_id": sim_id,
                    "q": float(q),
                    "chi1_mag": chi1_mag,
                    "chi2_mag": chi2_mag,
                    "ecc": ecc_val,
                    "num": num,
                })
        except Exception:
            continue

    # Per unique q (rounded to 0.01): pick lowest eccentricity, then newest
    from collections import defaultdict
    q_groups = defaultdict(list)
    for s in candidates:
        q_groups[round(s["q"], 2)].append(s)

    selected = []
    for q_key in sorted(q_groups.keys()):
        group = q_groups[q_key]
        group.sort(key=lambda x: (x["ecc"], -x["num"]))
        selected.append(group[0])

    selected.sort(key=lambda x: x["q"])
    print(f"Selected {len(selected)} simulations (q in [{selected[0]['q']:.2f}, {selected[-1]['q']:.2f}])")
    return selected


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


def preprocess_sim(sxs_id, lev):
    """Load one SXS simulation and extract the (2,2) mode on a uniform grid."""
    sim = sxs.load(f"{sxs_id}/Lev{lev}", progress=False,
                   ignore_deprecation=True)
    strain = sim.strain

    t_raw = np.asarray(strain.t, dtype=float)
    h22_raw = np.asarray(strain.data[:, strain.index(2, 2)], dtype=np.complex128)

    # Drop junk radiation
    t_ref = float(sim.metadata.reference_time)
    keep = t_raw >= (t_ref + EXTRA_BUFFER_M)
    t_raw = t_raw[keep]
    h22_raw = h22_raw[keep]

    # Locate amplitude peak
    amp = np.abs(h22_raw)
    i_peak = np.argmax(amp)
    t_peak = t_raw[i_peak]

    # Shift so peak is at t=0
    t_shifted = t_raw - t_peak

    # Determine uniform grid bounds
    t_max = float(np.floor(
        min(t_shifted[-1], T_END_AFTER_PEAK_M) / DT_GEOMETRIC
    ) * DT_GEOMETRIC)
    t_min_candidate = -(MAX_LENGTH_M - t_max)
    t_min = float(np.ceil(
        max(t_shifted[0], t_min_candidate) / DT_GEOMETRIC
    ) * DT_GEOMETRIC)

    n_pts = int(round((t_max - t_min) / DT_GEOMETRIC)) + 1
    t_uniform = t_min + np.arange(n_pts) * DT_GEOMETRIC

    # Resample via linear interpolation (real and imag separately)
    h22_uniform = (
        np.interp(t_uniform, t_shifted, np.real(h22_raw))
        + 1j * np.interp(t_uniform, t_shifted, np.imag(h22_raw))
    )

    # Phase-align: phase = 0 at t = 0
    i0 = int(np.argmin(np.abs(t_uniform)))
    phi0 = float(np.angle(h22_uniform[i0]))
    h22_uniform *= np.exp(-1j * phi0)

    # Cosine taper at both ends
    w_start = smooth_taper(
        -t_uniform, t_start=-(t_uniform[0] + TAPER_START_M), t_end=-t_uniform[0]
    )
    w_end = smooth_taper(
        t_uniform, t_start=t_uniform[-1] - TAPER_END_M, t_end=t_uniform[-1]
    )
    h22_uniform *= w_start * w_end

    # Amplitude and unwrapped phase
    h22_amp = np.abs(h22_uniform).astype(np.float64)
    h22_phase = np.unwrap(np.angle(h22_uniform)).astype(np.float64)
    h22_phase -= h22_phase[i0]

    return {
        "t": t_uniform.astype(np.float64),
        "h22_amp": h22_amp,
        "h22_phase": h22_phase,
        "n_samples": n_pts,
        "lev": lev,
    }


def build_benchmark_file(selected, output):
    """Process all selected simulations and write the benchmark HDF5."""
    n_total = len(selected)
    n_done = 0
    n_failed = 0
    failed = {}
    t0 = time.time()

    with h5py.File(output, "w") as h5:
        h5.attrs["description"] = (
            "Non-spinning BBH (2,2) mode waveforms from SXS, q in [1, 20]. "
            "Uniform time grid in geometric units (M), peak-aligned at t=0."
        )
        h5.attrs["source"] = "SXS Gravitational Waveform Database"
        h5.attrs["url"] = "https://data.black-holes.org"
        h5.attrs["dt_geometric"] = DT_GEOMETRIC
        h5.attrs["t_align"] = "h22_amplitude_peak"
        h5.attrs["phase_convention"] = "phase=0 at t=0 (amplitude peak)"
        h5.attrs["taper_start_M"] = TAPER_START_M
        h5.attrs["taper_end_M"] = TAPER_END_M
        h5.attrs["non_spinning"] = True

        gz = dict(compression="gzip", compression_opts=4)

        for sim_info in selected:
            sid = sim_info["sxs_id"]
            n_done += 1
            try:
                lev = find_highest_lev(sid)
                if lev is None:
                    raise RuntimeError(f"No Lev found for {sid}")

                t_start = time.time()
                data = preprocess_sim(sid, lev)
                elapsed = time.time() - t_start

                grp = h5.create_group(f"sims/{sid}")
                grp.attrs["q"] = sim_info["q"]
                grp.attrs["eccentricity"] = sim_info["ecc"]
                grp.attrs["chi1_mag"] = sim_info["chi1_mag"]
                grp.attrs["chi2_mag"] = sim_info["chi2_mag"]
                grp.attrs["lev_used"] = data["lev"]
                grp.attrs["n_samples"] = data["n_samples"]

                grp.create_dataset("t", data=data["t"], **gz)
                grp.create_dataset("h22_amp", data=data["h22_amp"], **gz)
                grp.create_dataset("h22_phase", data=data["h22_phase"], **gz)
                h5.flush()

                rate = n_done / (time.time() - t0)
                eta = (n_total - n_done) / rate / 60 if rate > 0 else 0
                print(
                    f"[{n_done}/{n_total}] {sid} q={sim_info['q']:.4f} "
                    f"Lev{lev} n={data['n_samples']} ({elapsed:.1f}s, ETA {eta:.0f}min)"
                )

            except Exception as e:
                n_failed += 1
                failed[sid] = str(e)
                print(f"[FAIL] {sid}: {e}")
                traceback.print_exc()

        # Metadata table
        sxs_ids = sorted(h5["sims"].keys()) if "sims" in h5 else []
        if sxs_ids:
            meta = h5.create_group("metadata")
            qs = np.array([h5[f"sims/{s}"].attrs["q"] for s in sxs_ids])
            eccs = np.array([h5[f"sims/{s}"].attrs["eccentricity"] for s in sxs_ids])
            meta.create_dataset("sxs_id", data=np.array(sxs_ids, dtype="S"))
            meta.create_dataset("q", data=qs)
            meta.create_dataset("eccentricity", data=eccs)
            meta.attrs["param_columns"] = ["q"]
            meta.attrs["n_simulations"] = len(sxs_ids)

    elapsed_total = (time.time() - t0) / 60
    print(
        f"\nDone: total={n_total} processed={n_done - n_failed} "
        f"failed={n_failed} time={elapsed_total:.1f}min"
    )
    if failed:
        print("Failed simulations:")
        for sid, err in failed.items():
            print(f"  {sid}: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Curate Analytic Bench dataset from SXS non-spinning BBH"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "analytic_benchmark.h5",
    )
    args = parser.parse_args()

    selected = select_simulations()
    print()
    build_benchmark_file(selected, args.output)


if __name__ == "__main__":
    main()
