"""Curate the ringdown benchmark dataset from Cook & Zalutskiy QNM tables.

Extracts (spin, omega_real, omega_imag, A_lm) for each (l, m, n) mode
from the raw Zenodo HDF5 files and writes a single consolidated HDF5
benchmark file.

Source: https://zenodo.org/records/2650358
Reference: Cook & Zalutskiy, Phys. Rev. D 90 (2014) 124021

Usage:
    python curate_qnm_dataset.py --raw-dir ../  --output ../ringdown_benchmark.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def extract_modes(raw_dir: Path, l_range=(2, 16), n_range=(0, 7)):
    """Extract QNM frequencies from raw Cook & Zalutskiy HDF5 files.

    Parameters
    ----------
    raw_dir : Path
        Directory containing KerrQNM_00.h5 through KerrQNM_07.h5.
    l_range : tuple
        (l_min, l_max) inclusive.
    n_range : tuple
        (n_min, n_max) inclusive.

    Yields
    ------
    dict with keys: l, m, n, spin, omega_real, omega_imag, A_lm_real, A_lm_imag
    """
    l_min, l_max = l_range
    n_min, n_max = n_range

    for n in range(n_min, n_max + 1):
        fname = raw_dir / f"KerrQNM_{n:02d}.h5"
        if not fname.exists():
            print(f"  Skipping n={n}, file not found: {fname}")
            continue

        with h5py.File(fname, "r") as f:
            n_group = f[f"n{n:02d}"]

            for m_key in sorted(n_group.keys()):
                m_group = n_group[m_key]

                for ds_key in sorted(m_group.keys()):
                    # Parse {l, m, n} from dataset name
                    parts = ds_key.strip("{}").split(",")
                    l_val = int(parts[0])
                    m_val = int(parts[1])
                    n_val = int(parts[2])

                    if l_val < l_min or l_val > l_max:
                        continue

                    data = m_group[ds_key][:]
                    yield {
                        "l": l_val,
                        "m": m_val,
                        "n": n_val,
                        "spin": data[:, 0],
                        "omega_real": data[:, 1],
                        "omega_imag": data[:, 2],
                        "A_lm_real": data[:, 3],
                        "A_lm_imag": data[:, 4],
                    }


def build_benchmark_file(raw_dir: Path, output: Path, l_range=(2, 16), n_range=(0, 7)):
    """Build the consolidated ringdown benchmark HDF5 file."""
    print(f"Reading raw QNM data from {raw_dir}")
    print(f"Output: {output}")
    print(f"l range: {l_range}, n range: {n_range}")

    mode_count = 0
    with h5py.File(output, "w") as out:
        out.attrs["source"] = "Cook & Zalutskiy, Phys. Rev. D 90 (2014) 124021"
        out.attrs["zenodo_doi"] = "10.5281/zenodo.2650358"
        out.attrs["spin_parameter"] = "s = -2 (gravitational)"
        out.attrs["l_range"] = list(l_range)
        out.attrs["n_range"] = list(n_range)
        out.attrs["description"] = (
            "Kerr QNM frequencies for s=-2 gravitational perturbations. "
            "Each mode group contains arrays indexed by dimensionless spin a/M."
        )

        for mode in extract_modes(raw_dir, l_range, n_range):
            l, m, n = mode["l"], mode["m"], mode["n"]
            group_name = f"l{l}/m{m:+d}/n{n}"
            g = out.create_group(group_name)
            g.attrs["l"] = l
            g.attrs["m"] = m
            g.attrs["n"] = n

            g.create_dataset("spin", data=mode["spin"])
            g.create_dataset("omega_real", data=mode["omega_real"])
            g.create_dataset("omega_imag", data=mode["omega_imag"])
            g.create_dataset("A_lm_real", data=mode["A_lm_real"])
            g.create_dataset("A_lm_imag", data=mode["A_lm_imag"])

            mode_count += 1

    print(f"Wrote {mode_count} modes to {output}")


def main():
    parser = argparse.ArgumentParser(description="Curate ringdown benchmark dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Directory with KerrQNM_*.h5 files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "ringdown_benchmark.h5",
        help="Output HDF5 file",
    )
    parser.add_argument("--l-min", type=int, default=2)
    parser.add_argument("--l-max", type=int, default=16)
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=7)
    args = parser.parse_args()

    build_benchmark_file(
        args.raw_dir,
        args.output,
        l_range=(args.l_min, args.l_max),
        n_range=(args.n_min, args.n_max),
    )


if __name__ == "__main__":
    main()
