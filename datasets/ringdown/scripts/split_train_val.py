"""Split the ringdown benchmark dataset into training and validation sets.

Splits the spin grid roughly 50-50 for every mode: odd-indexed spin
points go to training, even-indexed to validation. This interleaved
split tests interpolation in spin space.

Usage:
    python split_train_val.py
"""

from pathlib import Path

import h5py
import numpy as np

DATASET_DIR = Path(__file__).parent.parent


def split_dataset():
    src = DATASET_DIR / "ringdown_benchmark.h5"
    train_path = DATASET_DIR / "ringdown_training.h5"
    val_path = DATASET_DIR / "ringdown_validation.h5"

    print(f"Reading {src}")
    f_src = h5py.File(src, "r")

    # Copy root attributes
    root_attrs = dict(f_src.attrs)

    n_modes = 0
    n_spin_train = 0
    n_spin_val = 0

    with h5py.File(train_path, "w") as f_train, h5py.File(val_path, "w") as f_val:
        # Copy root attributes
        for k, v in root_attrs.items():
            f_train.attrs[k] = v
            f_val.attrs[k] = v
        f_train.attrs["split"] = "training"
        f_train.attrs["split_method"] = "odd spin indices"
        f_val.attrs["split"] = "validation"
        f_val.attrs["split_method"] = "even spin indices"

        # Walk all mode groups
        def process(name, obj):
            nonlocal n_modes, n_spin_train, n_spin_val
            if not isinstance(obj, h5py.Group):
                return
            if "omega_real" not in obj:
                return

            n_modes += 1
            l = obj.attrs["l"]
            m = obj.attrs["m"]
            n = obj.attrs["n"]

            spin = obj["spin"][:]
            n_total = len(spin)
            train_idx = np.arange(1, n_total, 2)  # odd indices
            val_idx = np.arange(0, n_total, 2)     # even indices

            n_spin_train = len(train_idx)
            n_spin_val = len(val_idx)

            datasets = ["spin", "omega_real", "omega_imag", "A_lm_real", "A_lm_imag"]

            for f_out, idx in [(f_train, train_idx), (f_val, val_idx)]:
                g = f_out.create_group(name)
                g.attrs["l"] = l
                g.attrs["m"] = m
                g.attrs["n"] = n
                for ds_name in datasets:
                    data = obj[ds_name][:]
                    g.create_dataset(ds_name, data=data[idx])

        f_src.visititems(process)

    f_src.close()

    print(f"Modes: {n_modes}")
    print(f"Training: {n_spin_train} spin points per mode -> {train_path.name}")
    print(f"Validation: {n_spin_val} spin points per mode -> {val_path.name}")


if __name__ == "__main__":
    split_dataset()
