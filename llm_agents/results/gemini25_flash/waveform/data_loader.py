import h5py
import numpy as np
import os

def load_waveform_data(file_path):
    """
    Loads waveform data from an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        dict: A dictionary containing parameters and waveform data.
    """
    data = {"parameters": [], "waveforms": [], "times": []}
    with h5py.File(file_path, "r") as f:
        n_simulations = f.attrs["n_simulations"]
        for i in range(n_simulations):
            group_name = f"sim_{i:04d}"
            g = f[group_name]

            # Extract parameters
            params = {
                "q": g.attrs["q"],
                "chi1x": g.attrs["chi1x"],
                "chi1y": g.attrs["chi1y"],
                "chi1z": g.attrs["chi1z"],
                "chi2x": g.attrs["chi2x"],
                "chi2y": g.attrs["chi2y"],
                "chi2z": g.attrs["chi2z"],
                "omega0": g.attrs["omega0"],
            }
            data["parameters"].append(params)

            # Extract time and waveform
            data["times"].append(g["t"][:])
            data["waveforms"].append(g["h22_real"][:] + 1j * g["h22_imag"][:])
    return data

if __name__ == "__main__":
    # Example usage:
    # Assuming the script is run from the gwBenchmarks/ root
    training_file = "datasets/waveform/waveform_training.h5"
    validation_file = "datasets/waveform/waveform_validation.h5"

    print(f"Loading training data from: {training_file}")
    training_data = load_waveform_data(training_file)
    print(f"Loaded {len(training_data['parameters'])} training simulations.")

    print(f"Loading validation data from: {validation_file}")
    validation_data = load_waveform_data(validation_file)
    print(f"Loaded {len(validation_data['parameters'])} validation simulations.")

    # You can now access data like:
    # training_data["parameters"][0]["q"]
    # training_data["waveforms"][0]
    # training_data["times"][0]
