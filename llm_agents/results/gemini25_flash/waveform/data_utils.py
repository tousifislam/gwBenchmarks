import h5py
import numpy as np

def load_waveform_data(file_path):
    """
    Loads waveform data from an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              parameters and waveform data for a single simulation.
    """
    data = []
    with h5py.File(file_path, "r") as f:
        n_simulations = f.attrs["n_simulations"]
        for i in range(n_simulations):
            group_name = f"sim_{i:04d}"
            g = f[group_name]

            # Extract parameters
            q = g.attrs["q"]
            chi1 = np.array([g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"]])
            chi2 = np.array([g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"]])
            omega0 = g.attrs["omega0"]

            # Extract waveform data
            t = g["t"][:]
            h22_real = g["h22_real"][:]
            h22_imag = g["h22_imag"][:]
            h22 = h22_real + 1j * h22_imag

            data.append({
                "q": q,
                "chi1": chi1,
                "chi2": chi2,
                "omega0": omega0,
                "t": t,
                "h22": h22
            })
    return data

def apply_time_convention(t, h22, convention):
    """
    Applies a specified time convention to the waveform data.

    Args:
        t (np.ndarray): Time array.
        h22 (np.ndarray): Complex waveform data.
        convention (str): The time convention to apply ('t0_at_peak', 't0_at_start', 'reversed_time').

    Returns:
        tuple: A tuple containing the modified time array and waveform data.
    """
    if convention == "t0_at_peak":
        # As stored, t=0 is already at peak amplitude
        return t, h22
    elif convention == "t0_at_start":
        t_shift = t[0]
        return t - t_shift, h22
    elif convention == "reversed_time":
        # t=0 at the last point, so reverse time and waveform
        t_reversed = t[-1] - t
        return t_reversed, h22[::-1]  # Reverse waveform data as well
    else:
        raise ValueError(f"Unknown time convention: {convention}")

def apply_phase_alignment(t, h22, alignment):
    """
    Applies a specified phase alignment to the waveform data.

    Args:
        t (np.ndarray): Time array.
        h22 (np.ndarray): Complex waveform data.
        alignment (str): The phase alignment to apply ('phase0_at_peak', 'initial_phase0', 'pn_relative').

    Returns:
        np.ndarray: The phase-aligned complex waveform data.
    """
    if alignment == "phase0_at_peak":
        # As stored, phase=0 is already at peak amplitude
        return h22
    elif alignment == "initial_phase0":
        initial_phase = np.angle(h22[0])
        return h22 * np.exp(-1j * initial_phase)
    elif alignment == "pn_relative":
        # This would require a PN baseline, which is not readily available here.
        # For now, we'll treat it as no specific alignment, or raise an error if strictly required.
        # TODO: Implement actual PN baseline subtraction if needed.
        print("Warning: PN relative phase alignment is not fully implemented.")
        return h22
    else:
        raise ValueError(f"Unknown phase alignment: {alignment}")

def reparameterize(q, chi1, chi2, omega0, param_set):
    """
    Reparameterizes the input parameters.

    Args:
        q (float): Mass ratio.
        chi1 (np.ndarray): Spin vector of the primary black hole.
        chi2 (np.ndarray): Spin vector of the secondary black hole.
        omega0 (float): Reference orbital frequency.
        param_set (str): The parameterization set to use.

    Returns:
        np.ndarray: Reparameterized parameters.
    """
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    mtot = m1 + m2 # Assuming mtot = 1 for now, or just using q as a ratio
    eta = m1 * m2 / (mtot * mtot) # Symmetric mass ratio

    chi1z = chi1[2]
    chi2z = chi2[2]

    # Effective inspiral spin parameter
    chi_eff = (m1 * chi1z + m2 * chi2z) / mtot

    # Precessing spin parameter
    chi_p = np.sqrt(
        (m1 * np.linalg.norm(chi1[:2]))**2 + (m2 * np.linalg.norm(chi2[:2]))**2
    ) / mtot # Approximation for chi_p

    if param_set == "raw_7d":
        return np.array([q, chi1[0], chi1[1], chi1[2], chi2[0], chi2[1], chi2[2]])
    elif param_set == "effective_spins":
        # Need magnitudes and angles for a full reparameterization,
        # but for now, provide a simplified version with common effective spins
        return np.array([eta, chi_eff, chi_p, np.linalg.norm(chi1), np.linalg.norm(chi2)])
    elif param_set == "mass_difference_spins":
        # delta_m, chi_eff, chi_p, |chi1|, |chi2|, phi1, phi2
        # Need to define phi1, phi2 from chi1x, chi1y, chi2x, chi2y
        delta_m = (m1 - m2) / mtot
        phi1 = np.arctan2(chi1[1], chi1[0])
        phi2 = np.arctan2(chi2[1], chi2[0])
        return np.array([delta_m, chi_eff, chi_p, np.linalg.norm(chi1), np.linalg.norm(chi2), phi1, phi2])
    elif param_set == "spherical_spins":
        # (eta, |chi1|, theta1, phi1, |chi2|, theta2, phi2)
        # Need to convert Cartesian chi to spherical
        r1, theta1, phi1 = cartesian_to_spherical(chi1)
        r2, theta2, phi2 = cartesian_to_spherical(chi2)
        return np.array([eta, r1, theta1, phi1, r2, theta2, phi2])
    elif param_set == "raw_7d_omega0":
        return np.array([q, chi1[0], chi1[1], chi1[2], chi2[0], chi2[1], chi2[2], omega0])
    else:
        raise ValueError(f"Unknown parameterization set: {param_set}")

def cartesian_to_spherical(vec):
    """
    Converts Cartesian coordinates to spherical coordinates (r, theta, phi).
    theta is the polar angle (from z-axis), phi is the azimuthal angle (from x-axis).
    """
    x, y, z = vec
    r = np.linalg.norm(vec)
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return r, theta, phi

class SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.U = None
        self.s = None
        self.Vh = None
        self.mean_waveform = None

    def fit(self, waveforms):
        """
        Fits the SVD basis to a set of waveforms.

        Args:
            waveforms (np.ndarray): A 2D array of complex waveforms,
                                    where rows are time samples and columns are different waveforms.
        """
        self.mean_waveform = np.mean(waveforms, axis=1)
        centered_waveforms = waveforms - self.mean_waveform[:, np.newaxis]

        # Perform SVD
        U, s, Vh = np.linalg.svd(centered_waveforms, full_matrices=False)
        self.U = U[:, :self.n_components]
        self.s = s[:self.n_components]
        self.Vh = Vh[:self.n_components, :]

    def transform(self, waveforms):
        """
        Transforms waveforms into SVD coefficients.

        Args:
            waveforms (np.ndarray): A 2D array of complex waveforms.

        Returns:
            np.ndarray: SVD coefficients.
        """
        if self.U is None:
            raise RuntimeError("SVD model has not been fitted yet.")
        centered_waveforms = waveforms - self.mean_waveform[:, np.newaxis]
        return np.dot(self.U.conj().T, centered_waveforms)

    def inverse_transform(self, coefficients):
        """
        Reconstructs waveforms from SVD coefficients.

        Args:
            coefficients (np.ndarray): SVD coefficients.

        Returns:
            np.ndarray: Reconstructed waveforms.
        """
        if self.U is None:
            raise RuntimeError("SVD model has not been fitted yet.")
        return np.dot(self.U, coefficients) + self.mean_waveform[:, np.newaxis]

