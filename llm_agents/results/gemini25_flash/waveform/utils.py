import numpy as np
from sklearn.decomposition import TruncatedSVD

def parameters_to_array(param_list):
    """
    Converts a list of parameter dictionaries into a NumPy array.
    Order of parameters: q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega0
    """
    param_names = ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z", "omega0"]
    return np.array([[p[name] for name in param_names] for p in param_list])

def perform_svd(waveforms, n_components):
    """
    Performs Truncated SVD on a list of complex waveforms.

    Args:
        waveforms (list): A list of complex numpy arrays, each representing a waveform.
        n_components (int): The number of SVD components to keep.

    Returns:
        tuple: (svd_model, coefficients)
    """
    # Stack real and imaginary parts of all waveforms into a single 2D array
    # Each row is a flattened waveform (real part concatenated with imaginary part)
    stacked_waveforms = []
    for w in waveforms:
        stacked_waveforms.append(np.concatenate([np.real(w), np.imag(w)]))
    stacked_waveforms = np.array(stacked_waveforms)

    svd = TruncatedSVD(n_components=n_components)
    coefficients = svd.fit_transform(stacked_waveforms)

    return svd, coefficients

def align_waveforms(waveforms, times, target_length=None, t_start=None, t_end=None):
    """
    Aligns waveforms to a common time base and interpolates them to a target length.

    Args:
        waveforms (list): List of complex numpy arrays representing waveforms.
        times (list): List of numpy arrays representing time series for each waveform.
        target_length (int, optional): The desired length of the interpolated waveforms.
                                       If None, uses the maximum length among the input waveforms.
        t_start (float, optional): The common start time for all waveforms. If None, uses the minimum
                                   start time among all waveforms.
        t_end (float, optional): The common end time for all waveforms. If None, uses the maximum
                                 end time among all waveforms.

    Returns:
        tuple: (aligned_waveforms, aligned_times)
    """
    if not waveforms:
        return [], []

    # Determine common time base
    if t_start is None:
        t_start = min(t[0] for t in times)
    if t_end is None:
        t_end = max(t[-1] for t in times)

    if target_length is None:
        # If no target_length, find the maximum length required for interpolation
        # based on the new common time range and original sampling rates.
        # This is a simplification; a more robust approach would consider the highest sampling rate.
        target_length = max(len(w) for w in waveforms)


    aligned_times = np.linspace(t_start, t_end, target_length)
    aligned_waveforms = []

    for i in range(len(waveforms)):
        real_interp = np.interp(aligned_times, times[i], np.real(waveforms[i]), left=0.0, right=0.0)
        imag_interp = np.interp(aligned_times, times[i], np.imag(waveforms[i]), left=0.0, right=0.0)
        aligned_waveforms.append(real_interp + 1j * imag_interp)

    return aligned_waveforms, aligned_times

def reparameterize_eta_chi_eff(params_array):
    """
    Reparameterizes raw parameters to (eta, chi_eff).
    This is a simplified example; full implementation requires more details about spin vectors.
    """
    # Placeholder: for demonstration, just return q and a dummy chi_eff
    q = params_array[:, 0]
    eta = q / (1 + q)**2 # Symmetric mass ratio

    # This is a very simplified placeholder for chi_eff
    # A proper calculation needs full spin vectors and orbital angular momentum
    chi_eff = (params_array[:, 4] + params_array[:, 7]) / 2 # Example: avg of chi1z and chi2z

    return np.vstack([eta, chi_eff]).T

# Add more reparameterization functions as needed
