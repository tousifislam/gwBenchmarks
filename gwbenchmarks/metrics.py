"""Physically meaningful metrics for gravitational wave benchmarks."""

import numpy as np
from scipy.integrate import trapezoid


def mismatch(h_pred: np.ndarray, h_true: np.ndarray, dt: float = 1.0) -> float:
    """Compute mismatch 1 - <h1|h2> / sqrt(<h1|h1> <h2|h2>) in time domain.

    Parameters
    ----------
    h_pred : np.ndarray
        Complex predicted waveform.
    h_true : np.ndarray
        Complex reference waveform.
    dt : float
        Time step for integration.

    Returns
    -------
    float
        Mismatch in [0, 1].
    """
    inner_12 = trapezoid(np.real(h_pred * np.conj(h_true)), dx=dt)
    inner_11 = trapezoid(np.real(h_pred * np.conj(h_pred)), dx=dt)
    inner_22 = trapezoid(np.real(h_true * np.conj(h_true)), dx=dt)

    if inner_11 == 0 or inner_22 == 0:
        return 1.0

    overlap = inner_12 / np.sqrt(inner_11 * inner_22)
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def phase_rmse(h_pred: np.ndarray, h_true: np.ndarray) -> float:
    """RMSE of the unwrapped phase difference between two complex waveforms."""
    phi_pred = np.unwrap(np.angle(h_pred))
    phi_true = np.unwrap(np.angle(h_true))
    return float(np.sqrt(np.mean((phi_pred - phi_true) ** 2)))


def log_amplitude_rmse(h_pred: np.ndarray, h_true: np.ndarray) -> float:
    """RMSE of log-amplitude difference between two complex waveforms."""
    amp_pred = np.abs(h_pred)
    amp_true = np.abs(h_true)

    mask = (amp_pred > 0) & (amp_true > 0)
    if not np.any(mask):
        return float("inf")

    log_diff = np.log(amp_pred[mask]) - np.log(amp_true[mask])
    return float(np.sqrt(np.mean(log_diff**2)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized RMSE: RMSE divided by the range of true values."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    val_range = np.ptp(true)
    if val_range == 0:
        return rmse(pred, true)
    return rmse(pred, true) / val_range


def circular_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean circular error: mean(1 - cos(pred - true))."""
    diff = np.asarray(pred) - np.asarray(true)
    return float(np.mean(1.0 - np.cos(diff)))


def relative_error(pred: float, true: float) -> float:
    """Relative error |pred - true| / |true|."""
    if true == 0:
        return float("inf") if pred != 0 else 0.0
    return abs(pred - true) / abs(true)


def expected_calibration_error(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error for probability predictions.

    For the validity benchmark: bin predictions by predicted mismatch,
    compare mean predicted vs actual mismatch in each bin.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities / values.
    true_labels : np.ndarray
        True binary labels or values.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    float
        ECE value.
    """
    predicted_probs = np.asarray(predicted_probs)
    true_labels = np.asarray(true_labels)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(predicted_probs)
    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (predicted_probs == bin_edges[i + 1])
        n_bin = np.sum(mask)
        if n_bin == 0:
            continue
        avg_pred = np.mean(predicted_probs[mask])
        avg_true = np.mean(true_labels[mask])
        ece += (n_bin / n_total) * abs(avg_pred - avg_true)

    return float(ece)
