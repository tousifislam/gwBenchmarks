"""Physically meaningful metrics for gravitational wave benchmarks."""

import numpy as np


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


def relative_error(pred: float, true: float) -> float:
    """Relative error |pred - true| / |true|."""
    if true == 0:
        return float("inf") if pred != 0 else 0.0
    return abs(pred - true) / abs(true)


def rms_relative_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Pointwise RMS relative error: sqrt(mean((pred - true)^2 / true^2)).

    Treats every time step equally in fractional terms, avoiding the
    late-time bias of the global L2 relative error when the signal grows
    monotonically (e.g. the PN frequency parameter x(t)).
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = true != 0
    if not np.any(mask):
        return float("inf")
    return float(np.sqrt(np.mean(((pred[mask] - true[mask]) / true[mask]) ** 2)))


MSUN_SEC = 4.925491025543576e-06

FD_MASSES_MSUN = [40.0, 80.0, 120.0, 160.0, 200.0]


def frequency_domain_mismatch(
    h_pred,
    h_ref,
    dt_geometric: float,
    mtot_msun: float,
    f_low: float = 15.0,
    f_high: float = 990.0,
) -> float:
    """Frequency-domain mismatch (1 - match) using the aLIGO PSD via PyCBC.

    Converts geometric-unit waveforms to physical units, then computes the
    PyCBC match (maximised over time shift and orbital phase).

    Parameters
    ----------
    h_pred : array_like
        Predicted complex waveform in geometric units.
    h_ref : array_like
        Reference complex waveform in geometric units.
    dt_geometric : float
        Time step in units of M (total mass).
    mtot_msun : float
        Total mass in solar masses used for unit conversion.
    f_low : float
        Low-frequency cutoff in Hz (default 15 Hz).
    f_high : float
        High-frequency cutoff in Hz (default 990 Hz).

    Returns
    -------
    float
        Mismatch in [0, 1].

    Raises
    ------
    ImportError
        If PyCBC is not installed.
    """
    try:
        from pycbc.filter import match
        from pycbc.psd import aLIGOZeroDetHighPower
        from pycbc.types import TimeSeries
    except ImportError:
        raise ImportError(
            "PyCBC is required for frequency-domain mismatch. "
            "Install it with: pip install pycbc"
        )

    dt_sec = dt_geometric * mtot_msun * MSUN_SEC

    hp_pred = TimeSeries(np.real(np.asarray(h_pred, dtype=np.float64)), delta_t=dt_sec)
    hp_ref  = TimeSeries(np.real(np.asarray(h_ref,  dtype=np.float64)), delta_t=dt_sec)

    tlen = max(len(hp_pred), len(hp_ref))
    hp_pred.resize(tlen)
    hp_ref.resize(tlen)

    delta_f = 1.0 / hp_ref.duration
    flen = tlen // 2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)

    m, _ = match(
        hp_ref, hp_pred, psd=psd,
        low_frequency_cutoff=f_low,
        high_frequency_cutoff=f_high,
    )
    return float(1.0 - m)


def mean_fd_mismatch(
    h_pred,
    h_ref,
    dt_geometric: float,
    masses: list = FD_MASSES_MSUN,
    f_low: float = 15.0,
    f_high: float = 990.0,
) -> float:
    """Mean frequency-domain mismatch over a list of total masses.

    Parameters
    ----------
    h_pred : array_like
        Predicted complex waveform in geometric units.
    h_ref : array_like
        Reference complex waveform in geometric units.
    dt_geometric : float
        Time step in units of M (total mass).
    masses : list of float
        Total masses in solar masses at which to evaluate the mismatch.
        Defaults to FD_MASSES_MSUN = [40, 80, 120, 160, 200] M_sun.
    f_low : float
        Low-frequency cutoff in Hz (default 15 Hz).
    f_high : float
        High-frequency cutoff in Hz (default 990 Hz).

    Returns
    -------
    float
        Arithmetic mean of per-mass mismatches, in [0, 1].

    Raises
    ------
    ImportError
        If PyCBC is not installed.
    """
    values = [
        frequency_domain_mismatch(h_pred, h_ref, dt_geometric, m, f_low, f_high)
        for m in masses
    ]
    return float(np.mean(values))