"""Starter detection statistic: the conventional template-bank maximum."""

from __future__ import annotations

import numpy as np


def detection_statistic(
    complex_overlaps: np.ndarray,
    bank_waveforms: np.ndarray,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return max_i |<d, b_i>|^2 over the unit-norm bank rows for every trial.

    complex_overlaps has shape (n_trials, n_rows); entry [t, i] is
    sum(d_t * conj(b_i) * weights) for the data d_t of trial t.
    """

    z = np.asarray(complex_overlaps, dtype=np.complex128)
    if z.ndim != 2 or z.shape[1] != len(bank_waveforms):
        raise ValueError("complex_overlaps must have shape (n_trials, n_rows)")
    return np.max(np.abs(z) ** 2, axis=1)
