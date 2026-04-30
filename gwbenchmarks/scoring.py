"""Combined scoring: accuracy loss penalized by evaluation cost."""

import numpy as np


def compute_score(loss: float, runtime: float, t0: float, alpha: float) -> float:
    """Compute benchmark score S_b = L_b * [1 + alpha * log(1 + t_b / t0)].

    Parameters
    ----------
    loss : float
        Accuracy loss L_b from the benchmark-specific loss function.
    runtime : float
        Evaluation runtime t_b in seconds.
    t0 : float
        Reference runtime scale in seconds.
    alpha : float
        Cost penalty weight.

    Returns
    -------
    float
        Combined score (lower is better).
    """
    cost_penalty = 1.0 + alpha * np.log(1.0 + runtime / t0)
    return loss * cost_penalty
