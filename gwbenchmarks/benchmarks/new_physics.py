"""Benchmark 7: New Physics Bench (RG-tail inspiral waveform).

Unlike Benchmarks 1-6 which are data-driven (train on HDF5, evaluate on
validation set), this benchmark is formula-driven: the agent receives physics
formulas and must implement h_of_f(), which is then scored against a reference
implementation using PyCBC frequency-domain mismatch.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark


# Default test-case parameter grid (4 Mc x 4 eta x 3 dL x 3 lambda_RG = 144)
DEFAULT_CHIRP_MASSES = [12.0, 20.0, 28.3, 40.0]
DEFAULT_ETAS = [0.12, 0.16, 0.22, 0.247]
DEFAULT_DISTANCES = [200.0, 410.0, 1000.0]
DEFAULT_LAMBDA_RGS = [0.8, 1.0, 1.2]


def make_test_cases(config: dict | None = None) -> list[dict]:
    """Generate the parameter grid of test cases."""
    if config is None:
        config = {}
    params = config.get("parameters", {})
    mcs = params.get("chirp_mass_msun", DEFAULT_CHIRP_MASSES)
    etas = params.get("eta", DEFAULT_ETAS)
    dls = params.get("distance_mpc", DEFAULT_DISTANCES)
    rgs = params.get("lambda_RG", DEFAULT_LAMBDA_RGS)

    fixed = config.get("fixed", {})
    tc = fixed.get("tc", 0.0)
    phic = fixed.get("phic", 0.0)
    fmax_over_fisco = fixed.get("fmax_over_fisco", 1.0)
    sigma_taper_over_fisco = fixed.get("sigma_taper_over_fisco", 0.01)

    cases = []
    for Mc in mcs:
        for eta in etas:
            for dL in dls:
                for rg in rgs:
                    cases.append({
                        "Mc": Mc,
                        "eta": eta,
                        "dL": dL,
                        "lambda_RG": rg,
                        "tc": tc,
                        "phic": phic,
                        "fmax_over_fisco": fmax_over_fisco,
                        "sigma_taper_over_fisco": sigma_taper_over_fisco,
                    })
    return cases


def fd_mismatch_from_fd_arrays(
    h_pred: np.ndarray,
    h_ref: np.ndarray,
    delta_f: float,
    f_low: float = 15.0,
    f_high: float = 990.0,
) -> float:
    """Compute frequency-domain mismatch using PyCBC match.

    Parameters
    ----------
    h_pred : ndarray
        Candidate complex FD strain array (starting at f=0).
    h_ref : ndarray
        Reference complex FD strain array (starting at f=0).
    delta_f : float
        Frequency resolution in Hz.
    f_low : float
        Low-frequency cutoff for the match integral.
    f_high : float
        High-frequency cutoff for the match integral.

    Returns
    -------
    float
        Mismatch in [0, 1].
    """
    from pycbc.filter import match
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import FrequencySeries

    n = max(len(h_pred), len(h_ref))
    hp = np.zeros(n, dtype=np.complex128)
    hr = np.zeros(n, dtype=np.complex128)
    hp[: len(h_pred)] = h_pred
    hr[: len(h_ref)] = h_ref

    hp_series = FrequencySeries(hp, delta_f=delta_f)
    hr_series = FrequencySeries(hr, delta_f=delta_f)

    psd = aLIGOZeroDetHighPower(n, delta_f, f_low)

    m, _ = match(
        hr_series,
        hp_series,
        psd=psd,
        low_frequency_cutoff=f_low,
        high_frequency_cutoff=f_high,
    )
    return float(1.0 - m)


class NewPhysicsBench(Benchmark):
    """Evaluate RG-tail inspiral waveform implementations.

    Input:  Physics formulas (source packet with arXiv:2602.08833 ingredients)
    Output: h_of_f(f, Mc, eta, dL, ...) implementation

    Loss: mean PyCBC frequency-domain mismatch over 144 test cases.
    """

    name = "new_physics"

    def __init__(self, config_path=None):
        super().__init__(config_path)
        self._f_low = self.config.get("f_low_hz", 15.0)
        self._f_high = self.config.get("f_high_hz", 990.0)
        self._delta_f = self.config.get("delta_f", 0.125)
        self._cases = make_test_cases(self.config)

    @property
    def test_cases(self) -> list[dict]:
        return self._cases

    def frequency_array(self) -> np.ndarray:
        """Return the frequency array used for evaluation."""
        return np.arange(0.0, self._f_high + self._delta_f, self._delta_f)

    def generate_reference_waveforms(self) -> list[np.ndarray]:
        """Generate reference waveforms for all test cases."""
        from gwbenchmarks.rg_tail_reference import h_of_f

        f = self.frequency_array()
        refs = []
        for case in self._cases:
            h = h_of_f(
                f,
                Mc=case["Mc"],
                eta=case["eta"],
                dL=case["dL"],
                tc=case["tc"],
                phic=case["phic"],
                lambda_RG=case["lambda_RG"],
                f_low=self._f_low,
                fmax_over_fisco=case["fmax_over_fisco"],
                sigma_taper_over_fisco=case["sigma_taper_over_fisco"],
            )
            refs.append(h)
        return refs

    def score_h_of_f(self, candidate_h_of_f) -> tuple[float, Dict[str, float]]:
        """Score a candidate h_of_f callable against the reference.

        Parameters
        ----------
        candidate_h_of_f : callable
            Must have the same signature as the reference h_of_f.

        Returns
        -------
        loss : float
            Mean mismatch across all test cases.
        components : dict
            Per-case mismatches and summary statistics.
        """
        from gwbenchmarks.rg_tail_reference import h_of_f as ref_h_of_f

        f = self.frequency_array()
        mismatches = []
        per_case = {}

        for i, case in enumerate(self._cases):
            wf_kwargs = dict(
                Mc=case["Mc"],
                eta=case["eta"],
                dL=case["dL"],
                tc=case["tc"],
                phic=case["phic"],
                lambda_RG=case["lambda_RG"],
                f_low=self._f_low,
                fmax_over_fisco=case["fmax_over_fisco"],
                sigma_taper_over_fisco=case["sigma_taper_over_fisco"],
            )

            try:
                h_ref = ref_h_of_f(f, **wf_kwargs)
                h_cand = candidate_h_of_f(f, **wf_kwargs)

                if not (np.all(np.isfinite(h_ref)) and np.all(np.isfinite(h_cand))):
                    mm = 1.0
                else:
                    mm = fd_mismatch_from_fd_arrays(
                        h_cand, h_ref, self._delta_f, self._f_low, self._f_high
                    )
            except Exception:
                mm = 1.0

            mismatches.append(mm)
            case_id = (
                f"mc{case['Mc']:g}_eta{case['eta']:g}"
                f"_dl{case['dL']:g}_rg{case['lambda_RG']:g}"
            )
            per_case[f"mismatch_{case_id}"] = mm

        loss = float(np.mean(mismatches))
        components = {
            "mean_fd_mismatch": loss,
            "median_fd_mismatch": float(np.median(mismatches)),
            "max_fd_mismatch": float(np.max(mismatches)),
            "n_cases": len(mismatches),
            "n_failed": sum(1 for m in mismatches if m >= 1.0),
            **per_case,
        }
        return loss, components

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        """Compute loss from pre-evaluated waveform arrays.

        predictions and targets should each contain keys "case_0", "case_1", ...
        with complex FD strain arrays.
        """
        mismatches = []
        per_case = {}

        for i in range(len(self._cases)):
            key = f"case_{i}"
            h_pred = predictions.get(key)
            h_ref = targets.get(key)

            if h_pred is None or h_ref is None:
                mm = 1.0
            else:
                try:
                    mm = fd_mismatch_from_fd_arrays(
                        h_pred, h_ref, self._delta_f, self._f_low, self._f_high
                    )
                except Exception:
                    mm = 1.0

            mismatches.append(mm)
            case = self._cases[i]
            case_id = (
                f"mc{case['Mc']:g}_eta{case['eta']:g}"
                f"_dl{case['dL']:g}_rg{case['lambda_RG']:g}"
            )
            per_case[f"mismatch_{case_id}"] = mm

        loss = float(np.mean(mismatches))
        components = {
            "mean_fd_mismatch": loss,
            "median_fd_mismatch": float(np.median(mismatches)),
            "max_fd_mismatch": float(np.max(mismatches)),
            "n_cases": len(mismatches),
            "n_failed": sum(1 for m in mismatches if m >= 1.0),
            **per_case,
        }
        return loss, components
