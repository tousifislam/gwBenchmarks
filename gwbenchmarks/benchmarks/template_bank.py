"""Template Bank Bench: compact frequency-domain waveform template banks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from gwbenchmarks.benchmarks.base import Benchmark


FULL_HM_MODES = [
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (2, -2),
    (2, -1),
    (3, -3),
    (3, -2),
    (4, -4),
]

ORTHOGONAL_MODE_GROUPS = [
    [(2, 2), (2, -2)],
    [(3, 3), (3, -3)],
    [(4, 4), (4, -4)],
]

PUBLIC_EVAL_PHASE_SEED = 1729
HIDDEN_TEST_PHASE_SEED = 2718

DEFAULT_SOURCE_PARAMS = {
    "d_luminosity": 1.0,
    "f_ref": 50.0,
    "iota": np.pi / 2.0,
    "l1": 0.0,
    "l2": 0.0,
    "m1": 50.0,
    "m2": 3.4,
    "s1x": 0.0,
    "s1y": 0.0,
    "s1z": -0.24,
    "s2x": 0.0,
    "s2y": 0.0,
    "s2z": -0.2,
    "phi_ref": 0.0,
}


@dataclass(frozen=True)
class TemplateBankData:
    """Frequency grid, normalization, weights, and public parameter pool."""

    frequencies_hz: np.ndarray
    amplitude_reference: np.ndarray
    weights: np.ndarray
    public_params: np.ndarray


def load_template_bank_data(data_dir: str | Path) -> TemplateBankData:
    """Load public template-bank arrays from ``data_dir``."""

    path = Path(data_dir)
    frequencies_hz = np.load(path / "f_amp.npy")
    amplitude_reference, weights = np.load(path / "Aref_weights.npy")
    public_params = np.load(path / "bank_wf_params.npy")
    return TemplateBankData(
        frequencies_hz=frequencies_hz,
        amplitude_reference=amplitude_reference,
        weights=weights,
        public_params=public_params,
    )


def append_phase_column(
    params_array: np.ndarray,
    seed: int,
    phase_min: float = 0.0,
    phase_max: float = 2.0 * np.pi,
) -> np.ndarray:
    """Append a deterministic ``phi_ref`` column to a 4D parameter array."""

    params = np.asarray(params_array, dtype=float)
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError(f"Expected shape (n, 4), got {params.shape}.")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(phase_min, phase_max, size=len(params))
    return np.column_stack([params, phases])


def parameter_dict_from_row(params: np.ndarray, phi_ref: float | None = None) -> dict:
    """Convert one 4D or 5D template-bank row to LALSimulation parameters."""

    row = np.asarray(params, dtype=float)
    if row.shape not in {(4,), (5,)}:
        raise ValueError(
            f"Expected parameter row with shape (4,) or (5,), got {row.shape}."
        )
    par_dict = DEFAULT_SOURCE_PARAMS.copy()
    par_dict["m1"] = float(row[0])
    par_dict["m2"] = float(row[1])
    par_dict["s1z"] = float(row[2])
    par_dict["s2z"] = float(row[3])
    par_dict["phi_ref"] = float(row[4]) if row.shape == (5,) else float(phi_ref or 0.0)
    return par_dict


def compute_hplus_hcross(
    frequencies_hz: np.ndarray,
    par_dict: dict,
    approximant: str = "IMRPhenomXHM",
    harmonic_modes: Iterable[tuple[int, int]] | None = None,
    force_nnlo: bool = True,
) -> np.ndarray:
    """Generate frequency-domain plus/cross waveforms with LALSimulation."""

    try:
        import lal
        import lalsimulation
    except ImportError as exc:
        raise ImportError(
            "TemplateBankBench requires lalsuite. Install it with `pip install lalsuite`."
        ) from exc

    local = dict(par_dict)
    local["d_luminosity_meters"] = local["d_luminosity"] * 1e6 * lal.PC_SI
    local["m1_kg"] = local["m1"] * lal.MSUN_SI
    local["m2_kg"] = local["m2"] * lal.MSUN_SI

    lal_dict = lal.CreateDict()
    if force_nnlo:
        lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(
            lal_dict, 102
        )
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(lal_dict, local["l1"])
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(lal_dict, local["l2"])

    if harmonic_modes is not None:
        mode_array = lalsimulation.SimInspiralCreateModeArray()
        for l_value, m_value in harmonic_modes:
            lalsimulation.SimInspiralModeArrayActivateMode(
                mode_array, int(l_value), int(m_value)
            )
        lalsimulation.SimInspiralWaveformParamsInsertModeArray(lal_dict, mode_array)

    local["lal_dic"] = lal_dict
    local["approximant"] = lalsimulation.GetApproximantFromString(approximant)
    local["f"] = lal.CreateREAL8Sequence(len(frequencies_hz))
    local["f"].data = np.asarray(frequencies_hz, dtype=np.float64)

    arg_order = [
        "phi_ref",
        "m1_kg",
        "m2_kg",
        "s1x",
        "s1y",
        "s1z",
        "s2x",
        "s2y",
        "s2z",
        "f_ref",
        "d_luminosity_meters",
        "iota",
        "lal_dic",
        "approximant",
        "f",
    ]
    hplus, hcross = lalsimulation.SimInspiralChooseFDWaveformSequence(
        *[local[key] for key in arg_order]
    )
    return np.stack([hplus.data.data, hcross.data.data])


def overlap(h1: np.ndarray, h2: np.ndarray, weights: np.ndarray) -> float:
    """Weighted absolute complex overlap."""

    return float(np.abs(np.sum(h1 * np.conj(h2) * weights, axis=-1)))


def normalize_waveform(waveform: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norm = np.sqrt(overlap(waveform, waveform, weights))
    if norm == 0:
        raise ValueError("Encountered zero-norm waveform.")
    return waveform / norm


def make_full_reference_waveform(
    params: np.ndarray,
    data: TemplateBankData,
    phi_ref: float | None = None,
) -> np.ndarray:
    hplus_hcross = compute_hplus_hcross(
        data.frequencies_hz,
        parameter_dict_from_row(params, phi_ref=phi_ref),
        approximant="IMRPhenomXHM",
        harmonic_modes=FULL_HM_MODES,
    )
    return hplus_hcross[0] / data.amplitude_reference


def generate_normalized_waveforms(
    params_array: np.ndarray,
    data: TemplateBankData,
    phi_ref: float | None = None,
) -> np.ndarray:
    rows = []
    for params in np.asarray(params_array):
        waveform = make_full_reference_waveform(params, data, phi_ref=phi_ref)
        rows.append(normalize_waveform(waveform, data.weights))
    return np.asarray(rows, dtype=np.complex128)


def summarize_prefix_coverage(
    bank_waveforms: np.ndarray,
    validation_waveforms: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.97,
    target_coverage: float = 0.5,
) -> dict:
    """Compute coverage as the ordered template bank grows."""

    n_validation = validation_waveforms.shape[0]
    running_best = np.zeros(n_validation, dtype=np.float64)
    prefix_coverages = []
    prefix_mins = []
    overlap_evaluations = 0

    for waveform in bank_waveforms:
        current = np.asarray(
            [overlap(vwf, waveform, weights) for vwf in validation_waveforms],
            dtype=np.float64,
        )
        overlap_evaluations += n_validation
        running_best = np.maximum(running_best, current)
        prefix_coverages.append(float(np.mean(running_best >= threshold)))
        prefix_mins.append(float(np.min(running_best)))

    prefix_length_to_target = None
    for idx, coverage in enumerate(prefix_coverages, start=1):
        if prefix_length_to_target is None and coverage >= target_coverage:
            prefix_length_to_target = idx

    return {
        "threshold": float(threshold),
        "target_coverage": float(target_coverage),
        "coverage_fraction": float(np.mean(running_best >= threshold)),
        "min_best_overlap": float(np.min(running_best)),
        "median_best_overlap": float(np.median(running_best)),
        "prefix_coverages": prefix_coverages,
        "prefix_min_best_overlaps": prefix_mins,
        "prefix_length_to_target": prefix_length_to_target,
        "prefix_length_to_50pct": (
            prefix_length_to_target if target_coverage == 0.5 else None
        ),
        "overlap_evaluations": overlap_evaluations,
    }


class TemplateBankBench(Benchmark):
    """Evaluate ordered template-bank submissions.

    Predictions contain ``bank_params`` with shape ``(n_bank, 5)`` and rows
    ``[m1, m2, s1z, s2z, phi_ref]``. Targets contain 5D evaluation parameters.
    The loss is the prefix length required to reach the target coverage. If the
    submitted bank never reaches target coverage, the loss is ``n_bank + 1``.
    """

    name = "template_bank"

    def evaluate_bank(
        self,
        bank_params: np.ndarray,
        evaluation_params: np.ndarray,
        data: TemplateBankData,
    ) -> dict:
        threshold = float(self.config.get("threshold", 0.97))
        target_coverage = float(self.config.get("target_coverage", 0.5))
        bank_waveforms = generate_normalized_waveforms(bank_params, data)
        evaluation_waveforms = generate_normalized_waveforms(evaluation_params, data)
        return summarize_prefix_coverage(
            bank_waveforms,
            evaluation_waveforms,
            data.weights,
            threshold=threshold,
            target_coverage=target_coverage,
        )

    def compute_loss(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> tuple[float, Dict[str, float]]:
        data = TemplateBankData(
            frequencies_hz=targets["frequencies_hz"],
            amplitude_reference=targets["amplitude_reference"],
            weights=targets["weights"],
            public_params=targets.get("public_params", np.empty((0, 4))),
        )
        metrics = self.evaluate_bank(
            np.asarray(predictions["bank_params"]),
            np.asarray(targets["evaluation_params"]),
            data,
        )
        prefix_length = metrics["prefix_length_to_target"]
        loss = float(prefix_length if prefix_length is not None else len(predictions["bank_params"]) + 1)
        return loss, metrics
