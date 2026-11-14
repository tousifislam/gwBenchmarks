"""Public helpers for the Template Bank Bench (detection in Gaussian noise).

This module is everything an agent needs to build and self-score a submission:

* waveform generation on the benchmark frequency grid (IMRPhenomXHM, h_plus,
  edge-on, full higher-mode set), see ``make_full_reference_waveform`` and
  ``generate_normalized_waveforms``;
* the benchmark inner product ``<a, b> = sum(a * conj(b) * weights)``;
* the Gaussian-noise model for filter outputs and ``score_detection``, which
  implements the hidden scoring procedure exactly (only the targets, trial
  counts and random seeds differ in the hidden evaluation).

The file is deliberately self-contained (it imports nothing from
``gwbenchmarks``) so that ``llm_agents/make_template_bank_workspace.py`` can copy
it verbatim into an isolated agent workspace as ``template_bank_core.py``.
``gwbenchmarks.benchmarks.template_bank`` wraps it in the suite's
``Benchmark`` interface.

Dependencies: numpy, scipy, lalsuite (imported lazily, only for waveform
generation) and, optionally, numba (only accelerates the inner product).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy import stats as _scipy_stats

try:  # numba is optional: it only speeds up complex_overlap.
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without numba
    def njit(function):
        return function


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
class PublicData:
    """Public benchmark payload exposed to the agent."""

    frequencies_hz: np.ndarray
    amplitude_reference: np.ndarray
    weights: np.ndarray
    benchmark_params: np.ndarray


# Seed for the reference phases of the public evaluation split. Hidden targets
# use a different, evaluator-only seed.
PUBLIC_EVAL_PHASE_SEED = 1729


def public_data_dir() -> Path:
    """Default location of the public arrays.

    In an isolated agent workspace the arrays sit in ``data/`` next to this
    file. Inside the gwBenchmarks repository they live in
    ``datasets/template_bank/``.
    """

    here = Path(__file__).resolve().parent
    if (here / "data" / "f_amp.npy").exists():
        return here / "data"
    return here.parents[1] / "datasets" / "template_bank"


def load_public_data(data_dir: str | Path | None = None) -> PublicData:
    data_path = Path(data_dir) if data_dir is not None else public_data_dir()
    frequencies_hz = np.load(data_path / "f_amp.npy")
    amplitude_reference, weights = np.load(data_path / "Aref_weights.npy")
    benchmark_params = np.load(data_path / "bank_wf_params.npy")
    return PublicData(
        frequencies_hz=frequencies_hz,
        amplitude_reference=amplitude_reference,
        weights=weights,
        benchmark_params=benchmark_params,
    )


def parameter_dict_from_row(params: np.ndarray, phi_ref: float | None = None) -> dict:
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
    if row.shape == (5,):
        par_dict["phi_ref"] = float(row[4])
    else:
        par_dict["phi_ref"] = 0.0 if phi_ref is None else float(phi_ref)
    return par_dict


def append_phase_column(
    params_array: np.ndarray,
    seed: int,
    phase_min: float = 0.0,
    phase_max: float = 2.0 * np.pi,
) -> np.ndarray:
    """Append a deterministic phi_ref column to a 4D parameter array."""

    params = np.asarray(params_array, dtype=float)
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError(f"Expected shape (n, 4), got {params.shape}.")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(phase_min, phase_max, size=len(params))
    return np.column_stack([params, phases])


def compute_hplus_hcross(
    frequencies_hz: np.ndarray,
    par_dict: dict,
    approximant: str = "IMRPhenomXHM",
    harmonic_modes: Iterable[tuple[int, int]] | None = None,
    force_nnlo: bool = True,
) -> np.ndarray:
    """Generate a frequency-domain waveform using LALSimulation."""

    try:
        import lal
        import lalsimulation
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Waveform generation needs lalsuite: `pip install lalsuite` "
            "(or `pip install -e .[template_bank]`)."
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


# The source is edge-on (iota = pi/2) with only h_plus observed, where the
# higher harmonics are strongest. Rows of bank_wf_params*.npy are
# [m1, m2, s1z, s2z]; an optional fifth column is the reference phase phi_ref.


def make_full_reference_waveform(
    params: np.ndarray,
    public_data: PublicData,
    phi_ref: float | None = None,
) -> np.ndarray:
    """Return the public full-mode h_plus waveform normalized by A_ref."""

    return make_reference_waveform(
        params,
        public_data,
        phi_ref=phi_ref,
        harmonic_modes=FULL_HM_MODES,
    )


def make_reference_waveform(
    params: np.ndarray,
    public_data: PublicData,
    phi_ref: float | None = None,
    harmonic_modes: Iterable[tuple[int, int]] | None = FULL_HM_MODES,
) -> np.ndarray:
    """Return a normalized-by-A_ref h_plus waveform for the requested modes."""

    hplus_hcross = compute_hplus_hcross(
        public_data.frequencies_hz,
        parameter_dict_from_row(params, phi_ref=phi_ref),
        approximant="IMRPhenomXHM",
        harmonic_modes=harmonic_modes,
    )
    return hplus_hcross[0] / public_data.amplitude_reference


@njit
def complex_overlap(h1: np.ndarray, h2: np.ndarray, weights: np.ndarray) -> complex:
    """Return the phase-preserving benchmark inner product."""

    return np.sum(h1 * np.conj(h2) * weights, axis=-1)


@njit
def overlap(h1: np.ndarray, h2: np.ndarray, weights: np.ndarray) -> float:
    return np.abs(complex_overlap(h1, h2, weights))


def normalize_waveform(waveform: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norm = np.sqrt(overlap(waveform, waveform, weights))
    if norm == 0:
        raise ValueError("Encountered zero-norm waveform.")
    return waveform / norm


def generate_normalized_waveforms(
    params_array: np.ndarray,
    public_data: PublicData,
    phi_ref: float | None = None,
    harmonic_modes: Iterable[tuple[int, int]] | None = FULL_HM_MODES,
) -> np.ndarray:
    rows = []
    for params in np.asarray(params_array):
        waveform = make_reference_waveform(
            params,
            public_data,
            phi_ref=phi_ref,
            harmonic_modes=harmonic_modes,
        )
        rows.append(normalize_waveform(waveform, public_data.weights))
    return np.asarray(rows, dtype=np.complex128)


# Small helpers used by the starter bank (farthest-first physical templates).


def pairwise_overlap_matrix(waveforms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_waveforms = waveforms.shape[0]
    matrix = np.empty((n_waveforms, n_waveforms), dtype=np.float64)
    for i in range(n_waveforms):
        for j in range(n_waveforms):
            matrix[i, j] = float(overlap(waveforms[i], waveforms[j], weights))
    return matrix


def greedy_farthest_first_order(overlap_matrix: np.ndarray, seed_index: int = 0) -> list[int]:
    """Order items so each new item is least covered by the current set."""

    n_items = overlap_matrix.shape[0]
    if overlap_matrix.shape != (n_items, n_items):
        raise ValueError("Expected a square overlap matrix.")
    if n_items == 0:
        return []

    remaining = set(range(n_items))
    seed = int(seed_index) % n_items
    order = [seed]
    remaining.remove(seed)
    best_cover = overlap_matrix[:, seed].copy()

    while remaining:
        next_index = min(remaining, key=lambda idx: best_cover[idx])
        order.append(next_index)
        remaining.remove(next_index)
        best_cover = np.maximum(best_cover, overlap_matrix[:, next_index])
    return order


# ---------------------------------------------------------------------------
# Detection-in-noise scoring
# ---------------------------------------------------------------------------

FALSE_ALARM_PROBABILITY = 1.0e-4   # threshold is set at this noise-only exceedance rate
SNR_MIN = 5.0                      # signals have optimal SNR rho >= SNR_MIN ...
MAX_BANK_ROWS = 32                 # ... and the bank may have at most this many filter rows
HIDDEN_NOISE_TRIALS = 2_000_000    # noise-only trials in the hidden evaluation
HIDDEN_SIGNAL_REPEATS = 10         # signal trials per hidden target (1,000 targets)
STATISTIC_BATCH_SIZE = 50_000      # trials passed to detection_statistic per call


def bank_gram_matrix(bank_waveforms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return G[i, j] = <b_i, b_j> = sum(b_i * conj(b_j) * weights)."""

    bank = np.asarray(bank_waveforms, dtype=np.complex128)
    return (bank * weights[None, :]) @ bank.conj().T


def signal_overlaps(
    target_waveforms: np.ndarray,
    bank_waveforms: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return S[t, i] = <h_t, b_i> for unit-norm targets h_t."""

    targets = np.asarray(target_waveforms, dtype=np.complex128)
    bank = np.asarray(bank_waveforms, dtype=np.complex128)
    return (targets * weights[None, :]) @ bank.conj().T


def noise_overlap_sampler(gram: np.ndarray) -> np.ndarray:
    """Return a matrix L such that z = xi @ L.T has the noise-overlap covariance.

    Detector noise n is stationary Gaussian. On the benchmark frequency grid it
    is modelled as independent circular complex Gaussian values with
    E[|n_f|^2] = 2 / weights_f, so that for any rows b_i, b_j the overlaps
    z_i = <n, b_i> satisfy E[z_i conj(z_j)] = 2 <b_j, b_i> = 2 conj(G[i, j]).
    For a unit-norm row, Re z and Im z are independent unit normals and
    |z|^2 follows a chi-squared distribution with 2 degrees of freedom.
    With xi_k = (N(0,1) + i N(0,1)) / sqrt(2), z = xi @ L.T has
    E[z conj(z)^T] = L L^H = 2 conj(G).
    """

    covariance = 2.0 * np.conj(np.asarray(gram, dtype=np.complex128))
    covariance = 0.5 * (covariance + covariance.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors * np.sqrt(eigenvalues)[None, :]


def draw_noise_overlaps(sampler: np.ndarray, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    k = sampler.shape[1]
    xi = (rng.standard_normal((n_trials, k)) + 1j * rng.standard_normal((n_trials, k))) / np.sqrt(2.0)
    return xi @ sampler.T


def draw_snrs(n_signals: int, rng: np.random.Generator, snr_min: float = SNR_MIN) -> np.ndarray:
    """Draw optimal SNRs from p(rho) proportional to rho^-4 for rho >= snr_min.

    This is the SNR distribution of sources uniformly distributed in volume.
    """

    return snr_min * rng.uniform(size=n_signals) ** (-1.0 / 3.0)


def oracle_detection_fraction(
    snrs: np.ndarray, false_alarm_probability: float = FALSE_ALARM_PROBABILITY
) -> float:
    """Detection fraction of a single exactly known template (an upper bound).

    The known-template statistic |z|^2 is chi^2_2 in noise and non-central
    chi^2_2 with non-centrality rho^2 for a signal, so the fraction is exact.
    """

    threshold = -2.0 * np.log(false_alarm_probability)
    return float(np.mean(_scipy_stats.ncx2.sf(threshold, 2, np.asarray(snrs) ** 2)))


def validate_bank(bank_waveforms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    bank = np.asarray(bank_waveforms, dtype=np.complex128)
    if bank.ndim != 2 or bank.shape[1] != len(weights):
        raise ValueError(
            f"bank_waveforms must have shape (n_rows, {len(weights)}), got {bank.shape}."
        )
    if not 1 <= len(bank) <= MAX_BANK_ROWS:
        raise ValueError(f"The bank must have between 1 and {MAX_BANK_ROWS} rows, got {len(bank)}.")
    if not np.all(np.isfinite(bank)):
        raise ValueError("bank_waveforms contains non-finite values.")
    return np.asarray([normalize_waveform(row, weights) for row in bank])


def evaluate_statistic_on_trials(
    detection_statistic: Callable,
    complex_overlaps: np.ndarray,
    bank_waveforms: np.ndarray,
    indices: np.ndarray,
    batch_size: int = STATISTIC_BATCH_SIZE,
) -> np.ndarray:
    """Call detection_statistic in batches and validate its output."""

    values = np.empty(len(complex_overlaps), dtype=np.float64)
    for start in range(0, len(complex_overlaps), batch_size):
        batch = complex_overlaps[start:start + batch_size]
        out = np.asarray(
            detection_statistic(batch.copy(), bank_waveforms, indices), dtype=np.float64
        )
        if out.shape != (len(batch),):
            raise ValueError(
                f"detection_statistic must return shape ({len(batch)},), got {out.shape}."
            )
        if not np.all(np.isfinite(out)):
            raise ValueError("detection_statistic returned non-finite values.")
        values[start:start + len(batch)] = out
    return values


def score_detection(
    bank_waveforms: np.ndarray,
    target_waveforms: np.ndarray,
    weights: np.ndarray,
    detection_statistic: Callable,
    waveform_indices: np.ndarray | None = None,
    seed: int = 0,
    n_noise_trials: int = 200_000,
    n_signal_repeats: int = 4,
    false_alarm_probability: float = FALSE_ALARM_PROBABILITY,
    snr_min: float = SNR_MIN,
    batch_size: int = STATISTIC_BATCH_SIZE,
) -> dict:
    """Score a bank and detection statistic exactly as the hidden evaluator does.

    1. Draw ``n_noise_trials`` noise-only overlap vectors and, for every target
       waveform, ``n_signal_repeats`` signal-plus-noise overlap vectors
       z = rho * <h, b> + n with rho drawn from p(rho) ~ rho^-4, rho >= snr_min.
    2. Shuffle all trials together and evaluate ``detection_statistic`` on them.
    3. Set the threshold so that a fraction ``false_alarm_probability`` of the
       noise-only trials exceed it, and report the fraction of signal trials
       that exceed it.

    The hidden evaluator uses 2,000,000 noise trials, 10 repeats per hidden
    target, and private random seeds.
    """

    bank = validate_bank(bank_waveforms, weights)
    indices = (
        np.arange(len(bank), dtype=np.int64)
        if waveform_indices is None
        else np.asarray(waveform_indices, dtype=np.int64)
    )
    if indices.shape != (len(bank),):
        raise ValueError("waveform_indices must have shape (n_rows,).")
    targets = np.asarray([normalize_waveform(row, weights) for row in target_waveforms])
    rng = np.random.default_rng(seed)
    sampler = noise_overlap_sampler(bank_gram_matrix(bank, weights))

    snrs = draw_snrs(len(targets) * n_signal_repeats, rng, snr_min)
    signal = np.tile(signal_overlaps(targets, bank, weights), (n_signal_repeats, 1))
    signal = snrs[:, None] * signal + draw_noise_overlaps(sampler, len(signal), rng)
    noise = draw_noise_overlaps(sampler, n_noise_trials, rng)

    trials = np.concatenate([noise, signal])
    is_signal = np.r_[np.zeros(len(noise), bool), np.ones(len(signal), bool)]
    order = rng.permutation(len(trials))
    statistic = np.empty(len(trials))
    statistic[order] = evaluate_statistic_on_trials(
        detection_statistic, trials[order], bank, indices, batch_size
    )

    noise_stat = np.sort(statistic[~is_signal])
    n_false = int(np.floor(false_alarm_probability * len(noise_stat)))
    if n_false < 1:
        raise ValueError("Too few noise trials for the requested false-alarm probability.")
    threshold = noise_stat[-(n_false + 1)]
    detected = statistic[is_signal] > threshold
    fraction = float(np.mean(detected))
    oracle = oracle_detection_fraction(snrs, false_alarm_probability)
    return {
        "n_rows": int(len(bank)),
        "false_alarm_probability": float(false_alarm_probability),
        "realized_false_alarm_fraction": float(np.mean(noise_stat > threshold)),
        "snr_min": float(snr_min),
        "n_noise_trials": int(n_noise_trials),
        "n_signal_trials": int(len(signal)),
        "detection_fraction": fraction,
        "detection_fraction_stderr": float(np.sqrt(fraction * (1 - fraction) / len(signal))),
        "oracle_detection_fraction": oracle,
        "relative_to_oracle": fraction / oracle,
        "threshold": float(threshold),
    }


def write_submission(
    output_dir: str | Path,
    bank_waveforms: np.ndarray,
    summary: dict,
    waveform_indices: np.ndarray | None = None,
    train_indices: np.ndarray | None = None,
    eval_indices: np.ndarray | None = None,
) -> None:
    """Write every required output file except ``detection_statistic.py``."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "bank_waveforms.npy", np.asarray(bank_waveforms, dtype=np.complex128))
    if waveform_indices is not None:
        np.save(output_path / "waveform_indices.npy", np.asarray(waveform_indices, dtype=np.int64))
    if train_indices is not None:
        np.save(output_path / "train_indices.npy", np.asarray(train_indices, dtype=np.int64))
    if eval_indices is not None:
        np.save(output_path / "eval_indices.npy", np.asarray(eval_indices, dtype=np.int64))
    with open(output_path / "run_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
