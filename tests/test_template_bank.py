"""Unit tests for the Template Bank Bench scoring (no lalsuite or data files needed).

Run from the repository root:  python -m unittest tests.test_template_bank
"""

import unittest

import numpy as np

from gwbenchmarks.benchmarks.template_bank import TemplateBankBench
from gwbenchmarks.benchmarks.template_bank_core import (
    FALSE_ALARM_PROBABILITY,
    bank_gram_matrix,
    draw_noise_overlaps,
    noise_overlap_sampler,
    oracle_detection_fraction,
    score_detection,
)


def max_abs2(complex_overlaps, bank_waveforms, indices=None):
    return np.max(np.abs(complex_overlaps) ** 2, axis=1)


class TemplateBankTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.weights = rng.uniform(0.5, 2.0, size=64)
        raw = rng.standard_normal((3, 64)) + 1j * rng.standard_normal((3, 64))
        norms = np.sqrt(np.sum(np.abs(raw) ** 2 * self.weights, axis=1))
        self.bank = raw / norms[:, None]

    def test_noise_overlaps_have_covariance_two_conj_gram(self):
        gram = bank_gram_matrix(self.bank, self.weights)
        z = draw_noise_overlaps(noise_overlap_sampler(gram), 400_000, np.random.default_rng(1))
        empirical = z.T @ z.conj() / len(z)
        np.testing.assert_allclose(empirical, 2.0 * np.conj(gram), atol=0.02)

    def test_exact_template_reaches_oracle(self):
        # With the target itself as the only filter, |z|^2 is the known-template statistic.
        metrics = score_detection(
            self.bank[:1], self.bank[:1], self.weights, max_abs2,
            seed=2, n_noise_trials=200_000, n_signal_repeats=4000,
        )
        self.assertAlmostEqual(metrics["detection_fraction"], metrics["oracle_detection_fraction"], delta=0.02)
        self.assertAlmostEqual(metrics["threshold"], -2 * np.log(FALSE_ALARM_PROBABILITY), delta=1.0)

    def test_oracle_is_monotonic_in_snr(self):
        self.assertLess(oracle_detection_fraction(np.full(10, 5.0)), oracle_detection_fraction(np.full(10, 8.0)))

    def test_bad_statistic_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must return shape"):
            score_detection(self.bank, self.bank, self.weights, lambda z, b, i: np.zeros(3),
                            n_noise_trials=20_000, n_signal_repeats=1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_detection(self.bank, self.bank, self.weights, lambda z, b, i: np.full(len(z), np.nan),
                            n_noise_trials=20_000, n_signal_repeats=1)

    def test_bank_size_limit(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 32"):
            score_detection(np.tile(self.bank, (11, 1)), self.bank, self.weights, max_abs2,
                            n_noise_trials=20_000, n_signal_repeats=1)

    def test_benchmark_loss_is_volume_loss(self):
        loss, parts = TemplateBankBench().compute_loss(
            {"bank_waveforms": self.bank, "detection_statistic": max_abs2},
            {"waveforms": self.bank, "weights": self.weights, "n_noise_trials": 50_000, "n_signal_repeats": 50},
        )
        self.assertAlmostEqual(loss, 1 - parts["detection_fraction"] / parts["oracle_detection_fraction"])
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
