import unittest
import sys
import types

import numpy as np

# The current repository registry references two optional backends whose source
# modules are not shipped. Stub them so this benchmark-only test can import the
# package without exercising model integrations.
for module_name, class_name in (
    ("gwbenchmarks.models.gemini", "GeminiModel"),
    ("gwbenchmarks.models.opencode", "OpenCodeModel"),
):
    module = types.ModuleType(module_name)
    setattr(module, class_name, type(class_name, (), {}))
    sys.modules.setdefault(module_name, module)

from gwbenchmarks.benchmarks.template_bank import (
    match_reconstructed_waveform,
    summarize_prefix_reconstruction_coverage,
)


def project_from_overlaps(complex_overlaps, bank_waveforms, indices=None):
    return np.sum(complex_overlaps[:, None] * bank_waveforms, axis=0)


class TemplateBankTests(unittest.TestCase):
    def test_coverage_target_is_strict_but_match_threshold_is_inclusive(self):
        bank = np.eye(2, dtype=np.complex128)
        targets = np.eye(2, dtype=np.complex128)

        metrics = summarize_prefix_reconstruction_coverage(
            bank,
            targets,
            np.ones(2),
            reconstruct_from_overlaps=project_from_overlaps,
            threshold=1.0,
            target_coverage=0.5,
        )

        self.assertEqual(metrics["prefix_coverages"], [0.5, 1.0])
        self.assertEqual(metrics["prefix_length_to_target"], 2)
        self.assertEqual(metrics["coverage_operator"], ">")
        self.assertEqual(metrics["match_operator"], ">=")
        self.assertEqual(metrics["required_coverage_count"], 2)

    def test_evaluator_normalizes_reconstructed_waveform(self):
        weights = np.array([1.0, 2.0])
        target = np.array([1.0 + 1.0j, 2.0 - 0.5j])

        match = match_reconstructed_waveform(target, 17.0 * target, weights)

        self.assertAlmostEqual(match, 1.0)

    def test_scalar_reconstruction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must return one waveform"):
            match_reconstructed_waveform(
                np.array([1.0, 0.0]),
                np.asarray(1.0),
                np.ones(2),
            )


if __name__ == "__main__":
    unittest.main()
