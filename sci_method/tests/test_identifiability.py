import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sci_method.identifiability import (  # noqa: E402
    fisher_information, information_report, add_prior_precision,
)


class IdentifiabilityTests(unittest.TestCase):
    def test_unobserved_fine_region_yields_exact_null_contrast(self):
        kernels = np.zeros((8, 4, 4, 2))
        kernels[:, :, :, 0] = np.eye(4)[None]
        covariances = np.repeat(np.eye(4)[None], 8, axis=0)
        fisher = fisher_information(covariances, kernels)
        report = information_report(fisher)
        self.assertEqual(report["rank"], 1)
        self.assertLess(report["eigenvalues"][0], 1e-12)
        self.assertGreater(report["eigenvalues"][-1], 0.0)

    def test_prior_does_not_change_data_fisher(self):
        data_fisher = np.diag([2.0, 0.0])
        prior = np.diag([0.0, 5.0])
        regularized = add_prior_precision(data_fisher, prior)
        self.assertTrue(np.array_equal(data_fisher, np.diag([2.0, 0.0])))
        self.assertEqual(information_report(data_fisher)["rank"], 1)
        self.assertEqual(information_report(regularized)["rank"], 2)
        self.assertEqual(information_report(data_fisher)["eigenvalues"][0], 0.0)

    def test_contrast_information_is_profiled_against_common_nuisance(self):
        fisher = np.array([[4.0, 3.0], [3.0, 4.0]])
        report = information_report(fisher, nuisance_indices=(0,))
        self.assertAlmostEqual(report["profiled_information"][0, 0], 4.0 - 9.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
