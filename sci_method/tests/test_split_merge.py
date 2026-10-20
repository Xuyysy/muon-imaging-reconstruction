import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sci_method.identifiability import contrast_information  # noqa: E402
from sci_method.split_merge import (  # noqa: E402
    child_levels_from_parent_contrast,
    expected_two_sided_power,
    calibrate_lrt_threshold,
    decide_split,
    GaussianObservationGroup,
    grouped_gaussian_nll_and_gradient,
    fit_covariance_model,
)


class ContrastInformationTests(unittest.TestCase):
    def test_estimable_contrast_reports_full_joint_cramer_rao_variance(self):
        fisher = np.array([[10.0, 2.0, 0.0], [2.0, 8.0, 3.0], [0.0, 3.0, 5.0]])
        contrast = np.array([-1.0, 1.0, 0.0])

        result = contrast_information(fisher, contrast)

        expected_variance = contrast @ np.linalg.inv(fisher) @ contrast
        self.assertTrue(result["estimable"])
        self.assertAlmostEqual(result["variance"], expected_variance, places=12)
        self.assertAlmostEqual(result["information"], 1.0 / expected_variance, places=12)
        self.assertEqual(result["null_projection_norm"], 0.0)

    def test_rejects_nonestimable_contrast_even_when_pseudoinverse_is_finite(self):
        fisher = np.diag([4.0, 0.0])
        contrast = np.array([1.0, 1.0])
        finite_but_misleading = contrast @ np.linalg.pinv(fisher) @ contrast
        self.assertEqual(finite_but_misleading, 0.25)

        result = contrast_information(fisher, contrast)

        self.assertFalse(result["estimable"])
        self.assertTrue(np.isinf(result["variance"]))
        self.assertEqual(result["information"], 0.0)
        self.assertAlmostEqual(result["null_projection_norm"], 1.0)

    def test_joint_fisher_rejects_target_contrast_confounded_with_background(self):
        # Both columns encode the same local covariance direction.
        fisher = np.array([[5.0, 5.0], [5.0, 5.0]])
        result = contrast_information(fisher, np.array([1.0, 0.0]))
        self.assertFalse(result["estimable"])
        self.assertTrue(np.isinf(result["variance"]))

    def test_invalid_zero_or_nonfinite_contrasts_are_rejected(self):
        with self.assertRaises(ValueError):
            contrast_information(np.eye(2), np.zeros(2))
        with self.assertRaises(ValueError):
            contrast_information(np.eye(2), np.array([np.nan, 1.0]))


class SplitRuleTests(unittest.TestCase):
    def test_parent_and_signed_contrast_define_two_nonnegative_child_levels(self):
        self.assertEqual(child_levels_from_parent_contrast(1.0, 0.5), (0.75, 1.25))
        self.assertEqual(child_levels_from_parent_contrast(1.0, -0.5), (1.25, 0.75))
        with self.assertRaises(ValueError):
            child_levels_from_parent_contrast(0.2, 0.5)

    def test_design_power_is_bounded_and_increases_with_information(self):
        low = expected_two_sided_power(0.5, variance=4.0, alpha=0.05)
        high = expected_two_sided_power(0.5, variance=0.01, alpha=0.05)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertGreater(high, low)

    def test_threshold_uses_higher_empirical_null_quantile(self):
        threshold = calibrate_lrt_threshold(np.arange(20, dtype=float), alpha=0.25)
        self.assertEqual(threshold, 14.0)

    def test_split_requires_estimability_power_and_calibrated_evidence(self):
        report = {"estimable": True, "variance": 0.01, "information": 100.0}
        result = decide_split(report, lrt=10.0, threshold=5.0,
                              predicted_power=0.95, min_power=0.8)
        self.assertTrue(result["split"])
        self.assertFalse(decide_split({**report, "estimable": False}, 10, 5, 0.95)["split"])
        self.assertFalse(decide_split(report, 10, 5, 0.2)["split"])
        self.assertFalse(decide_split(report, 2, 5, 0.95)["split"])
        self.assertFalse(decide_split(report, 10, 5, 0.95, fit_converged=False)["split"])


class GroupedLikelihoodTests(unittest.TestCase):
    def test_grouped_sufficient_statistic_matches_event_level_likelihood(self):
        from sci_method.forward_model import gaussian_nll_and_gradient

        rng = np.random.default_rng(42)
        y = rng.multivariate_normal(np.zeros(4), np.eye(4), size=30)
        base = np.eye(4) * 0.8
        derivative = np.eye(4) * 0.2
        theta = np.array([0.7])
        group = GaussianObservationGroup("g", y, base, derivative[None, :, :])
        grouped = grouped_gaussian_nll_and_gradient([group], theta)
        event = gaussian_nll_and_gradient(
            y, np.repeat(base[None], len(y), axis=0),
            np.repeat(derivative[None, :, :, None], len(y), axis=0), theta
        )
        self.assertAlmostEqual(grouped[0], event[0], places=11)
        self.assertTrue(np.allclose(grouped[1], event[1], atol=1e-11))

    def test_projected_fit_recovers_single_covariance_scale(self):
        rng = np.random.default_rng(81)
        covariance = np.eye(4) * 1.2
        observations = rng.multivariate_normal(np.zeros(4), covariance, size=500)
        group = GaussianObservationGroup("g", observations, np.eye(4), np.eye(4)[None])
        fitted, converged, _ = fit_covariance_model([group], initial=np.array([0.1]))
        self.assertTrue(converged)
        self.assertAlmostEqual(fitted[0], 0.2, delta=0.12)


if __name__ == "__main__":
    unittest.main()
