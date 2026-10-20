import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "studies"))

from run_adaptive_split_merge import (  # noqa: E402
    SCENARIOS, build_likelihood_groups, contrast_report_for_decision,
    EVENT_BUDGET, fit_covariance_model, simulate_empty_field, simulate_target, _wilson,
    truth_free_scenario_inputs,
)


class AdaptiveStudyDesignTests(unittest.TestCase):
    def test_scenarios_cover_requested_positive_and_negative_conditions(self):
        self.assertEqual(SCENARIOS["uniform_known_background"]["child_levels"], (1.0, 1.0))
        self.assertGreater(
            SCENARIOS["strong_contrast_calibrated_background"]["child_levels"][1]
            - SCENARIOS["strong_contrast_calibrated_background"]["child_levels"][0],
            0.5,
        )
        self.assertEqual(SCENARIOS["right_child_uncovered"]["counts"]["right"], 0)
        self.assertEqual(SCENARIOS["weak_contrast_large_hit_error"]["sigma_hit_mm"], 1.0)
        self.assertTrue(SCENARIOS["weak_contrast_large_hit_error"]["target_split"])
        self.assertFalse(SCENARIOS["weak_contrast_large_hit_error"]["effect_meets_design_minimum"])
        self.assertEqual(SCENARIOS["algebraic_background_contrast_degeneracy"]["background_mode"],
                         "algebraic_confounded")

    def test_zero_false_splits_have_zero_lower_confidence_bound(self):
        interval = _wilson(0, 200)
        self.assertEqual(interval["lower"], 0.0)
        self.assertGreater(interval["upper"], 0.0)

    def test_inference_design_input_does_not_expose_truth_fields(self):
        inputs = truth_free_scenario_inputs("strong_contrast_calibrated_background")
        self.assertNotIn("child_levels", inputs)
        self.assertNotIn("target_split", inputs)
        self.assertNotIn("truth_mask", inputs)

    def test_each_scenario_uses_the_declared_equal_event_budget(self):
        for scenario in SCENARIOS.values():
            self.assertEqual(sum(scenario["counts"].values()), EVENT_BUDGET)

    def test_null_calibration_can_override_the_measurement_error_stratum(self):
        low_error = simulate_target(
            "uniform_known_background", 202610777,
            child_levels_override=(1.0, 1.0), sigma_hit_mm_override=0.3,
        )
        high_error = simulate_target(
            "uniform_known_background", 202610777,
            child_levels_override=(1.0, 1.0), sigma_hit_mm_override=1.0,
        )
        self.assertFalse(np.array_equal(low_error["left"], high_error["left"]))

    def test_right_uncovered_and_algebraic_confounding_are_nonestimable(self):
        blind = SCENARIOS["right_child_uncovered"]
        blind_data = simulate_target("right_child_uncovered", seed=812)
        blind_groups = build_likelihood_groups(
            blind_data, sigma_hit_mm=blind["sigma_hit_mm"],
            mode=blind["background_mode"], model="fine",
        )
        self.assertFalse(contrast_report_for_decision(blind_groups, blind["background_mode"])["estimable"])

        ambiguous = SCENARIOS["algebraic_background_contrast_degeneracy"]
        ambiguous_data = simulate_target("algebraic_background_contrast_degeneracy", seed=913)
        ambiguous_groups = build_likelihood_groups(
            ambiguous_data, sigma_hit_mm=ambiguous["sigma_hit_mm"],
            mode=ambiguous["background_mode"], model="fine",
        )
        self.assertFalse(contrast_report_for_decision(
            ambiguous_groups, ambiguous["background_mode"]
        )["estimable"])

    def test_empty_field_observations_are_separate_and_add_information(self):
        scenario = SCENARIOS["strong_contrast_calibrated_background"]
        target = simulate_target("strong_contrast_calibrated_background", seed=101)
        empty = simulate_empty_field(scenario["sigma_hit_mm"], scenario["counts"], seed=102)
        without = build_likelihood_groups(
            target, sigma_hit_mm=scenario["sigma_hit_mm"],
            mode=scenario["background_mode"], model="fine",
        )
        with_empty = build_likelihood_groups(
            target, sigma_hit_mm=scenario["sigma_hit_mm"],
            mode=scenario["background_mode"], model="fine", empty_field_data=empty,
        )
        self.assertEqual(len(with_empty), len(without) + 3)
        info_without = contrast_report_for_decision(without, scenario["background_mode"])
        info_with = contrast_report_for_decision(with_empty, scenario["background_mode"])
        self.assertTrue(info_without["estimable"])
        self.assertTrue(info_with["estimable"])
        self.assertGreaterEqual(info_with["information"] + 1e-12, info_without["information"])

    def test_roundoff_limited_calibration_fit_is_not_reported_as_nonconvergence(self):
        scenario = SCENARIOS["strong_contrast_calibrated_background"]
        target_seed = 203610214
        target = simulate_target(
            "strong_contrast_calibrated_background", target_seed,
            child_levels_override=(1.0, 1.0),
        )
        empty = simulate_empty_field(
            scenario["sigma_hit_mm"], scenario["counts"],
            seed=target_seed + 500_000_000,
        )
        groups = build_likelihood_groups(
            target, sigma_hit_mm=scenario["sigma_hit_mm"],
            mode=scenario["background_mode"], model="coarse", empty_field_data=empty,
        )
        parameters, converged, iterations = fit_covariance_model(
            groups, initial=np.array([1.0, 1.0, 0.5])
        )
        self.assertTrue(converged)
        self.assertLess(iterations, 500)
        self.assertTrue(np.all(np.isfinite(parameters)))


if __name__ == "__main__":
    unittest.main()
