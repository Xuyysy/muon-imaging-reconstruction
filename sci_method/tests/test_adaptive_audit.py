import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies"))

from adaptive_audit_analysis import (  # noqa: E402
    ablation_decisions, lrt_numerical_diagnostic,
)
from run_adaptive_audit_scenarios import LOW_BUDGET_SCENARIOS, NULL_STRATA  # noqa: E402


class AdaptiveAuditAnalysisTests(unittest.TestCase):
    def test_estimability_gate_rejects_near_zero_lrt_that_passes_its_empirical_cut(self):
        row = {
            "coarse_converged": "True", "fine_converged": "True",
            "lrt_train": "1e-10", "lrt_threshold": "6.9e-11",
            "contrast_estimable": "False", "predicted_power": "0.0",
            "n_left": "1600", "n_right": "1600", "n_other": "800",
            "background_mode": "algebraic_confounded", "empty_field_seed": "",
        }
        decisions = ablation_decisions(row)
        self.assertTrue(decisions["lrt_only"])
        self.assertFalse(decisions["estimability_plus_lrt"])
        self.assertFalse(decisions["full_composite"])

    def test_lrt_report_labels_scale_relative_to_objective_without_tuning_cutoff(self):
        diagnostic = lrt_numerical_diagnostic(1.7e-10, 11115.0)
        self.assertAlmostEqual(diagnostic["relative_to_objective_scale"], 1.7e-10 / 11115.0)
        self.assertGreater(diagnostic["machine_epsilon_units"], 0.0)
        self.assertFalse(diagnostic["threshold_changed"])

    def test_nested_gates_separate_estimability_power_and_observed_evidence(self):
        row = {
            "coarse_converged": "True", "fine_converged": "True",
            "lrt_train": "5.0", "lrt_threshold": "3.0",
            "contrast_estimable": "True", "predicted_power": "0.79",
            "n_left": "200", "n_right": "200", "n_other": "100",
            "background_mode": "known_fixed", "empty_field_seed": "",
        }
        decisions = ablation_decisions(row)
        self.assertTrue(decisions["lrt_only"])
        self.assertTrue(decisions["estimability_plus_lrt"])
        self.assertFalse(decisions["estimability_power_only"])
        self.assertFalse(decisions["estimability_lrt_power"])
        self.assertFalse(decisions["bic_only"])

    def test_low_budget_scenarios_are_small_predeclared_grid_not_a_scan(self):
        self.assertEqual(len(LOW_BUDGET_SCENARIOS), 6)
        self.assertEqual(NULL_STRATA, ((500, 0.3), (1000, 0.3), (500, 1.0), (1000, 1.0)))
        self.assertEqual({row["budget"] for row in LOW_BUDGET_SCENARIOS}, {500, 1000})
        self.assertEqual({row["delta"] for row in LOW_BUDGET_SCENARIOS}, {0.0, 0.5, 0.75})


if __name__ == "__main__":
    unittest.main()
