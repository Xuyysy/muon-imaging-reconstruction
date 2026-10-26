"""确定性审查测试：测试工具边界与独立解析期望，不训练/抽样。"""
from fractions import Fraction
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import audit_checks as audit


class StageZeroTests(unittest.TestCase):
    def test_moments_match_hand_derived_rational_values(self):
        moments = audit.exact_moments()
        self.assertEqual(moments[0], [Fraction(50)]*4)
        self.assertEqual(moments[1], list(map(Fraction, [-3750, -1250, 1250, 3750])))
        self.assertEqual(moments[2][0], Fraction(875000, 3))
        for row in moments:
            self.assertEqual(sum(x*v for x,v in zip(row, [-1,3,-3,1])), 0)

    def test_correlated_detector_block_is_not_diagonalized(self):
        block = audit.analyze()['detector_covariance_mm2']
        np.testing.assert_allclose(np.array(block)[:2,:2], [[7.74,8.82],[8.82,10.26]], atol=1e-13)

    def test_two_positive_fields_have_equal_covariance_but_different_depth_contrast(self):
        r = audit.analyze()
        self.assertAlmostEqual(r['depth_contrast_u0'], 0)
        self.assertAlmostEqual(r['depth_contrast_u1'], .1)
        self.assertTrue(r['vertical']['covariance_equal'])
        self.assertTrue(r['vertical']['kernel_matches_exact'])
        self.assertLess(r['vertical']['nll_abs_difference'], 1e-10)

    def test_full_crossing_tilt_does_not_remove_exact_null(self):
        self.assertTrue(audit.analyze()['tilted_full']['covariance_equal'])

    def test_clipping_breaks_null_but_retains_failed_effect_expectation(self):
        r = audit.analyze()['tilted_clipped']
        # 原来“总协方差>0.001”预期未达，保留原门槛及失败，不能放宽它。
        self.assertFalse(r['initial_clipping_threshold_met'])
        self.assertEqual(r['initial_clipping_threshold'], .001)
        self.assertGreater(r['covariance_abs_difference_mm2'], r['covariance_rounding_budget_mm2'])
        # 独立积分得材料行和比=(1500*750)/(28,000,000+500,000/3)=27/676。
        self.assertAlmostEqual(r['relative_material_covariance_change'], 27/676, places=12)

    def test_joint_fisher_flags_depth_contrast_unestimable(self):
        r = audit.analyze()['vertical']
        self.assertEqual(r['fisher_rank'], 3)
        self.assertFalse(r['depth_contrast_estimable'])
        self.assertTrue(r['total_density_estimable'])

    def test_independent_calibration_resolves_only_added_nuisance_not_intrinsic_null(self):
        r = audit.analyze()['nuisance']
        self.assertFalse(r['unknown_common_total_estimable'])
        self.assertTrue(r['calibrated_common_total_estimable'])
        self.assertFalse(r['calibrated_depth_contrast_estimable'])
        self.assertFalse(r['unknown_common_total_estimable_after_duplicating_object'])

    def test_prior_changes_posterior_precision_not_data_null(self):
        r = audit.analyze()['prior']
        self.assertAlmostEqual(r['data_curvature_null_direction'], 0, places=12)
        self.assertAlmostEqual(r['added_prior_curvature'], 20)

    def test_zero_failure_upper_bound_is_not_zero(self):
        self.assertAlmostEqual(audit.zero_failure_upper(1), .95)
        self.assertGreater(audit.zero_failure_upper(64), .04)
        self.assertLess(audit.zero_failure_upper(64), .05)

    def test_mcnemar_exact_power_has_independent_small_case(self):
        # 6 discordant outcomes all in one direction: two-sided p=2/64<.05.
        self.assertAlmostEqual(audit.paired_power(6, 1.0, 1.0), 1)
        self.assertAlmostEqual(audit.paired_power(5, 1.0, 1.0), 0)
        self.assertAlmostEqual(audit.paired_power(6, 1.0, 0.0), 2/64)

    def test_probability_validation_rejects_impossible_discordance(self):
        with self.assertRaises(ValueError):
            audit.paired_power(128, .01, .05)

    def test_existing_result_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = set(Path(tmp).iterdir())
            with self.assertRaises(FileExistsError):
                audit.run(Path(tmp))
            self.assertEqual(before, set(Path(tmp).iterdir()))

    def test_runner_rejects_kernel_scale_regression_and_preserves_failure(self):
        original = audit.path_kernels
        # 故障注入：共同核放大仍保留covariance等式，但应被独立解析核检查拒绝。
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fault'
            with patch.object(audit, 'path_kernels', side_effect=lambda *a,**k:100*original(*a,**k)):
                with self.assertRaises(AssertionError):
                    audit.run(out)
            self.assertEqual(json.loads((out/'run_state.json').read_text())['status'], 'failed')
            self.assertTrue((out/'failure.json').exists())
            self.assertTrue((out/'checks.json').exists())
            self.assertTrue((out/'evidence.json').exists())

    def test_runner_completion_status_is_derived_not_hardcoded(self):
        # 仅状态机夹具，不是新物理数值结果，也不改实际inputs/threshold。
        report = audit.analyze()
        report['tilted_clipped']['relative_covariance_change'] = .002
        report['tilted_clipped']['initial_clipping_threshold_met'] = True
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'status-fixture'
            with patch.object(audit, 'analyze', return_value=report):
                audit.run(out)
            state = json.loads((out/'run_state.json').read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['failed_diagnostics'], [])


if __name__ == '__main__':
    unittest.main()
