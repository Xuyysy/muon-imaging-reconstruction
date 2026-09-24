"""回退决策仅依赖可观测统计量，边界不随真值或评分变化。"""
import unittest

import numpy as np
import pandas as pd

from diagnose_refinement_fallback import choose_reconstruction, policy_table


class RefinementFallbackTests(unittest.TestCase):
    def test_threshold_and_noise_boundaries(self):
        self.assertEqual(choose_reconstruction(255, 0.0, 100.0), 'coarse16')
        self.assertEqual(choose_reconstruction(256, 0.0, 100.0), 'adaptive40')
        self.assertEqual(choose_reconstruction(500, 0.05, 100.0), 'coarse16')

    def test_policy_uses_matching_condition_only(self):
        data = pd.DataFrame([
            {'method': method, 'event_count': 5000, 'repeat': 1,
             'position_sigma_mm': 0.0, 'selected_events': 300,
             'iou': iou, 'global_error_mm': 1.0, 'auto_error_mm': 2.0}
            for method, iou in [('coarse16', .5), ('adaptive40', .7),
                                ('local_shrink', .9)]
        ])
        selected, _ = policy_table(data, 100.0)
        self.assertEqual(selected.iloc[0].chosen_method, 'adaptive40')
        self.assertAlmostEqual(selected.iloc[0].iou, .7)
        self.assertAlmostEqual(selected.iloc[0].iou_gain_vs_coarse, .2)

    def test_duplicate_pair_is_rejected(self):
        data = pd.DataFrame([
            {'method': 'coarse16', 'event_count': 1000, 'repeat': 1,
             'position_sigma_mm': 0.0, 'selected_events': 80,
             'iou': .5, 'global_error_mm': 1.0, 'auto_error_mm': 2.0}
        ])
        with self.assertRaises(ValueError):
            policy_table(pd.concat([data, data], ignore_index=True), 100.0)


if __name__ == '__main__':
    unittest.main()
