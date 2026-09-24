"""Small synthetic checks for the development-only factorial ablation."""
import unittest

import numpy as np
import pandas as pd

from confidence_refinement import confidence_refined_volume, split_half_consistency
from run_split_gate_ablation import VARIANTS, ablated_volume, consistency_at_gate


class GateAblationTests(unittest.TestCase):
    def test_design_is_exact_two_by_two(self):
        self.assertEqual(set(VARIANTS), {'frozen4_shrink', 'gate2_shrink',
                                         'gate4_no_shrink', 'gate2_no_shrink'})

    def test_frozen_arm_matches_original_volume_and_gate(self):
        events = pd.DataFrame({'poca_x_mm': np.r_[np.ones(24), np.ones(8) * 20],
                               'poca_y_mm': np.ones(32),
                               'poca_z_mm': np.r_[np.zeros(16), np.ones(16) * 20],
                               'capped_weight': np.ones(32)})
        xy = events[['poca_x_mm', 'poca_y_mm']].to_numpy()
        weights = events.capped_weight.to_numpy()
        expected_score, expected_support = split_half_consistency(xy, weights)
        observed_score, observed_support = consistency_at_gate(xy, weights, 4.0)
        np.testing.assert_allclose(observed_score, expected_score)
        np.testing.assert_array_equal(observed_support, expected_support)
        original, _ = confidence_refined_volume(events, 0.05, 100.0)
        ablated, _ = ablated_volume(events, 0.05, 100.0, 4.0, True)
        np.testing.assert_allclose(ablated, original)

    def test_relaxing_gate_cannot_reduce_supported_cells(self):
        xy = np.column_stack((np.r_[np.ones(5), np.ones(7) * 20], np.ones(12)))
        weights = np.ones(12)
        _, support4 = consistency_at_gate(xy, weights, 4.0)
        _, support2 = consistency_at_gate(xy, weights, 2.0)
        self.assertTrue(np.all(support2 >= support4))


if __name__ == '__main__':
    unittest.main()
