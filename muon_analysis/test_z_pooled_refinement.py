"""沿 z 汇总判据的守恒与空间范围测试。"""
import unittest

import numpy as np
import pandas as pd

from run_noise_aware_refinement import noise_aware_volume
from run_z_pooled_refinement import z_pooled_volume


class ZPooledRefinementTests(unittest.TestCase):
    def setUp(self):
        # 四条事件分别位于四个相邻粗 z 层；单层统计不足，四层共享后足够。
        self.events = pd.DataFrame({
            'poca_x_mm': [1.0] * 4, 'poca_y_mm': [1.0] * 4,
            'poca_z_mm': [-95.0, -82.0, -69.0, -56.0],
            'capped_weight': [4.0] * 4,
        })

    def test_pooled_selection_preserves_each_z_slice_weight(self):
        one, info_one = z_pooled_volume(self.events, 0.0, 100.0, 1)
        four, info_four = z_pooled_volume(self.events, 0.0, 100.0, 4)
        self.assertEqual(info_one['selected_regions'], 0)
        self.assertEqual(info_four['selected_regions'], 1)
        self.assertEqual(info_four['refined_coarse_slices'], 4)
        self.assertTrue(np.allclose(one.sum(axis=(0, 1)), four.sum(axis=(0, 1))))
        self.assertAlmostEqual(four.sum(), 16.0)
        self.assertEqual(one.shape, (32, 32, 16))

    def test_single_z_group_matches_prior_3d_xy_projection(self):
        pooled, _ = z_pooled_volume(self.events, 0.0, 100.0, 1)
        old, _, _ = noise_aware_volume(self.events, 0.0, 100.0)
        self.assertTrue(np.allclose(pooled.sum(axis=2), old.sum(axis=2)))

    def test_invalid_group_is_rejected(self):
        with self.assertRaises(ValueError):
            z_pooled_volume(self.events, 0.0, 100.0, 3)


if __name__ == '__main__':
    unittest.main()
