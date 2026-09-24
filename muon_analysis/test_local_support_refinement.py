"""邻域提议不能代替本地事件，且收缩后每个 z 层权重不变。"""
import unittest

import numpy as np
import pandas as pd

from run_local_support_refinement import local_support_volumes


class LocalSupportRefinementTests(unittest.TestCase):
    def test_singleton_is_not_refined_even_with_neighbor_support(self):
        # 本地 (x=1,y=1,z=1) 只有一条；相邻粗格放入 3 条。
        events = pd.DataFrame({
            'poca_x_mm': [1.0, 12.0, 13.0, 14.0],
            'poca_y_mm': [1.0] * 4,
            'poca_z_mm': [1.0] * 4,
            'capped_weight': [4.0] * 4,
        })
        volumes, info = local_support_volumes(events, 0.0, 100.0)
        self.assertGreater(info['proposed_coarse_slices'], info['refined_coarse_slices'])
        for volume in volumes.values():
            self.assertTrue(np.allclose(volume[16:18, 16:18, 8], 1.0))
            self.assertAlmostEqual(volume.sum(), 16.0)

    def test_two_local_events_allow_partial_detail_without_z_leakage(self):
        events = pd.DataFrame({
            'poca_x_mm': [1.0, 7.0, 12.0, 13.0],
            'poca_y_mm': [1.0] * 4,
            'poca_z_mm': [1.0] * 4,
            'capped_weight': [4.0] * 4,
        })
        volumes, info = local_support_volumes(events, 0.0, 100.0)
        self.assertGreater(info['refined_coarse_slices'], 0)
        self.assertGreater(info['mean_shrink_alpha'], 0.0)
        self.assertLess(info['mean_shrink_alpha'], 1.0)
        self.assertFalse(np.allclose(volumes['local_hard'], volumes['local_shrink']))
        for volume in volumes.values():
            self.assertEqual(volume.shape, (32, 32, 16))
            self.assertAlmostEqual(volume.sum(), 16.0)
            self.assertAlmostEqual(volume[:, :, 8].sum(), 16.0)
            self.assertEqual(np.count_nonzero(volume.sum(axis=(0, 1))), 1)

    def test_empty_input_does_not_create_false_signal(self):
        empty = pd.DataFrame(columns=['poca_x_mm', 'poca_y_mm',
                                      'poca_z_mm', 'capped_weight'])
        volumes, info = local_support_volumes(empty, 0.10, 100.0)
        self.assertEqual(info['refined_coarse_slices'], 0)
        self.assertTrue(all(np.all(volume == 0) for volume in volumes.values()))


if __name__ == '__main__':
    unittest.main()
