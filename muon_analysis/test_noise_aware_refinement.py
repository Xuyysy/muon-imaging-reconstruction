"""新细分规则的最小物理/数值约束测试。"""
import unittest

import numpy as np
import pandas as pd

from run_noise_aware_refinement import noise_aware_volume


class NoiseAwareRefinementTests(unittest.TestCase):
    def test_single_event_stays_coarse_and_conserves_weight(self):
        events = pd.DataFrame({'poca_x_mm': [1.0], 'poca_y_mm': [1.0],
                               'poca_z_mm': [1.0], 'capped_weight': [4.0]})
        volume, selected, threshold = noise_aware_volume(events, 0.0, 100.0)
        self.assertEqual(selected, 0)
        self.assertEqual(threshold, 2.0)
        self.assertAlmostEqual(volume.sum(), 4.0)
        self.assertEqual(np.count_nonzero(volume), 8)

    def test_noise_can_stop_refinement_without_changing_total_weight(self):
        # 八条同权事件落在同一个粗体素的两个不同子体素中。
        events = pd.DataFrame({'poca_x_mm': [1.0] * 4 + [7.0] * 4,
                               'poca_y_mm': [1.0] * 8,
                               'poca_z_mm': [1.0] * 8,
                               'capped_weight': [4.0] * 8})
        ideal, ideal_selected, _ = noise_aware_volume(events, 0.0, 100.0)
        noisy, noisy_selected, _ = noise_aware_volume(events, 0.30, 100.0)
        self.assertEqual(ideal_selected, 1)
        self.assertEqual(noisy_selected, 0)
        self.assertAlmostEqual(ideal.sum(), 32.0)
        self.assertAlmostEqual(noisy.sum(), 32.0)
        self.assertEqual(np.count_nonzero(ideal), 2)
        self.assertEqual(np.count_nonzero(noisy), 8)

    def test_empty_events_remain_zero(self):
        events = pd.DataFrame(columns=['poca_x_mm', 'poca_y_mm',
                                       'poca_z_mm', 'capped_weight'])
        volume, selected, _ = noise_aware_volume(events, 0.10, 100.0)
        self.assertEqual(selected, 0)
        self.assertTrue(np.all(volume == 0))


if __name__ == '__main__':
    unittest.main()
