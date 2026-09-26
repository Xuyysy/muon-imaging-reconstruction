"""回退判据及新目标几何的无数据轻量测试。"""
import unittest

import numpy as np

from run_noise_floor_fallback_diagnostic import choose_propagated
from run_plus_iron_fallback_validation import plus_mask, score_plus


class NoiseFloorFallbackTests(unittest.TestCase):
    def test_frozen_decision_at_existing_layer_spacing(self):
        # 同侧探测器间距为 100 mm；三个非零误差档位的选择已在协议中固定。
        self.assertTrue(choose_propagated(0., 100.))
        self.assertTrue(choose_propagated(.05, 100.))
        self.assertTrue(choose_propagated(.10, 100.))
        self.assertFalse(choose_propagated(.30, 100.))

    def test_invalid_calibration(self):
        with self.assertRaises(ValueError):
            choose_propagated(-.01, 100.)
        with self.assertRaises(ValueError):
            choose_propagated(.05, 0.)

    def test_plus_footprint_and_empty_corners(self):
        axis = np.array([-70., -50., -25., 0., 25., 50., 70.])
        mask = plus_mask(axis, axis)
        self.assertTrue(mask[3, 3])
        self.assertTrue(mask[1, 3])
        self.assertTrue(mask[3, 5])
        self.assertFalse(mask[1, 1])
        self.assertFalse(mask[0, 3])
        metric = score_plus(mask.astype(float), mask, axis, axis)
        self.assertEqual(metric['iou'], 1.)
        self.assertEqual(metric['corner_false_positive_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
