"""方框几何掩膜及孔洞评价的轻量回归检查。"""
import unittest

import numpy as np

from run_frame_z_gate_validation import frame_mask, score_frame


class FrameValidationTests(unittest.TestCase):
    def test_footprint_matches_four_geant4_boxes(self):
        axis = np.array([-70., -50., -30., 0., 30., 50., 70.])
        mask = frame_mask(axis, axis)
        self.assertTrue(mask[1, 3])   # 左梁
        self.assertTrue(mask[3, 1])   # 下梁
        self.assertTrue(mask[5, 3])   # 右梁
        self.assertTrue(mask[3, 5])   # 上梁
        self.assertFalse(mask[3, 3])  # 中孔必须为空
        self.assertFalse(mask[0, 0])  # 外部空气

    def test_perfect_frame_has_no_hole_false_positive(self):
        axis = np.array([-50., -25., 0., 25., 50.])
        truth = frame_mask(axis, axis)
        metrics = score_frame(truth.astype(float), truth, axis, axis)
        self.assertEqual(metrics['iou'], 1.)
        self.assertEqual(metrics['hole_false_positive_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
