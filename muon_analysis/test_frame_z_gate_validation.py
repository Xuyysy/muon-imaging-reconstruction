"""方框几何掩膜及孔洞评价的轻量回归检查。"""
import unittest

import numpy as np

from run_frame_z_gate_validation import (
    frame_mask, score_frame, smooth_hard_image, test_smoothing_control,
)


class FrameValidationTests(unittest.TestCase):
    def test_smoothing_control_invariants(self):
        test_smoothing_control()

    def test_native16_edge_impulse_matches_fixed_kernel(self):
        # 原生16格一单位质量冲激，映射后的四像素各1/4。
        image = np.zeros((32, 32))
        image[:2, :2] = .25
        column = np.exp(-.5 * np.square(np.arange(16)/.5))
        column[3:] = 0.  # ceil(3*0.5)=2的支持半径。
        column /= column.sum()
        native = np.outer(column, column)
        expected = np.repeat(np.repeat(native/4, 2, axis=0), 2, axis=1)
        np.testing.assert_allclose(smooth_hard_image(image, 16), expected, atol=1e-14)

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
