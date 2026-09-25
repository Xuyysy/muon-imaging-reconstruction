"""软 Z 入选权重和固定 XY 投影的基本守恒测试。"""
import unittest

import numpy as np
import pandas as pd

from run_soft_z_gate_experiment import (
    candidate_events, xy_image, z_acceptance_weights,
)


class SoftZGateTests(unittest.TestCase):
    def test_hard_soft_zero_noise_identical(self):
        z = np.array([-101., -100., 0., 100., 101.])
        hard, soft = z_acceptance_weights(z, np.repeat(z[:, None], 4, axis=1))
        np.testing.assert_array_equal(hard, soft)
        np.testing.assert_array_equal(hard, [0., 1., 1., 1., 0.])

    def test_fractional_boundary_acceptance(self):
        hard, soft = z_acceptance_weights(
            np.array([110.]), np.array([[90., 95., 105., 115.]]))
        self.assertEqual(hard[0], 0.)
        self.assertEqual(soft[0], .5)

    def test_same_weight_in_both_grid_projections(self):
        points = np.array([[0., 0.], [20., -10.]])
        weights = np.array([2., 3.])
        for bins in (16, 32):
            image = xy_image(points, weights, bins)
            self.assertEqual(image.shape, (32, 32))
            self.assertAlmostEqual(image.sum(), 5.)

    def test_candidate_angle_and_xy_are_independent_of_z(self):
        data = pd.DataFrame({
            'theta_mrad': [4., 4., 1.],
            'poca_x_mm': [0., 91., 0.],
            'poca_y_mm': [0., 0., 0.],
            'poca_z_mm': [999., 0., 0.],
        })
        ids, weights = candidate_events(data, 0., 100.)
        np.testing.assert_array_equal(ids, [0])
        self.assertAlmostEqual(weights[0], 9.)


if __name__ == '__main__':
    unittest.main()
