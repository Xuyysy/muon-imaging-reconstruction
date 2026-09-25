"""简易 Z 连续门的边界与数值控制；不依赖 Geant4 大样本。"""
import unittest

import numpy as np

from run_adaptive_voxel_experiment import Z_LIMIT_MM
from run_simple_z_gate_baseline import simple_z_gate


class SimpleZGateTests(unittest.TestCase):
    def test_zero_noise_is_exact_hard_gate(self):
        z = np.array([-Z_LIMIT_MM - 1, -Z_LIMIT_MM, 0,
                      Z_LIMIT_MM, Z_LIMIT_MM + 1], float)
        np.testing.assert_array_equal(simple_z_gate(z, np.full(5, 2.), 0),
                                      (np.abs(z) <= Z_LIMIT_MM).astype(float))

    def test_boundary_half_and_symmetric(self):
        z = np.array([-Z_LIMIT_MM, Z_LIMIT_MM, 0.])
        result = simple_z_gate(z, np.full(3, 5.), .05)
        self.assertEqual(result[0], .5)
        self.assertEqual(result[1], .5)
        self.assertGreater(result[2], .5)

    def test_near_parallel_track_softens_margin(self):
        z = np.array([Z_LIMIT_MM - 10., Z_LIMIT_MM + 10.])
        low_angle = simple_z_gate(z, np.full(2, 2.), .05)
        high_angle = simple_z_gate(z, np.full(2, 10.), .05)
        self.assertLess(low_angle[0], high_angle[0])
        self.assertGreater(low_angle[1], high_angle[1])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            simple_z_gate(np.array([0.]), np.array([1., 2.]), .05)
        with self.assertRaises(ValueError):
            simple_z_gate(np.array([0.]), np.array([0.]), .05)
        with self.assertRaises(ValueError):
            simple_z_gate(np.array([0.]), np.array([2.]), -.01)


if __name__ == '__main__':
    unittest.main()
