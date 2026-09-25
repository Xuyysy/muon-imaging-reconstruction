"""软 Z 边界三组消融的逐事件代数与图像守恒测试。"""
import unittest

import numpy as np

from diagnose_soft_z_mechanism import decompose_gates
from run_soft_z_gate_experiment import xy_image


class SoftZMechanismTests(unittest.TestCase):
    def test_event_and_image_decomposition(self):
        hard = np.array([1., 0., 1., 0.])
        soft = np.array([.8, .4, .25, 0.])
        angle_weight = np.array([2., 3., 4., 5.])
        xy = np.array([[0., 0.], [5., 5.], [20., 0.], [0., 20.]])
        gates = decompose_gates(hard, soft)
        np.testing.assert_allclose(gates['suppress_only'], [.8, 0., .25, 0.])
        np.testing.assert_allclose(gates['rescue_only'], [0., .4, 0., 0.])
        for bins in (16, 32):
            full = xy_image(xy, angle_weight * gates['full_soft'], bins)
            suppress = xy_image(xy, angle_weight * gates['suppress_only'], bins)
            rescue = xy_image(xy, angle_weight * gates['rescue_only'], bins)
            np.testing.assert_allclose(full, suppress + rescue, atol=1e-12)

    def test_zero_noise_reproduces_hard(self):
        hard = np.array([1., 0., 1.])
        gates = decompose_gates(hard, hard.copy())
        np.testing.assert_array_equal(gates['hard'], gates['suppress_only'])
        np.testing.assert_array_equal(gates['hard'], gates['full_soft'])
        np.testing.assert_array_equal(gates['rescue_only'], np.zeros_like(hard))

    def test_rejects_invalid_weight(self):
        with self.assertRaises(ValueError):
            decompose_gates(np.array([1., 0.]), np.array([.5, 1.1]))


if __name__ == '__main__':
    unittest.main()
