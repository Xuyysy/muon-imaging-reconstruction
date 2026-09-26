"""冻结 Q 门的零噪声、空格、非真值及输入约束测试。"""
import unittest

import numpy as np

from local_support_soft_z_gate import local_support_gate


class LocalSupportGateTests(unittest.TestCase):
    def test_supported_rescue_and_empty_cell(self):
        xy = np.array([[0., 0.], [1., 1.], [0.5, 0.5], [80., 80.]])
        w = np.ones(4)
        h = np.array([1., 1., 0., 0.])
        p = np.array([.8, .4, .6, .6])
        # 首个粗格的硬门有效数为 2，所以救回事件 q=2/(2+2)=0.5。
        np.testing.assert_allclose(local_support_gate(xy, w, h, p),
                                   [.8, .4, .3, 0.])

    def test_zero_noise_equals_hard(self):
        xy = np.array([[0., 0.], [10., 10.], [80., 80.]])
        h = np.array([1., 0., 0.])
        np.testing.assert_array_equal(local_support_gate(xy, np.ones(3), h, h), h)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            local_support_gate(np.zeros((2, 2)), np.ones(2),
                               np.ones(2), np.array([.5, 1.1]))


if __name__ == '__main__':
    unittest.main()
