"""局部支撑事件数机制诊断的质量守恒与空回流测试。"""
import unittest

import numpy as np

from diagnose_local_support_count_mechanism import support_details
from local_support_soft_z_gate import local_support_gate


class CountMechanismTests(unittest.TestCase):
    def test_rescue_retention_matches_frozen_gate(self):
        xy = np.array([[0., 0.], [1., 1.], [.5, .5], [80., 80.]])
        w = np.ones(4)
        h = np.array([1., 1., 0., 0.])
        p = np.array([.8, .4, .6, .6])
        q = local_support_gate(xy, w, h, p)
        f = support_details(xy, w, h, p, q)
        self.assertAlmostEqual(f['rescue_mass'], 1.2)
        self.assertAlmostEqual(f['kept_rescue_mass'], .3)
        self.assertAlmostEqual(f['kept_rescue_fraction'], .25)
        self.assertAlmostEqual(f['rescue_mass_in_empty_fraction'], .5)

    def test_zero_rescue_is_missing_not_zero(self):
        xy = np.array([[0., 0.]])
        w = np.array([2.])
        h = np.array([1.])
        p = np.array([1.])
        q = local_support_gate(xy, w, h, p)
        f = support_details(xy, w, h, p, q)
        self.assertEqual(f['rescue_mass'], 0.)
        self.assertTrue(np.isnan(f['kept_rescue_fraction']))


if __name__ == '__main__':
    unittest.main()
