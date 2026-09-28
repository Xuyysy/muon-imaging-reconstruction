"""支撑特征不变量：只使用事件位置/权重，不需要真实目标。"""
import unittest
import numpy as np
import pandas as pd
from diagnose_smoothing_support import observed_features, neighborhood_sum, rank_correlation


class SmoothingSupportTests(unittest.TestCase):
    def test_equal_weights_have_exact_event_neff(self):
        xy = np.zeros((8, 2))
        features = observed_features(xy, np.ones(8), 32)
        self.assertEqual(features['global_neff'], 8.)
        self.assertEqual(features['occupied_cell_neff_median'], 8.)
        self.assertEqual(features['split_half_l1'], 0.)

    def test_weight_scale_invariance(self):
        xy = np.array([[-20., -20.], [-20., -20.], [20., 20.], [20., 20.]])
        w = np.array([1., 2., 3., 4.])
        a, b = observed_features(xy, w, 16), observed_features(xy, w*17, 16)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=12)

    def test_local_sum_does_not_wrap(self):
        image = np.zeros((4, 4))
        image[0, 0] = 1.
        result = neighborhood_sum(image)
        self.assertEqual(result.sum(), 4.)
        self.assertEqual(result[-1, -1], 0.)

    def test_constant_correlation_is_missing(self):
        self.assertTrue(np.isnan(rank_correlation(pd.Series([1., 1.]), pd.Series([1., 2.]))))
        self.assertAlmostEqual(rank_correlation(pd.Series([1., 2., 3.]), pd.Series([3., 2., 1.])), -1.)


if __name__ == '__main__':
    unittest.main()
