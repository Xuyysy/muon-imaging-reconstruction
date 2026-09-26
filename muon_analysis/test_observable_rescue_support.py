"""无真值空间支撑特征的确定性示例。"""
import unittest

import numpy as np

from diagnose_observable_rescue_support import spatial_support_features


class ObservableRescueSupportTests(unittest.TestCase):
    def test_supported_and_empty_rescue(self):
        xy = np.array([[0., 0.], [1., 1.], [80., 80.]])
        hard = np.array([2., 2., 0.])
        rescue = np.array([1., 1., 2.])
        f = spatial_support_features(xy, hard, rescue)
        self.assertAlmostEqual(f['supported_rescue_fraction'], .5)
        self.assertAlmostEqual(f['empty_rescue_fraction'], .5)
        self.assertEqual(f['supported_cells'], 1)
        self.assertEqual(f['rescue_mass_on_coarse_grid'], 4.)

    def test_zero_rescue_and_invalid_input(self):
        xy = np.array([[0., 0.]])
        f = spatial_support_features(xy, np.array([1.]), np.array([0.]))
        self.assertTrue(np.isnan(f['hard_rescue_cosine']))
        with self.assertRaises(ValueError):
            spatial_support_features(xy, np.array([1.]), np.array([-1.]))


if __name__ == '__main__':
    unittest.main()
