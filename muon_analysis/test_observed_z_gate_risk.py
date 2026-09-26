"""观测侧特征不得读取目标真值；测试简单门权重算术。"""
import unittest

import numpy as np
import pandas as pd

from diagnose_observed_z_gate_risk import observed_features


class ObservedRiskTests(unittest.TestCase):
    def test_rescued_weight_fraction(self):
        observed = pd.DataFrame({'poca_z_mm': [0., 150.],
                                 'theta_mrad': [5., 5.]})
        # 第一条稳定入界；第二条名义越界，但 2 次抽样有 1 次回流。
        sampled_z = np.array([[0., 0.], [0., 150.]])
        feature = observed_features(observed, np.array([0, 1]),
                                    np.array([4., 4.]), sampled_z,
                                    sigma=.05, spacing=100., input_count=10)
        self.assertAlmostEqual(feature['rescue_mass_fraction'], 1 / 3)
        self.assertEqual(feature['suppression_mass_fraction'], 0.)
        self.assertEqual(feature['fractional_gate_fraction'], .5)
        self.assertEqual(feature['candidate_fraction'], .2)


if __name__ == '__main__':
    unittest.main()
