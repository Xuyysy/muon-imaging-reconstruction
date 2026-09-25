"""Small tests for paired POCA-Z movement and track parallelism diagnostics."""
import unittest

import numpy as np
import pandas as pd

from diagnose_poca_z_stability import analyze_pair, parallelism_denominator


class PocaZStabilityTests(unittest.TestCase):
    def setUp(self):
        self.hits = pd.DataFrame({
            'up1_x_mm': [0., 0.], 'up1_y_mm': [0., 0.], 'up1_z_mm': [2., 2.],
            'up2_x_mm': [0., 0.], 'up2_y_mm': [0., 0.], 'up2_z_mm': [1., 1.],
            'down1_x_mm': [0., 0.], 'down1_y_mm': [0., 0.],
            'down1_z_mm': [-1., -1.],
            'down2_x_mm': [0., 1.], 'down2_y_mm': [0., 0.],
            'down2_z_mm': [-2., -2.],
        })
        self.reference = pd.DataFrame({
            'poca_x_mm': [0., 0.], 'poca_y_mm': [0., 0.],
            'poca_z_mm': [95., 70.], 'theta_mrad': [2.5, 6.]})

    def test_parallelism_denominator_zero_for_parallel(self):
        denominator = parallelism_denominator(self.hits)
        self.assertAlmostEqual(denominator[0], 0.)
        self.assertGreater(denominator[1], .4)

    def test_z_exit_and_predeclared_strata(self):
        noisy = self.reference.copy()
        noisy.loc[0, 'poca_z_mm'] = 105.
        noisy.loc[1, 'poca_z_mm'] = 72.
        result, rows = analyze_pair(self.reference, noisy, self.hits, .05, 100.)
        self.assertEqual(result['baseline_selected'], 2)
        self.assertEqual(result['z_exit'], 1)
        self.assertEqual(result['z_exit_fraction'], .5)
        self.assertEqual(result['abs_dz_selected_median_mm'], 6.)
        self.assertEqual(len(rows), 6)
        by_feature = {name: [row for row in rows if row['feature'] == name]
                      for name in ('z_margin_mm', 'theta0_mrad')}
        self.assertEqual([r['events'] for r in by_feature['z_margin_mm']], [1, 0, 1])
        self.assertEqual([r['events'] for r in by_feature['theta0_mrad']], [1, 0, 1])

    def test_sigma_zero_control(self):
        result, _ = analyze_pair(self.reference, self.reference.copy(),
                                 self.hits, 0., 100.)
        self.assertEqual(result['z_exit'], 0)
        self.assertEqual(result['abs_dz_selected_p90_mm'], 0.)


if __name__ == '__main__':
    unittest.main()
