"""Checks that angle/geometry failure states form exclusive partitions."""
import unittest

import pandas as pd

from diagnose_selection_gate_transitions import classify_transitions, gate_masks


def events(theta, x, z=None):
    if z is None:
        z = [0.] * len(theta)
    return pd.DataFrame({'theta_mrad': theta, 'poca_x_mm': x,
                         'poca_y_mm': [0.] * len(theta), 'poca_z_mm': z})


class SelectionGateTransitionTests(unittest.TestCase):
    def test_exclusive_loss_and_gain_reasons(self):
        # First three events are lost for angle, geometry, both, respectively;
        # next three enter from those same baseline failure states.
        ideal = events([3., 3., 3., 1., 3., 1., 3.],
                       [0., 0., 0., 0., 1000., 1000., 0.])
        noisy = events([1., 3., 1., 3., 3., 3., 3.],
                       [0., 1000., 1000., 0., 0., 0., 0.])
        result = classify_transitions(ideal, noisy, 0., 100.)
        self.assertEqual((result['baseline_selected'], result['noisy_selected'],
                          result['kept'], result['lost'], result['gained']),
                         (4, 4, 1, 3, 3))
        for direction in ('lost', 'gained'):
            self.assertEqual([result[f'{direction}_{cause}'] for cause in
                              ('angle_only', 'geometry_only', 'both')], [1, 1, 1])
            self.assertEqual(result[f'{direction}_geometry_xy_only'], 2)
            self.assertEqual(result[f'{direction}_geometry_z_only'], 0)

    def test_z_and_xy_z_subcategories(self):
        ideal = events([3., 3.], [0., 0.])
        noisy = events([3., 3.], [0., 1000.], [1000., 1000.])
        result = classify_transitions(ideal, noisy, 0., 100.)
        self.assertEqual(result['lost_geometry_only'], 2)
        self.assertEqual(result['lost_geometry_z_only'], 1)
        self.assertEqual(result['lost_geometry_xy_and_z'], 1)

    def test_zero_error_control(self):
        ideal = events([1., 3.], [0., 0.])
        result = classify_transitions(ideal, ideal.copy(), 0., 100.)
        self.assertEqual(result['lost'], 0)
        self.assertEqual(result['gained'], 0)
        self.assertEqual(result['retained_fraction'], 1.)
        self.assertEqual(gate_masks(ideal, 0., 100.)['selected'].tolist(), [False, True])


if __name__ == '__main__':
    unittest.main()
