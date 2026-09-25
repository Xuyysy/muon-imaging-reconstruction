"""Synthetic checks for truth-only boundary scoring and same-row POCA pairing."""
import unittest

import numpy as np
import pandas as pd

from diagnose_boundary_poca_scale import boundary_metrics, displacement_metrics


class BoundaryPocaScaleTests(unittest.TestCase):
    def test_concentrated_truth_child_beats_uniform(self):
        truth = np.zeros((32, 32), dtype=bool)
        truth[16, 16] = True
        events = pd.DataFrame({'poca_x_mm': np.ones(8),
                               'poca_y_mm': np.ones(8),
                               'capped_weight': np.ones(8)})
        result = boundary_metrics(events, truth)
        self.assertEqual(result['boundary_cells_neff_ge_4'], 1)
        self.assertEqual(result['boundary_cells_neff_ge_8'], 1)
        self.assertGreater(result['boundary_l1_improvement_neff_ge_4'], 0)
        self.assertGreater(result['boundary_inside_enrichment_neff_ge_4'], 0)

    def test_no_events_has_no_invented_boundary_score(self):
        truth = np.zeros((32, 32), dtype=bool)
        truth[16, 16] = True
        events = pd.DataFrame({'poca_x_mm': [], 'poca_y_mm': [],
                               'capped_weight': []})
        result = boundary_metrics(events, truth)
        self.assertEqual(result['boundary_cells_neff_ge_4'], 0)
        self.assertTrue(np.isnan(result['boundary_l1_improvement_neff_ge_4']))

    def test_selected_pairing_uses_shared_row_index(self):
        ref = pd.DataFrame({'poca_x_mm': [0., 10., 20.],
                            'poca_y_mm': [0., 0., 0.],
                            'poca_z_mm': [0., 0., 0.]})
        noisy = ref.copy()
        noisy['poca_x_mm'] += [1., 2., 9.]
        result = displacement_metrics(ref, noisy, ref.iloc[[0, 2]],
                                      noisy.iloc[[1, 2]], 6.25)
        self.assertEqual(result['both_selected_pairs'], 1)
        self.assertEqual(result['selected_xy_median_mm'], 9.)
        self.assertEqual(result['selected_xy_ge_one_fine_cell_fraction'], 1.)


if __name__ == '__main__':
    unittest.main()
