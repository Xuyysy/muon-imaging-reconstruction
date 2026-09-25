"""Small reproducibility and truth-isolation checks for the new diagnostic."""
import unittest

import numpy as np
import pandas as pd

from diagnose_heldout_subcell_score import evaluate_cells, heldout_subcell_score


class HeldoutSubcellScoreTests(unittest.TestCase):
    def test_repeated_single_child_has_positive_score(self):
        xy = np.ones((24, 2))
        score, support = heldout_subcell_score(xy)
        self.assertTrue(np.isfinite(score).any())
        self.assertGreater(score[8, 8], 0)
        self.assertEqual(support[8, 8], 4)

    def test_one_event_is_unsupported(self):
        score, support = heldout_subcell_score(np.ones((1, 2)))
        self.assertTrue(np.isnan(score).all())
        self.assertEqual(int(support.sum()), 0)

    def test_score_does_not_depend_on_truth_mask(self):
        events = pd.DataFrame({'poca_x_mm': np.ones(24),
                               'poca_y_mm': np.ones(24),
                               'capped_weight': np.ones(24)})
        truth_a = np.zeros((32, 32), dtype=bool)
        truth_b = truth_a.copy()
        truth_a[16, 16] = True
        truth_b[0, 0] = True
        _, cells_a = evaluate_cells(events, truth_a)
        _, cells_b = evaluate_cells(events, truth_b)
        score_a = cells_a.loc[(cells_a.ix == 8) & (cells_a.iy == 8),
                              'heldout_log_gain'].iloc[0]
        score_b = cells_b.loc[(cells_b.ix == 8) & (cells_b.iy == 8),
                              'heldout_log_gain'].iloc[0]
        self.assertEqual(score_a, score_b)


if __name__ == '__main__':
    unittest.main()
