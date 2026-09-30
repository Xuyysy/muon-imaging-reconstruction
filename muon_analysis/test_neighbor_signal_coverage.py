"""邻格证据审计的口径与缺失值检查。"""
import unittest

import numpy as np
import pandas as pd

from audit_neighbor_signal_coverage import best_neighbor, summarize_condition


class NeighborSignalCoverageTests(unittest.TestCase):
    def test_only_cardinal_neighbor_and_deterministic_tie(self):
        counts = {(2, 2): 1, (2, 3): 8, (3, 2): 8, (3, 3): 99}
        self.assertEqual(best_neighbor(counts, 2, 2, 1), (2, 3, 8))
        self.assertIsNone(best_neighbor({(2, 2): 1, (3, 3): 99}, 2, 2, 1))

    def test_neighbor_must_be_denser_and_at_least_four(self):
        self.assertIsNone(best_neighbor({(2, 2): 3, (2, 3): 3}, 2, 2, 3))
        self.assertEqual(best_neighbor({(2, 2): 3, (2, 3): 4}, 2, 2, 3),
                         (2, 3, 4))

    def test_missing_score_is_not_zero_score(self):
        cells = pd.DataFrame([
            {"ix": 2, "iy": 2, "cell_event_count": 1,
             "heldout_log_gain": np.nan},
            {"ix": 2, "iy": 3, "cell_event_count": 4,
             "heldout_log_gain": 0.0},
        ])
        result = summarize_condition(cells)
        self.assertEqual(result["sparse_cells"], 2)
        self.assertEqual(result["neighbor_supported"], 1)
        self.assertEqual(result["local_score_available"], 1)
        self.assertEqual(result["joint_coverage"], 0)

    def test_truth_fields_do_not_affect_audit(self):
        cells = pd.DataFrame([
            {"ix": 1, "iy": 1, "cell_event_count": 1,
             "heldout_log_gain": np.nan},
            {"ix": 1, "iy": 2, "cell_event_count": 5,
             "heldout_log_gain": 0.25},
        ])
        with_truth = cells.assign(target_child_count=[0, 4],
                                  single_cell_iou_gain=[-1.0, 1.0])
        self.assertEqual(summarize_condition(cells), summarize_condition(with_truth))


if __name__ == "__main__":
    unittest.main()
