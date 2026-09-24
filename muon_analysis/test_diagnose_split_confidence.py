"""Check gate accounting without using the held-out T scene."""
import unittest

import numpy as np
import pandas as pd

from diagnose_split_confidence import SCENES, diagnose_events
from run_adaptive_voxel_experiment import imaging_edges
from run_baseline_experiment import u_target_mask


class SplitConfidenceDiagnosisTests(unittest.TestCase):
    def setUp(self):
        edges = imaging_edges(32)
        self.cx, self.cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
        self.truth = u_target_mask(self.cx, self.cy)

    def test_only_development_scenes_listed(self):
        self.assertEqual(set(SCENES), {'u_center', 'l_shape', 'double_block'})

    def test_single_event_cannot_pass_half_count_gate(self):
        events = pd.DataFrame({'poca_x_mm': [1.0], 'poca_y_mm': [1.0],
                               'poca_z_mm': [1.0], 'capped_weight': [4.0]})
        result = diagnose_events(events, 0.0, 100.0,
                                 self.truth, self.cx, self.cy)
        self.assertEqual(result['occupied_xy_cells'], 1)
        self.assertEqual(result['split_supported_xy_cells'], 0)
        self.assertEqual(result['positive_consistency_xy_cells'], 0)
        self.assertEqual(result['relative_image_l1'], 0.0)

    def test_repeated_pattern_passes_gate_and_changes_image(self):
        events = pd.DataFrame({'poca_x_mm': np.ones(24),
                               'poca_y_mm': np.ones(24),
                               'poca_z_mm': np.zeros(24),
                               'capped_weight': np.ones(24)})
        result = diagnose_events(events, 0.0, 100.0,
                                 self.truth, self.cx, self.cy)
        self.assertGreater(result['split_supported_xy_cells'], 0)
        self.assertGreater(result['positive_consistency_xy_cells'], 0)
        self.assertGreater(result['relative_image_l1'], 0)
        self.assertGreater(result['effective_detail_fraction'], 0)


if __name__ == '__main__':
    unittest.main()
