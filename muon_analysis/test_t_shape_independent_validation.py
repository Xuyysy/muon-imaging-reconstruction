"""Test the frozen validation decision before running the new T scene."""
import unittest

import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import imaging_edges
from run_t_shape_independent_validation import (
    assert_frozen_rule, preregistered_verdict, t_mask,
)


class TShapeValidationTests(unittest.TestCase):
    def test_frozen_hash_matches_protocol(self):
        assert_frozen_rule()

    def test_truth_has_bar_and_stem(self):
        edges = imaging_edges(32)
        cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
        truth = t_mask(cx, cy)
        x, y = np.meshgrid(cx, cy, indexing='ij')
        self.assertTrue(np.any(truth & (x < -30) & (y > 20)))
        self.assertTrue(np.any(truth & (x > 30) & (y > 20)))
        self.assertTrue(np.any(truth & (abs(x) < 15) & (y < -20)))
        self.assertFalse(np.any(truth & (x > 30) & (y < -20)))

    def test_verdict_requires_gain_wins_and_no_other_large_loss(self):
        def result(gains, wins):
            return pd.DataFrame({'method': ['split_confidence'] * 4,
                                 'event_count': [5000] * 4,
                                 'position_sigma_mm': [0.0, 0.05, 0.1, 0.3],
                                 'iou_gain_mean': gains,
                                 'iou_win_count': wins})
        self.assertTrue(preregistered_verdict(result([.021, 0, -.01, -.02],
                                                      [4, 0, 0, 0]))['criterion_met'])
        self.assertFalse(preregistered_verdict(result([.021, 0, -.021, 0],
                                                       [4, 0, 0, 0]))['criterion_met'])
        self.assertFalse(preregistered_verdict(result([.021, 0, 0, 0],
                                                       [3, 0, 0, 0]))['criterion_met'])


if __name__ == '__main__':
    unittest.main()
