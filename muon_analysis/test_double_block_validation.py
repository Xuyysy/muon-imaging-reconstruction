"""Check the independent truth mask and two-component scoring contract."""
import unittest

import numpy as np

from run_adaptive_voxel_experiment import imaging_edges
from run_double_block_validation import double_block_mask, shape_metrics


class DoubleBlockValidationTests(unittest.TestCase):
    def setUp(self):
        edges = imaging_edges(32)
        self.cx, self.cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
        self.truth = double_block_mask(self.cx, self.cy)

    def test_two_blocks_have_empty_gap(self):
        x, y = np.meshgrid(self.cx, self.cy, indexing='ij')
        gap = (x > -15) & (x < 15) & (y >= -30) & (y <= 30)
        self.assertFalse(np.any(self.truth & gap))
        self.assertTrue(np.any(self.truth & (x < 0)))
        self.assertTrue(np.any(self.truth & (x > 0)))

    def test_perfect_prediction_has_full_recall_and_no_gap_signal(self):
        metrics = shape_metrics(self.truth.astype(float), self.truth,
                                self.cx, self.cy)
        self.assertEqual(metrics['min_block_recall'], 1.0)
        self.assertEqual(metrics['gap_false_positive_fraction'], 0.0)
        self.assertEqual(metrics['gap_mass_fraction'], 0.0)


if __name__ == '__main__':
    unittest.main()
