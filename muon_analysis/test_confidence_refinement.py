"""Tests of split-half 2.5D refinement before any new-scene data exist."""
import unittest

import numpy as np
import pandas as pd

from confidence_refinement import confidence_refined_volume


def events(x, y=None, z=None, weight=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return pd.DataFrame({
        'poca_x_mm': x,
        'poca_y_mm': np.zeros(n) if y is None else y,
        'poca_z_mm': np.zeros(n) if z is None else z,
        'capped_weight': np.ones(n) if weight is None else weight,
    })


class ConfidenceRefinementTests(unittest.TestCase):
    def test_empty_input_is_zero(self):
        volume, info = confidence_refined_volume(events([]), 0.1, 100.0)
        self.assertEqual(volume.shape, (32, 32, 16))
        self.assertEqual(volume.sum(), 0.0)
        self.assertEqual(info['active_refined_xy_cells'], 0)

    def test_singleton_stays_coarse(self):
        volume, info = confidence_refined_volume(events([1.0]), 0.0, 100.0)
        self.assertEqual(info['active_refined_xy_cells'], 0)
        self.assertEqual(np.count_nonzero(volume), 4)
        self.assertAlmostEqual(volume.sum(), 1.0)

    def test_repeated_subcell_pattern_refines_and_conserves_each_z(self):
        # 24 events all favour the same subcell, spread over two coarse z bins.
        e = events(np.full(24, 1.0), z=np.r_[np.full(12, -6.0), np.full(12, 6.0)])
        volume, info = confidence_refined_volume(e, 0.0, 100.0)
        self.assertGreater(info['active_refined_xy_cells'], 0)
        self.assertGreater(info['mean_alpha_nonzero'], 0.0)
        self.assertAlmostEqual(volume.sum(), 24.0)
        self.assertEqual(np.count_nonzero(volume.sum(axis=(0, 1))), 2)
        np.testing.assert_allclose(sorted(volume.sum(axis=(0, 1))[volume.sum(axis=(0, 1)) > 0]),
                                   [12.0, 12.0])

    def test_noise_reduces_detail_not_total_weight(self):
        e = events(np.full(24, 1.0))
        quiet, qinfo = confidence_refined_volume(e, 0.0, 100.0)
        noisy, ninfo = confidence_refined_volume(e, 0.3, 100.0)
        self.assertGreater(qinfo['mean_alpha_nonzero'], ninfo['mean_alpha_nonzero'])
        self.assertAlmostEqual(quiet.sum(), noisy.sum())
        self.assertFalse(np.allclose(quiet, noisy))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            confidence_refined_volume(events([1.0]), -0.1, 100.0)
        with self.assertRaises(ValueError):
            confidence_refined_volume(events([1.0], weight=[-1.0]), 0.0, 100.0)


if __name__ == '__main__':
    unittest.main()
