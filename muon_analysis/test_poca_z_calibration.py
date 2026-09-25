"""PoCA-Z 不确定度校准的配对、零噪声及风险分组单元测试。"""
import unittest

import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import evaluate_calibration, estimate_z_samples


class PocaZCalibrationTests(unittest.TestCase):
    def test_zero_noise_samples_are_identical(self):
        hits = pd.DataFrame({
            'up1_x_mm': [0.], 'up1_y_mm': [0.], 'up1_z_mm': [200.],
            'up2_x_mm': [0.], 'up2_y_mm': [0.], 'up2_z_mm': [100.],
            'down1_x_mm': [1.], 'down1_y_mm': [0.], 'down1_z_mm': [-100.],
            'down2_x_mm': [2.], 'down2_y_mm': [0.], 'down2_z_mm': [-200.],
        })
        samples = estimate_z_samples(hits, 0., seed=42, draws=4)
        self.assertEqual(samples.shape, (1, 4))
        np.testing.assert_array_equal(samples[:, 0], samples[:, 1])

    def test_coverage_exit_and_tertile_partition(self):
        reference = np.array([0., 1., 2., 3., 4., 5.])
        observed = np.array([0., 1., 2., 3., 4., 105.])
        samples = np.array([
            [-1., 1., -1., 1.], [0., 2., 0., 2.],
            [-1., 5., -1., 5.], [0., 6., 0., 6.],
            [-6., 14., -6., 14.], [80., 130., 80., 130.],
        ])
        result, strata = evaluate_calibration(reference, observed, samples)
        self.assertEqual(result['reference_selected'], 6)
        self.assertEqual(result['z_exit'], 1)
        self.assertEqual(sum(row['events'] for row in strata), 6)
        self.assertEqual([row['events'] for row in strata], [2, 2, 2])

    def test_rejects_unpaired_rows(self):
        with self.assertRaises(ValueError):
            evaluate_calibration(np.array([0., 1.]), np.array([0., 1.]),
                                 np.zeros((1, 4)))


if __name__ == '__main__':
    unittest.main()
