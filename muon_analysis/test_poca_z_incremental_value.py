"""检验 PoCA-Z 深度不确定度是否提供超出角度与边界位置的信息。"""
import unittest

import numpy as np

from evaluate_poca_z_incremental_value import ridge_predict, regression_metrics


class IncrementalValueTests(unittest.TestCase):
    def test_ridge_predicts_constant_training_target(self):
        x_train = np.array([[1., 2.], [2., 3.], [3., 4.]])
        prediction = ridge_predict(x_train, np.full(3, 5.),
                                   np.array([[4., 5.], [5., 6.]]))
        np.testing.assert_allclose(prediction, 5., atol=1e-10)

    def test_metrics(self):
        metrics = regression_metrics(np.array([1., 3.]), np.array([2., 2.]))
        self.assertAlmostEqual(metrics['mae'], 1.)
        self.assertAlmostEqual(metrics['rmse'], 1.)

    def test_rejects_nonfinite_input(self):
        with self.assertRaises(ValueError):
            ridge_predict(np.array([[np.nan]]), np.array([1.]),
                          np.array([[1.]]))


if __name__ == '__main__':
    unittest.main()
