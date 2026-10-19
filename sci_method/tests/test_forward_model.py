import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sci_method.forward_model import (  # noqa: E402
    BoxVoxel, ForwardModelError, detector_residual_covariance,
    residual_jacobian, residuals_from_hits, scattering_covariance_per_density,
    gaussian_nll_and_gradient,
)


class ForwardModelTests(unittest.TestCase):
    def setUp(self):
        self.z = np.array([-400.0, -300.0, 300.0, 400.0])
        self.sigma = 0.3

    def test_residual_map_matches_direct_line_extrapolation(self):
        hits = np.array([[1, 2, -400], [3, 4, -300], [8, 10, 300], [9, 11, 400]], dtype=float)
        y = residuals_from_hits(hits, self.z)
        self.assertTrue(np.allclose(y, [-7.0, -8.0, -6.0, -7.0]))
        self.assertEqual(residual_jacobian(self.z).shape, (4, 12))

    def test_detector_covariance_contains_shared_upstream_correlations(self):
        cov = detector_residual_covariance(self.z, self.sigma)
        self.assertAlmostEqual(cov[0, 1], cov[1, 0])
        self.assertNotEqual(cov[0, 1], 0.0)
        self.assertAlmostEqual(cov[0, 2], 0.0)
        self.assertTrue(np.allclose(cov, cov.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-10)

    def test_detector_covariance_matches_monte_carlo(self):
        rng = np.random.default_rng(20261019)
        j = residual_jacobian(self.z)
        errors = rng.normal(0.0, self.sigma, size=(120_000, 4, 2))
        flat = np.zeros((len(errors), 12))
        flat[:, 0::3] = errors[:, :, 0]
        flat[:, 1::3] = errors[:, :, 1]
        empirical = np.cov(flat @ j.T, rowvar=False, ddof=1)
        analytic = detector_residual_covariance(self.z, self.sigma)
        mask = np.abs(analytic) > 1e-12
        self.assertLess(np.max(np.abs(empirical[mask] - analytic[mask]) / np.abs(analytic[mask])), 0.02)
        self.assertTrue(np.all(analytic[~mask] == 0.0))

    def test_path_integral_has_cross_downstream_covariance_and_zero_miss(self):
        voxel = BoxVoxel(-10, 10, -10, 10, -50, 50)
        path = (0.0, 0.0, 0.0, 0.0, -100.0, 100.0)
        kernel = scattering_covariance_per_density(path, voxel, self.z, momentum_gev=4.0)
        self.assertGreater(kernel[0, 1], 0.0)
        self.assertTrue(np.allclose(kernel[:2, :2], kernel[2:, 2:]))
        self.assertTrue(np.allclose(kernel[:2, 2:], 0.0))
        self.assertTrue(np.allclose(kernel, kernel.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-12)
        miss = (30.0, 0.0, 0.0, 0.0, -100.0, 100.0)
        self.assertTrue(np.allclose(scattering_covariance_per_density(miss, voxel, self.z), 0.0))

    def test_zero_material_and_zero_measurement_noise_limits(self):
        det = detector_residual_covariance(self.z, self.sigma)
        y = np.array([[0.1, -0.1, 0.2, 0.1]])
        nll, grad = gaussian_nll_and_gradient(y, det[None, :, :], np.zeros((1, 4, 4, 1)), np.array([0.0]))
        self.assertTrue(np.isfinite(nll))
        self.assertTrue(np.all(np.isfinite(grad)))
        zero_hit_cov = detector_residual_covariance(self.z, 0.0)
        self.assertTrue(np.allclose(zero_hit_cov, 0.0))
        voxel = BoxVoxel(-10, 10, -10, 10, -50, 50)
        path = (0.0, 0.0, 0.0, 0.0, -100.0, 100.0)
        material_kernel = scattering_covariance_per_density(path, voxel, self.z)
        gaussian_nll_and_gradient(np.zeros((1, 4)), zero_hit_cov[None], material_kernel[None, :, :, None], np.array([1e-3]))
        with self.assertRaises(ForwardModelError):
            gaussian_nll_and_gradient(np.zeros((1, 4)), zero_hit_cov[None], material_kernel[None, :, :, None], np.array([0.0]))
        with self.assertRaises(ForwardModelError):
            gaussian_nll_and_gradient(np.zeros((0, 4)), np.zeros((0, 4, 4)), np.zeros((0, 4, 4, 1)), np.array([0.0]))

    def test_likelihood_gradient_agrees_with_central_finite_difference(self):
        voxel = BoxVoxel(-10, 10, -10, 10, -50, 50)
        path = (0.0, 0.0, 0.0, 0.0, -100.0, 100.0)
        kernel = scattering_covariance_per_density(path, voxel, self.z)
        det = detector_residual_covariance(self.z, self.sigma)
        y = np.array([[0.03, -0.02, 0.01, 0.04], [-0.01, 0.02, 0.0, -0.03]])
        kernels = np.repeat(kernel[None, :, :, None], 2, axis=0)
        detector = np.repeat(det[None], 2, axis=0)
        theta = np.array([1e-4])
        nll, gradient = gaussian_nll_and_gradient(y, detector, kernels, theta)
        eps = 1e-8
        plus = gaussian_nll_and_gradient(y, detector, kernels, theta + eps)[0]
        minus = gaussian_nll_and_gradient(y, detector, kernels, theta - eps)[0]
        finite_difference = (plus - minus) / (2 * eps)
        self.assertTrue(np.isfinite(nll))
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertAlmostEqual(gradient[0], finite_difference, delta=max(1e-5, abs(finite_difference) * 2e-5))


if __name__ == "__main__":
    unittest.main()
