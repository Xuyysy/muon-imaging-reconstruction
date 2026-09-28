"""物理支撑与局部尺度核的回归测试，不运行仿真。"""
import unittest
import numpy as np
from local_scale_reconstruction import physical_support, local_scale_image
from run_frame_z_gate_validation import smooth_hard_image


class LocalScaleTests(unittest.TestCase):
    def test_same_physical_query_and_weight_scale(self):
        xy = np.array([[0., 0.], [0., 0.], [25., 25.]])
        w = np.array([1., 2., 3.])
        query = np.array([[0., 0.], [25., 25.]])
        neff, guide = physical_support(xy, w, query)
        scaled, density = physical_support(xy, w*7, query.copy())
        np.testing.assert_allclose(neff, [9/5, 1.])
        np.testing.assert_allclose(scaled, neff)
        np.testing.assert_allclose(density, guide*7)

    def test_cancel_factors_reproduces_fixed_s(self):
        rng = np.random.default_rng(52)
        for bins in (16, 32):
            native = rng.random((bins, bins))
            image = native if bins == 32 else np.repeat(np.repeat(native/4, 2, axis=0), 2, axis=1)
            n, g = np.ones_like(native), np.ones_like(native)
            output, _ = local_scale_image(image, bins, n, g, support=False, edge=False)
            np.testing.assert_allclose(output, smooth_hard_image(image, bins), rtol=1e-12, atol=1e-12)

    def test_edge_impulse_conserves_and_does_not_wrap(self):
        image = np.zeros((32, 32)); image[0, 0] = 1.
        output, _ = local_scale_image(image, 32, np.ones((32, 32)), np.ones((32, 32)))
        self.assertAlmostEqual(output.sum(), 1.)
        self.assertEqual(output[-1, -1], 0.)
        self.assertGreaterEqual(output.min(), 0.)

    def test_zero_image_and_scaled_guide(self):
        zero = np.zeros((32, 32)); n = np.ones((32, 32)); g = np.ones_like(n)
        output, _ = local_scale_image(zero, 32, n, g)
        np.testing.assert_array_equal(output, zero)
        image = np.arange(1024.).reshape(32, 32)
        g = image/100+1
        a, _ = local_scale_image(image, 32, n, g)
        b, _ = local_scale_image(image*9, 32, n, g*9)
        np.testing.assert_allclose(b, a*9, rtol=1e-12, atol=1e-10)

    def test_strong_observed_contrast_reduces_cross_edge_transfer(self):
        image = np.zeros((32, 32)); image[15, 15] = 1.
        n = np.ones_like(image); g = np.ones_like(image); g[16:, :] = 1000.
        isotropic, _ = local_scale_image(image, 32, n, g, support=False, edge=False)
        edged, _ = local_scale_image(image, 32, n, g, support=False, edge=True)
        self.assertLess(edged[16:, :].sum(), isotropic[16:, :].sum())
        self.assertAlmostEqual(edged.sum(), 1.)

    def test_boundary_guide_uses_clipped_physical_area(self):
        neff, guide = physical_support(np.array([[90., 90.]]), np.array([2.]), np.array([[90., 90.]]))
        self.assertAlmostEqual(neff[0], 1.)
        self.assertAlmostEqual(guide[0], 2/(16.875**2))


if __name__ == '__main__':
    unittest.main()
