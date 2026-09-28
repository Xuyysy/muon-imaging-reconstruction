"""形状空区与分数评价的最小回归测试，不运行Geant4。"""
import unittest
import numpy as np

from diagnose_split_confidence import SCENES
from run_adaptive_voxel_experiment import imaging_edges
from run_shape_smoothing_control import empty_region, fractional_membership, shape_scores


class ShapeSmoothingTests(unittest.TestCase):
    def setUp(self):
        self.cx = (imaging_edges(32)[0][:-1]+imaging_edges(32)[0][1:])/2

    def test_empty_regions_do_not_overlap_targets(self):
        for scene, (_, mask) in SCENES.items():
            empty = empty_region(scene, self.cx, self.cx)
            self.assertTrue(empty.any())
            self.assertFalse((empty & mask(self.cx, self.cx)).any())

    def test_ties_are_fractional_and_permutation_invariant(self):
        image = np.array([4., 2., 2., 2., 0., 0.])
        expected = np.array([1., 2/3, 2/3, 2/3, 0., 0.])
        np.testing.assert_allclose(fractional_membership(image, 3), expected)
        perm = [5, 1, 3, 0, 4, 2]
        np.testing.assert_allclose(fractional_membership(image[perm], 3), expected[perm])

    def test_ideal_shapes_have_zero_empty_occupancy_and_mass(self):
        for scene, (_, mask) in SCENES.items():
            truth = mask(self.cx, self.cx)
            scores = shape_scores(truth.astype(float), truth,
                                  empty_region(scene, self.cx, self.cx), self.cx, scene)
            self.assertEqual(scores['empty_mass_fraction'], 0.)
            self.assertEqual(scores['empty_fractional_occupancy'], 0.)
            if scene == 'double_block':
                self.assertEqual(scores['fractional_min_block_recall'], 1.)

    def test_empty_mass_uses_whole_image_denominator(self):
        scene = 'double_block'
        truth = SCENES[scene][1](self.cx, self.cx)
        empty = empty_region(scene, self.cx, self.cx)
        image = truth.astype(float)
        image[empty] = 2.
        image[0, 0] = 7.  # 目标足迹外权重也必须进入全图分母。
        metrics = shape_scores(image, truth, empty, self.cx, scene)
        self.assertAlmostEqual(metrics['empty_mass_fraction'], 2*empty.sum()/image.sum())
        scaled = shape_scores(image*9, truth, empty, self.cx, scene)
        self.assertAlmostEqual(metrics['empty_mass_fraction'], scaled['empty_mass_fraction'])


if __name__ == '__main__':
    unittest.main()
