"""Unit checks for paired counterfactual image construction."""
import unittest

import numpy as np
import pandas as pd

from decompose_selection_attribute_effect import counterfactual_images, score_images


class SelectionAttributeDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.edges = [np.array([-2., 0., 2.]), np.array([-2., 0., 2.])]
        self.ideal = pd.DataFrame({
            'poca_x_mm': [-1., 1., -1.], 'poca_y_mm': [-1., 1., 1.],
            'theta_mrad': [1., 2., 3.]})
        self.truth = np.array([[True, False], [False, True]])

    def test_zero_error_control_and_truth_is_posthoc(self):
        selected = self.ideal.iloc[[0, 1]]
        images, counts = counterfactual_images(
            self.ideal, self.ideal.copy(), selected, selected, 0., 100., self.edges)
        for image in images.values():
            np.testing.assert_array_equal(image, images['baseline'])
        self.assertEqual((counts['lost'], counts['gained']), (0, 0))
        score_a = score_images(images, self.truth)
        score_b = score_images(images, ~self.truth)
        self.assertEqual(score_a['full_raw_l1_over_baseline_mass'], 0.)
        self.assertEqual(score_a['selection_raw_l1_over_baseline_mass'], 0.)
        self.assertEqual(score_a['attribute_raw_l1_over_baseline_mass'], 0.)
        self.assertEqual(score_a['interaction_raw_l1_over_baseline_mass'], 0.)
        self.assertEqual(score_a['full_mass'], score_b['full_mass'])

    def test_two_factor_identity_and_membership_counts(self):
        noisy = self.ideal.copy()
        noisy.loc[0, 'poca_x_mm'] = 1.
        noisy.loc[0, 'theta_mrad'] = 2.
        noisy.loc[2, 'theta_mrad'] = 4.
        images, counts = counterfactual_images(
            self.ideal, noisy, self.ideal.iloc[[0, 1]], noisy.iloc[[1, 2]],
            0., 100., self.edges)
        self.assertEqual((counts['kept'], counts['lost'], counts['gained']), (1, 1, 1))
        np.testing.assert_array_equal(images['baseline'], [[1., 0.], [0., 4.]])
        np.testing.assert_array_equal(images['selection_only'], [[0., 9.], [0., 4.]])
        np.testing.assert_array_equal(images['attribute_only'], [[0., 0.], [4., 4.]])
        np.testing.assert_array_equal(images['full'], [[0., 9.], [0., 4.]])
        interaction = (images['full'] - images['selection_only']
                       - images['attribute_only'] + images['baseline'])
        np.testing.assert_array_equal(interaction, [[1., 0.], [-4., 0.]])
        self.assertGreater(score_images(images, self.truth)['interaction_raw_l1_over_baseline_mass'], 0.)


if __name__ == '__main__':
    unittest.main()
