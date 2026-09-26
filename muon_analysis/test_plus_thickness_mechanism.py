"""十字目标深度门机制诊断的观测侧特征核对。"""
import unittest

import numpy as np

from diagnose_plus_thickness_mechanism import gate_features


class GateFeaturesTests(unittest.TestCase):
    def test_counts_and_mass_identity(self):
        hard = np.array([1., 1., 0., 0.])
        soft = np.array([.5, 1., .25, 0.])
        weight = np.array([2., 3., 4., 5.])
        z = np.array([0., 50., 120., -150.])
        f = gate_features(hard, soft, weight, z, 10)
        self.assertEqual(f['candidate_count'], 4)
        self.assertEqual(f['hard_z_count'], 2)
        self.assertEqual(f['rescued_event_count'], 1)
        self.assertEqual(f['suppressed_event_count'], 1)
        self.assertEqual(f['hard_mass'], 5.)
        self.assertEqual(f['suppression_mass_loss'], 1.)
        self.assertEqual(f['rescue_mass'], 1.)
        self.assertEqual(f['soft_mass'], 5.)

    def test_rejects_unaligned_arrays(self):
        with self.assertRaises(ValueError):
            gate_features(np.ones(2), np.ones(3), np.ones(2), np.ones(2), 10)


if __name__ == '__main__':
    unittest.main()
