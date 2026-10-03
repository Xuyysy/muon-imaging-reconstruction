"""检查不重叠事件空间审计的格计数与真值隔离。"""
import unittest

import numpy as np

from diagnose_disjoint_spatial_support import support_from_positions, truth_cell_type


class DisjointSpatialSupportTests(unittest.TestCase):
    def test_coarse_truth_partition(self):
        truth = np.zeros((32, 32), dtype=bool)
        truth[0, 0] = True                  # 混合边界
        truth[2:4, 2:4] = True              # 目标内部
        kinds = truth_cell_type(truth, 16)
        self.assertEqual(kinds[0, 0], 1)
        self.assertEqual(kinds[1, 1], 2)
        self.assertEqual(kinds[2, 2], 0)

    def test_two_halves_and_strict_support(self):
        types = np.zeros((16, 16), dtype=int)
        types[8, 8] = 1
        a = np.array([[1., 1.], [1., 1.], [20., 20.]])
        b = np.array([[1., 1.], [1., 1.], [70., 70.]])
        result = support_from_positions(a, b, 16, types)
        self.assertEqual(result["shared_1each"], 1)
        self.assertEqual(result["shared_2each"], 1)
        self.assertEqual(result["shared_1each_mixed_boundary"], 1)
        self.assertEqual(result["union_cells"], 3)
        self.assertEqual(result["union_mixed_boundary"], 1)
        self.assertEqual(sum(result[f"union_{name}"] for name in
                             ("background", "mixed_boundary", "target_interior")), 3)

    def test_no_union_is_missing_fraction_not_zero(self):
        empty = np.empty((0, 2))
        result = support_from_positions(empty, empty, 16,
                                        np.zeros((16, 16), dtype=int))
        self.assertEqual(result["shared_1each"], 0)
        self.assertIsNone(result["shared_1each_fraction_of_union"])


if __name__ == "__main__":
    unittest.main()
