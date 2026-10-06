"""空场负对照的关键分母与方向测试；不依赖 Geant4 数据文件。"""
import unittest

from run_empty_target_control import region_summary, sum_conditions


class EmptyControlTests(unittest.TestCase):
    def test_region_partition_and_denominator(self):
        row = {"union_cells": 8, "union_background": 3,
               "union_mixed_boundary": 2, "union_target_interior": 3,
               "shared_1each_background": 1,
               "shared_1each_mixed_boundary": 1,
               "shared_1each_target_interior": 2}
        got = region_summary(row)
        self.assertEqual(got["inside_union_cells"], 5)
        self.assertEqual(got["inside_shared_cells"], 3)
        self.assertAlmostEqual(got["inside_shared_per_occupied"], 3/5)
        self.assertAlmostEqual(got["outside_shared_per_occupied"], 1/3)

    def test_empty_denominator_is_missing_not_zero(self):
        row = {"union_cells": 0, "union_background": 0,
               "union_mixed_boundary": 0, "union_target_interior": 0,
               "shared_1each_background": 0,
               "shared_1each_mixed_boundary": 0,
               "shared_1each_target_interior": 0}
        got = region_summary(row)
        self.assertIsNone(got["inside_shared_per_occupied"])

    def test_pooled_ratio_uses_pooled_counts(self):
        base = {"a_selected": 1, "b_selected": 2,
                "inside_union_cells": 3, "inside_shared_cells": 1,
                "outside_union_cells": 4, "outside_shared_cells": 2,
                "shared_2each": 0}
        other = dict(base, inside_union_cells=1, inside_shared_cells=1)
        got = sum_conditions([base, other])
        self.assertEqual(got["inside_shared_per_occupied"], .5)
        self.assertEqual(got["conditions"], 2)


if __name__ == "__main__":
    unittest.main()
