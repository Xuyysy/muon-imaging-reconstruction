"""四层 CSV 观测量审计的最小数值与字段检查。"""
import unittest

import numpy as np
import pandas as pd

from audit_observable_event_schema import (
    ANGLE_COLUMNS, EXPECTED_COLUMNS, HIT_COLUMNS, angles_from_hits,
)


class ObservableSchemaTests(unittest.TestCase):
    def test_expected_groups_have_no_extra_measurement(self):
        self.assertEqual(len(EXPECTED_COLUMNS), 19)
        self.assertEqual(len(HIT_COLUMNS), 12)
        self.assertNotIn("momentum", " ".join(EXPECTED_COLUMNS))

    def test_angles_follow_four_hits(self):
        # 两条 z 向轨迹均斜率为零时三种角度均为零。
        row = {}
        for name, z in (("up1", -405.), ("up2", -305.),
                        ("down1", 295.), ("down2", 395.)):
            row[f"{name}_x_mm"] = 0.
            row[f"{name}_y_mm"] = 0.
            row[f"{name}_z_mm"] = z
        frame = pd.DataFrame([row])
        np.testing.assert_allclose(angles_from_hits(frame)[0], [0., 0., 0.])
        # 只有下游 x 位移 1 mm / 100 mm 时，偏角由两点唯一确定。
        frame.loc[0, "down2_x_mm"] = 1.
        observed = angles_from_hits(frame)[0]
        expected = np.arctan2(1., 100.) * 1000.
        self.assertAlmostEqual(observed[0], expected)
        self.assertAlmostEqual(observed[1], 0.)
        self.assertAlmostEqual(observed[2], expected)
        self.assertEqual(len(ANGLE_COLUMNS), 3)


if __name__ == "__main__":
    unittest.main()
