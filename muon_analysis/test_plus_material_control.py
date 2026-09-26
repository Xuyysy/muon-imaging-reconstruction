"""材料控制中的事件匹配限制与文件名控制。"""
import unittest

import pandas as pd

from run_plus_material_control import csv_for, upstream_exact_match_fraction


class PlusMaterialControlTests(unittest.TestCase):
    def test_upstream_match_requires_all_four_coordinates(self):
        al = pd.DataFrame({'event': [0, 1],
                           'up1_x_mm': [1., 1.], 'up1_y_mm': [2., 2.],
                           'up2_x_mm': [3., 3.], 'up2_y_mm': [4., 4.]})
        iron = al.copy()
        iron.loc[1, 'up2_y_mm'] = 5.
        matched, fraction = upstream_exact_match_fraction(al, iron)
        self.assertEqual(matched, 2)
        self.assertEqual(fraction, .5)

    def test_material_outputs_are_separate(self):
        self.assertNotEqual(csv_for('al', 'a'), csv_for('iron', 'a'))
        self.assertNotEqual(csv_for('al', 'a'), csv_for('al', 'b'))


if __name__ == '__main__':
    unittest.main()
