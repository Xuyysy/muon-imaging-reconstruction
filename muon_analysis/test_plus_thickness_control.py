"""厚度控制的输入文件映射与随机流重现辅助函数。"""
import unittest

import numpy as np

from run_adaptive_voxel_experiment import RANDOM_SEED
from run_plus_thickness_control import advance_to_count_5000, csv_for


class PlusThicknessControlTests(unittest.TestCase):
    def test_input_files_distinct_and_old_file_preserved(self):
        self.assertTrue(str(csv_for('al', 100, 'a')).endswith('muon_tracks_plus_al_seed_a.csv'))
        paths = {csv_for(material, depth, stream)
                 for material in ('al', 'iron') for depth in (50, 100)
                 for stream in ('a', 'b')}
        self.assertEqual(len(paths), 8)
        with self.assertRaises(ValueError):
            csv_for('lead', 50, 'a')

    def test_rng_advance_matches_original_count_loops(self):
        total = 9971
        actual = np.random.default_rng(RANDOM_SEED + 8107)
        advance_to_count_5000(actual, total)
        expected = np.random.default_rng(RANDOM_SEED + 8107)
        for count in (1000, 3000):
            for _ in range(5):
                expected.choice(total, count, replace=False)
                for _ in range(8):
                    expected.normal(0, 1, count)
        np.testing.assert_array_equal(actual.choice(total, 5000, replace=False),
                                      expected.choice(total, 5000, replace=False))


if __name__ == '__main__':
    unittest.main()
