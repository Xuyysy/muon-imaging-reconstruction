"""归档核对不能误配或吞掉缺失记录；不运行Geant4。"""
import unittest
import numpy as np
import pandas as pd
from audit_refinement_checkpoint import sign_counts, aligned_archives, KEY


class CheckpointTests(unittest.TestCase):
    def test_sign_tolerance(self):
        self.assertEqual(sign_counts([1, -1, 0, 1e-13]),
                         {'positive': 1, 'negative': 1, 'zero': 2})

    def test_missing_gain_rejected(self):
        with self.assertRaises(ValueError):
            sign_counts([np.nan])

    def archive(self):
        rows = [{'scenario': 'toy', 'event_count': 1000, 'repeat': i,
                 'position_sigma_mm': .05, 'method': 'coarse',
                 'fractional_iou': .2, 'fractional_gain': 0.,
                 'gain_bound_min': 0., 'gain_bound_max': 0.,
                 'iou': .2, 'iou_gain': 0., 'selected_cells': 0.}
                for i in range(300)]
        return pd.DataFrame(rows)

    def test_alignment_not_row_order(self):
        frame = self.archive()
        aligned = aligned_archives(frame, frame.iloc[::-1], frame)
        self.assertEqual(len(aligned), 300)

    def test_duplicate_or_metric_change_rejected(self):
        frame = self.archive()
        duplicate = frame.copy()
        duplicate.loc[1, KEY] = duplicate.loc[0, KEY]
        with self.assertRaises(ValueError):
            aligned_archives(frame, duplicate, frame)
        changed = frame.copy()
        changed.loc[0, 'fractional_gain'] = .01
        with self.assertRaises(ValueError):
            aligned_archives(frame, changed, frame)


if __name__ == '__main__':
    unittest.main()
