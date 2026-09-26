"""冻结判据与未见文件名核对。"""
import unittest

import pandas as pd

from run_local_support_gate_validation import decide_success, source_csv


class FrozenValidationTests(unittest.TestCase):
    def test_files_are_separate(self):
        paths = {source_csv(material, stream) for material in ('al', 'iron')
                 for stream in ('a', 'b')}
        self.assertEqual(len(paths), 4)
        self.assertTrue(all('_mid_seed_' in str(path) for path in paths))

    def test_frozen_verdict_rejects_single_high_noise_failure(self):
        rows = []
        for sigma in (.05, .30):
            for material in ('al', 'iron'):
                for stream in ('a', 'b'):
                    rows.append({'event_count': 5000, 'position_sigma_mm': sigma,
                                 'material': material, 'stream': stream,
                                 'q_minus_h': .01, 'q_minus_p': 0.})
        summary = pd.DataFrame(rows)
        self.assertTrue(decide_success(summary)['overall_pass'])
        summary.loc[(summary.position_sigma_mm == .30)
                    & (summary.material == 'al')
                    & (summary.stream == 'a'), 'q_minus_p'] = -.001
        self.assertFalse(decide_success(summary)['overall_pass'])


if __name__ == '__main__':
    unittest.main()
