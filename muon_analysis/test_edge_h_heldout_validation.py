"""H几何及冻结判据测试，不能依赖验证结果调整判据。"""
import unittest
import numpy as np
import pandas as pd
from run_edge_h_heldout_validation import h_mask,h_empty,frozen_verdict


class HeldoutHTests(unittest.TestCase):
    def test_boxes_and_open_regions(self):
        axis = np.array([-50.,-30.,0.,15.,60.,80.])
        truth = h_mask(axis,axis); empty = h_empty(axis,axis)
        self.assertFalse((truth & empty).any())
        self.assertTrue(truth[1,1])
        self.assertTrue(truth[4,2])
        self.assertTrue(empty[3,1])  # 下开口
        self.assertTrue(empty[3,3])  # 上开口
        self.assertFalse(truth[0,0])

    def test_middle_bridge(self):
        x=np.array([0.,15.,30.]); y=np.array([-30.,-10.,10.])
        self.assertTrue(h_mask(x,y)[:,1].all())
        self.assertFalse(h_empty(x,y)[:,1].any())

    def test_all_four_groups_required(self):
        frame=pd.DataFrame({'edge_only_minus_matched_edge_fractional_iou':[.01]*4,
                            'edge_only_minus_matched_edge_empty_mass_fraction':[-.001]*4,
                            'valid_matches':[5]*4})
        self.assertTrue(frozen_verdict(frame)[0]['mechanism_reproduced'])
        frame.loc[0,'edge_only_minus_matched_edge_fractional_iou']=-.01
        self.assertFalse(frozen_verdict(frame)[0]['mechanism_reproduced'])
        self.assertFalse(frozen_verdict(frame.iloc[1:])[0]['mechanism_reproduced'])

    def test_invalid_match_prevents_success(self):
        frame=pd.DataFrame({'edge_only_minus_matched_edge_fractional_iou':[.01]*4,
                            'edge_only_minus_matched_edge_empty_mass_fraction':[-.001]*4,
                            'valid_matches':[5,5,4,5]})
        self.assertFalse(frozen_verdict(frame)[0]['mechanism_reproduced'])


if __name__=='__main__':
    unittest.main()
