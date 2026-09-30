"""借用逻辑的质量、对称、无真值与边界检查。"""
import unittest
import numpy as np
from neighbor_borrowing import reconstruct_pair


class NeighborBorrowingTests(unittest.TestCase):
    def test_no_truth_or_new_weight(self):
        a=np.array([[1.,0,0,0],[0,1,0,0]])
        b=np.array([[8.,0,0,0],[0,0,8.,0]])
        methods,d=reconstruct_pair(a,b)
        for output_a,output_b in methods.values():
            np.testing.assert_allclose((output_a*1).sum(axis=1),1.)
            np.testing.assert_allclose((output_b*8).sum(axis=1),8.)
        self.assertTrue(np.all(d['borrowed_a']<=4.))
        self.assertTrue(np.all(d['borrowed_b']<=1.))
        self.assertTrue(np.all((d['gate']>=0)&(d['gate']<=1)))

    def test_swap_cells_and_children(self):
        a=np.array([[1.,1,0,0],[0,2,0,0]])
        b=np.array([[1.,3,1,3],[4,2,2,0]])
        methods,d=reconstruct_pair(a,b)
        swapped,sd=reconstruct_pair(b,a)
        for name in methods:
            np.testing.assert_allclose(methods[name][0],swapped[name][1])
            np.testing.assert_allclose(methods[name][1],swapped[name][0])
        perm=np.array([2,0,3,1])
        changed,cd=reconstruct_pair(a[:,perm],b[:,perm])
        for name in methods:
            for cell in (0,1):
                np.testing.assert_allclose(methods[name][cell][:,perm],changed[name][cell])
        np.testing.assert_allclose(d['gate'],cd['gate'])

    def test_identical_empirical_patterns_share(self):
        a=np.array([[1.,0,0,0]])
        b=np.array([[8.,0,0,0]])
        methods,d=reconstruct_pair(a,b)
        self.assertEqual(float(d['gate'][0]),1.)
        self.assertGreater(methods['gated'][0][0,0],methods['independent'][0][0,0])

    def test_invalid_counts(self):
        a=np.array([[1.,0,0,0]])
        for b in (np.zeros((1,4)),np.array([[-1.,2.,0,0]]),np.zeros((2,4))):
            with self.assertRaises(ValueError):
                reconstruct_pair(a,b)

    def test_full_pool_uses_same_budget_not_free_events(self):
        a=np.array([[1.,0,0,0]])
        b=np.array([[2.,3,1,2]])
        methods,_=reconstruct_pair(a,b)
        # 若B的八条事件真与A同分布，那么在A定向采集相同事件的
        # 后验数值公式相同；但现实里不能因此说八条是免费数据。
        expected=(1.+a+b)/(4.+a.sum()+b.sum())
        np.testing.assert_allclose(methods['pooled'][0],expected)

    def test_single_event_boundary_still_borrows_wrong_direction(self):
        a=np.array([[1.,0,0,0]])
        b=np.array([[0.,0,8.,0]])
        methods,d=reconstruct_pair(a,b)
        self.assertGreater(float(d['gate'][0]),0.)
        self.assertGreater(float(methods['gated'][0][0,2]),float(methods['independent'][0][0,2]))


if __name__=='__main__':
    unittest.main()
