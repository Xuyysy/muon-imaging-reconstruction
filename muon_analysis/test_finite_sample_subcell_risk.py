"""解析枚举检验修正条件，不把方差无偏误称为风险改善。"""
import unittest
import numpy as np
from finite_sample_subcell_risk import repaired_reconstruct


class FiniteSampleTests(unittest.TestCase):
    def test_conditional_variance_exact(self):
        p=np.array([.7,.1,.1,.1])
        for weights in (np.array([1.,1.]),np.array([1.,2.])):
            total_variance,corrected_variance=0.,0.
            for first in range(4):
                for second in range(4):
                    child=np.array([first,second]); probability=p[first]*p[second]
                    moments=[np.bincount(child,weights=weights**power,minlength=4) for power in (1,2,3)]
                    images,d=repaired_reconstruct(*moments)
                    total_variance+=probability*np.square(images['raw']-p).sum()
                    corrected_variance+=probability*float(d['variance_corrected'])
            self.assertAlmostEqual(total_variance,corrected_variance)

    def test_single_event_fallback_and_all_same_cap(self):
        for count in (1,2,8):
            moments=[np.array([float(count),0,0,0])]*3
            images,d=repaired_reconstruct(*moments)
            np.testing.assert_allclose(images['guarded'],images['neff'])
            if count==1:
                self.assertFalse(bool(d['correction_valid']))
                np.testing.assert_allclose(images['corrected'],images['neff'])
            else:
                self.assertEqual(float(d['corrected_alpha']),1.)

    def test_empty_and_equal_factor(self):
        images,d=repaired_reconstruct(np.zeros(4),np.zeros(4),np.zeros(4))
        np.testing.assert_allclose(images['guarded'],np.full(4,.25))
        _,d=repaired_reconstruct([1,1,0,0],[1,1,0,0],[1,1,0,0])
        self.assertAlmostEqual(float(d['correction_factor']),2.)

    def test_scale_and_permutation(self):
        w=np.array([3.,4.,0,0]);w2=np.array([5.,16.,0,0]);w3=np.array([9.,64.,0,0])
        images,d=repaired_reconstruct(w,w2,w3)
        perm=np.array([2,0,3,1])
        changed,dc=repaired_reconstruct(w[perm]*1e4,w2[perm]*1e8,w3[perm]*1e12)
        for method in images:
            np.testing.assert_allclose(images[method][perm],changed[method])
        for name in ('s2','s3','correction_factor','guarded_alpha'):
            np.testing.assert_allclose(d[name],dc[name])

    def test_bad_third_moment_rejected(self):
        with self.assertRaises(ValueError):
            repaired_reconstruct([1,0,0,0],[1,0,0,0],[0,0,0,0])

    def test_same_observation_opposite_risk_preferences(self):
        images,_=repaired_reconstruct([1,1,0,0],[1,1,0,0],[1,1,0,0])
        uniform=np.full(4,.25); boundary=np.array([.5,.5,0,0])
        risk=lambda m,p:float(np.square(images[m]-p).sum())
        self.assertLess(risk('guarded',uniform),risk('neff',uniform))
        self.assertGreater(risk('guarded',boundary),risk('neff',boundary))
        self.assertAlmostEqual(risk('guarded',boundary),.25)
        self.assertAlmostEqual(risk('neff',boundary),1./9.)


if __name__ == '__main__':
    unittest.main()
