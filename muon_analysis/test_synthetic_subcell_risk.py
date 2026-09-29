"""检查候选实际公式及其退化；不将失败例子修成成功。"""
import unittest
import numpy as np
from synthetic_subcell_risk import reconstruct, weighted_population_target, squared_risk


class SyntheticRiskTests(unittest.TestCase):
    def test_empty_and_uniform(self):
        for w,q in ((np.zeros(4),np.zeros(4)), (np.ones(4),np.ones(4))):
            images,d=reconstruct(w,q)
            for image in images.values():
                np.testing.assert_allclose(image,np.full(4,.25))
            self.assertEqual(float(d['candidate_alpha']),0.)

    def test_single_event_failure_is_preserved(self):
        images,d=reconstruct([1,0,0,0],[1,0,0,0])
        self.assertEqual(float(d['variance_proxy']),0.)
        self.assertEqual(float(d['candidate_alpha']),1.)
        self.assertAlmostEqual(float(squared_risk(images['candidate'],np.full(4,.25))),.75)
        self.assertAlmostEqual(float(squared_risk(images['neff'],np.full(4,.25))),.03)

    def test_scale_and_permutation(self):
        w=np.array([3.,4.,5.,0.]); q=np.array([5.,10.,17.,0.])
        base,d=reconstruct(w,q)
        perm=np.array([2,0,3,1])
        changed,dc=reconstruct(w[perm]*1e6,q[perm]*1e12)
        for method in base:
            np.testing.assert_allclose(base[method][perm],changed[method])
        for key in d:
            np.testing.assert_allclose(d[key],dc[key])

    def test_proxy_matches_event_residuals(self):
        child=np.array([0,0,1,2]); weights=np.array([1.,2.,4.,3.])
        w=np.bincount(child,weights=weights,minlength=4)
        q=np.bincount(child,weights=weights**2,minlength=4)
        images,d=reconstruct(w,q)
        e=np.eye(4)[child]
        direct=((weights**2)*np.square(e-images['raw']).sum(axis=1)).sum()/weights.sum()**2
        self.assertAlmostEqual(float(d['variance_proxy']),float(direct))

    def test_correlated_target_is_weighted(self):
        q=weighted_population_target([.25]*4,[1,2,4,8])
        np.testing.assert_allclose(q,np.array([1,2,4,8])/15)

    def test_equal_two_event_variance_bias_exact(self):
        # 枚举均匀总体的全部16种两事件组合，核对理论偏小因子1/2。
        proxies, errors = [], []
        for first in range(4):
            for second in range(4):
                counts=np.bincount([first,second],minlength=4).astype(float)
                images,d=reconstruct(counts,counts)
                proxies.append(float(d['variance_proxy']))
                errors.append(float(squared_risk(images['raw'],np.full(4,.25))))
        self.assertAlmostEqual(float(np.mean(errors)),.375)
        self.assertAlmostEqual(float(np.mean(proxies)),.1875)

    def test_invalid_input(self):
        for w,q in (([1,0,0,0],[0,0,0,0]),([np.nan]*4,[1]*4),([-1]*4,[1]*4)):
            with self.assertRaises(ValueError):
                reconstruct(w,q)


if __name__ == '__main__':
    unittest.main()
