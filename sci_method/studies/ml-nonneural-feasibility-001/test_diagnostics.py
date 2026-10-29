"""Small mathematical and input-isolation checks; no neural/physical runs."""
import inspect
import math
import unittest
import numpy as np
from diagnostics import (chi2_sf_even, chi2_isf_even, pair_tail, map_mode,
                         propose, physical_modes, Q_pvalue, lr_weights,
                         oracle_indices)


class DiagnosticTests(unittest.TestCase):
    def test_df2_analytic_tail(self):
        self.assertAlmostEqual(chi2_sf_even(7, 2), math.exp(-3.5), places=14)

    def test_df4_analytic_tail(self):
        self.assertAlmostEqual(chi2_sf_even(7, 4), math.exp(-3.5)*4.5, places=14)

    def test_quantiles_round_trip(self):
        for n in (1, 2, 20, 40):
            for alpha in (.05, .025, .00625):
                self.assertAlmostEqual(chi2_sf_even(chi2_isf_even(alpha, 4*n),4*n),alpha,places=12)

    def test_tail_equal_weights(self):
        self.assertAlmostEqual(pair_tail(30,5,[.7,.7]),chi2_sf_even(30/.7,20),places=12)

    def test_pair_exponential_closed_form(self):
        a,b,q=.2,1.0,7.0
        expected=(b*math.exp(-q/(2*b))-a*math.exp(-q/(2*a)))/(b-a)
        self.assertAlmostEqual(pair_tail(q,1,[a,b]),expected,places=11)

    def test_psd_coupling_deterministic(self):
        c0,d,a,_=physical_modes(.3)
        self.assertGreater(np.linalg.eigvalsh(c0).min(),0)
        self.assertGreaterEqual(np.linalg.eigvalsh(d).min(),-1e-12)
        for u in (0,.5,1):
            self.assertTrue(np.all((1+u*a)/(1+a)<=1+1e-14))

    def test_covariance_correlations_retained(self):
        c0,_,_,_=physical_modes(.3)
        self.assertAlmostEqual(c0[0,1],8.82)

    def test_boundary_pvalue(self):
        _,_,a,_=physical_modes(.3)
        energy=np.array([4.,12.])*(1+a)
        self.assertAlmostEqual(Q_pvalue(energy,4,a),chi2_sf_even(16,16),places=12)

    def test_map_boundary_and_kkt(self):
        value,g=map_mode(np.zeros(2),10,np.array([.1,2.]))
        self.assertEqual(value,0)
        self.assertGreater(g,0)

    def test_map_interior_and_finite_difference(self):
        a=np.array([.1,2.]); t=20*(1+3*a)
        u,g=map_mode(t,10,a)
        self.assertGreater(u,0)
        self.assertLess(abs(g),1e-6)
        f=lambda v: np.sum(10*np.log1p(v*a)+.5*t/(1+v*a))+.125*v*v
        self.assertLess(abs((f(u+1e-4)-f(u-1e-4))/2e-4-g),1e-6)

    def test_map_global_candidate_check(self):
        a=np.array([.01,3.]); t=np.array([.1,100.])
        u,_=map_mode(t,4,a)
        f=lambda v: np.sum(4*np.log1p(v*a)+.5*t/(1+v*a))+.125*v*v
        self.assertLessEqual(f(u),min(f(x) for x in np.linspace(0,30,301))+1e-8)

    def test_proposer_cannot_receive_truth(self):
        self.assertEqual(tuple(inspect.signature(propose).parameters),('energy','counts','a'))
        a=np.array([.1,2.]); n=np.array([2,4,6,8,10,12,16,20])
        result=propose(2*n[:,None]*(1+a),n,a)
        self.assertEqual(set(result),{'fixed','map','score','fisher'})
        self.assertTrue(all(len(v)==2 for v in result.values()))

    def test_oracle_no_audit_random_input(self):
        self.assertEqual(tuple(inspect.signature(oracle_indices).parameters),('truth','powers'))
        self.assertEqual(oracle_indices([0,2,0,3],[.9,.2,.9,.8]),[3,1])

    def test_lr_weight_psd(self):
        _,_,a,_=physical_modes(.3)
        self.assertTrue(np.all(lr_weights(a)>0))

    def test_invalid_covariance_scale_rejected(self):
        with self.assertRaises(ValueError): physical_modes(0)

    def test_invalid_df_rejected(self):
        with self.assertRaises(ValueError): chi2_sf_even(2,3)

    def test_zero_kernel_no_contrast_information(self):
        a=np.zeros(2)
        self.assertTrue(np.all((1+4*a)/(1+a)==1))
        # Q still rejects noise at its nominal size: size is not material power.
        self.assertAlmostEqual(pair_tail(chi2_isf_even(.025,8),2,[1,1]),.025,places=12)


if __name__ == '__main__': unittest.main()
