"""Numerical behavior regressions, with analytic optima where available."""
import sys
from pathlib import Path
import unittest
from decimal import Decimal, localcontext

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "studies"))
from sci_method.split_merge import GaussianObservationGroup, fit_covariance_model
from sci_method.split_merge import grouped_gaussian_nll_and_gradient
import audit_optimizer_stability as audit
from sci_method.covariance_optimizer import CovarianceObjective, fit_covariance_detailed


class OptimizerStabilityTests(unittest.TestCase):
    @staticmethod
    def scalar_group(variance, derivatives=None, base=None):
        return GaussianObservationGroup(
            "analytic", np.sqrt(4*variance)*np.eye(4),
            np.eye(4) if base is None else base,
            np.eye(4)[None] if derivatives is None else derivatives)

    def test_tiny_preconditioned_step_cannot_certify_large_kkt_residual(self):
        # C=(1+10^8 theta)I; sample covariance=2I; exact optimum theta=1e-8.
        # The old preconditioned step is 1e-8, but the starting score is -8e8.
        group = GaussianObservationGroup("scaled", np.sqrt(8)*np.eye(4),
                                          np.eye(4), (1e8*np.eye(4))[None])
        theta, success, _ = fit_covariance_model([group], initial=np.zeros(1))
        gradient = grouped_gaussian_nll_and_gradient([group], theta)[1]
        self.assertTrue(success)
        self.assertLessEqual(abs(float(gradient[0])), 1e-6)
        self.assertAlmostEqual(theta[0], 1e-8, delta=1e-15)

    def test_two_known_failures_reach_fixed_scale_kkt_specification(self):
        for case in audit.fixed_cases()[:2]:
            with self.subTest(case=case["name"]):
                _, _, models, mode = audit.prepare_case(case)
                initial = audit.generator._initial_parameters("coarse", mode)
                theta, success, _ = fit_covariance_model(models["coarse"], initial=initial)
                self.assertTrue(success)
                diagnostic = audit.full_diagnostics(models["coarse"], theta, initial)
                self.assertLessEqual(diagnostic["fixed_scale_kkt_inf"], 1e-6)

    def test_zero_boundary_has_positive_score_but_zero_kkt(self):
        result = fit_covariance_detailed([self.scalar_group(0.1)], initial=[1.0])
        self.assertTrue(result.converged)
        self.assertEqual(result.parameters[0], 0.0)
        self.assertGreater(result.gradient[0], 0.0)
        self.assertEqual(result.kkt_inf, 0.0)

    def test_duplicate_kernel_optimum_is_covariance_equivalent_not_unique_parameters(self):
        coarse = fit_covariance_detailed([self.scalar_group(2.0)], initial=[0.2])
        fine = fit_covariance_detailed(
            [self.scalar_group(2.0, np.array([np.eye(4), np.eye(4)]))],
            initial=[0.1, 0.3])
        self.assertTrue(coarse.converged and fine.converged)
        self.assertAlmostEqual(sum(fine.parameters), 1.0, delta=1e-6)
        self.assertAlmostEqual(coarse.nll, fine.nll, delta=1e-8)

    def test_unobserved_parameter_is_not_forced_to_an_invented_unique_solution(self):
        group = self.scalar_group(2.0, np.array([np.eye(4), np.zeros((4, 4))]))
        result = fit_covariance_detailed([group], initial=[0.2, 7.0])
        self.assertTrue(result.converged)
        self.assertEqual(result.parameters[1], 7.0)
        self.assertAlmostEqual(result.parameters[0], 1.0, delta=1e-6)

    def test_near_singular_positive_covariance_remains_valid_and_is_reportable(self):
        base = np.diag([1e-10, 1, 2, 3])
        group = GaussianObservationGroup("ill_conditioned", np.sqrt(8)*np.sqrt(base),
                                          base, base[None])
        result = fit_covariance_detailed([group], initial=[0.2])
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.parameters[0], 1.0, delta=1e-6)

    def test_invalid_covariance_and_exhausted_iteration_budget_do_not_report_success(self):
        invalid = self.scalar_group(2, np.zeros((1, 4, 4)), np.zeros((4, 4)))
        failed = fit_covariance_detailed([invalid])
        self.assertEqual(failed.status, "failed")
        group = self.scalar_group(2)
        exhausted = fit_covariance_detailed([group], initial=[0.1], max_iter=1)
        self.assertFalse(exhausted.converged)
        self.assertGreater(exhausted.kkt_inf, 1e-6)
        self.assertEqual(exhausted.status, "unproven")

    def test_small_positive_iterate_is_not_treated_as_an_exact_active_bound(self):
        group = self.scalar_group(0.1)
        result = fit_covariance_detailed([group], initial=[1e-12])
        self.assertTrue(result.converged)
        self.assertEqual(result.parameters[0], 0.0)

    def test_stationary_local_maximum_is_not_certified_as_a_likelihood_minimum(self):
        # At theta=1, C1=2I with S1=0 and C2=11I with S2/n=71.5I.
        # Score/8 = 1/2 + 1/11 - 71.5/121 = 0, but Hessian < 0.
        groups = [self.scalar_group(0.0), self.scalar_group(71.5, base=10*np.eye(4))]
        result = fit_covariance_detailed(groups, initial=[1.0])
        self.assertFalse(result.converged)
        self.assertEqual(result.status, "unproven")
        self.assertIn("negative_curvature", result.reason)

    def test_stable_difference_matches_high_precision_scalar_formula_below_full_nll_ulp(self):
        objective = CovarianceObjective([self.scalar_group(2)])
        theta = np.array([1.0])
        candidate = np.array([1.0+1e-8])
        with localcontext() as context:
            context.prec = 70
            ratio = (Decimal.from_float(float(candidate[0]))+1)/2
            expected = float(Decimal(8)*(ratio.ln()+1/ratio-1))
        measured = objective.difference(theta, candidate)
        self.assertGreater(measured, 0)
        self.assertAlmostEqual(measured/expected, 1.0, delta=1e-6)

    def test_whitened_score_and_observed_hessian_match_independent_finite_differences(self):
        case = audit.fixed_cases()[1]
        _, _, models, _ = audit.prepare_case(case)
        groups = models["fine"]
        objective = CovarianceObjective(groups)
        theta = np.array([0.6, 1.3, 1.1])
        value, gradient, hessian, _ = objective.evaluate(theta)
        self.assertAlmostEqual(value, grouped_gaussian_nll_and_gradient(groups, theta)[0], delta=1e-8)
        for i in range(3):
            step = np.zeros(3)
            step[i] = 1e-4
            plus, gp = grouped_gaussian_nll_and_gradient(groups, theta+step)
            minus, gm = grouped_gaussian_nll_and_gradient(groups, theta-step)
            self.assertAlmostEqual(gradient[i], (plus-minus)/2e-4, delta=5e-5)
            np.testing.assert_allclose(hessian[:, i], (gp-gm)/2e-4, atol=5e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
