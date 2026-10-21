"""Execution-entry faults on deterministic unit fixtures, never study runs."""
from dataclasses import replace
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"studies"))
import run_adaptive_audit_scenarios as runner
from sci_method.covariance_optimizer import CovarianceObjective, fit_covariance_detailed


class ExecutionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.output=Path(self.temp.name)/"fixture"
        self.output.mkdir()
        try:
            self.api=importlib.import_module("low_budget_execution_gate")
        except ModuleNotFoundError:
            self.api=None
        self.addCleanup(self.cleanup)
        self.assertIsNotNone(self.api, "durable execution gate is not implemented")
        (self.output/"unit_fixture.json").write_text(json.dumps(dict(test=self.id(),
            purpose="fault injection / deterministic unit fixtures; not a six-cell experiment",
            observations="zero arrays; no RNG generation or physical data")))
        self.gate=self.api.ExecutionGate(self.output)
        self.data={name:np.zeros((4,4)) for name in ("left","right","other")}
        self.coarse=runner._build_groups(self.data,0.3,"coarse")
        self.fine=runner._build_groups(self.data,0.3,"fine")
        self.context=dict(phase="calibration",replicate=0,seed=123,
                          counts={name:4 for name in self.data})

    def cleanup(self):
        destination=os.environ.get("MUON_GATE_TEST_EVIDENCE_DIR")
        if destination and any(self.output.iterdir()):
            target=Path(destination)/self._testMethodName
            shutil.copytree(self.output,target)
        self.temp.cleanup()

    def records(self,name):
        path=self.output/name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_normal_pair_and_sufficient_statistics_are_persisted(self):
        self.gate.record_sample(self.data,self.context,role="train")
        coarse,fine,lrt=self.gate.fit_pair(self.coarse,self.fine,self.context)
        self.assertTrue(coarse["certified"] and fine["certified"])
        self.assertEqual(lrt["raw_lrt"],0.0)
        self.assertEqual(lrt["stable_lrt"],0.0)
        self.gate.check_lrt(lrt,{"calibrated":dict(value=3.0,allowance=4e-8)},self.context)
        self.assertEqual(len(self.records("fits.jsonl")),2)
        inputs=self.records("inputs.jsonl")
        self.assertEqual([r["n_events"] for r in inputs],[4,4,4])
        self.assertEqual(inputs[0]["scatter_mm2"],np.zeros((4,4)).tolist())
        self.assertEqual(len(inputs[0]["observation_sha256"]),64)

    def test_calibration_failure_persists_and_does_not_attempt_fine_fit(self):
        calls=[]
        def fail(groups,**kwargs):
            calls.append(len(groups[0].covariance_derivatives_mm2))
            result=fit_covariance_detailed(groups,**kwargs)
            return replace(result,status="unproven",reason="injected_iteration_budget")
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.fit_pair(self.coarse,self.fine,self.context,solver=fail)
        self.assertEqual(calls,[2])
        self.assertFalse(self.records("fits.jsonl")[0]["certified"])
        self.assertEqual(self.records("failures.jsonl")[0]["context"]["seed"],123)
        self.assertEqual(self.records("not_attempted.jsonl")[0]["model"],"fine")

    def test_claimed_success_with_forged_zero_kkt_is_recomputed_and_rejected(self):
        def false_success(groups,**kwargs):
            result=fit_covariance_detailed(groups,**kwargs)
            theta=np.ones_like(result.parameters)
            value=CovarianceObjective(groups).evaluate(theta)[0]
            return replace(result,parameters=theta,nll=value,gradient=np.zeros_like(theta),kkt_inf=0.0)
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.fit_pair(self.coarse,self.fine,self.context,solver=false_success)
        record=self.records("fits.jsonl")[0]
        self.assertEqual(record["reported_kkt_inf"],0.0)
        self.assertGreater(record["recomputed_kkt_inf"],1e-6)
        self.assertIn("kkt_not_satisfied",record["certification_issues"])

    def test_unknown_status_and_nonfinite_report_do_not_become_success(self):
        for name,mutation in (("unknown",dict(status="success")),("nan",dict(kkt_inf=float("nan")))):
            with self.subTest(name=name):
                folder=self.output/name;folder.mkdir()
                gate=self.api.ExecutionGate(folder)
                def invalid(groups,**kwargs):
                    return replace(fit_covariance_detailed(groups,**kwargs),**mutation)
                with self.assertRaises(self.api.ExecutionGateError):
                    gate.fit_pair(self.coarse,self.fine,self.context,solver=invalid)
                failure=json.loads((folder/"fits.jsonl").read_text().splitlines()[0])
                self.assertFalse(failure["certified"])
                self.assertTrue(failure["certification_issues"])
                self.assertNotIn("NaN",(folder/"fits.jsonl").read_text())

    def test_solver_exception_is_durable_before_propagation(self):
        def fail(*args,**kwargs):
            raise ArithmeticError("injected solver exception")
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.fit_pair(self.coarse,self.fine,self.context,solver=fail)
        self.assertEqual(self.records("fits.jsonl")[0]["status"],"exception")
        self.assertIn("ArithmeticError",self.records("fits.jsonl")[0]["exception"])

    def test_near_cutoff_stops_and_saves_both_lrt_values(self):
        _,_,lrt=self.gate.fit_pair(self.coarse,self.fine,self.context)
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.check_lrt(lrt,{"calibrated":dict(value=1e-9,allowance=0.0)},self.context)
        record=self.records("lrt_checks.jsonl")[-1]
        self.assertTrue(record["ambiguous"])
        self.assertEqual(record["cutoffs"]["calibrated"]["value"],1e-9)
        self.assertIn("raw_lrt",record)
        self.assertIn("stable_lrt",record)

    def test_nonfinite_stable_lrt_stops_before_any_statistical_decision(self):
        with patch.object(CovarianceObjective,"difference",return_value=float("nan")):
            # Use already certified real solutions so the injected difference
            # fault concerns the LRT, rather than the solver line search.
            fitted={2:fit_covariance_detailed(self.coarse,initial=[0,0]),
                    3:fit_covariance_detailed(self.fine,initial=[0,0,0])}
            def solver(groups,**kwargs):
                return fitted[groups[0].covariance_derivatives_mm2.shape[0]]
            with self.assertRaises(self.api.ExecutionGateError):
                self.gate.fit_pair(self.coarse,self.fine,self.context,solver=solver)
        self.assertEqual(self.records("lrt_checks.jsonl")[-1]["stable_lrt"],None)

    def test_known_fixed_mapping_rejects_changed_derivative_or_observation(self):
        valid=self.api.validate_known_fixed_nesting(self.coarse,self.fine)
        np.testing.assert_array_equal(valid@np.array([2.,3.]),[2.,2.,3.])
        changed=list(self.fine)
        changed[0]=replace(changed[0],base_covariance_mm2=changed[0].base_covariance_mm2+np.eye(4))
        with self.assertRaises(ValueError):
            self.api.validate_known_fixed_nesting(self.coarse,changed)
        derivative=np.copy(self.fine[0].covariance_derivatives_mm2)
        derivative[0,0,0]+=1.0
        changed[0]=replace(self.fine[0],covariance_derivatives_mm2=derivative)
        with self.assertRaises(ValueError):
            self.api.validate_known_fixed_nesting(self.coarse,changed)
        observations=np.copy(self.fine[0].residuals_mm);observations[0,0]+=1.0
        changed[0]=replace(self.fine[0],residuals_mm=observations)
        with self.assertRaises(ValueError):
            self.api.validate_known_fixed_nesting(self.coarse,changed)

    def test_stable_lrt_matches_direct_nll_for_distinct_models(self):
        # These are feasible hand-chosen parameters, not fitted model outputs.
        theta_c=np.array([1.,0.5]);theta_f=np.array([0.8,1.2,0.5])
        mapping=self.api.validate_known_fixed_nesting(self.coarse,self.fine)
        objective=CovarianceObjective(self.fine)
        expected=2*(CovarianceObjective(self.coarse).evaluate(theta_c)[0]-objective.evaluate(theta_f)[0])
        actual=2*objective.difference(theta_f,mapping@theta_c)
        self.assertAlmostEqual(actual,expected,delta=1e-8)

    def test_real_low_budget_calibration_loop_stops_first_injected_failure(self):
        count=[]
        def fixture(budget,*args):
            count.append(budget)
            return {name:np.zeros((n,4)) for name,n in runner._counts(budget).items()}
        def fail(groups,**kwargs):
            return replace(fit_covariance_detailed(groups,**kwargs),status="unproven",reason="injected")
        with patch.object(runner,"_simulate_low_budget",side_effect=fixture), \
             patch.object(self.api,"fit_covariance_detailed",side_effect=fail), \
             patch.object(runner,"calibrate_lrt_threshold") as calibrate:
            with self.assertRaises(self.api.ExecutionGateError):
                runner._low_budget_study(self.output)
        self.assertEqual(count,[500])
        calibrate.assert_not_called()
        self.assertTrue((self.output/"failures.jsonl").exists())

    def test_second_calibration_failure_keeps_first_success_without_calibrating_subset(self):
        generated=[];fits=[]
        def fixture(budget,*args):
            generated.append(args[-1])
            return {name:np.zeros((n,4)) for name,n in runner._counts(budget).items()}
        def fail_second_pair(groups,**kwargs):
            fits.append(len(groups[0].covariance_derivatives_mm2))
            result=fit_covariance_detailed(groups,**kwargs)
            return replace(result,status="unproven",reason="injected_second_pair") if len(fits)==3 else result
        with patch.object(runner,"_simulate_low_budget",side_effect=fixture), \
             patch.object(self.api,"fit_covariance_detailed",side_effect=fail_second_pair), \
             patch.object(runner,"calibrate_lrt_threshold") as calibrate:
            with self.assertRaises(self.api.ExecutionGateError):
                runner._low_budget_study(self.output)
        self.assertEqual(generated,[runner.CAL_SEED_BASE,runner.CAL_SEED_BASE+1])
        self.assertEqual(fits,[2,3,2])
        calibrate.assert_not_called()
        with (self.output/"low_budget_null_calibration.csv").open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),1)
        self.assertEqual(int(rows[0]["seed"]),runner.CAL_SEED_BASE)
        self.assertEqual(self.records("failures.jsonl")[-1]["context"]["replicate"],1)

    def test_fine_failure_is_persisted_after_certified_coarse(self):
        def fail_fine(groups,**kwargs):
            result=fit_covariance_detailed(groups,**kwargs)
            if len(result.parameters)==3:
                return replace(result,status="precision_limited",reason="injected_resolution_limit")
            return result
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.fit_pair(self.coarse,self.fine,self.context,solver=fail_fine)
        records=self.records("fits.jsonl")
        self.assertEqual([r["model"] for r in records],["coarse","fine"])
        self.assertTrue(records[0]["certified"])
        self.assertFalse(records[1]["certified"])

    def test_array_status_is_recorded_as_malformed_instead_of_bypassing_journal(self):
        def malformed(groups,**kwargs):
            return replace(fit_covariance_detailed(groups,**kwargs),status=np.array(["converged","failed"]))
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.fit_pair(self.coarse,self.fine,self.context,solver=malformed)
        self.assertIn("unknown_status",self.records("fits.jsonl")[0]["certification_issues"])

    def test_invalid_cutoff_states_are_persisted_and_rejected(self):
        _,_,lrt=self.gate.fit_pair(self.coarse,self.fine,self.context)
        for value in (None,"not-a-number",float("nan"),-1.0):
            with self.subTest(value=value),self.assertRaises(self.api.ExecutionGateError):
                self.gate.check_lrt(lrt,{"calibrated":dict(value=value)},self.context)
        self.assertEqual(len(self.records("failures.jsonl")),4)

    def test_aborted_entry_has_startup_config_snapshots_and_final_hash_manifest(self):
        destination=self.output/"lifecycle"
        def fixture(budget,*args):
            return {name:np.zeros((n,4)) for name,n in runner._counts(budget).items()}
        def fail(groups,**kwargs):
            return replace(fit_covariance_detailed(groups,**kwargs),status="failed",reason="injected")
        with patch.object(runner,"_output_path",return_value=destination), \
             patch.object(runner,"_run_contents",side_effect=lambda path,**kwargs:runner._low_budget_study(path)), \
             patch.object(runner,"_simulate_low_budget",side_effect=fixture), \
             patch.object(self.api,"fit_covariance_detailed",side_effect=fail):
            with self.assertRaises(self.api.ExecutionGateError):
                runner.run("run-unit-abort")
        state=json.loads((destination/"run_state.json").read_text())
        evidence=json.loads((destination/"evidence.json").read_text())
        self.assertEqual(state["status"],"aborted")
        self.assertFalse(state["study_body_completed"])
        for filename in ("config.json","environment.json","failure.json","fits.jsonl","inputs.jsonl"):
            self.assertIn(filename,evidence["result_file_sha256"])
            self.assertEqual(hashlib.sha256((destination/filename).read_bytes()).hexdigest(),evidence["result_file_sha256"][filename])
        for filename in ("covariance_optimizer.py","low_budget_execution_gate.py"):
            self.assertTrue((destination/"source_snapshot"/filename).is_file())
            self.assertTrue(any(p.endswith(filename) for p in evidence["source_file_sha256"]))

    def test_completed_ablation_only_entry_has_explicit_status_without_sampling(self):
        destination=self.output/"ablation"
        with patch.object(runner,"_output_path",return_value=destination), \
             patch.object(runner,"_run_contents",return_value=({},dict(run001_ablation_rows=0))), \
             patch.object(runner,"_simulate_low_budget") as draw:
            runner.run("run-unit-ablation",include_low_budget=False)
        draw.assert_not_called()
        self.assertEqual(json.loads((destination/"run_state.json").read_text())["status"],"completed")
        self.assertFalse(json.loads((destination/"config.json").read_text())["low_budget_scenarios_requested"])

    def test_existing_output_is_rejected_without_altering_files(self):
        before={p.name:p.read_bytes() for p in self.output.iterdir() if p.is_file()}
        with patch.object(runner,"_output_path",return_value=self.output):
            with self.assertRaises(FileExistsError):
                runner.run("run-existing")
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.output.iterdir() if p.is_file()})

    def test_bic_and_calibration_uncertainty_are_both_checked(self):
        _,_,lrt=self.gate.fit_pair(self.coarse,self.fine,self.context)
        # BIC-side uncertainty must stop even if the calibrated comparison is safe.
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.check_lrt(lrt,{"calibrated":dict(value=3.,allowance=0.),
                                     "bic":dict(value=1e-9,allowance=0.)},self.context)
        self.assertTrue(self.records("lrt_checks.jsonl")[-1]["cutoffs"]["bic"]["ambiguous"])
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.check_lrt(lrt,{"calibrated":dict(value=1e-4,allowance=2e-4)},self.context)
        self.assertEqual(self.records("lrt_checks.jsonl")[-1]["cutoffs"]["calibrated"]["calibration_allowance"],2e-4)

    def test_raw_and_stable_decision_disagreement_is_preserved(self):
        _,_,lrt=self.gate.fit_pair(self.coarse,self.fine,self.context)
        injected={**lrt,"raw_lrt":2.,"stable_lrt":2.+2e-9}
        with self.assertRaises(self.api.ExecutionGateError):
            self.gate.check_lrt(injected,{"calibrated":dict(value=2.+1e-9)},self.context)
        check=self.records("lrt_checks.jsonl")[-1]["cutoffs"]["calibrated"]
        self.assertNotEqual(check["raw_exceeds"],check["stable_exceeds"])

    def test_persisted_observations_can_reconstruct_the_likelihood_without_raw_rows(self):
        self.gate.record_sample(self.data,self.context,role="train")
        saved=self.records("inputs.jsonl")
        reconstructed=0.
        theta=np.array([0.8,1.2,0.5])
        for group,row in zip(self.fine,saved):
            covariance=group.base_covariance_mm2+np.einsum("p,pab->ab",theta,group.covariance_derivatives_mm2)
            reconstructed+=(row["n_events"]*np.linalg.slogdet(covariance)[1]+
                            np.trace(np.linalg.solve(covariance,np.array(row["scatter_mm2"]))))/2
        self.assertAlmostEqual(reconstructed,CovarianceObjective(self.fine).evaluate(theta)[0],delta=1e-8)

    def test_existing_fit_time_metric_remains_solver_only(self):
        with patch.object(self.api.time,"perf_counter",side_effect=[0.,1.,3.,10.,12.,15.]):
            coarse,fine,_=self.gate.fit_pair(self.coarse,self.fine,self.context)
        self.assertEqual(coarse["fit_ms"],1000.)
        self.assertEqual(fine["fit_ms"],2000.)
        self.assertEqual(coarse["certification_and_scoring_ms"],2000.)
        self.assertEqual(fine["certification_and_scoring_ms"],3000.)


if __name__=="__main__":
    unittest.main()
