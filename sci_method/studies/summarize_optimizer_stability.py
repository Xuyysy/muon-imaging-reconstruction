"""Supplement fixed audit fits with full matrices and analytic stress records.

Reads the saved run-004 observations; no historical stochastic data are drawn.
Only tiny analytic control fits are executed. No low-budget grid or Geant4.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from audit_optimizer_stability import (
    ROOT, generator, legacy, full_diagnostics, protected_history_hashes, save, sha,
)
from sci_method.covariance_optimizer import CovarianceObjective, fit_covariance_detailed
from sci_method.split_merge import GaussianObservationGroup


def group(variance, *, base=None, derivatives=None):
    return GaussianObservationGroup("analytic", np.sqrt(4*variance)*np.eye(4),
        np.eye(4) if base is None else base,
        np.eye(4)[None] if derivatives is None else derivatives)


def stress_models():
    base = np.diag([1e-10, 1.0, 2.0, 3.0])
    ill = GaussianObservationGroup("ill_conditioned", np.sqrt(8)*np.sqrt(base), base, base[None])
    return (
        ("tiny_preconditioned_step", [group(2, derivatives=(1e8*np.eye(4))[None])], [0.0], {}),
        ("small_positive_not_active", [group(0.1)], [1e-12], {}),
        ("zero_boundary", [group(0.1)], [1.0], {}),
        ("duplicate_parameters", [group(2, derivatives=np.array([np.eye(4),np.eye(4)]))], [0.1,0.3], {}),
        ("unobserved_parameter", [group(2, derivatives=np.array([np.eye(4),np.zeros((4,4))]))], [0.2,7.0], {}),
        ("near_singular_positive", [ill], [0.2], {}),
        ("stationary_local_maximum", [group(0),group(71.5,base=10*np.eye(4))], [1.0], {}),
        ("iteration_exhaustion", [group(2)], [0.1], {"max_iter":1}),
        ("parameter_resolution_limit", [GaussianObservationGroup("unrepresentable_step",
            np.sqrt(4*(1+1e-12))*np.eye(4), (1-1e12)*np.eye(4),
            (1e12*np.eye(4))[None])], [1.0], {}),
        ("invalid_covariance", [group(2,base=np.zeros((4,4)),derivatives=np.zeros((1,4,4)))], [1.0], {}),
    )


def safe_result(result):
    # JSON uses null for undefined objective/score on an explicitly failed fit.
    data = asdict(result)
    data.pop("history")
    for key in ("nll", "kkt_inf"):
        if not np.isfinite(data[key]):
            data[key] = None
    if not np.isfinite(data["gradient"]).all():
        data["gradient"] = None
    return data


def run(run_id):
    if not run_id.startswith("run-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("invalid run ID")
    source = ROOT/"studies/optimizer-stability-001/results/run-004"
    before = protected_history_hashes()
    output = source.parent/run_id
    output.mkdir(exist_ok=False)
    rows = json.loads((source/"summary.json").read_text())
    matrices = []
    for row in rows:
        case = row["case"]
        spec = generator.SCENARIOS[case["scenario"]]
        with np.load(source/f"{case['name']}_observations.npz", allow_pickle=False) as saved:
            target = {k: saved[k] for k in ("left","right","other")}
            empty = {k: saved[f"empty_{k}"] for k in target} if f"empty_left" in saved else None
        for model in ("coarse","fine"):
            groups = generator.build_likelihood_groups(target, sigma_hit_mm=case["sigma"],
                mode=spec["background_mode"], model=model, empty_field_data=empty)
            objective = CovarianceObjective(groups)
            initial = generator._initial_parameters(model,spec["background_mode"])
            old_trace = json.loads((source/f"{case['name']}_{model}_legacy_trace.json").read_text())
            comparison = json.loads((source/f"{case['name']}_{model}_comparison.json").read_text())
            records = []
            for entry in comparison["candidate"]["history"]:
                theta = np.array(entry["parameters"])
                value, gradient, hessian, fisher = objective.evaluate(theta)
                free = (theta>0)|(gradient<0)
                eigen,vectors = np.linalg.eigh(hessian[np.ix_(free,free)])
                cutoff = 64*np.finfo(float).eps*max(1.0,float(np.max(np.abs(eigen))))
                keep = eigen>cutoff
                operator = np.zeros((len(theta),len(theta)))
                if entry["method"] == "observed_hessian":
                    operator[np.ix_(free,free)] = (vectors[:,keep]/eigen[keep])@vectors[:,keep].T
                else:
                    operator = np.diag(1/np.maximum(np.diag(fisher),64*np.finfo(float).eps))
                records.append(dict(iteration=entry["iteration"], step=entry["step"],
                    parameters=theta, nll_without_constant=value,
                    full_nll=value+2*sum(len(g.residuals_mm) for g in groups)*np.log(2*np.pi),
                    analytic_gradient=gradient, coordinate_scale=np.maximum(1,initial),
                    observed_hessian=hessian, fisher=fisher, free_mask=free,
                    direction_operator=operator, proposed_direction=-operator@gradient,
                    actual_delta=entry["delta"], stable_nll_change=entry["stable_nll_change"],
                    armijo_change=entry["armijo_change"], accepted=entry["accepted"]))
            old = np.array(old_trace["final"]["parameters"])
            final = np.array(comparison["candidate"]["parameters"])
            raw_trials = []
            for entry in old_trace["trials"]:
                if "objective_change" not in entry:
                    continue
                stable = objective.difference(np.array(entry["theta"]),np.array(entry["candidate"]))
                raw_trials.append(dict(iteration=entry["iteration"],step=entry["trial_step"],
                    legacy_absolute_nll_change=entry["objective_change"],
                    stable_nll_change=stable, accepted_by_legacy=entry["accepted"]))
            matrices.append(dict(case=case["name"],model=model,iterations=records,
                legacy_trial_precision=raw_trials,
                stable_remaining_decrease=objective.difference(old,final),
                legacy_objective_ulp=float(np.spacing(old_trace["final"]["nll_without_constant"]))))
    save(output/"full_iteration_matrices.json",matrices)
    stresses = []
    for name,groups,initial,options in stress_models():
        initial = np.array(initial)
        candidate = fit_covariance_detailed(groups,initial=initial,**options)
        record = dict(name=name,initial=initial,candidate=safe_result(candidate))
        try:
            old,success,iterations = legacy.fit_covariance_model(groups,initial=initial,**options)
            record["legacy"] = dict(success=success,iterations=iterations,
                                      diagnostic=full_diagnostics(groups,old,initial))
        except (ValueError,np.linalg.LinAlgError) as exc:
            record["legacy"] = dict(error_type=type(exc).__name__,message=str(exc))
        if name=="stationary_local_maximum":
            value,g,h,_ = CovarianceObjective(groups).evaluate(initial)
            record["analytic_negative_example"] = dict(score=g,hessian=h,
                nll_at_stationary=value,nll_at_feasible_zero=CovarianceObjective(groups).evaluate(np.zeros(1))[0])
        stresses.append(record)
    save(output/"analytic_stress_controls.json",stresses)
    fits=[m for row in rows for m in row["models"].values()]
    summary=dict(fixed_panel_cases=len(rows),fixed_panel_fits=len(fits),
        legacy_success=sum(m["converged"] for m in fits),
        legacy_stalled=sum(not m["converged"] for m in fits),
        legacy_success_kkt_above_frozen_tolerance=sum(m["converged"] and m["final"]["fixed_scale_kkt_inf"]>1e-6 for m in fits),
        candidate_status_counts={s:sum(m["candidate"]["status"]==s for m in fits)
                                 for s in ("converged","precision_limited","unproven","failed")},
        all_reference_checks_passed=all(m["comparison"]["accepted"] for m in fits),
        max_candidate_kkt=max(m["candidate"]["kkt_inf"] for m in fits),
        max_nll_reference_difference=max(abs(m["comparison"]["nll_reference_difference"]) for m in fits),
        max_raw_lrt_difference=max(abs(r["lrt"]["new_raw"]-r["lrt"]["old_raw"]) for r in rows),
        composite_decision_changes=sum(r["lrt"]["old_composite"]["split"]!=r["lrt"]["new_composite"]["split"] for r in rows),
        objective_scale_ambiguous_lrt_cases=[r["case"]["name"] for r in rows if r["lrt"]["threshold_in_allowance"]],
        mean_of_per_fit_median_ms={key:float(np.mean([np.median(m["comparison"]["timing_ms"][key]) for m in fits]))
                                   for key in ("legacy","candidate")},
        historical_directories_unchanged=before==protected_history_hashes(),
        six_cell_grid_run=False,geant4_run=False)
    save(output/"summary.json",summary)
    shutil.copyfile(Path(__file__),output/"source_snapshot.py")
    shutil.copyfile(ROOT/"src/sci_method/covariance_optimizer.py",output/"covariance_optimizer_snapshot.py")
    suite = subprocess.run([sys.executable,"-m","unittest","discover","-s","sci_method/tests","-v"],
        cwd=ROOT.parent,capture_output=True,text=True)
    (output/"test.log").write_text(suite.stdout+suite.stderr)
    save(output/"test_status.json",dict(exit_code=suite.returncode,command=f"{sys.executable} -m unittest discover -s sci_method/tests -v"))
    save(output/"evidence.json",dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        input_run004_evidence_sha256=sha(source/"evidence.json"),
        source_sha256=sha(Path(__file__)),
        command=f"PYTHONPATH=sci_method/src {sys.executable} sci_method/studies/summarize_optimizer_stability.py --run-id {run_id}",
        results_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file()}))
    print(json.dumps(summary,indent=2))
    print(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    run(parser.parse_args().run_id)
