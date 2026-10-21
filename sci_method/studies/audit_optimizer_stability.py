"""Fixed-panel numerical audit; never invokes Geant4 or the six-cell grid."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "studies"))
HISTORY = ROOT / "studies/adaptive-prototype-001/results/run-002"
BASE_SHA = "5334f3533e1ada58f0afb3e98f8de10a6f52aeb8"


def load_snapshot(name, filename):
    spec = importlib.util.spec_from_file_location(name, HISTORY / "source_snapshot" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        # Historical source snapshots are read-only research inputs. Prevent
        # importlib from depositing interpreter caches beside them.
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


legacy = load_snapshot("optimizer_audit_legacy", "split_merge.py")
generator = load_snapshot("optimizer_audit_generator", "run_adaptive_split_merge.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save(path, value):
    path.write_text(json.dumps(value, default=native, indent=2, allow_nan=False) + "\n")


def fixed_cases():
    cases = [
        dict(name="A", scenario="uniform_known_background", phase="calibration_null",
             sigma=1.0, replicate=54, seed=5028102055, null=True),
        dict(name="B", scenario="weak_contrast_large_hit_error", phase="evaluation_train",
             sigma=1.0, replicate=173, seed=5030103731, null=False),
    ]
    for i, (scenario, config) in enumerate(generator.SCENARIOS.items()):
        cases.append(dict(name=f"control_{i}", scenario=scenario, phase="evaluation_train",
                          sigma=config["sigma_hit_mm"], replicate=0,
                          seed=5027102001 + i * 1000000, null=False))
    cases.append(dict(name="control_calibration", scenario="uniform_known_background",
                      phase="calibration_null", sigma=1.0, replicate=0,
                      seed=5028102001, null=True))
    return cases


def prepare_case(case):
    config = generator.SCENARIOS[case["scenario"]]
    data = generator.simulate_target(
        case["scenario"], case["seed"], sigma_hit_mm_override=case["sigma"],
        child_levels_override=(1.0, 1.0) if case["null"] else None,
    )
    empty = None
    if config["background_mode"] == "empty_field_calibrated":
        empty = generator.simulate_empty_field(case["sigma"], config["counts"], case["seed"] + 2)
    models = {
        model: generator.build_likelihood_groups(
            data, sigma_hit_mm=case["sigma"], mode=config["background_mode"],
            model=model, empty_field_data=empty,
        )
        for model in ("coarse", "fine")
    }
    return data, empty, models, config["background_mode"]


def check_statistics(case, data):
    with (HISTORY / "sample_covariance_sufficient_statistics.csv").open() as stream:
        saved = [row for row in csv.DictReader(stream)
                 if row["phase"] == case["phase"] and int(row["seed"]) == case["seed"]]
    checks = []
    for row in saved:
        scatter = data[row["group"]].T @ data[row["group"]]
        old = np.array([float(row[f"s{i}{j}_mm2"]) for i in range(4) for j in range(i, 4)])
        new = np.array([scatter[i, j] for i in range(4) for j in range(i, 4)])
        checks.append(dict(group=row["group"], exact_equal=np.array_equal(new, old),
                           max_absolute_difference=float(np.max(np.abs(new-old)))))
    if len(checks) != len(data):
        raise RuntimeError("historical sufficient-statistic rows missing")
    if not all(check["exact_equal"] for check in checks):
        raise RuntimeError("regenerated likelihood inputs differ; stop before fit comparison")
    return checks


def kkt(theta, gradient, scale):
    residual = np.where(theta > 0, gradient, np.minimum(gradient, 0.0))
    return residual, float(np.max(np.abs(scale * residual)))


def full_diagnostics(groups, theta, initial):
    value, gradient = legacy.grouped_gaussian_nll_and_gradient(groups, theta)
    diag = np.zeros(len(theta))
    covariances = []
    for group in groups:
        cov = group.base_covariance_mm2 + np.einsum("p,pab->ab", theta, group.covariance_derivatives_mm2)
        inv = np.linalg.inv(cov)
        for i, derivative in enumerate(group.covariance_derivatives_mm2):
            diag[i] += len(group.residuals_mm)/2 * np.trace(inv @ derivative @ inv @ derivative)
        eigen = np.linalg.eigvalsh(cov)
        covariances.append(dict(group=group.name, eigenvalues=eigen,
                                condition=float(eigen[-1]/eigen[0]), covariance=cov))
    preconditioner = np.maximum(diag, max(float(diag.max()), 1.0)*1e-12)
    projected = theta - np.maximum(0, theta-gradient/preconditioner)
    residual, norm = kkt(theta, gradient, np.maximum(1.0, initial))
    n = sum(len(g.residuals_mm) for g in groups)
    return dict(parameters=theta, nll_without_constant=value,
                full_nll=value + 2*n*np.log(2*np.pi), analytic_gradient=gradient,
                kkt_residual=residual, fixed_scale_kkt_inf=norm,
                fisher_diagonal=diag, inverse_diagonal_preconditioner=1/preconditioner,
                preconditioned_projected_gradient=projected, covariances=covariances)


def trace_legacy(groups, initial):
    """Observe unchanged historical function through Python's line tracer."""
    lines, start_line = inspect.getsourcelines(legacy.fit_covariance_model)
    text = {start_line+i: line.strip() for i, line in enumerate(lines)}
    trials, iterates, exits = [], [], []
    def tracer(frame, event, arg):
        if frame.f_code is not legacy.fit_covariance_model.__code__:
            return None
        if event == "line":
            local = frame.f_locals
            source = text[frame.f_lineno]
            if source.startswith("if not np.any(delta)") and not np.any(local["delta"]):
                trials.append(dict(iteration=local["iteration"], theta=local["theta"].copy(),
                    trial_step=local["trial_step"], delta=local["delta"].copy(),
                    candidate=local["candidate"].copy(), accepted=False,
                    skip_reason="parameter_change_rounded_to_zero"))
            if source.startswith("if candidate_value <= armijo"):
                record = {key: np.copy(local[key]) if isinstance(local[key], np.ndarray) else local[key]
                          for key in ("iteration", "theta", "value", "gradient", "preconditioner",
                                      "scaled_gradient", "trial_step", "candidate", "delta",
                                      "candidate_value", "candidate_gradient", "armijo")}
                record["objective_change"] = local["candidate_value"]-local["value"]
                record["accepted"] = bool(local["candidate_value"] <= local["armijo"])
                trials.append(record)
            if source.startswith("if np.linalg.norm(projected_gradient"):
                iterates.append({key: np.copy(local[key]) if isinstance(local[key], np.ndarray) else local[key]
                                 for key in ("iteration", "theta", "value", "gradient",
                                             "fisher_diagonal", "preconditioner", "projected_gradient")})
            if source.startswith("return theta,"):
                exits.append(dict(source_line=frame.f_lineno, expression=source))
        return tracer
    old_trace = sys.gettrace()
    begin = time.perf_counter()
    try:
        sys.settrace(tracer)
        theta, converged, iterations = legacy.fit_covariance_model(groups, initial=initial)
    finally:
        sys.settrace(old_trace)
    return dict(initial=full_diagnostics(groups, initial, initial),
                final=full_diagnostics(groups, theta, initial),
                converged=converged, iterations=iterations, exits=exits,
                iterates=iterates, trials=trials,
                traced_runtime_ms=1000*(time.perf_counter()-begin))


def independent_value_score(groups, parameters):
    """Independent direct solve / slogdet / per-event quadratic evaluation.

    Does not call the production objective, whitening, gradient or Hessian.
    Used by SciPy as a separate numerical implementation of the same NLL.
    """
    value, gradient = 0.0, np.zeros(len(parameters))
    for group in groups:
        covariance = np.array(group.base_covariance_mm2, copy=True)
        for theta, derivative in zip(parameters, group.covariance_derivatives_mm2):
            covariance += theta*derivative
        np.linalg.cholesky(covariance)
        _, logdet = np.linalg.slogdet(covariance)
        inverse = np.linalg.solve(covariance, np.eye(4))
        solved_y = np.linalg.solve(covariance, group.residuals_mm.T)
        n = len(group.residuals_mm)
        value += (n*logdet+np.sum(group.residuals_mm.T*solved_y))/2
        for j, derivative in enumerate(group.covariance_derivatives_mm2):
            gradient[j] += (n*np.trace(inverse@derivative)
                           -np.sum(solved_y*(derivative@solved_y)))/2
    return float(value), gradient


def reference_fit(groups, initial, multiple_starts=False):
    """L-BFGS-B followed by an independent free-score root least-squares check.

    Every solver return flag and the recomputed KKT are kept; reference selection
    uses the frozen KKT specification rather than trusting success flags.
    """
    from scipy.optimize import minimize, least_squares
    candidates = []
    begin = time.perf_counter()
    for multiplier in ((0.5, 1.0, 2.0) if multiple_starts else (1.0,)):
        start = initial*multiplier
        offset = independent_value_score(groups, initial)[0]
        def fun(x):
            try:
                f, g = independent_value_score(groups, x)
                return f-offset, g
            except np.linalg.LinAlgError:
                return 1e100, np.zeros(len(x))
        bounded = minimize(fun, start, jac=True, method="L-BFGS-B",
            bounds=[(0, None)]*len(start),
            options=dict(ftol=0.0, gtol=1e-9, maxiter=1000, maxls=50))
        value, gradient = independent_value_score(groups, bounded.x)
        norm = kkt(bounded.x, gradient, np.maximum(1, initial))[1]
        candidates.append(dict(method="L-BFGS-B", start=start, parameters=bounded.x,
            nll=value, gradient=gradient, kkt_inf=norm, solver_success=bool(bounded.success),
            solver_message=str(bounded.message), nfev=bounded.nfev))
        free = (bounded.x > 1e-10) | (gradient < 0)
        def root_score(x):
            full = bounded.x.copy()
            full[free] = x
            return independent_value_score(groups, full)[1][free]
        if np.any(free):
            try:
                root = least_squares(root_score, bounded.x[free], bounds=(0, np.inf),
                    jac="3-point", ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=500)
                polished = bounded.x.copy()
                polished[free] = root.x
                value, gradient = independent_value_score(groups, polished)
                norm = kkt(polished, gradient, np.maximum(1, initial))[1]
                candidates.append(dict(method="bounded_score_least_squares",
                    start=bounded.x, parameters=polished, nll=value, gradient=gradient,
                    kkt_inf=norm, solver_success=bool(root.success),
                    solver_message=str(root.message), nfev=root.nfev))
            except (ValueError, np.linalg.LinAlgError) as exc:
                candidates.append(dict(method="bounded_score_least_squares",
                                       error=str(exc), kkt_inf=None))
    valid = [c for c in candidates if c.get("kkt_inf") is not None and c["kkt_inf"] <= 1e-6]
    # Rank by residual first within the predeclared NLL agreement budget, so
    # one-ulp objective ordering is not mistaken for superiority.
    if valid:
        best_nll = min(c["nll"] for c in valid)
        valid = [c for c in valid if c["nll"]-best_nll <= 1e-8]
        selected = min(valid, key=lambda c:c["kkt_inf"])
    else:
        selected = min((c for c in candidates if "nll" in c), key=lambda c:c["nll"])
    return dict(selected=selected, candidates=candidates, certified_kkt=bool(valid),
                runtime_ms=1000*(time.perf_counter()-begin))


def gradient_check(groups, parameters):
    value, gradient = independent_value_score(groups, parameters)
    steps = np.finfo(float).eps**(1/3)*np.maximum(1, parameters)
    records = []
    for factor in (0.1, 1.0, 10.0):
        numeric = np.zeros(len(parameters))
        for i, h in enumerate(steps*factor):
            e = np.zeros(len(parameters)); e[i] = h
            plus = independent_value_score(groups, parameters+e)[0]
            if parameters[i] >= h:
                minus = independent_value_score(groups, parameters-e)[0]
                numeric[i] = (plus-minus)/(2*h)
            else:
                plus2 = independent_value_score(groups, parameters+2*e)[0]
                numeric[i] = (-3*value+4*plus-plus2)/(2*h)
        records.append(dict(step_factor=factor, steps=steps*factor, numeric_gradient=numeric,
                            absolute_error=np.abs(numeric-gradient)))
    errors = np.min([r["absolute_error"] for r in records], axis=0)
    return dict(analytic_gradient=gradient, estimates=records, best_absolute_error=errors,
                passed=bool(np.all(errors <= 5e-5+2e-5*np.abs(gradient))))


def compare_fit(groups, initial, old, multiple_starts):
    from dataclasses import asdict
    from sci_method.covariance_optimizer import CovarianceObjective, fit_covariance_detailed
    begin = time.perf_counter()
    result = fit_covariance_detailed(groups, initial=initial, record_history=True)
    new_time = 1000*(time.perf_counter()-begin)
    # Equal timing treatment: no tracing/history for either solver; median of
    # five repetitions of the identical fixed inputs (not new data/experiments).
    timings = {"legacy":[], "candidate":[]}
    for _ in range(5):
        begin = time.perf_counter()
        legacy.fit_covariance_model(groups, initial=initial)
        timings["legacy"].append(1000*(time.perf_counter()-begin))
        begin = time.perf_counter()
        fit_covariance_detailed(groups, initial=initial)
        timings["candidate"].append(1000*(time.perf_counter()-begin))
    reference = reference_fit(groups, initial, multiple_starts)
    selected = reference["selected"]
    objective = CovarianceObjective(groups)
    value, gradient, hessian, fisher = objective.evaluate(result.parameters)
    ref_theta = selected["parameters"]
    ref_value, ref_gradient = independent_value_score(groups, result.parameters)
    eigen = np.linalg.eigvalsh(hessian)
    rank = int(np.sum(eigen > 1e-10*max(1, np.max(np.abs(eigen)))))
    covariance_errors = []
    for group in groups:
        d = group.covariance_derivatives_mm2
        a = group.base_covariance_mm2 + np.einsum("i,iab->ab", result.parameters, d)
        b = group.base_covariance_mm2 + np.einsum("i,iab->ab", ref_theta, d)
        covariance_errors.append(float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300)))
    parameter_error = float(np.max(np.abs(result.parameters-ref_theta)/np.maximum(1, np.abs(ref_theta))))
    objective_budget = max(1e-8, 512*np.finfo(float).eps*max(1, abs(value)))
    reference_error = abs(ref_value-selected["nll"])
    out = dict(candidate=asdict(result), candidate_traced_ms=new_time,
        candidate_final=full_diagnostics(groups, result.parameters, initial),
        reference=reference, observed_hessian_eigenvalues=eigen, fisher_eigenvalues=np.linalg.eigvalsh(fisher),
        observed_hessian_rank=rank, local_negative_curvature=bool(eigen.min() < -1e-8),
        parameter_scaled_reference_difference=parameter_error,
        max_covariance_relative_reference_difference=max(covariance_errors),
        nll_reference_difference=ref_value-selected["nll"], nll_agreement_budget=objective_budget,
        old_to_new_nll_difference=ref_value-old["final"]["nll_without_constant"],
        old_to_new_stable_nll_difference=objective.difference(
            np.array(old["final"]["parameters"]), result.parameters),
        old_to_new_parameter_difference=result.parameters-old["final"]["parameters"],
        independent_vs_whitened_gradient_max_error=float(np.max(np.abs(gradient-ref_gradient))),
        gradient_checks=dict(initial=gradient_check(groups, initial),
                             final=gradient_check(groups, result.parameters)),
        timing_ms=timings,
        acceptance=dict(kkt=result.converged and result.kkt_inf <=1e-6,
             reference_kkt=reference["certified_kkt"], objective=reference_error <= objective_budget,
             parameter_or_covariance=(parameter_error <=1e-5 if rank==len(initial)
                                      else max(covariance_errors) <=1e-6)))
    out["acceptance"]["finite_difference_gradient"] = all(
        check["passed"] for check in out["gradient_checks"].values())
    out["acceptance"]["independent_score"] = out["independent_vs_whitened_gradient_max_error"] <= 1e-6
    out["accepted"] = all(out["acceptance"].values()) and not out["local_negative_curvature"]
    return out


def protected_history_hashes():
    records = {}
    for study in ("adaptive-prototype-001", "adaptive-audit-fix-001"):
        for run_id in ("run-001", "run-002"):
            directory = ROOT/"studies"/study/"results"/run_id
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    records[str(path.relative_to(ROOT))] = sha(path)
    return records


def run(run_id, stage):
    if not run_id.startswith("run-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("invalid run ID")
    protection_before = protected_history_hashes()
    output = ROOT / "studies/optimizer-stability-001/results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT/"OPTIMIZER_STABILITY_PROTOCOL.md",
             ROOT/"src/sci_method/split_merge.py", ROOT/"src/sci_method/forward_model.py",
             ROOT/"src/sci_method/identifiability.py", ROOT/"src/sci_method/study_io.py",
             ROOT/"tests/test_optimizer_stability.py"]
    for name in ("covariance_optimizer.py",):
        if (ROOT/"src/sci_method"/name).exists():
            paths.append(ROOT/"src/sci_method"/name)
    snapshots = output/"source_snapshot"
    snapshots.mkdir()
    for path in paths:
        shutil.copyfile(path, snapshots/path.name)
    shutil.copyfile(HISTORY/"source_snapshot/split_merge.py", snapshots/"legacy_split_merge.py")
    shutil.copyfile(HISTORY/"source_snapshot/run_adaptive_split_merge.py", snapshots/"historical_generator.py")
    cases = fixed_cases()
    save(output/"config.json", dict(stage=stage, cases=cases, base_commit=BASE_SHA,
         rng="numpy.default_rng/PCG64; multivariate_normal default SVD; left,right,other order",
         six_cell_grid_run=False, geant4_run=False,
         kkt_tolerance=1e-6, max_iterations=500,
         frozen_acceptance_commit="da315c0", scipy_is_audit_only=True))
    results = []
    for case in cases:
        data, empty, models, mode = prepare_case(case)
        checks = check_statistics(case, data)
        np.savez(output/f"{case['name']}_observations.npz", **data,
                 **({f"empty_{k}":v for k,v in empty.items()} if empty is not None else {}))
        record = dict(case=case, historical_statistics=checks, models={})
        for model, groups in models.items():
            initial = generator._initial_parameters(model, mode)
            old = trace_legacy(groups, initial)
            save(output/f"{case['name']}_{model}_legacy_trace.json", old)
            record["models"][model] = dict(converged=old["converged"],
                iterations=old["iterations"], final=old["final"], exit=old["exits"])
            if stage == "comparison":
                comparison = compare_fit(groups, initial, old, case["name"] in ("A", "B"))
                save(output/f"{case['name']}_{model}_comparison.json", comparison)
                record["models"][model]["comparison"] = {
                    k:v for k,v in comparison.items() if k not in ("reference", "candidate", "gradient_checks")}
                record["models"][model]["candidate"] = {
                    k:v for k,v in comparison["candidate"].items() if k != "history"}
                record["models"][model]["reference_parameters"] = comparison["reference"]["selected"]["parameters"]
            print(case["name"], model, old["converged"], old["iterations"],
                  "KKT", old["final"]["fixed_scale_kkt_inf"], flush=True)
        if stage == "comparison":
            from sci_method.covariance_optimizer import CovarianceObjective
            coarse = record["models"]["coarse"]
            fine = record["models"]["fine"]
            c, f = coarse["candidate"], fine["candidate"]
            embedded = np.insert(c["parameters"], 1, c["parameters"][0])
            stable_lrt = 2*CovarianceObjective(models["fine"]).difference(f["parameters"], embedded)
            raw_lrt = 2*(c["nll"]-f["nll"])
            old_lrt = 2*(coarse["final"]["nll_without_constant"]-fine["final"]["nll_without_constant"])
            thresholds = json.loads((HISTORY/"summary.json").read_text())["calibration_thresholds_by_stratum"]
            threshold = thresholds[f"{mode}|sigma={case['sigma']:.1f}"]
            error = 2*sum(v["comparison"]["nll_agreement_budget"] +
                          abs(v["comparison"]["nll_reference_difference"])
                          for v in (coarse, fine))
            info = generator.contrast_report_for_decision(models["fine"], mode)
            power = info["predicted_power_for_design_contrast"]
            old_decision = generator.decide_split(info, max(0,old_lrt), threshold, power,
                                fit_converged=coarse["converged"] and fine["converged"])
            new_decision = generator.decide_split(info, max(0,raw_lrt), threshold, power,
                                fit_converged=c["status"]=="converged" and f["status"]=="converged")
            record["lrt"] = dict(old_raw=old_lrt, new_raw=raw_lrt,
                new_covariance_difference=stable_lrt, frozen_threshold=threshold,
                numerical_allowance=error, threshold_in_allowance=abs(raw_lrt-threshold)<=error,
                score_below_numerical_resolution=abs(raw_lrt)<=error,
                old_lrt_only=old_lrt>threshold, new_lrt_only=raw_lrt>threshold,
                stable_lrt_only=stable_lrt>threshold,
                old_composite=old_decision, new_composite=new_decision,
                estimable=info["estimable"], predicted_power=power)
        results.append(record)
    save(output/"summary.json", results)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        np.show_config()
    environment = dict(python=sys.version, numpy=np.__version__,
         platform=platform.platform(), machine=platform.machine(), numpy_config=capture.getvalue())
    if stage == "comparison":
        import scipy
        environment["scipy"] = scipy.__version__
    save(output/"environment.json", environment)
    suite = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "sci_method/tests", "-v"],
                           cwd=ROOT.parent, capture_output=True, text=True)
    (output/"test.log").write_text(suite.stdout+suite.stderr)
    save(output/"test_status.json", dict(exit_code=suite.returncode,
         command=f"{sys.executable} -m unittest discover -s sci_method/tests -v"))
    protection_after = protected_history_hashes()
    save(output/"historical_protection.json", dict(unchanged=protection_before==protection_after,
         before_sha256=protection_before, after_sha256=protection_after))
    source_hashes = {str(p.relative_to(ROOT.parent)):sha(p) for p in paths}
    evidence = dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        source_file_sha256=source_hashes, historical_evidence_sha256=sha(HISTORY/"evidence.json"),
        historical_scatter_sha256=sha(HISTORY/"sample_covariance_sufficient_statistics.csv"),
        command=f"PYTHONPATH=sci_method/src {sys.executable} sci_method/studies/audit_optimizer_stability.py --stage {stage} --run-id {run_id}",
        results_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    save(output/"evidence.json", evidence)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("legacy", "comparison"), default="legacy")
    args = parser.parse_args()
    run(args.run_id, args.stage)
