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
    spec.loader.exec_module(module)
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
    if len(checks) != sum(len(v) > 0 for v in data.values()):
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


def run(run_id, stage):
    output = ROOT / "studies/optimizer-stability-001/results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT/"OPTIMIZER_STABILITY_PROTOCOL.md",
             ROOT/"src/sci_method/split_merge.py"]
    for name in ("covariance_optimizer.py",):
        if (ROOT/"src/sci_method"/name).exists():
            paths.append(ROOT/"src/sci_method"/name)
    snapshots = output/"source_snapshot"
    snapshots.mkdir()
    for path in paths:
        shutil.copyfile(path, snapshots/path.name)
    cases = fixed_cases()
    save(output/"config.json", dict(stage=stage, cases=cases, base_commit=BASE_SHA,
         rng="numpy.default_rng/PCG64; multivariate_normal default SVD; left,right,other order",
         six_cell_grid_run=False, geant4_run=False))
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
            print(case["name"], model, old["converged"], old["iterations"],
                  "KKT", old["final"]["fixed_scale_kkt_inf"], flush=True)
        results.append(record)
    save(output/"summary.json", results)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        np.show_config()
    save(output/"environment.json", dict(python=sys.version, numpy=np.__version__,
         platform=platform.platform(), machine=platform.machine(), numpy_config=capture.getvalue()))
    source_hashes = {str(p.relative_to(ROOT.parent)):sha(p) for p in paths}
    evidence = dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        source_file_sha256=source_hashes, historical_evidence_sha256=sha(HISTORY/"evidence.json"),
        historical_scatter_sha256=sha(HISTORY/"sample_covariance_sufficient_statistics.csv"),
        command=f"PYTHONPATH=sci_method/src python3 sci_method/studies/audit_optimizer_stability.py --stage {stage} --run-id {run_id}",
        results_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    save(output/"evidence.json", evidence)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("legacy",), default="legacy")
    args = parser.parse_args()
    run(args.run_id, args.stage)
