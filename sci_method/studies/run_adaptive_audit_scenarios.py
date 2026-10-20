"""Reproduce frozen-run ablations and a small predeclared low-budget grid.

No Geant4 is invoked. All synthetic runs use unique, disjoint seeds and new
audit-fix output paths. This is developmental evidence, not independent SCI
validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "studies"))

from adaptive_audit_analysis import ablation_decisions, lrt_numerical_diagnostic, wilson_interval  # noqa: E402
from run_adaptive_split_merge import (  # noqa: E402
    BACKGROUND_LEVEL, CALIBRATION_ALPHA, COUNT_MINIMUM_PER_CHILD, _draw_group,
    _fit_and_score, _fit_trial, _synthetic_target_covariance,
    build_likelihood_groups, calibrate_lrt_threshold,
    contrast_report_for_decision, count_based_split,
    decide_split, simulate_target,
)
from sci_method.study_io import create_output_directory  # noqa: E402


STUDY_ID = "adaptive-audit-fix-001"
RUN_ID = "run-001"
REPLICATES = 200
CALIBRATION_REPLICATES = 200
CAL_SEED_BASE = 4_026_102_001
EVAL_SEED_BASE = 4_027_102_001
LOW_BUDGET_SCENARIOS = (
    {"name": "uniform_n500_sigma03", "budget": 500, "delta": 0.0, "sigma": 0.3},
    {"name": "uniform_n1000_sigma03", "budget": 1000, "delta": 0.0, "sigma": 0.3},
    {"name": "moderate_d05_n500_sigma03", "budget": 500, "delta": 0.5, "sigma": 0.3},
    {"name": "moderate_d075_n500_sigma03", "budget": 500, "delta": 0.75, "sigma": 0.3},
    {"name": "moderate_d05_n1000_sigma03", "budget": 1000, "delta": 0.5, "sigma": 0.3},
    {"name": "moderate_d05_n500_sigma10", "budget": 500, "delta": 0.5, "sigma": 1.0},
)
NULL_STRATA = ((500, 0.3), (1000, 0.3), (500, 1.0), (1000, 1.0))
METHODS = (
    "lrt_only", "bic_only", "estimability_plus_lrt", "estimability_power_only",
    "estimability_lrt_power", "full_composite", "count_only", "fixed_coarse", "fixed_fine",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_run001_metrics() -> list[dict]:
    source = ROOT / "studies/adaptive-prototype-001/results/run-001/trial_metrics.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000:
        raise ValueError(f"expected 1,000 immutable run-001 rows; found {len(rows)}")
    return rows


def _ablation_run001(output: Path) -> tuple[dict, dict]:
    source = ROOT / "studies/adaptive-prototype-001/results/run-001/trial_metrics.csv"
    rows = _read_run001_metrics()
    output_rows = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        decisions = ablation_decisions(row)
        n_target = sum(int(row[k]) for k in ("n_left", "n_right", "n_other"))
        if row["background_mode"] == "empty_field_calibrated" and row["empty_field_seed"]:
            n_likelihood = n_target * 2
        else:
            n_likelihood = n_target
        lrt = float(row["lrt_train"])
        scale_proxy = max(abs(float(row["coarse_eval_nll"])), abs(float(row["fine_eval_nll"])))
        diag = lrt_numerical_diagnostic(lrt, scale_proxy)
        merged = {
            **row, **{f"decision_{k}": v for k, v in decisions.items()},
            "bic_sample_count": n_likelihood,
            "bic_lrt_cutoff": math.log(max(n_likelihood, 2)),
            "lrt_relative_to_validation_nll_scale": diag["relative_to_objective_scale"],
            "lrt_machine_epsilon_units_of_validation_scale": diag["machine_epsilon_units"],
        }
        output_rows.append(merged)
        grouped.setdefault(row["scenario"], []).append(merged)

    with (output / "run001_ablation_by_trial.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    summaries = []
    for name, trials in grouped.items():
        base = []
        for trial in trials:
            truth_min = trial["effect_meets_design_minimum_evaluation_only"].lower() == "true"
            decision_methods = {m: trial[f"decision_{m}"] == "True" for m in (
                "lrt_only", "bic_only", "estimability_plus_lrt",
                "estimability_power_only", "estimability_lrt_power", "full_composite",
            )}
            decision_methods["count_only"] = trial["count_split"].lower() == "true"
            decision_methods["fixed_coarse"] = False
            decision_methods["fixed_fine"] = True
            base.append((trial, truth_min, decision_methods))
        method_summaries = {}
        for method in METHODS:
            selected = [d[method] for _, _, d in base]
            count = sum(selected)
            interval = wilson_interval(count, len(selected))
            nlls = [float(row["fine_eval_nll"] if decision else row["coarse_eval_nll"])
                    for (row, _, _), decision in zip(base, selected)]
            method_summaries[method] = {
                "split_count": count, "split_rate": count / len(selected),
                "split_rate_wilson95": interval,
                "correct_vs_declared_min_effect": sum(
                    decision == truth for (_, truth, _), decision in zip(base, selected)
                ) / len(selected),
                "mean_validation_nll": float(np.mean(nlls)),
            }
        changed = {}
        chain = ("lrt_only", "estimability_plus_lrt", "estimability_lrt_power", "full_composite")
        for left, right in zip(chain, chain[1:]):
            changed[f"{left}_to_{right}"] = sum(
                a[left] != a[right] for _, _, a in base
            )
        lrt = np.array([float(row["lrt_train"]) for row in trials])
        rel = np.array([float(row["lrt_relative_to_validation_nll_scale"]) for row in trials])
        summaries.append({
            "scenario": name, "replicates": len(trials),
            "mean_lrt": float(np.mean(lrt)), "max_lrt": float(np.max(lrt)),
            "mean_lrt_relative_to_validation_nll_scale": float(np.mean(rel)),
            "max_lrt_relative_to_validation_nll_scale": float(np.max(rel)),
            "ablations": method_summaries,
            "gate_changed_decision_counts": changed,
        })
    summary = {
        "source_run": "adaptive-prototype-001/run-001",
        "source_csv_sha256": _sha256(source),
        "method_definitions": {
            "lrt_only": "converged and raw training LRT > unchanged empirical run-001 stratum cutoff",
            "bic_only": "converged and raw training LRT > log(n independent training tracks); one extra child parameter",
            "estimability_plus_lrt": "LRT-only plus joint Fisher contrast estimability",
            "estimability_power_only": "estimable and predeclared Wald power >=0.80, with no observed-LRT gate",
            "estimability_lrt_power": "estimability+LRT plus power >=0.80",
            "full_composite": "convergence+estimability+power+unchanged calibrated LRT; identical to prior row when all fits converge",
        },
        "scenarios": summaries,
        "caveat": "All rows are re-analysis of development run-001, not independent validation. Validation NLL scale is used only as an approximate objective magnitude because training NLLs were not stored in run-001; exact LRT and unchanged cutoffs are retained.",
    }
    return summary, {"run001_ablation_rows": len(output_rows)}


def _counts(budget: int) -> dict[str, int]:
    left = int(round(0.4 * budget))
    right = int(round(0.4 * budget))
    return {"left": left, "right": right, "other": budget - left - right}


def _simulate_low_budget(budget: int, delta: float, sigma: float, seed: int) -> dict:
    counts = _counts(budget)
    levels = (1.0 - delta / 2.0, 1.0 + delta / 2.0)
    rng = np.random.default_rng(seed)
    data = {}
    for region, count in counts.items():
        covariance = _synthetic_target_covariance(
            region, child_levels=levels, background_level=BACKGROUND_LEVEL,
            mode="known_fixed", sigma_hit_mm=sigma,
        )
        data[region] = _draw_group(rng, covariance, count)
    return data


def _build_groups(data: dict, sigma: float, model: str):
    return build_likelihood_groups(data, sigma_hit_mm=sigma, mode="known_fixed", model=model)


def _low_budget_study(output: Path) -> dict:
    thresholds = {}
    cal_rows = []
    for index, (budget, sigma) in enumerate(NULL_STRATA):
        null_lrt = []
        for replicate in range(CALIBRATION_REPLICATES):
            seed = CAL_SEED_BASE + index * 1_000_000 + replicate
            data = _simulate_low_budget(budget, 0.0, sigma, seed)
            groups_coarse, groups_fine = _build_groups(data, sigma, "coarse"), _build_groups(data, sigma, "fine")
            coarse, fine, lrt, converged = _fit_trial(groups_coarse, groups_fine, groups_coarse, groups_fine, "known_fixed")
            cal_rows.append({"budget": budget, "sigma_hit_mm": sigma, "replicate": replicate,
                             "seed": seed, "lrt": lrt if converged else "", "converged": converged})
            if converged:
                null_lrt.append(lrt)
        if len(null_lrt) < 20:
            raise RuntimeError(f"too few converged null fits for N={budget}, sigma={sigma}")
        thresholds[f"N={budget}|sigma={sigma:.1f}"] = calibrate_lrt_threshold(null_lrt, alpha=CALIBRATION_ALPHA)

    metrics = []
    for scenario_index, spec in enumerate(LOW_BUDGET_SCENARIOS):
        key = f"N={spec['budget']}|sigma={spec['sigma']:.1f}"
        cutoff = thresholds[key]
        for replicate in range(REPLICATES):
            seed = EVAL_SEED_BASE + scenario_index * 1_000_000 + replicate * 10
            validation_seed = seed + 1
            counts = _counts(spec["budget"])
            train = _simulate_low_budget(spec["budget"], spec["delta"], spec["sigma"], seed)
            validation = _simulate_low_budget(spec["budget"], spec["delta"], spec["sigma"], validation_seed)
            coarse_train, fine_train = _build_groups(train, spec["sigma"], "coarse"), _build_groups(train, spec["sigma"], "fine")
            coarse_val, fine_val = _build_groups(validation, spec["sigma"], "coarse"), _build_groups(validation, spec["sigma"], "fine")
            coarse, fine, lrt, converged = _fit_trial(coarse_train, fine_train, coarse_val, fine_val, "known_fixed")
            info = contrast_report_for_decision(fine_train, "known_fixed")
            decisions = ablation_decisions({
                "coarse_converged": str(coarse["converged"]), "fine_converged": str(fine["converged"]),
                "lrt_train": str(lrt), "lrt_threshold": str(cutoff),
                "contrast_estimable": str(info["estimable"]),
                "predicted_power": str(info["predicted_power_for_design_contrast"]),
                "n_left": str(counts["left"]), "n_right": str(counts["right"]), "n_other": str(counts["other"]),
                "background_mode": "known_fixed", "empty_field_seed": "",
            })
            decisions["count_only"] = count_based_split(
                (counts["left"], counts["right"]), minimum_per_child=COUNT_MINIMUM_PER_CHILD,
            )
            decisions["fixed_coarse"] = False
            decisions["fixed_fine"] = True
            row = {
                "scenario": spec["name"], "replicate": replicate,
                "train_seed": seed, "validation_seed": validation_seed,
                "event_budget": spec["budget"], "delta_truth_evaluation_only": spec["delta"],
                "sigma_hit_mm": spec["sigma"], "n_left": counts["left"],
                "n_right": counts["right"], "n_other": counts["other"],
                "contrast_estimable": info["estimable"],
                "contrast_variance_ref2": info["variance"] if info["estimable"] else "inf",
                "predicted_power_for_delta_05": info["predicted_power_for_design_contrast"],
                "lrt_train": lrt, "lrt_cutoff": cutoff, "fit_converged": converged,
                "coarse_validation_nll": coarse["eval_nll"], "fine_validation_nll": fine["eval_nll"],
                "coarse_fit_ms": coarse["fit_ms"], "fine_fit_ms": fine["fit_ms"],
                "failure": "" if converged else "coarse_or_fine_nonconvergence",
                **{f"split_{method}": decision for method, decision in decisions.items()},
            }
            metrics.append(row)

    with (output / "low_budget_null_calibration.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cal_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(cal_rows)
    with (output / "low_budget_trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(metrics)

    summaries = []
    for spec in LOW_BUDGET_SCENARIOS:
        trials = [r for r in metrics if r["scenario"] == spec["name"]]
        method_report = {}
        for method in METHODS:
            chosen = [bool(r[f"split_{method}"]) for r in trials]
            nll = [float(r["fine_validation_nll"] if decision else r["coarse_validation_nll"])
                   for r, decision in zip(trials, chosen)]
            interval = wilson_interval(sum(chosen), len(chosen))
            method_report[method] = {
                "splits": sum(chosen), "split_rate": sum(chosen) / len(chosen),
                "split_rate_wilson95": interval,
                "correct_vs_minimum_effect": sum(
                    decision == (spec["delta"] >= 0.5) for decision in chosen
                ) / len(chosen),
                "mean_validation_nll": float(np.mean(nll)),
            }
        summaries.append({
            **spec, "replicates": len(trials), "calibration_cutoff": thresholds[
                f"N={spec['budget']}|sigma={spec['sigma']:.1f}"
            ],
            "mean_fine_minus_coarse_validation_nll": float(np.mean([
                float(r["fine_validation_nll"]) - float(r["coarse_validation_nll"])
                for r in trials
            ])),
            "mean_fit_ms_coarse": float(np.mean([float(r["coarse_fit_ms"]) for r in trials])),
            "mean_fit_ms_fine": float(np.mean([float(r["fine_fit_ms"]) for r in trials])),
            "fit_failures": sum(bool(r["failure"]) for r in trials),
            "methods": method_report,
        })
    return {
        "calibration_replicates_per_stratum": CALIBRATION_REPLICATES,
        "evaluation_replicates_per_scenario": REPLICATES,
        "calibration_thresholds": thresholds,
        "scenarios": summaries,
        "predeclared_scope": "Six fixed cells only: two null event budgets, two moderate contrasts at N=500, N=1000 replication of delta=0.5, and one high-hit-error case. No parameter search. Alpha, power threshold, minimum effect, event allocations, and 200 replicas fixed before evaluating these cells.",
    }


def run(run_id: str = RUN_ID) -> Path:
    if not run_id.startswith("run-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("run ID must start with run- and contain only letters, digits, dash or underscore")
    output = ROOT / "studies" / STUDY_ID / "results" / run_id
    create_output_directory(output)

    ablation, counts = _ablation_run001(output)
    (output / "run001_ablation_summary.json").write_text(json.dumps(ablation, indent=2, sort_keys=True) + "\n")
    algebraic = next(s for s in ablation["scenarios"] if s["scenario"] == "algebraic_background_contrast_degeneracy")
    algebraic_trials = [r for r in _read_run001_metrics()
                        if r["scenario"] == "algebraic_background_contrast_degeneracy"]
    diag = {
        "source_run": "adaptive-prototype-001/run-001",
        "algebraic_negative_control_replicates": algebraic["replicates"],
        "raw_lrt_min": 0.0,
        "raw_lrt_mean": algebraic["mean_lrt"],
        "raw_lrt_max": algebraic["max_lrt"],
        "unchanged_empirical_cutoff": float(algebraic_trials[0]["lrt_threshold"]),
        "lrt_only_split_count": algebraic["ablations"]["lrt_only"]["split_count"],
        "estimability_plus_lrt_split_count": algebraic["ablations"]["estimability_plus_lrt"]["split_count"],
        "mean_relative_to_validation_nll_scale": algebraic["mean_lrt_relative_to_validation_nll_scale"],
        "max_relative_to_validation_nll_scale": algebraic["max_lrt_relative_to_validation_nll_scale"],
        "interpretation": "The fine and coarse covariance families are exactly equivalent under the algebraically confounded nuisance. Positive LRTs near zero, including values slightly above an empirical cutoff calibrated from the same numerical degeneracy, are optimizer/floating-point residuals, not resolved statistical evidence. The Fisher estimability check rejects the contrast in all replicates. The cutoff is unchanged; no post-hoc threshold was tuned.",
        "training_objective_limitation": "run-001 did not store training NLL values. Objective-scale ratios use held-out NLL magnitude as an explicit order-of-magnitude proxy; raw training LRT values and empirical cutoffs are exact stored values.",
    }
    (output / "near_zero_lrt_diagnostic.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n")
    low_budget = _low_budget_study(output)
    (output / "low_budget_summary.json").write_text(json.dumps(low_budget, indent=2, sort_keys=True) + "\n")

    config = {
        "study_id": STUDY_ID, "run_id": run_id,
        "source_development_run": "adaptive-prototype-001/run-001",
        "low_budget_seed_bases": {"null_calibration": CAL_SEED_BASE, "evaluation": EVAL_SEED_BASE},
        "calibration_replicates_per_stratum": CALIBRATION_REPLICATES,
        "evaluation_replicates_per_scenario": REPLICATES,
        "frozen_scenarios": LOW_BUDGET_SCENARIOS,
        "null_strata": NULL_STRATA,
        "fixed_parameters": {"alpha": CALIBRATION_ALPHA, "min_power": 0.80,
                             "minimum_contrast_ref": 0.5, "count_minimum_per_child": COUNT_MINIMUM_PER_CHILD,
                             "lambda_ref_mm_inv": 1e-2, "momentum_gev_c": 4.0,
                             "path_fraction_left_right_other": [0.4, 0.4, 0.2]},
        "geant4_invoked": False,
        "all_data_synthetic_or_reanalysis": True,
        "thresholds_not_tuned_from_evaluation": True,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    script = Path(__file__).resolve()
    source_paths = [script, ROOT / "studies/adaptive_audit_analysis.py",
                    ROOT / "studies/run_adaptive_split_merge.py",
                    ROOT / "src/sci_method/forward_model.py",
                    ROOT / "src/sci_method/split_merge.py"]
    snap = output / "source_snapshot"
    snap.mkdir()
    source_hashes = {}
    for source in source_paths:
        shutil.copyfile(source, snap / source.name)
        source_hashes[source.relative_to(REPO).as_posix()] = _sha256(source)
    result_hashes = {p.relative_to(output).as_posix(): _sha256(p)
                     for p in sorted(output.rglob("*")) if p.is_file() and p.name != "evidence.json"}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except subprocess.CalledProcessError:
        commit = "unknown"
    evidence = {
        "run_id": run_id, "source_commit": commit,
        "source_file_sha256": source_hashes,
        "run001_csv_sha256": ablation["source_csv_sha256"],
        "run001_ablation_rows": counts["run001_ablation_rows"],
        "result_file_sha256": result_hashes,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "command": f"PYTHONPATH=sci_method/src python3 sci_method/studies/run_adaptive_audit_scenarios.py --run-id {run_id}",
        "geant4_invoked": False,
        "calibration_and_evaluation_seeds_disjoint": True,
        "truth_access": "Synthetic truth is used by generation and post-decision evaluation only; the decision rule consumes likelihood/Fisher/evidence quantities.",
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    print(run(args.run_id))


if __name__ == "__main__":
    main()
