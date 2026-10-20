"""Run a small reproducible two-child split/merge synthetic prototype.

This study uses only hand-specified straight paths, fixed 4 GeV/c Gaussian
scattering, and independent Gaussian hit-position errors. It does not invoke
Geant4 and is not an SCI independent validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # sci_method/; the repository is ROOT.parent.
sys.path.insert(0, str(ROOT / "src"))

from sci_method.forward_model import (  # noqa: E402
    BoxVoxel, detector_residual_covariance, scattering_covariance_per_density,
)
from sci_method.identifiability import (  # noqa: E402
    contrast_information,
)
from sci_method.split_merge import (  # noqa: E402
    GaussianObservationGroup, calibrate_lrt_threshold, count_based_split,
    decide_split, expected_two_sided_power, fit_covariance_model,
    grouped_gaussian_nll_and_gradient,
)
from sci_method.study_io import create_output_directory  # noqa: E402


STUDY_ID = "adaptive-prototype-001"
RUN_ID = "run-001"
Z_PLANES_MM = np.array([-400.0, -300.0, 300.0, 400.0])
MOMENTUM_GEV_C = 4.0
LAMBDA_REF_MM_INV = 1.0e-2
EVENT_BUDGET = 4000
CALIBRATION_REPLICATES = 200
EVALUATION_REPLICATES = 200
CALIBRATION_ALPHA = 0.05
MIN_DESIGN_POWER = 0.80
DESIGN_CONTRAST_REF = 0.5
COUNT_MINIMUM_PER_CHILD = 40
BASE_CALIBRATION_SEED = 5_026_102_001
BASE_EVALUATION_SEED = 5_027_102_001
BASELINE_LEVEL = 1.0
BACKGROUND_LEVEL = 0.5

CHILD_VOXELS = {
    "left": BoxVoxel(-20, 0, -10, 10, -50, 50),
    "right": BoxVoxel(0, 20, -10, 10, -50, 50),
    "other": BoxVoxel(20, 40, -10, 10, -50, 50),
}
BACKGROUND_VOXEL = BoxVoxel(-100, 100, -100, 100, -200, -150)
PATH_X_MM = {"left": -10.0, "right": 10.0, "other": 30.0}

SCENARIOS = {
    "uniform_known_background": {
        "background_mode": "known_fixed", "sigma_hit_mm": 0.3,
        "child_levels": (1.0, 1.0), "background_level": BACKGROUND_LEVEL,
        "counts": {"left": 1600, "right": 1600, "other": 800},
        "target_split": False,
        "effect_meets_design_minimum": False,
    },
    "strong_contrast_calibrated_background": {
        "background_mode": "empty_field_calibrated", "sigma_hit_mm": 0.3,
        "child_levels": (0.25, 1.75), "background_level": BACKGROUND_LEVEL,
        "counts": {"left": 1600, "right": 1600, "other": 800},
        "target_split": True,
        "effect_meets_design_minimum": True,
    },
    "right_child_uncovered": {
        "background_mode": "known_fixed", "sigma_hit_mm": 0.3,
        "child_levels": (1.0, 1.0), "background_level": BACKGROUND_LEVEL,
        "counts": {"left": 3200, "right": 0, "other": 800},
        "target_split": None,
        "effect_meets_design_minimum": None,
    },
    "weak_contrast_large_hit_error": {
        "background_mode": "known_fixed", "sigma_hit_mm": 1.0,
        "child_levels": (0.9, 1.1), "background_level": BACKGROUND_LEVEL,
        "counts": {"left": 1600, "right": 1600, "other": 800},
        "target_split": True,
        "effect_meets_design_minimum": False,
    },
    "algebraic_background_contrast_degeneracy": {
        "background_mode": "algebraic_confounded", "sigma_hit_mm": 0.3,
        "child_levels": (1.0, 1.0), "background_level": 1.0,
        "counts": {"left": 1600, "right": 1600, "other": 800},
        "target_split": None,
        "effect_meets_design_minimum": None,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(name: str):
    return (PATH_X_MM[name], 0.0, 0.0, 0.0, -400.0, 400.0)


def _unit_density_kernels():
    """Return covariance derivatives in mm^2 per dimensionless lambda_ref."""
    return {
        name: LAMBDA_REF_MM_INV * scattering_covariance_per_density(
            _path(name), voxel, Z_PLANES_MM, momentum_gev=MOMENTUM_GEV_C
        )
        for name, voxel in CHILD_VOXELS.items()
    }


KERNELS = _unit_density_kernels()
BACKGROUND_KERNELS = {
    name: LAMBDA_REF_MM_INV * scattering_covariance_per_density(
        _path(name), BACKGROUND_VOXEL, Z_PLANES_MM, momentum_gev=MOMENTUM_GEV_C
    )
    for name in PATH_X_MM
}


def _algebraic_confounded_background_kernel(region: str) -> np.ndarray:
    """Return the signed nuisance derivative for the algebraic negative control.

    The nuisance exactly matches the local child-contrast direction and is
    deliberately zero for the independent ``other`` path class.
    """
    if region == "left":
        return -0.5 * KERNELS["left"]
    if region == "right":
        return 0.5 * KERNELS["right"]
    if region == "other":
        return np.zeros((4, 4), dtype=float)
    raise ValueError(f"unknown path class: {region}")


def _synthetic_target_covariance(region: str, *, child_levels, background_level: float,
                                 mode: str, sigma_hit_mm: float) -> np.ndarray:
    """Covariance used by the synthetic target generator for one path class."""
    covariance = detector_residual_covariance(Z_PLANES_MM, sigma_hit_mm)
    if region in ("left", "right"):
        covariance += float(child_levels[0 if region == "left" else 1]) * KERNELS[region]
    elif region == "other":
        covariance += BASELINE_LEVEL * KERNELS["other"]
    else:
        raise ValueError(f"unknown path class: {region}")

    if mode in ("known_fixed", "empty_field_calibrated"):
        covariance += background_level * BACKGROUND_KERNELS[region]
    elif mode == "algebraic_confounded":
        covariance += background_level * _algebraic_confounded_background_kernel(region)
    else:
        raise ValueError(f"unknown background mode: {mode}")
    return covariance


def truth_free_scenario_inputs(name: str) -> dict:
    """Expose only design inputs needed by inference; truth remains evaluator-side."""
    scenario = SCENARIOS[name]
    return {
        "background_mode": scenario["background_mode"],
        "sigma_hit_mm": scenario["sigma_hit_mm"],
        "path_counts": dict(scenario["counts"]),
        "event_budget": EVENT_BUDGET,
    }


def _draw_group(rng, covariance, count):
    if count == 0:
        return np.empty((0, 4), dtype=float)
    return rng.multivariate_normal(np.zeros(4), 0.5 * (covariance + covariance.T), size=count)


def simulate_target(name: str, seed: int, *, child_levels_override=None,
                    sigma_hit_mm_override: float | None = None):
    """Create synthetic residuals; scenario truth is used only in this generator."""
    scenario = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    sigma = scenario["sigma_hit_mm"] if sigma_hit_mm_override is None else float(sigma_hit_mm_override)
    detector = detector_residual_covariance(Z_PLANES_MM, sigma)
    result = {}
    child_levels = scenario["child_levels"] if child_levels_override is None else child_levels_override
    for region, count in scenario["counts"].items():
        mode = scenario["background_mode"]
        cov = _synthetic_target_covariance(
            region, child_levels=child_levels,
            background_level=scenario["background_level"], mode=mode,
            sigma_hit_mm=sigma,
        )
        np.linalg.cholesky(cov)
        result[region] = _draw_group(rng, cov, count)
    return result


def simulate_empty_field(sigma_hit_mm: float, counts: dict, seed: int):
    """Independent empty-field calibration sample; it contains no target region."""
    rng = np.random.default_rng(seed)
    detector = detector_residual_covariance(Z_PLANES_MM, sigma_hit_mm)
    return {
        region: _draw_group(rng, detector + BACKGROUND_LEVEL * BACKGROUND_KERNELS[region], count)
        for region, count in counts.items()
    }


def _parameter_names(model: str, mode: str):
    names = ["parent", "other"] if model == "coarse" else ["left", "right", "other"]
    if mode != "known_fixed":
        names.append("background")
    return names


def _mode_background_kernel(region: str, mode: str):
    if mode == "algebraic_confounded":
        return _algebraic_confounded_background_kernel(region)
    return BACKGROUND_KERNELS[region]


def build_likelihood_groups(target_data: dict, *, sigma_hit_mm: float, mode: str,
                            model: str, empty_field_data: dict | None = None):
    """Construct the same Gaussian likelihood for coarse/fine alternatives."""
    names = _parameter_names(model, mode)
    index = {name: i for i, name in enumerate(names)}
    detector = detector_residual_covariance(Z_PLANES_MM, sigma_hit_mm)
    groups = []
    for region, residuals in target_data.items():
        if len(residuals) == 0:
            continue
        derivatives = np.zeros((len(names), 4, 4))
        if model == "fine" and region in ("left", "right"):
            derivatives[index[region]] = KERNELS[region]
        elif model == "coarse" and region in ("left", "right"):
            derivatives[index["parent"]] = KERNELS[region]
        elif region == "other":
            derivatives[index["other"]] = KERNELS["other"]
        base = detector.copy()
        if mode == "known_fixed":
            base += BACKGROUND_LEVEL * BACKGROUND_KERNELS[region]
        else:
            derivatives[index["background"]] = _mode_background_kernel(region, mode)
        groups.append(GaussianObservationGroup(region, residuals, base, derivatives))

    if mode == "empty_field_calibrated" and empty_field_data is not None:
        for region, residuals in empty_field_data.items():
            if len(residuals) == 0:
                continue
            derivatives = np.zeros((len(names), 4, 4))
            derivatives[index["background"]] = BACKGROUND_KERNELS[region]
            groups.append(GaussianObservationGroup(
                f"empty_{region}", residuals, detector, derivatives
            ))
    elif mode != "empty_field_calibrated" and empty_field_data is not None:
        raise ValueError("empty-field calibration is only valid for that background mode")
    return groups


def _initial_parameters(model: str, mode: str):
    if model == "coarse":
        values = [BASELINE_LEVEL, BASELINE_LEVEL]
    else:
        values = [BASELINE_LEVEL, BASELINE_LEVEL, BASELINE_LEVEL]
    if mode != "known_fixed":
        values.append(BACKGROUND_LEVEL if mode == "empty_field_calibrated" else 1.0)
    return np.asarray(values, dtype=float)


def _fisher_groups(groups, parameters):
    n_parameters = len(parameters)
    fisher = np.zeros((n_parameters, n_parameters))
    for group in groups:
        covariance = group.base_covariance_mm2 + np.einsum(
            "p,pab->ab", parameters, group.covariance_derivatives_mm2
        )
        count = len(group.residuals_mm)
        if not count:
            continue
        np.linalg.cholesky(covariance)
        inv = np.linalg.inv(covariance)
        for a, derivative_a in enumerate(group.covariance_derivatives_mm2):
            for b, derivative_b in enumerate(group.covariance_derivatives_mm2):
                fisher[a, b] += 0.5 * count * np.trace(
                    inv @ derivative_a @ inv @ derivative_b
                )
    if not np.any(fisher):
        raise ValueError("Fisher analysis requires nonempty groups")
    return 0.5 * (fisher + fisher.T)


def contrast_report_for_decision(groups, mode: str):
    """Evaluate child-difference information in the full target+nuisance Fisher."""
    fine_names = _parameter_names("fine", mode)
    reference = _initial_parameters("fine", mode)
    fisher = _fisher_groups(groups, reference)
    contrast = np.zeros(len(fine_names))
    contrast[fine_names.index("left")] = -1.0
    contrast[fine_names.index("right")] = 1.0
    report = contrast_information(fisher, contrast)
    power = 0.0
    if report["estimable"]:
        power = expected_two_sided_power(
            DESIGN_CONTRAST_REF, report["variance"], alpha=CALIBRATION_ALPHA
        )
    report["predicted_power_for_design_contrast"] = power
    report["design_contrast_ref"] = DESIGN_CONTRAST_REF
    report["parameter_names"] = fine_names
    report["joint_data_fisher"] = fisher
    return report


def _fit_and_score(train_groups, eval_groups, model, mode):
    initial = _initial_parameters(model, mode)
    start = time.perf_counter()
    params, converged, iterations = fit_covariance_model(train_groups, initial=initial)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    train_nll = grouped_gaussian_nll_and_gradient(train_groups, params)[0]
    eval_nll = grouped_gaussian_nll_and_gradient(eval_groups, params)[0]
    return {
        "parameters": params, "converged": converged, "iterations": iterations,
        "train_nll": train_nll, "eval_nll": eval_nll, "fit_ms": elapsed_ms,
    }


def _summary_sufficient_statistics(writer, phase, scenario_name, replicate, seed, data):
    fields = tuple((i, j) for i in range(4) for j in range(i, 4))
    for group_name, values in data.items():
        scatter = values.T @ values
        row = {
            "phase": phase, "scenario": scenario_name, "replicate": replicate,
            "seed": seed, "group": group_name, "n_events": len(values),
        }
        for i, j in fields:
            row[f"s{i}{j}_mm2"] = float(scatter[i, j])
        writer.writerow(row)


def _wilson(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return {"lower": None, "upper": None}
    if successes == 0:
        denominator = 1.0 + z*z/total
        return {"lower": 0.0, "upper": (z*z/total) / denominator}
    if successes == total:
        denominator = 1.0 + z*z/total
        return {"lower": 1.0 / denominator, "upper": 1.0}
    p = successes / total
    denominator = 1.0 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return {"lower": max(0.0, center-half), "upper": min(1.0, center+half)}


def _calibration_key(mode, sigma):
    return f"{mode}|sigma={sigma:.1f}"


def _train_and_validation(scenario_name: str, train_seed: int, validation_seed: int,
                          calibration_seed: int | None):
    scenario = SCENARIOS[scenario_name]
    mode, sigma = scenario["background_mode"], scenario["sigma_hit_mm"]
    train = simulate_target(scenario_name, train_seed)
    validation = simulate_target(scenario_name, validation_seed)
    empty_train = empty_validation = None
    if mode == "empty_field_calibrated":
        empty_train = simulate_empty_field(sigma, scenario["counts"], int(calibration_seed))
    train_fit_groups_fine = build_likelihood_groups(
        train, sigma_hit_mm=sigma, mode=mode, model="fine", empty_field_data=empty_train
    )
    train_fit_groups_coarse = build_likelihood_groups(
        train, sigma_hit_mm=sigma, mode=mode, model="coarse", empty_field_data=empty_train
    )
    val_groups_fine = build_likelihood_groups(
        validation, sigma_hit_mm=sigma, mode=mode, model="fine"
    )
    val_groups_coarse = build_likelihood_groups(
        validation, sigma_hit_mm=sigma, mode=mode, model="coarse"
    )
    return train, validation, empty_train, train_fit_groups_coarse, train_fit_groups_fine, val_groups_coarse, val_groups_fine


def _fit_trial(train_coarse, train_fine, val_coarse, val_fine, mode):
    coarse = _fit_and_score(train_coarse, val_coarse, "coarse", mode)
    fine = _fit_and_score(train_fine, val_fine, "fine", mode)
    converged = coarse["converged"] and fine["converged"]
    lrt = max(0.0, 2.0 * (coarse["train_nll"] - fine["train_nll"])) if converged else math.nan
    return coarse, fine, lrt, converged


def _calibrate(calibration_writer, sufficient_writer, calibration_replicates):
    keys = (
        ("known_fixed", 0.3),
        ("empty_field_calibrated", 0.3),
        ("known_fixed", 1.0),
        ("algebraic_confounded", 0.3),
    )
    thresholds, raw = {}, {}
    for stratum_index, (mode, sigma) in enumerate(keys):
        lrt_values = []
        scenario_name = {
            "known_fixed": "uniform_known_background",
            "empty_field_calibrated": "strong_contrast_calibrated_background",
            "algebraic_confounded": "algebraic_background_contrast_degeneracy",
        }[mode]
        scenario = SCENARIOS[scenario_name]
        # Calibration always uses equal child levels, independent of evaluation labels.
        for replicate in range(calibration_replicates):
            seed = BASE_CALIBRATION_SEED + stratum_index * 1_000_000 + replicate
            rng_seed = seed
            target = simulate_target(
                scenario_name, rng_seed,
                child_levels_override=(BASELINE_LEVEL, BASELINE_LEVEL),
                sigma_hit_mm_override=sigma,
            )
            empty = None
            empty_seed = seed + 500_000_000
            if mode == "empty_field_calibrated":
                empty = simulate_empty_field(sigma, scenario["counts"], empty_seed)
                _summary_sufficient_statistics(sufficient_writer, "calibration_empty_field", scenario_name, replicate, empty_seed, empty)
            coarse_groups = build_likelihood_groups(target, sigma_hit_mm=sigma, mode=mode,
                                                    model="coarse", empty_field_data=empty)
            fine_groups = build_likelihood_groups(target, sigma_hit_mm=sigma, mode=mode,
                                                  model="fine", empty_field_data=empty)
            coarse, fine, lrt, converged = _fit_trial(
                coarse_groups, fine_groups, coarse_groups, fine_groups, mode
            )
            # NLL evaluation here uses the same null samples only for threshold
            # calibration; these rows are never included in final performance.
            if converged:
                lrt_values.append(lrt)
            _summary_sufficient_statistics(sufficient_writer, "calibration_null", scenario_name, replicate, seed, target)
            calibration_writer.writerow({
                "stratum": _calibration_key(mode, sigma), "scenario": scenario_name,
                "replicate": replicate, "target_seed": seed, "empty_field_seed": empty_seed if empty is not None else "",
                "lrt": lrt if converged else "", "converged": converged,
                "coarse_iterations": coarse["iterations"], "fine_iterations": fine["iterations"],
            })
        if len(lrt_values) < 20:
            raise RuntimeError(f"too few converged null calibration fits for {mode}, sigma={sigma}")
        thresholds[_calibration_key(mode, sigma)] = calibrate_lrt_threshold(
            lrt_values, alpha=CALIBRATION_ALPHA
        )
        raw[_calibration_key(mode, sigma)] = lrt_values
    return thresholds, raw


def run_study(run_id: str = RUN_ID, calibration_replicates: int = CALIBRATION_REPLICATES,
              evaluation_replicates: int = EVALUATION_REPLICATES):
    if not run_id.startswith("run-") or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id
    ):
        raise ValueError("run-id must start with 'run-' and contain only letters, numbers, '-' or '_'")
    if calibration_replicates < 20 or evaluation_replicates < 1:
        raise ValueError("calibration needs >=20 null replicates; evaluation needs >=1")
    output = ROOT / "studies" / STUDY_ID / "results" / run_id
    create_output_directory(output)

    config = {
        "study_id": STUDY_ID, "run_id": run_id,
        "detector_z_mm": Z_PLANES_MM.tolist(), "momentum_gev_c": MOMENTUM_GEV_C,
        "beta": 1.0, "event_budget_per_target_sample": EVENT_BUDGET,
        "lambda_ref_mm_inv": LAMBDA_REF_MM_INV,
        "hit_error_model": "independent isotropic Gaussian x/y hit error per plane; known z geometry",
        "calibration_replicates_per_stratum": calibration_replicates,
        "evaluation_replicates_per_scenario": evaluation_replicates,
        "calibration_alpha": CALIBRATION_ALPHA, "min_design_power": MIN_DESIGN_POWER,
        "design_contrast_ref": DESIGN_CONTRAST_REF,
        "count_baseline_minimum_per_child": COUNT_MINIMUM_PER_CHILD,
        "threshold_calibration": "empirical higher 95th percentile of development null likelihood-ratio values, stratified by background mode and sigma; disjoint seeds",
        "scenarios": SCENARIOS,
        "background_models": {
            "known_fixed": "background covariance fixed at the declared synthetic value",
            "empty_field_calibrated": "unknown background scale jointly fitted using independent target-free empty-field observations",
            "algebraic_confounded": "negative control: nuisance derivative exactly equals target contrast direction; not claimed to be a physical background",
        },
        "random_seed_schedule": {
            "calibration_base": BASE_CALIBRATION_SEED,
            "evaluation_base": BASE_EVALUATION_SEED,
            "calibration_stratum_stride": 1_000_000,
            "empty_field_offset": 500_000_000,
            "evaluation_replicate_stride": 10,
            "evaluation_seed_offsets": {"train": 0, "validation": 1, "empty_field": 2},
        },
        "parameterization": "dimensionless child scales theta=lambda/lambda_ref; parent mean=(theta_L+theta_R)/2; signed contrast delta=theta_R-theta_L; densities must be nonnegative",
        "comparators": ["fixed_coarse", "fixed_fine", "event_count_only", "information_evidence_adaptive"],
        "truth_mask_or_oracle_used_for_inference": False,
        "geant4_invoked": False,
        "scope_note": "small synthetic prototype; not SCI independent validation or physical Geant4 evidence",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sufficient_path = output / "sample_covariance_sufficient_statistics.csv"
    calibration_path = output / "calibration_null_lrt.csv"
    metrics_path = output / "trial_metrics.csv"
    with sufficient_path.open("w", newline="", encoding="utf-8") as sufficient_file, \
         calibration_path.open("w", newline="", encoding="utf-8") as calibration_file:
        suff_fields = ["phase", "scenario", "replicate", "seed", "group", "n_events"] + [f"s{i}{j}_mm2" for i in range(4) for j in range(i, 4)]
        sufficient_writer = csv.DictWriter(sufficient_file, fieldnames=suff_fields, lineterminator="\n")
        calibration_writer = csv.DictWriter(calibration_file, fieldnames=[
            "stratum", "scenario", "replicate", "target_seed", "empty_field_seed", "lrt", "converged", "coarse_iterations", "fine_iterations"
        ], lineterminator="\n")
        sufficient_writer.writeheader(); calibration_writer.writeheader()
        thresholds, calibration_values = _calibrate(
            calibration_writer, sufficient_writer, calibration_replicates
        )

    metric_rows = []
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file, \
         sufficient_path.open("a", newline="", encoding="utf-8") as sufficient_file:
        fields = [
            "scenario", "replicate", "train_seed", "validation_seed", "empty_field_seed",
            "target_split_truth_evaluation_only", "effect_meets_design_minimum_evaluation_only",
            "background_mode", "sigma_hit_mm", "n_left", "n_right", "n_other",
            "contrast_estimable", "contrast_variance_ref2", "predicted_power",
            "lrt_train", "lrt_threshold", "adaptive_split", "adaptive_reason",
            "count_split", "fixed_coarse_split", "fixed_fine_split",
            "coarse_converged", "fine_converged", "coarse_iterations", "fine_iterations",
            "coarse_eval_nll", "fine_eval_nll", "adaptive_eval_nll", "count_eval_nll",
            "coarse_fit_ms", "fine_fit_ms", "failure",
        ]
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=fields, lineterminator="\n")
        sufficient_writer = csv.DictWriter(sufficient_file, fieldnames=suff_fields, lineterminator="\n")
        metrics_writer.writeheader()
        for scenario_index, (scenario_name, scenario) in enumerate(SCENARIOS.items()):
            mode, sigma = scenario["background_mode"], scenario["sigma_hit_mm"]
            threshold = thresholds[_calibration_key(mode, sigma)]
            for replicate in range(evaluation_replicates):
                seed_base = BASE_EVALUATION_SEED + scenario_index * 1_000_000 + replicate * 10
                train_seed, validation_seed, empty_seed = seed_base, seed_base + 1, seed_base + 2
                train, validation, empty_train, coarse_train_groups, fine_train_groups, coarse_val_groups, fine_val_groups = _train_and_validation(
                    scenario_name, train_seed, validation_seed,
                    empty_seed if mode == "empty_field_calibrated" else None,
                )
                _summary_sufficient_statistics(sufficient_writer, "evaluation_train", scenario_name, replicate, train_seed, train)
                _summary_sufficient_statistics(sufficient_writer, "evaluation_validation", scenario_name, replicate, validation_seed, validation)
                if empty_train is not None:
                    _summary_sufficient_statistics(sufficient_writer, "evaluation_empty_field", scenario_name, replicate, empty_seed, empty_train)

                coarse, fine, lrt, fits_converged = _fit_trial(
                    coarse_train_groups, fine_train_groups, coarse_val_groups, fine_val_groups, mode
                )
                info = contrast_report_for_decision(fine_train_groups, mode)
                adaptive = decide_split(
                    info, lrt if fits_converged else 0.0, threshold,
                    info["predicted_power_for_design_contrast"],
                    min_power=MIN_DESIGN_POWER, fit_converged=fits_converged,
                )
                counts = scenario["counts"]
                count_split = count_based_split(
                    (counts["left"], counts["right"]),
                    minimum_per_child=COUNT_MINIMUM_PER_CHILD,
                )
                adaptive_nll = fine["eval_nll"] if adaptive["split"] else coarse["eval_nll"]
                count_nll = fine["eval_nll"] if count_split else coarse["eval_nll"]
                failure = "" if fits_converged else "coarse_or_fine_fit_nonconvergence"
                row = {
                    "scenario": scenario_name, "replicate": replicate,
                    "train_seed": train_seed, "validation_seed": validation_seed,
                    "empty_field_seed": empty_seed if empty_train is not None else "",
                    "target_split_truth_evaluation_only": scenario["target_split"],
                    "effect_meets_design_minimum_evaluation_only": scenario["effect_meets_design_minimum"],
                    "background_mode": mode, "sigma_hit_mm": sigma,
                    "n_left": counts["left"], "n_right": counts["right"], "n_other": counts["other"],
                    "contrast_estimable": info["estimable"],
                    "contrast_variance_ref2": info["variance"] if info["estimable"] else "inf",
                    "predicted_power": info["predicted_power_for_design_contrast"],
                    "lrt_train": lrt if fits_converged else "", "lrt_threshold": threshold,
                    "adaptive_split": adaptive["split"], "adaptive_reason": adaptive["reason"],
                    "count_split": count_split, "fixed_coarse_split": False, "fixed_fine_split": True,
                    "coarse_converged": coarse["converged"], "fine_converged": fine["converged"],
                    "coarse_iterations": coarse["iterations"], "fine_iterations": fine["iterations"],
                    "coarse_eval_nll": coarse["eval_nll"], "fine_eval_nll": fine["eval_nll"],
                    "adaptive_eval_nll": adaptive_nll, "count_eval_nll": count_nll,
                    "coarse_fit_ms": coarse["fit_ms"], "fine_fit_ms": fine["fit_ms"],
                    "failure": failure,
                }
                metrics_writer.writerow(row)
                metric_rows.append(row)

    summaries = []
    for scenario_name, scenario in SCENARIOS.items():
        rows = [row for row in metric_rows if row["scenario"] == scenario_name]
        decisions = [bool(row["adaptive_split"]) for row in rows]
        count_decisions = [bool(row["count_split"]) for row in rows]
        n = len(rows)
        correct = None if scenario["target_split"] is None else sum(
            decision == scenario["target_split"] for decision in decisions
        )
        relevant_correct = None if scenario["effect_meets_design_minimum"] is None else sum(
            decision == scenario["effect_meets_design_minimum"] for decision in decisions
        )
        summaries.append({
            "scenario": scenario_name,
            "replicates": n,
            "target_split_truth_for_evaluation_only": scenario["target_split"],
            "effect_meets_design_minimum_for_evaluation_only": scenario["effect_meets_design_minimum"],
            "adaptive_split_count": sum(decisions),
            "adaptive_split_rate": sum(decisions) / n,
            "adaptive_split_rate_wilson_95": _wilson(sum(decisions), n),
            "adaptive_correct_decision_rate": None if correct is None else correct / n,
            "adaptive_correct_decision_rate_at_declared_minimum_effect": (
                None if relevant_correct is None else relevant_correct / n
            ),
            "adaptive_reason_counts": {
                reason: sum(row["adaptive_reason"] == reason for row in rows)
                for reason in sorted({row["adaptive_reason"] for row in rows})
            },
            "count_only_split_rate": sum(count_decisions) / n,
            "fixed_coarse_split_rate": 0.0,
            "fixed_fine_split_rate": 1.0,
            "mean_validation_nll_coarse": float(np.mean([float(row["coarse_eval_nll"]) for row in rows])),
            "mean_validation_nll_fine": float(np.mean([float(row["fine_eval_nll"]) for row in rows])),
            "mean_validation_nll_adaptive": float(np.mean([float(row["adaptive_eval_nll"]) for row in rows])),
            "mean_validation_nll_count_only": float(np.mean([float(row["count_eval_nll"]) for row in rows])),
            "mean_fit_ms_coarse": float(np.mean([float(row["coarse_fit_ms"]) for row in rows])),
            "mean_fit_ms_fine": float(np.mean([float(row["fine_fit_ms"]) for row in rows])),
            "fit_failure_count": sum(bool(row["failure"]) for row in rows),
            "unidentifiable_count": sum(not bool(row["contrast_estimable"]) for row in rows),
        })

    summary = {
        "study_id": STUDY_ID, "run_id": run_id,
        "calibration_thresholds_by_stratum": thresholds,
        "calibration_converged_counts": {key: len(values) for key, values in calibration_values.items()},
        "scenarios": summaries,
        "interpretation_limit": "All rates and predictive scores are synthetic prototype/development evidence. Truth labels are evaluated only after inference decisions. No Geant4 or independent SCI validation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    script_path = Path(__file__).resolve()
    source_paths = [script_path, ROOT / "src/sci_method/forward_model.py",
                    ROOT / "src/sci_method/identifiability.py",
                    ROOT / "src/sci_method/split_merge.py",
                    ROOT / "src/sci_method/study_io.py"]
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    source_hashes = {}
    for path in source_paths:
        name = "run_adaptive_split_merge.py" if path == script_path else path.name
        shutil.copyfile(path, snapshots / name)
        source_hashes[str(path.relative_to(ROOT.parent))] = sha256_file(path)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent, text=True)
        code_diff = subprocess.check_output(["git", "diff", "HEAD", "--", *[str(p.relative_to(ROOT.parent)) for p in source_paths]], cwd=ROOT.parent)
        diff_sha = hashlib.sha256(code_diff).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, status, diff_sha = "unknown", "unknown", "unknown"
    result_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "evidence.json"
    }
    evidence = {
        "study_id": STUDY_ID, "run_id": run_id,
        "source_commit_before_run": commit,
        "source_worktree_clean_before_run": not bool(status.strip()),
        "source_status_before_run": status.splitlines(),
        "uncommitted_source_diff_sha256": diff_sha,
        "source_file_sha256": source_hashes,
        "source_snapshot_directory": "source_snapshot/",
        "config_sha256": sha256_file(output / "config.json"),
        "result_file_sha256": result_hashes,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "command": f"python3 sci_method/studies/run_adaptive_split_merge.py --run-id {run_id}",
        "geant4_invoked": False,
        "independent_evaluation_seeds": True,
        "calibration_seeds_disjoint_from_evaluation": True,
        "inference_truth_isolation": "scenario truth is accessed only by synthetic generator and post-decision evaluator; decision functions receive local likelihood/Fisher/evidence only",
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, summary, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--calibration-replicates", type=int, default=CALIBRATION_REPLICATES)
    parser.add_argument("--evaluation-replicates", type=int, default=EVALUATION_REPLICATES)
    args = parser.parse_args()
    output, summary, _ = run_study(args.run_id, args.calibration_replicates, args.evaluation_replicates)
    print(f"Study output: {output}")
    for row in summary["scenarios"]:
        print(row["scenario"], "adaptive split rate", row["adaptive_split_rate"],
              "unidentifiable", row["unidentifiable_count"], "fit failures", row["fit_failure_count"])


if __name__ == "__main__":
    main()
