"""Run the predeclared hand-built identifiability study; no Geant4 is invoked."""

from __future__ import annotations
import csv
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "sci_method"
sys.path.insert(0, str(PACKAGE / "src"))
from sci_method.forward_model import (  # noqa: E402
    BoxVoxel, detector_residual_covariance, scattering_covariance_per_density,
)
from sci_method.identifiability import (  # noqa: E402
    add_prior_precision, fisher_information, information_report,
)
from sci_method.study_io import create_output_directory  # noqa: E402

Z_PLANES_MM = [-400.0, -300.0, 300.0, 400.0]
SEED = 20261019
EVENTS = 200
SIGMAS_MM = (0.0, 0.3)
LAMBDA0_MM_INV = 1.0e-4
RANK_RTOL = 1.0e-10
PRIOR_DEMO_PRECISION = 1.0  # mm^2 in lambda parameter coordinates; illustrative only.
VOXELS = (
    BoxVoxel(-40, -20, -10, 10, -50, 50),
    BoxVoxel(-20, 0, -10, 10, -50, 50),
    BoxVoxel(0, 20, -10, 10, -50, 50),
    BoxVoxel(20, 40, -10, 10, -50, 50),
)
DEPTH_VOXELS = tuple(
    BoxVoxel(-10, 10, -10, 10, -50 + 25*i, -50 + 25*(i+1))
    for i in range(4)
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _report_to_json(report):
    return {
        "eigenvalues": np.asarray(report["eigenvalues"]).tolist(),
        "eigenvectors_columns": np.asarray(report["eigenvectors_columns"]).tolist(),
        "rank": report["rank"],
        "rank_tolerance": float(report["rank_tolerance"]),
        "rank_rtol": float(report["rank_rtol"]),
        "null_directions": np.asarray(report["null_directions"]).tolist(),
        "nuisance_indices": list(report["nuisance_indices"]),
        "profiled_parameter_indices": list(report["profiled_parameter_indices"]),
        "profiled_information": np.asarray(report["profiled_information"]).tolist(),
    }


def _scenario_paths(name: str):
    """Return fixed path set: balanced lateral coverage or a known blind case."""
    if name == "balanced_four_strips":
        x_values = np.repeat([-30.0, -10.0, 10.0, 30.0], EVENTS // 4)
    elif name == "single_strip_only":
        x_values = np.full(EVENTS, -30.0)
    elif name == "longitudinal_four_voxels":
        x_values = np.zeros(EVENTS)
    else:
        raise ValueError(name)
    # Vertical, parallel nominal tracks cross target depth [-50,50] mm.
    return [(float(x), 0.0, 0.0, 0.0, -400.0, 400.0) for x in x_values]


def _model_groups(n_regions: int):
    if n_regions == 1:
        return [(0, 1, 2, 3)]
    if n_regions == 2:
        return [(0, 1), (2, 3)]
    if n_regions == 4:
        return [(0,), (1,), (2,), (3,)]
    raise ValueError(n_regions)


def _run_model(paths, groups, sigma_mm, rng, scenario, voxels):
    n = len(paths)
    p = len(groups)
    detector_single = detector_residual_covariance(Z_PLANES_MM, sigma_mm)
    kernels = np.zeros((n, 4, 4, p))
    for i, path in enumerate(paths):
        for a, group in enumerate(groups):
            for region in group:
                kernels[i, :, :, a] += scattering_covariance_per_density(
                    path, voxels[region], Z_PLANES_MM, momentum_gev=4.0
                )
    density = np.full(p, LAMBDA0_MM_INV)
    covariances = np.repeat(detector_single[None], n, axis=0)
    covariances += np.einsum("abv,v->ab", kernels[0], density)[None] if n == 1 else np.einsum("iabv,v->iab", kernels, density)
    # A separate common-scatter nuisance is exactly the sum of local responses.
    nuisance_kernel = kernels.sum(axis=3, keepdims=True)
    joint_kernels = np.concatenate((kernels, nuisance_kernel), axis=3)
    joint_fisher = fisher_information(covariances, joint_kernels)
    raw_fisher = joint_fisher[:p, :p]
    profiled_report = information_report(joint_fisher, nuisance_indices=(p,), rank_rtol=RANK_RTOL)
    raw_report = information_report(raw_fisher, rank_rtol=RANK_RTOL)
    # Generate Gaussian observations using the fixed seed; Fisher is expected information.
    observations = np.vstack([rng.multivariate_normal(np.zeros(4), cov) for cov in covariances])
    return {
        "scenario": scenario,
        "sigma_hit_mm": sigma_mm,
        "n_events": n,
        "n_parameters": p,
        "groups": [list(group) for group in groups],
        "lambda0_mm_inv_per_parameter": density.tolist(),
        "observation_sample_mean_mm": observations.mean(axis=0).tolist(),
        "observation_sample_covariance_mm2": np.cov(observations, rowvar=False).tolist(),
        "data_fisher_without_nuisance": raw_fisher.tolist(),
        "data_fisher_eigensystem_without_nuisance": _report_to_json(raw_report),
        "prior_regularization_demo": {
            "prior_precision": (np.eye(p) * PRIOR_DEMO_PRECISION).tolist(),
            "data_plus_prior_eigenvalues": np.linalg.eigvalsh(
                add_prior_precision(raw_fisher, np.eye(p) * PRIOR_DEMO_PRECISION)
            ).tolist(),
            "data_fisher_eigenvalues_unchanged": raw_report["eigenvalues"].tolist(),
            "interpretation": "Illustrative prior precision only; it is not additional likelihood information and is not tuned to improve a result.",
        },
        "joint_data_fisher_with_common_scatter_nuisance": joint_fisher.tolist(),
        "profiled_local_information_after_common_nuisance": _report_to_json(profiled_report),
        "interpretation": "Fisher information is from the likelihood only; the common-scatter nuisance is profiled, not counted as a prior.",
    }, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run-003", help="new unique result folder name; existing paths are refused")
    args = parser.parse_args()
    if not args.run_id.startswith("run-") or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in args.run_id):
        raise ValueError("run-id must start with 'run-' and contain only letters, numbers, '-' or '_'")
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        source_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_commit, source_status = "unknown", "unknown"
    source_worktree_clean = not bool(source_status.strip())
    output = PACKAGE / "studies" / "forward-prototype-001" / "results" / args.run_id
    create_output_directory(output)
    config = {
        "study_id": "forward-prototype-001",
        "run_id": args.run_id,
        "detector_z_mm": Z_PLANES_MM,
        "momentum_gev_c": 4.0,
        "beta": 1.0,
        "hit_error_model": "independent Gaussian x/y per plane, common sigma; known z geometry",
        "sigmas_hit_mm": list(SIGMAS_MM),
        "event_budget_per_scenario": EVENTS,
        "rng_seed": SEED,
        "nominal_density_mm_inv": LAMBDA0_MM_INV,
        "rank_relative_tolerance": RANK_RTOL,
        "prior_precision_demo_mm2": PRIOR_DEMO_PRECISION,
        "lateral_region_bounds_mm": [[v.x_min_mm, v.x_max_mm, v.y_min_mm, v.y_max_mm,
                                       v.z_min_mm, v.z_max_mm] for v in VOXELS],
        "longitudinal_region_bounds_mm": [[v.x_min_mm, v.x_max_mm, v.y_min_mm, v.y_max_mm,
                                           v.z_min_mm, v.z_max_mm] for v in DEPTH_VOXELS],
        "scenarios": ["balanced_four_strips", "single_strip_only", "longitudinal_four_voxels"],
        "models": [1, 2, 4],
        "note": "Hand-defined paths and Gaussian observations only; not Geant4 or a natural-spectrum inference.",
    }
    config_bytes = json.dumps(config, sort_keys=True, indent=2).encode()
    (output / "config.json").write_bytes(config_bytes)
    path_rows = []
    result_rows = []
    observation_rows = []
    rng = np.random.default_rng(SEED)
    for scenario in config["scenarios"]:
        paths = _scenario_paths(scenario)
        voxels = DEPTH_VOXELS if scenario == "longitudinal_four_voxels" else VOXELS
        for i, path in enumerate(paths):
            path_rows.append({"scenario": scenario, "event_id": i, **dict(zip(
                ("x0_mm", "y0_mm", "dx_dz", "dy_dz", "z_start_mm", "z_end_mm"), path))})
        for sigma in SIGMAS_MM:
            for n_regions in (1, 2, 4):
                result, obs = _run_model(paths, _model_groups(n_regions), sigma, rng, scenario, voxels)
                result_rows.append(result)
                for event_id, resid in enumerate(obs):
                    observation_rows.append({"scenario": scenario, "sigma_hit_mm": sigma,
                                             "n_regions": n_regions, "event_id": event_id,
                                             **dict(zip(("rx_down1_mm", "rx_down2_mm", "ry_down1_mm", "ry_down2_mm"), resid))})
    with (output / "synthetic_paths.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(path_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(path_rows)
    with (output / "synthetic_observations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(observation_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(observation_rows)
    (output / "information_summary.json").write_text(json.dumps(result_rows, indent=2, sort_keys=True))
    with (output / "information_summary.csv").open("w", newline="") as f:
        fields = ("scenario", "sigma_hit_mm", "n_events", "n_parameters", "raw_data_rank",
                  "raw_fisher_eigenvalues", "profiled_local_rank", "profiled_eigenvalues")
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in result_rows:
            raw = row["data_fisher_eigensystem_without_nuisance"]
            prof = row["profiled_local_information_after_common_nuisance"]
            prof_report = information_report(np.asarray(prof["profiled_information"]), rank_rtol=RANK_RTOL)
            writer.writerow({"scenario": row["scenario"], "sigma_hit_mm": row["sigma_hit_mm"],
                             "n_events": row["n_events"], "n_parameters": row["n_parameters"],
                             "raw_data_rank": raw["rank"],
                             "raw_fisher_eigenvalues": json.dumps(raw["eigenvalues"]),
                             "profiled_local_rank": prof_report["rank"],
                             "profiled_eigenvalues": json.dumps(prof_report["eigenvalues"].tolist())})
    paths_bytes = (output / "synthetic_paths.csv").read_bytes()
    code_path = Path(__file__).resolve()
    module_paths = [code_path, PACKAGE / "src" / "sci_method" / "data_contract.py",
                    PACKAGE / "src" / "sci_method" / "forward_model.py",
                    PACKAGE / "src" / "sci_method" / "identifiability.py",
                    PACKAGE / "src" / "sci_method" / "study_io.py"]
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir()
    for source in module_paths:
        snapshot_name = "run_forward_identifiability.py" if source == code_path else source.name
        shutil.copyfile(source, source_snapshot / snapshot_name)
    evidence = {
        "study_id": config["study_id"],
        "source_git_commit_before_run": source_commit,
        "source_worktree_clean_before_run": source_worktree_clean,
        "source_tree_note": "If false, the parent commit is not a complete source revision; use the exact source snapshots and hashes below.",
        "source_script": str(code_path.relative_to(ROOT)),
        "source_script_sha256": sha256_bytes(code_path.read_bytes()),
        "source_files_sha256": {str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes())
                                for path in module_paths},
        "config_sha256": sha256_bytes(config_bytes),
        "synthetic_path_table_sha256": sha256_bytes(paths_bytes),
        "seed": SEED,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_snapshot_directory": "source_snapshot/",
        "reproduction_command": f"python3 sci_method/studies/run_forward_identifiability.py --run-id {args.run_id}-reproduction",
        "physics_simulation": False,
        "truth_mask_or_oracle_used": False,
        "outputs": ["config.json", "synthetic_paths.csv", "synthetic_observations.csv",
                    "information_summary.csv", "information_summary.json", "evidence.json",
                    "source_snapshot/"],
        "warning": "This is a hand-built Gaussian identifiability diagnostic, not a reconstruction-performance experiment.",
    }
    (output / "source_script_snapshot.py").write_bytes(code_path.read_bytes())
    evidence["outputs"].append("source_script_snapshot.py")
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
    print(output)
    for row in result_rows:
        raw = row["data_fisher_eigensystem_without_nuisance"]
        prof = row["profiled_local_information_after_common_nuisance"]
        prof_info = np.asarray(prof["profiled_information"])
        peigs = np.linalg.eigvalsh(0.5*(prof_info+prof_info.T))
        print(f"{row['scenario']} sigma={row['sigma_hit_mm']:.1f}mm p={row['n_parameters']}: "
              f"rank={raw['rank']}/{row['n_parameters']}, eig={np.array(raw['eigenvalues'])}; "
              f"profiled-rank={np.count_nonzero(peigs > max(peigs[-1],0)*RANK_RTOL) if peigs.size and peigs[-1]>0 else 0}, "
              f"profiled-eig={peigs}")


if __name__ == "__main__":
    main()
