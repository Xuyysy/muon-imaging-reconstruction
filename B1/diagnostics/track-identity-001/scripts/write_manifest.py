#!/usr/bin/env python3
"""Write source and result hashes for a completed track-identity audit."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


SOURCE_PATHS = (
    "B1/CMakeLists.txt",
    "B1/exampleB1.cc",
    "B1/include/EventAction.hh",
    "B1/src/EventAction.cc",
    "B1/src/SteppingAction.cc",
    "B1/src/ActionInitialization.cc",
    "B1/src/PrimaryGeneratorAction.cc",
    "B1/src/DetectorConstruction.cc",
    "B1/src/RunAction.cc",
    "B1/diagnostics/track-identity-001/run_1000.mac",
    "B1/diagnostics/track-identity-001/run_10000.mac",
    "B1/diagnostics/track-identity-001/run_audit.sh",
    "B1/diagnostics/track-identity-001/README.md",
    "B1/diagnostics/track-identity-001/AUDIT_RESULTS.md",
    "B1/diagnostics/track-identity-001/scripts/summarize_identity.py",
    "B1/diagnostics/track-identity-001/scripts/validate_run.py",
    "B1/diagnostics/track-identity-001/scripts/write_manifest.py",
    "B1/diagnostics/track-identity-001/tests/test_summary.py",
    "B1/diagnostics/track-identity-001/tests/test_validate_run.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--script-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    cache = (args.build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
    match = re.search(r"^Geant4_DIR:[^=]*=(.+)$", cache, re.MULTILINE)
    if not match:
        raise SystemExit("Geant4_DIR not found in the build CMakeCache")
    geant4_dir = Path(match.group(1).strip())
    version_text = (geant4_dir / "Geant4ConfigVersion.cmake").read_text(encoding="utf-8")
    version_match = re.search(r'set\(PACKAGE_VERSION "([^"]+)"\)', version_text)
    if not version_match:
        raise SystemExit("Could not read Geant4 version from Geant4ConfigVersion.cmake")

    source_hashes = {
        relative: sha256(args.repo / relative) for relative in SOURCE_PATHS
    }
    previous_manifest_path = args.results_dir / "configuration.json"
    previous_manifest = (
        json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        if previous_manifest_path.exists() else {}
    )
    original_manifest_path = args.results_dir / "geant4_run_manifest.json"
    if previous_manifest and not original_manifest_path.exists():
        original_manifest_path.write_text(
            json.dumps(previous_manifest, indent=2) + "\n", encoding="utf-8"
        )
    result_hashes = {
        path.relative_to(args.results_dir).as_posix(): sha256(path)
        for path in sorted(args.results_dir.rglob("*"))
        if path.is_file() and path.name != "configuration.json"
    }
    source_commit = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "purpose": "Geant4 track-identity data-reliability audit only; not SCI algorithm validation",
        "geant4_source_commit": previous_manifest.get("source_commit", source_commit),
        "geant4_source_file_sha256": previous_manifest.get("source_file_sha256", source_hashes),
        "analysis_source_commit": source_commit,
        "analysis_source_file_sha256": source_hashes,
        "geant4_version": version_match.group(1),
        "physics_list": "QBBC (configured in B1/exampleB1.cc)",
        "source_mode": "Ideal fixed 4 GeV mu- parallel to +z (PrimaryGeneratorAction.cc)",
        "target_scenario": "centered (existing default U-shaped aluminium target)",
        "detector_geometry": "existing four-layer geometry; unchanged",
        "runs": [
            {"run_id": "run-1000", "events": 1000, "seed_pair": [318527, 742913]},
            {"run_id": "run-10000", "events": 10000, "seed_pair": [915731, 460297]},
        ],
        "event_ids_are_run_local": True,
        "reconstruction_input": "current_rule_tracks.csv only; identity diagnostic fields are separate",
        "configure_command": "cmake -S B1 -B B1/build-track-identity-audit -DGeant4_DIR=/Users/xuyaoyang/geant4-install/lib/cmake/Geant4 -DWITH_GEANT4_UIVIS=OFF",
        "build_command": "cmake --build B1/build-track-identity-audit --parallel 4",
        "run_command": "bash B1/diagnostics/track-identity-001/run_audit.sh",
        "results_directory": str(args.results_dir),
        "result_file_sha256": result_hashes,
    }
    target = args.results_dir / "configuration.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote provenance manifest: {target}")
    print(f"Geant4 source commit: {manifest['geant4_source_commit']}")
    print(f"Analysis source commit: {source_commit}")
    print(f"Hashed {len(source_hashes)} source/config files and {len(result_hashes)} result files.")


if __name__ == "__main__":
    main()
