"""只读审计四层缪子事件 CSV 的观测通道和派生列。

方法见 B1/OBSERVABLE_EVENT_SCHEMA_AUDIT_PLAN.md。脚本不使用仿真目标真值，
也不做成像质量或边界分类检验。结果只是一份可复现的数据结构核查。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "observable_event_schema_results"
DEVELOPMENT_FILES = (
    "muon_tracks.csv",
    "muon_tracks_l_shape.csv",
    "muon_tracks_double_block.csv",
)
HIT_PREFIXES = ("up1", "up2", "down1", "down2")
HIT_COLUMNS = tuple(f"{plane}_{axis}_mm" for plane in HIT_PREFIXES
                    for axis in ("x", "y", "z"))
ANGLE_COLUMNS = ("theta_x_mrad", "theta_y_mrad", "theta_mrad")
POCA_COLUMNS = ("poca_x_mm", "poca_y_mm", "poca_z_mm")
EXPECTED_COLUMNS = ("event",) + HIT_COLUMNS + POCA_COLUMNS + ANGLE_COLUMNS


def file_hash(path: Path) -> str:
    """记录来源字节哈希，防止以后误把别的版本当作同一份实验。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        return tuple(next(csv.reader(stream)))


def angles_from_hits(frame: pd.DataFrame) -> np.ndarray:
    """独立复算 C++ 中的两个投影角和合角；它们不是新探测通道。"""
    vectors = []
    for first, second in (("up1", "up2"), ("down1", "down2")):
        delta = np.column_stack([
            frame[f"{second}_{axis}_mm"].to_numpy(dtype=float)
            - frame[f"{first}_{axis}_mm"].to_numpy(dtype=float)
            for axis in ("x", "y", "z")
        ])
        vectors.append(delta)
    incoming, outgoing = vectors
    x = (np.arctan2(outgoing[:, 0], outgoing[:, 2])
         - np.arctan2(incoming[:, 0], incoming[:, 2])) * 1000.0
    y = (np.arctan2(outgoing[:, 1], outgoing[:, 2])
         - np.arctan2(incoming[:, 1], incoming[:, 2])) * 1000.0
    return np.column_stack((x, y, np.hypot(x, y)))


def audit_one(path: Path) -> dict:
    frame = pd.read_csv(path)
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected field order: {path.name}")
    if frame.empty:
        raise ValueError(f"Empty CSV: {path.name}")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    nonfinite = ~np.isfinite(numeric.to_numpy(dtype=float))
    if bool(nonfinite.any()):
        raise ValueError(f"Missing/non-finite numeric fields: {path.name}")
    if frame.event.duplicated().any():
        raise ValueError(f"Duplicate event id: {path.name}")
    actual = frame[list(ANGLE_COLUMNS)].to_numpy(dtype=float)
    recomputed = angles_from_hits(frame)
    difference = np.abs(actual - recomputed)
    z_values = {
        f"{plane}_z_mm": sorted(float(v) for v in frame[f"{plane}_z_mm"].unique())
        for plane in HIT_PREFIXES
    }
    return {
        "file": path.name,
        "sha256": file_hash(path),
        "complete_tracks": int(len(frame)),
        "missing_or_nonfinite_cells": int(nonfinite.sum()),
        "duplicate_event_ids": int(frame.event.duplicated().sum()),
        "z_values_mm": z_values,
        "angle_max_abs_difference_mrad": {
            key: float(difference[:, index].max())
            for index, key in enumerate(ANGLE_COLUMNS)
        },
        "angle_p99_abs_difference_mrad": {
            key: float(np.percentile(difference[:, index], 99))
            for index, key in enumerate(ANGLE_COLUMNS)
        },
    }


def run(output_dir: Path) -> dict:
    all_files = sorted(ROOT.glob("muon_tracks*.csv"))
    if not all_files:
        raise FileNotFoundError("No archived muon track CSV files")
    inconsistent = [path.name for path in all_files
                    if header(path) != EXPECTED_COLUMNS]
    if inconsistent:
        raise ValueError(f"Schema differs in archived files: {inconsistent}")
    records = [audit_one(ROOT / name) for name in DEVELOPMENT_FILES]
    report = {
        "status": "retrospective_schema_audit_only",
        "plan": "B1/OBSERVABLE_EVENT_SCHEMA_AUDIT_PLAN.md",
        "header_consistency_checked_files": len(all_files),
        "header_consistency": "all_match_cpp_output_header",
        "columns": list(EXPECTED_COLUMNS),
        "field_groups": {
            "event_identifier": ["event"],
            "recorded_plane_entry_positions": list(HIT_COLUMNS),
            "deterministic_derived_poca": list(POCA_COLUMNS),
            "deterministic_derived_angles": list(ANGLE_COLUMNS),
            "independently_measured_momentum_or_energy": [],
            "redundant_same_side_hits_or_fit_residual": [],
            "per_event_resolution_or_uncertainty": [],
            "per_plane_deposited_energy_or_time": [],
        },
        "development_files": records,
        "interpretation": (
            "Two recorded positions per side define the line exactly. "
            "Angle and PoCA columns are computed from those hits, not additional "
            "measurements. This does not rule out cross-event spatial evidence."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output_dir)
    print("matching_headers:", report["header_consistency_checked_files"])
    for row in report["development_files"]:
        print(row["file"], "tracks:", row["complete_tracks"],
              "angle max error (mrad):", row["angle_max_abs_difference_mrad"])


if __name__ == "__main__":
    main()
