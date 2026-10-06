"""空场负对照：固定预算的两半事件空间共同占用（不重建新图像）。

协议：B1/EMPTY_TARGET_CONTROL_PROTOCOL.md。U 足迹只在最后统计区域时使用；
空场的“U 区”只是预设的假想位置，不代表空场有目标。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_disjoint_spatial_support import (
    GRID_SIZES, REPEATS, COUNTS, SIGMAS_MM, selected_half,
    source_hash, support_from_positions, truth_cell_type,
)
from run_adaptive_voxel_experiment import imaging_edges
from run_baseline_experiment import u_target_mask
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "muon_analysis" / "empty_target_control_results"
STREAMS = ("a", "b")
SCENES = ("u", "empty")
RANDOM_SEED = 20261006


def region_summary(row: dict) -> dict:
    """把预设 U 足迹（混合+内部）与外区分开，明确每个分母。"""
    inside_names = ("mixed_boundary", "target_interior")
    inside_union = sum(row[f"union_{name}"] for name in inside_names)
    inside_shared = sum(row[f"shared_1each_{name}"] for name in inside_names)
    outside_union = row["union_background"]
    outside_shared = row["shared_1each_background"]
    if inside_union + outside_union != row["union_cells"]:
        raise AssertionError("Region counts do not partition occupied cells")
    return {
        "inside_union_cells": inside_union,
        "inside_shared_cells": inside_shared,
        "outside_union_cells": outside_union,
        "outside_shared_cells": outside_shared,
        "inside_shared_per_occupied": inside_shared / inside_union if inside_union else None,
        "outside_shared_per_occupied": outside_shared / outside_union if outside_union else None,
    }


def sum_conditions(rows: list[dict]) -> dict:
    """五次同源子抽样合并分子/分母；不计算虚假的独立样本误差条。"""
    fields = ("a_selected", "b_selected", "inside_union_cells",
              "inside_shared_cells", "outside_union_cells", "outside_shared_cells",
              "shared_2each")
    total = {field: sum(row[field] for row in rows) for field in fields}
    total["conditions"] = len(rows)
    for region in ("inside", "outside"):
        union = total[f"{region}_union_cells"]
        total[f"{region}_shared_per_occupied"] = (
            total[f"{region}_shared_cells"] / union if union else None
        )
    return total


def plot_readout(pooled: list[dict], path: Path) -> None:
    """只画两条独立 Geant4 流的主比较；不把重复抽样当误差条。"""
    rows = [row for row in pooled if row["event_count"] == 5000
            and row["sigma_mm"] == 0.30 and row["grid_bins"] == 16]
    if len(rows) != 4:
        raise AssertionError("Main comparison must have U/empty on two streams")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharey=True)
    for axis, region in zip(axes, ("inside", "outside")):
        for i, stream in enumerate(STREAMS):
            current = {r["scene"]: r for r in rows if r["stream"] == stream}
            values = [current[scene][f"{region}_shared_per_occupied"] for scene in SCENES]
            if any(value is None for value in values):
                continue
            axis.bar(np.array([0, 1]) + (i - .5) * .34, values,
                     width=.32, label=f"stream {stream.upper()}")
        axis.set_xticks([0, 1], ["U target", "Empty control"])
        axis.set_title("Prespecified U footprint" if region == "inside"
                       else "Outside footprint")
        axis.grid(axis="y", alpha=.22)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Shared cells / cells occupied by either half")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output_dir: Path) -> dict:
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = u_target_mask(cx, cy)
    types = {bins: truth_cell_type(truth, bins) for bins in GRID_SIZES}
    sources = {}
    conditions = []
    for stream in STREAMS:
        for scene in SCENES:
            csv_name = f"muon_tracks_control_{scene}_{stream}.csv"
            path = ROOT / "muon_analysis" / csv_name
            tracks = pd.read_csv(path)
            if len(tracks) < max(COUNTS) or tracks.event.duplicated().any():
                raise ValueError(f"Invalid event IDs or insufficient tracks: {path}")
            sources[f"{stream}_{scene}"] = {
                "csv": csv_name, "sha256": source_hash(path),
                "complete_tracks": len(tracks),
            }
            spacing = same_side_layer_spacing_mm(tracks)
            # 两种场景使用相同编号的分析种子，但各自独立抽自己的母流。
            # Geant4 几何改变会使随机消耗分叉，不能声称是事件逐条配对。
            rng = np.random.default_rng(RANDOM_SEED + 1000 * STREAMS.index(stream))
            for count in COUNTS:
                for repeat in range(1, REPEATS + 1):
                    raw = tracks.iloc[rng.choice(len(tracks), count, replace=False)].reset_index(drop=True)
                    half = count // 2
                    a_raw, b_raw = raw.iloc[:half].copy(), raw.iloc[half:].copy()
                    if set(a_raw.event) & set(b_raw.event):
                        raise AssertionError("The two raw halves share an event")
                    draws = {f"{prefix}_{axis}": rng.normal(size=count)
                             for prefix in ("up1", "up2", "down1", "down2")
                             for axis in ("x", "y")}
                    for sigma in SIGMAS_MM:
                        a = selected_half(a_raw, {k: v[:half] for k, v in draws.items()},
                                          sigma, spacing)
                        b = selected_half(b_raw, {k: v[half:] for k, v in draws.items()},
                                          sigma, spacing)
                        if set(a.event) & set(b.event):
                            raise AssertionError("Selected halves share an event")
                        xy_a = a[["poca_x_mm", "poca_y_mm"]].to_numpy(dtype=float)
                        xy_b = b[["poca_x_mm", "poca_y_mm"]].to_numpy(dtype=float)
                        for bins in GRID_SIZES:
                            summary = support_from_positions(xy_a, xy_b, bins, types[bins])
                            conditions.append({
                                "stream": stream, "scene": scene, "event_count": count,
                                "repeat": repeat, "sigma_mm": sigma, "grid_bins": bins,
                                "input_per_half": half, **summary,
                                **region_summary(summary),
                            })
    expected = len(STREAMS) * len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRID_SIZES)
    if len(conditions) != expected:
        raise AssertionError(f"Expected {expected} conditions, got {len(conditions)}")
    keys = [(r["stream"], r["scene"], r["event_count"], r["repeat"],
             r["sigma_mm"], r["grid_bins"]) for r in conditions]
    if len(set(keys)) != expected:
        raise AssertionError("Duplicate condition key")
    pooled = []
    for stream in STREAMS:
        for scene in SCENES:
            for count in COUNTS:
                for sigma in SIGMAS_MM:
                    for bins in GRID_SIZES:
                        rows = [r for r in conditions if r["stream"] == stream
                                and r["scene"] == scene and r["event_count"] == count
                                and r["sigma_mm"] == sigma and r["grid_bins"] == bins]
                        pooled.append({"stream": stream, "scene": scene,
                                       "event_count": count, "sigma_mm": sigma,
                                       "grid_bins": bins, **sum_conditions(rows)})
    deltas = []
    for stream in STREAMS:
        for count in COUNTS:
            for sigma in SIGMAS_MM:
                for bins in GRID_SIZES:
                    pair = {r["scene"]: r for r in pooled if r["stream"] == stream
                            and r["event_count"] == count and r["sigma_mm"] == sigma
                            and r["grid_bins"] == bins}
                    if any(pair[s]["inside_shared_per_occupied"] is None for s in SCENES):
                        raise AssertionError("Missing main comparison denominator")
                    deltas.append({"stream": stream, "event_count": count,
                                   "sigma_mm": sigma, "grid_bins": bins,
                                   "inside_ratio_u_minus_empty":
                                       pair["u"]["inside_shared_per_occupied"]
                                       - pair["empty"]["inside_shared_per_occupied"]})
    main = [r for r in deltas if r["event_count"] == 5000
            and r["sigma_mm"] == .30 and r["grid_bins"] == 16]
    report = {
        "protocol": "B1/EMPTY_TARGET_CONTROL_PROTOCOL.md",
        "status": "negative_control_not_a_new_reconstruction_method",
        "analysis_seed": RANDOM_SEED, "sources": sources,
        "unit": "coarse XY cell within one correlated subsample",
        "conditions": conditions, "pooled": pooled, "deltas": deltas,
        "main_comparison": main,
        "main_direction_consistent": all(r["inside_ratio_u_minus_empty"] > 0 for r in main),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_readout(pooled, output_dir / "negative_control.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output_dir)
    print("Main comparison:")
    for row in report["main_comparison"]:
        print(row)
    print("Both streams positive:", report["main_direction_consistent"])


if __name__ == "__main__":
    main()
