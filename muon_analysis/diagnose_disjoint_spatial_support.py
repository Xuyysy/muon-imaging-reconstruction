"""固定输入事件预算的两半空间共同占用审计；只用于已见开发场景。

运行前请读 B1/DISJOINT_SPATIAL_SUPPORT_PLAN.md。真值只在观测计数完成后
分层，不参与事件筛选、共同占用判定或任何候选重建。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_split_confidence import ROOT, SCENES
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm,
)


OUTPUT = ROOT / "muon_analysis" / "disjoint_spatial_support_results"
COUNTS = (1000, 5000)
SIGMAS_MM = (0.05, 0.30)
GRID_SIZES = (16, 8)  # 16 为主条件；8 只作更粗尺度的覆盖参照。
REPEATS = 5
SEED_OFFSET = 7021
PLAN = "B1/DISJOINT_SPATIAL_SUPPORT_PLAN.md"


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth_cell_type(truth32: np.ndarray, bins: int) -> np.ndarray:
    """仿真真值只用于事后分层；0/1/2 分别为背景/混合边界/内部。"""
    if truth32.shape != (32, 32) or 32 % bins:
        raise ValueError("Truth mask must align with the 32x32 XY grid")
    factor = 32 // bins
    target_pixels = truth32.reshape(bins, factor, bins, factor).sum(axis=(1, 3))
    return np.where(target_pixels == 0, 0,
                    np.where(target_pixels == factor * factor, 2, 1))


def support_from_positions(xy_a: np.ndarray, xy_b: np.ndarray, bins: int,
                           types: np.ndarray) -> dict:
    """共同占用只由两组观测 POCA 坐标决定，真值仅在最后数类型。"""
    if types.shape != (bins, bins):
        raise ValueError("Coarse truth type grid mismatch")
    for points in (xy_a, xy_b):
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Expected finite XY coordinates")
    edges = imaging_edges(bins)[:2]
    count_a = np.histogram2d(xy_a[:, 0], xy_a[:, 1], bins=edges)[0].astype(int)
    count_b = np.histogram2d(xy_b[:, 0], xy_b[:, 1], bins=edges)[0].astype(int)
    if count_a.sum() != len(xy_a) or count_b.sum() != len(xy_b):
        raise AssertionError("Selected events fell outside declared XY grid")
    union = (count_a > 0) | (count_b > 0)
    both1 = (count_a >= 1) & (count_b >= 1)
    both2 = (count_a >= 2) & (count_b >= 2)
    result = {
        "a_selected": int(len(xy_a)), "b_selected": int(len(xy_b)),
        "union_cells": int(union.sum()),
        "shared_1each": int(both1.sum()),
        "shared_2each": int(both2.sum()),
        "shared_1each_fraction_of_union": (float(both1.sum() / union.sum())
                                           if union.any() else None),
    }
    names = ("background", "mixed_boundary", "target_interior")
    for kind, name in enumerate(names):
        result[f"possible_{name}"] = int((types == kind).sum())
        result[f"union_{name}"] = int((union & (types == kind)).sum())
        if result[f"union_{name}"] > result[f"possible_{name}"]:
            raise AssertionError("Occupied count exceeds possible class cells")
    if sum(result[f"union_{name}"] for name in names) != result["union_cells"]:
        raise AssertionError("Truth types do not partition union cells")
    for prefix, shared in (("shared_1each", both1), ("shared_2each", both2)):
        for kind, name in enumerate(names):
            result[f"{prefix}_{name}"] = int((shared & (types == kind)).sum())
        if sum(result[f"{prefix}_{name}"] for name in names) != result[prefix]:
            raise AssertionError("Truth types do not partition shared cells")
    if result["shared_2each"] > result["shared_1each"]:
        raise AssertionError("Strict support exceeds weak support")
    return result


def selected_half(raw_half: pd.DataFrame, draws: dict[str, np.ndarray],
                  sigma: float, spacing: float) -> pd.DataFrame:
    measured = measured_hits(raw_half, sigma, draws)
    # 参数 0 表示不二次添加噪声；每事件只保留本次已测量的四层击中。
    observed = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
    observed["event"] = raw_half.event.to_numpy()
    return corrected_events(observed, sigma, spacing)


def aggregate(rows: list[dict]) -> dict:
    names = ("background", "mixed_boundary", "target_interior")
    count_fields = ("a_selected", "b_selected", "union_cells", "shared_1each",
                    "shared_2each") + tuple(
        f"shared_{minimum}each_{kind}"
        for minimum in (1, 2)
        for kind in names
    ) + tuple(f"{prefix}_{kind}" for prefix in ("possible", "union")
              for kind in names)
    totals = {name: sum(row[name] for row in rows) for name in count_fields}
    totals["condition_count"] = len(rows)
    totals["shared_1each_fraction_of_union"] = (
        totals["shared_1each"] / totals["union_cells"]
        if totals["union_cells"] else None
    )
    totals["shared_1each_background_fraction"] = (
        totals["shared_1each_background"] / totals["shared_1each"]
        if totals["shared_1each"] else None
    )
    for kind in names:
        totals[f"shared_1each_fraction_of_union_{kind}"] = (
            totals[f"shared_1each_{kind}"] / totals[f"union_{kind}"]
            if totals[f"union_{kind}"] else None
        )
    return totals


def plot_pooled(pooled: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    colors = {"background": "#718096", "mixed_boundary": "#ed8936",
              "target_interior": "#3b8c64"}
    for ax, bins in zip(axes, GRID_SIZES):
        rows = [row for row in pooled if row["grid_bins"] == bins]
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for name, label in (("background", "Background"),
                            ("mixed_boundary", "Mixed boundary"),
                            ("target_interior", "Target interior")):
            values = np.array([row[f"shared_1each_{name}"] / row["condition_count"]
                               for row in rows])
            ax.bar(x, values, bottom=bottom, color=colors[name], label=label)
            bottom += values
        ax.set_xticks(x, [f"N={row['event_count']}\nσ={row['sigma_mm']:.2f}"
                           for row in rows])
        ax.set_title(f"{bins}×{bins} coarse XY")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Cells seen by both disjoint halves (mean per condition)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output_dir: Path) -> dict:
    edges32 = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges32[:2]]
    conditions = []
    sources = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        path = ROOT / "muon_analysis" / csv_name
        tracks = pd.read_csv(path)
        if tracks.event.duplicated().any() or len(tracks) < max(COUNTS):
            raise ValueError(f"Invalid event IDs or insufficient tracks: {csv_name}")
        sources[scene] = {"csv": csv_name, "sha256": source_hash(path),
                          "complete_tracks": int(len(tracks))}
        truth32 = mask_fn(cx, cy)
        types = {bins: truth_cell_type(truth32, bins) for bins in GRID_SIZES}
        spacing = same_side_layer_spacing_mm(tracks)
        rng = np.random.default_rng(RANDOM_SEED + SEED_OFFSET + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                raw = tracks.iloc[rng.choice(len(tracks), count, replace=False)].reset_index(drop=True)
                # 先平分原始事件，再独立筛选；不能事后向稀疏的一半补事件。
                half = count // 2
                a_raw, b_raw = raw.iloc[:half].copy(), raw.iloc[half:].copy()
                a_ids, b_ids = set(a_raw.event), set(b_raw.event)
                if a_ids & b_ids or len(a_ids) + len(b_ids) != count:
                    raise AssertionError("Raw halves share an event or violate budget")
                draws = {f"{prefix}_{axis}": rng.normal(0, 1, count)
                         for prefix in ("up1", "up2", "down1", "down2")
                         for axis in ("x", "y")}
                a_draws = {key: value[:half] for key, value in draws.items()}
                b_draws = {key: value[half:] for key, value in draws.items()}
                for sigma in SIGMAS_MM:
                    a = selected_half(a_raw, a_draws, sigma, spacing)
                    b = selected_half(b_raw, b_draws, sigma, spacing)
                    if set(a.event) & set(b.event):
                        raise AssertionError("Selected halves share an event")
                    xy_a = a[["poca_x_mm", "poca_y_mm"]].to_numpy(dtype=float)
                    xy_b = b[["poca_x_mm", "poca_y_mm"]].to_numpy(dtype=float)
                    for bins in GRID_SIZES:
                        conditions.append({
                            "scenario": scene, "event_count": count,
                            "repeat": repeat, "sigma_mm": sigma, "grid_bins": bins,
                            "a_input_events": len(a_ids), "b_input_events": len(b_ids),
                            **support_from_positions(xy_a, xy_b, bins, types[bins]),
                        })
    expected = len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRID_SIZES)
    if len(conditions) != expected:
        raise AssertionError(f"Expected {expected} condition rows, got {len(conditions)}")
    keys = [(row["scenario"], row["event_count"], row["repeat"],
             row["sigma_mm"], row["grid_bins"]) for row in conditions]
    if len(set(keys)) != expected:
        raise AssertionError("Duplicate condition key")
    by_scene = []
    for scene in SCENES:
        for count in COUNTS:
            for sigma in SIGMAS_MM:
                for bins in GRID_SIZES:
                    rows = [row for row in conditions if row["scenario"] == scene
                            and row["event_count"] == count
                            and row["sigma_mm"] == sigma and row["grid_bins"] == bins]
                    by_scene.append({"scenario": scene, "event_count": count,
                                     "sigma_mm": sigma, "grid_bins": bins,
                                     **aggregate(rows)})
    pooled = []
    for count in COUNTS:
        for sigma in SIGMAS_MM:
            for bins in GRID_SIZES:
                rows = [row for row in conditions if row["event_count"] == count
                        and row["sigma_mm"] == sigma and row["grid_bins"] == bins]
                pooled.append({"event_count": count, "sigma_mm": sigma,
                               "grid_bins": bins, **aggregate(rows)})
    report = {
        "status": "retrospective_development_diagnosis_only",
        "plan": PLAN,
        "seed": RANDOM_SEED + SEED_OFFSET,
        "sources": sources,
        "unit": "coarse XY cell within one correlated scenario/subsample",
        "conditions": conditions,
        "by_scene": by_scene,
        "pooled": pooled,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_pooled(pooled, output_dir / "shared_cells.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output_dir)
    for row in report["pooled"]:
        print(row)


if __name__ == "__main__":
    main()
