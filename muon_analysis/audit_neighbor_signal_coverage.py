"""审计已见 Geant4 档案：稀疏粗格的邻格与本格留出证据覆盖率。

本脚本只读取既有逐格 CSV，不读取目标真值列，不改变任何重建算法。
请先阅读 B1/NEIGHBOR_SIGNAL_COVERAGE_PLAN.md；输出为事后开发诊断。
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


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "single_cell_refinement_signal_results"
DEFAULT_OUTPUT = ROOT / "neighbor_signal_coverage_results"
CONDITION = ["scenario", "event_count", "repeat", "position_sigma_mm"]
CELL_KEY = CONDITION + ["ix", "iy"]
SCENES = ("u_center", "l_shape", "double_block")
COUNTS = (1000, 5000)
SIGMAS = (0.05, 0.30)


def best_neighbor(counts: dict[tuple[int, int], int], ix: int, iy: int,
                  local_count: int) -> tuple[int, int, int] | None:
    """四邻接中选计数最多的 B；只读观测计数，不用仿真真值。"""
    candidates = []
    for x, y in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
        neighbor_count = counts.get((x, y), 0)
        if neighbor_count >= max(4, local_count + 1):
            candidates.append((neighbor_count, x, y))
    if not candidates:
        return None
    n, x, y = sorted(candidates, key=lambda row: (-row[0], row[1], row[2]))[0]
    return (x, y, n)


def summarize_condition(cells: pd.DataFrame) -> dict:
    """独立计算一组条件；缺失的分半分数不能被误算成零。"""
    counts = {(int(row.ix), int(row.iy)): int(row.cell_event_count)
              for row in cells.itertuples(index=False)}
    if len(counts) != len(cells):
        raise ValueError("Duplicate occupied XY cell")
    output = {"occupied_cells": len(cells), "sparse_cells": 0,
              "neighbor_supported": 0, "local_score_available": 0,
              "joint_coverage": 0, "event_count_bands": {}}
    for name in ("1", "2-3", "4-7"):
        output["event_count_bands"][name] = {
            "sparse_cells": 0, "neighbor_supported": 0,
            "local_score_available": 0, "joint_coverage": 0}
    for row in cells.itertuples(index=False):
        n = int(row.cell_event_count)
        if not 1 <= n <= 7:
            continue
        band = "1" if n == 1 else "2-3" if n <= 3 else "4-7"
        neighbor = best_neighbor(counts, int(row.ix), int(row.iy), n) is not None
        score = bool(np.isfinite(row.heldout_log_gain))
        for tally in (output, output["event_count_bands"][band]):
            tally["sparse_cells"] += 1
            tally["neighbor_supported"] += int(neighbor)
            tally["local_score_available"] += int(score)
            tally["joint_coverage"] += int(neighbor and score)
    if output["joint_coverage"] > min(output["neighbor_supported"],
                                      output["local_score_available"]):
        raise AssertionError("Joint coverage exceeds a marginal count")
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path) -> dict:
    cells_path, repeats_path = SOURCE / "cells.csv", SOURCE / "by_repeat.csv"
    # 原档案混有真值诊断列；显式列选择防止后续误把真值引入选择规则。
    cells = pd.read_csv(cells_path, usecols=CELL_KEY +
                        ["cell_event_count", "heldout_log_gain"])
    repeats = pd.read_csv(repeats_path, usecols=CONDITION + ["occupied_cells"])
    if cells[CELL_KEY].isna().any().any() or cells.duplicated(CELL_KEY).any():
        raise ValueError("Missing or duplicate cell key")
    if repeats[CONDITION].isna().any().any() or repeats.duplicated(CONDITION).any():
        raise ValueError("Missing or duplicate condition key")
    if ((cells.ix < 0) | (cells.ix > 15) | (cells.iy < 0) | (cells.iy > 15) |
            (cells.cell_event_count <= 0)).any():
        raise ValueError("Invalid grid coordinates or cell count")
    recorded = repeats.set_index(CONDITION).occupied_cells.to_dict()
    conditions = []
    for key, subset in cells.groupby(CONDITION, sort=True):
        if key not in recorded or len(subset) != int(recorded[key]):
            raise ValueError(f"Occupied-cell count mismatch: {key}")
        scene, n, repeat, sigma = key
        if scene not in SCENES or n not in COUNTS or sigma not in SIGMAS:
            continue
        scene, n, repeat, sigma = key
        conditions.append({"scenario": str(scene), "event_count": int(n),
                           "repeat": int(repeat), "position_sigma_mm": float(sigma),
                           **summarize_condition(subset)})
    expected = {(scene, n, repeat, sigma) for scene in SCENES
                for n in COUNTS for repeat in range(1, 6) for sigma in SIGMAS}
    actual = {tuple(row[field] for field in CONDITION) for row in conditions}
    if actual != expected:
        raise ValueError(f"Expected {len(expected)} conditions; got {len(actual)}")
    metrics = ("sparse_cells", "neighbor_supported", "local_score_available",
               "joint_coverage")

    def aggregate(rows: list[dict]) -> dict:
        total = {key: sum(row[key] for row in rows) for key in metrics}
        # 没有该类格时，比例不可定义；不能把缺失写成“0% 覆盖”。
        denominator = total["sparse_cells"]
        total["neighbor_rate"] = (total["neighbor_supported"] / denominator
                                  if denominator else None)
        total["local_score_rate"] = (total["local_score_available"] / denominator
                                     if denominator else None)
        total["joint_rate"] = (total["joint_coverage"] / denominator
                               if denominator else None)
        return total

    by_scene = [{"scenario": scene, "event_count": n, "position_sigma_mm": sigma,
                 "condition_count": 5,
                 **aggregate([row for row in conditions if row["scenario"] == scene
                              and row["event_count"] == n
                              and row["position_sigma_mm"] == sigma])}
                for scene in SCENES for n in COUNTS for sigma in SIGMAS]
    pooled = [{"event_count": n, "position_sigma_mm": sigma,
               "condition_count": 15,
               **aggregate([row for row in conditions if row["event_count"] == n
                            and row["position_sigma_mm"] == sigma])}
              for n in COUNTS for sigma in SIGMAS]
    bands = {}
    for n in COUNTS:
        for sigma in SIGMAS:
            relevant = [row for row in conditions if row["event_count"] == n
                        and row["position_sigma_mm"] == sigma]
            bands[f"N={n},sigma={sigma:.2f}"] = {
                band: aggregate([row["event_count_bands"][band] for row in relevant])
                for band in ("1", "2-3", "4-7")}
    result = {
        "status": "retrospective_development_archive_only",
        "protocol": "B1/NEIGHBOR_SIGNAL_COVERAGE_PLAN.md",
        "sources": {str(path.relative_to(ROOT.parent)): sha256(path)
                    for path in (cells_path, repeats_path)},
        "unit": "occupied coarse XY cell within one scenario/subsample/condition",
        "conditions": conditions, "by_scene": by_scene, "pooled": pooled,
        "by_event_count_band": bands,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    labels = [f"N={row['event_count']}\nσ={row['position_sigma_mm']:.2f} mm"
              for row in pooled]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for offset, field, label, color in [
        (-0.25, "neighbor_rate", "Densest adjacent support", "#4472c4"),
        (0, "local_score_rate", "Local held-out score", "#ed7d31"),
        (0.25, "joint_rate", "Both available", "#70ad47")]:
        bars = ax.bar(x + offset, [row[field] * 100 for row in pooled], width=0.23,
                      color=color, label=label)
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of sparse occupied cell-records (%)")
    ax.set_ylim(0, 45)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage.png", dpi=170)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output_dir)
    for row in result["pooled"]:
        print(row)


if __name__ == "__main__":
    main()
