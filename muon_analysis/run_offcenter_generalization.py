"""偏心 U 型目标的泛化验证。

本脚本不重新调节散射角阈值、截断上限或自适应细分比例；直接复用中心
目标实验选出的流程，检验重建规则在目标位置变化后是否仍然有效。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import (
    EVENT_COUNTS, RANDOM_SEED, REFINE_FRACTIONS, adaptive_volume,
    fixed_fine_volume, imaging_edges, prepare_events,
)


ROOT = Path('/Users/xuyaoyang/geant4-test')
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks_offcenter.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'offcenter_generalization_results'
REPEATS = 5
# 与 DetectorConstruction.cc 的 offcenter 场景严格一致。
TARGET_CENTER_X_MM = 25.0
TARGET_CENTER_Y_MM = -15.0
# 不扫描新参数，只验证中心场景中最有代表性的三种细分程度。
TEST_FRACTIONS = [0.0, 0.20, 0.40]


def shifted_u_mask(x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
    """按偏心 U 型几何建立真值掩膜，仅用于实验评价。"""
    x, y = np.meshgrid(x_centers, y_centers, indexing='ij')
    bottom = ((x >= TARGET_CENTER_X_MM - 60) & (x <= TARGET_CENTER_X_MM + 60)
              & (y >= TARGET_CENTER_Y_MM - 60) & (y <= TARGET_CENTER_Y_MM - 30))
    left = ((x >= TARGET_CENTER_X_MM - 60) & (x <= TARGET_CENTER_X_MM - 30)
            & (y >= TARGET_CENTER_Y_MM - 30) & (y <= TARGET_CENTER_Y_MM + 60))
    right = ((x >= TARGET_CENTER_X_MM + 30) & (x <= TARGET_CENTER_X_MM + 60)
             & (y >= TARGET_CENTER_Y_MM - 30) & (y <= TARGET_CENTER_Y_MM + 60))
    return bottom | left | right


def evaluate_shifted(image: np.ndarray, x_centers: np.ndarray, y_centers: np.ndarray):
    """沿用中心场景的等面积 IoU 与强度质心定位误差定义。"""
    truth = shifted_u_mask(x_centers, y_centers)
    k = int(truth.sum())
    predicted = np.zeros_like(truth, dtype=bool)
    predicted.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    iou = np.logical_and(predicted, truth).sum() / np.logical_or(predicted, truth).sum()
    x, y = np.meshgrid(x_centers, y_centers, indexing='ij')
    total = image.sum()
    rec_x = float((image * x).sum() / total) if total else np.nan
    rec_y = float((image * y).sum() / total) if total else np.nan
    true_x, true_y = float(x[truth].mean()), float(y[truth].mean())
    return float(iou), float(np.hypot(rec_x - true_x, rec_y - true_y))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    edges = imaging_edges(32)
    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    images = {}

    for count in [n for n in EVENT_COUNTS if n <= len(data)]:
        for repeat in range(REPEATS):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)]
            events = prepare_events(subset)
            fixed, _ = fixed_fine_volume(events)
            fixed_image = fixed.sum(axis=2)
            iou, error = evaluate_shifted(fixed_image, x_centers, y_centers)
            records.append({'method': 'fixed_32x32x32', 'event_count': count, 'repeat': repeat + 1,
                            'refine_fraction': np.nan, 'selected_events': len(events),
                            'iou_equal_area': iou, 'localization_error_mm': error})
            if count == 5000 and repeat == 0:
                images['fixed 32³'] = fixed_image
            for fraction in TEST_FRACTIONS:
                volume, _, refined, leaves = adaptive_volume(events, fraction)
                image = volume.sum(axis=2)
                iou, error = evaluate_shifted(image, x_centers, y_centers)
                records.append({'method': 'adaptive_16_to_32', 'event_count': count, 'repeat': repeat + 1,
                                'refine_fraction': fraction, 'selected_events': len(events),
                                'refined_coarse_voxels': refined, 'leaf_voxels': leaves,
                                'iou_equal_area': iou, 'localization_error_mm': error})
                if count == 5000 and repeat == 0:
                    images[f'adaptive {fraction:.0%}'] = image

    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT_DIR / 'offcenter_metrics_by_repeat.csv', index=False)
    summary = metrics.groupby(['method', 'event_count', 'refine_fraction'], dropna=False, as_index=False).agg(
        repeats=('repeat', 'count'), selected_events_mean=('selected_events', 'mean'),
        iou_mean=('iou_equal_area', 'mean'), iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
    )
    summary.to_csv(OUT_DIR / 'offcenter_metrics_summary.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    adaptive = summary[summary['method'] == 'adaptive_16_to_32']
    fixed = summary[summary['method'] == 'fixed_32x32x32']
    for count, group in adaptive.groupby('event_count'):
        axes[0].errorbar(group['refine_fraction'], group['iou_mean'], yerr=group['iou_std'], marker='o', capsize=3, label=f'{count} adaptive')
        axes[1].errorbar(group['refine_fraction'], group['localization_error_mm_mean'], yerr=group['localization_error_mm_std'], marker='o', capsize=3, label=f'{count} adaptive')
        base = fixed[fixed['event_count'] == count].iloc[0]
        axes[0].axhline(base['iou_mean'], linestyle='--', alpha=0.6)
        axes[1].axhline(base['localization_error_mm_mean'], linestyle='--', alpha=0.6)
    axes[0].set(xlabel='Refine fraction', ylabel='IoU (higher is better)')
    axes[1].set(xlabel='Refine fraction', ylabel='Localization error [mm] (lower is better)')
    for axis in axes: axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Off-centre U-target generalization: centre = (+25, -15) mm')
    fig.savefig(OUT_DIR / 'offcenter_metrics_vs_refinement.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, (label, image) in zip(axes, images.items()):
        axis.imshow(np.log1p(image).T, origin='lower', cmap='inferno', extent=[-90, 90, -90, 90], aspect='equal')
        axis.set(title=label, xlabel='PoCA x [mm]', ylabel='PoCA y [mm]')
    fig.suptitle('Off-centre U target: 5000 input events')
    fig.savefig(OUT_DIR / 'offcenter_poca_xy_5000_events.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()
