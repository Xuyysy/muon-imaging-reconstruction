"""小尺寸 U 型目标的泛化验证。

目标从 120 mm 外轮廓、30 mm 壁厚改为 80 mm 外轮廓、20 mm 壁厚。
算法参数保持为中心大 U 目标实验中的 theta >= 2 mrad、cap = 3 mrad，
不针对小目标重新搜索阈值。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import (
    EVENT_COUNTS, RANDOM_SEED, adaptive_volume, fixed_fine_volume,
    imaging_edges, prepare_events,
)

ROOT = Path('/Users/xuyaoyang/geant4-test')
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks_small.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'small_target_generalization_results'
REPEATS = 5
TEST_FRACTIONS = [0.0, 0.20, 0.40]


def small_u_mask(x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
    """与 small Geant4 场景一致的 U 型真值：80 mm 外轮廓、20 mm 壁厚。"""
    x, y = np.meshgrid(x_centers, y_centers, indexing='ij')
    bottom = (np.abs(x) <= 40) & (y >= -40) & (y <= -20)
    left = (x >= -40) & (x <= -20) & (y >= -20) & (y <= 40)
    right = (x >= 20) & (x <= 40) & (y >= -20) & (y <= 40)
    return bottom | left | right


def evaluate_small(image: np.ndarray, x_centers: np.ndarray, y_centers: np.ndarray):
    """与前两轮相同：等面积 Top-k IoU + 散射强度质心定位误差。"""
    truth = small_u_mask(x_centers, y_centers)
    k = int(truth.sum())
    predicted = np.zeros_like(truth, dtype=bool)
    predicted.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    iou = np.logical_and(predicted, truth).sum() / np.logical_or(predicted, truth).sum()
    x, y = np.meshgrid(x_centers, y_centers, indexing='ij')
    total = image.sum()
    rec_x = float((image * x).sum() / total) if total else np.nan
    rec_y = float((image * y).sum() / total) if total else np.nan
    return float(iou), float(np.hypot(rec_x - x[truth].mean(), rec_y - y[truth].mean()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    edges = imaging_edges(32)
    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    rng = np.random.default_rng(RANDOM_SEED)
    rows, images = [], {}
    for count in [n for n in EVENT_COUNTS if n <= len(data)]:
        for repeat in range(REPEATS):
            events = prepare_events(data.iloc[rng.choice(len(data), size=count, replace=False)])
            fixed, _ = fixed_fine_volume(events)
            fixed_image = fixed.sum(axis=2)
            iou, error = evaluate_small(fixed_image, x_centers, y_centers)
            rows.append({'method': 'fixed_32x32x32', 'event_count': count, 'repeat': repeat + 1,
                         'refine_fraction': np.nan, 'selected_events': len(events),
                         'iou_equal_area': iou, 'localization_error_mm': error})
            if count == 5000 and repeat == 0: images['fixed 32³'] = fixed_image
            for fraction in TEST_FRACTIONS:
                volume, _, refined, leaves = adaptive_volume(events, fraction)
                image = volume.sum(axis=2)
                iou, error = evaluate_small(image, x_centers, y_centers)
                rows.append({'method': 'adaptive_16_to_32', 'event_count': count, 'repeat': repeat + 1,
                             'refine_fraction': fraction, 'selected_events': len(events),
                             'refined_coarse_voxels': refined, 'leaf_voxels': leaves,
                             'iou_equal_area': iou, 'localization_error_mm': error})
                if count == 5000 and repeat == 0: images[f'adaptive {fraction:.0%}'] = image

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / 'small_target_metrics_by_repeat.csv', index=False)
    summary = metrics.groupby(['method', 'event_count', 'refine_fraction'], dropna=False, as_index=False).agg(
        repeats=('repeat', 'count'), selected_events_mean=('selected_events', 'mean'),
        iou_mean=('iou_equal_area', 'mean'), iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
    )
    summary.to_csv(OUT_DIR / 'small_target_metrics_summary.csv', index=False)

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
    fig.suptitle('Small U-target generalization: 80 mm footprint, 20 mm walls')
    fig.savefig(OUT_DIR / 'small_target_metrics_vs_refinement.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, (label, image) in zip(axes, images.items()):
        axis.imshow(np.log1p(image).T, origin='lower', cmap='inferno', extent=[-90, 90, -90, 90], aspect='equal')
        axis.set(title=label, xlabel='PoCA x [mm]', ylabel='PoCA y [mm]')
    fig.suptitle('Small U target: 5000 input events')
    fig.savefig(OUT_DIR / 'small_target_poca_xy_5000_events.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()
