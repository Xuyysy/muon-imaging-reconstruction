"""散射角截断加权的低统计量 PoCA 实验。

在 v0.3 的第一轮实验中，2 mrad 最小散射角筛选改善了 U 型轮廓 IoU，
但极端大散射角事件仍可能让少数体素的 theta² 权重异常偏大。本脚本使用：

    weight_i = min(theta_i², theta_cap²)

并在相同入射事件子集上比较“无筛选基线”“仅 2 mrad 筛选”和不同上限的
截断加权方案。它不改变 PoCA 位置、体素网格或评价指标。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_baseline_experiment import (
    BINS,
    RANDOM_SEED,
    XY_LIMIT_MM,
    Z_LIMIT_MM,
    evaluate_image,
)


ROOT = Path('/Users/xuyaoyang/geant4-test')
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'angle_weighting_results'
EVENT_COUNTS = [1000, 3000, 5000]
REPEATS = 5

# v0.3 第一轮实验表明 2 mrad 值得作为候选最小散射角。
THETA_MIN_MRAD = 2.0
# None 表示不截断，即仍然使用原始 theta²；其余值是扫描的 theta_cap。
THETA_CAPS_MRAD: list[float | None] = [None, 8.0, 6.0, 4.0, 3.0, 2.0]


def voxelize_weighted(data: pd.DataFrame, theta_cap_mrad: float | None):
    """固定体素累积，权重为 theta² 或截断后的 min(theta², cap²)。"""
    valid = data[
        (data['poca_x_mm'].abs() <= XY_LIMIT_MM)
        & (data['poca_y_mm'].abs() <= XY_LIMIT_MM)
        & (data['poca_z_mm'].abs() <= Z_LIMIT_MM)
    ]
    edges = [
        np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1),
        np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1),
        np.linspace(-Z_LIMIT_MM, Z_LIMIT_MM, BINS + 1),
    ]
    theta = valid['theta_mrad'].to_numpy()
    weights = theta ** 2
    if theta_cap_mrad is not None:
        # 不把极端大角度事件删除，而是限制其最大强度贡献。
        weights = np.minimum(weights, theta_cap_mrad ** 2)
    volume, edges = np.histogramdd(
        valid[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy(),
        bins=edges,
        weights=weights,
    )
    return volume, edges, valid


def centers() -> tuple[np.ndarray, np.ndarray]:
    """返回与基线评价函数相同的 XY 体素中心。"""
    x_edges = np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1)
    y_edges = np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1)
    return (x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2


def cap_label(cap: float | None) -> str:
    """让 CSV、图例和 Markdown 都使用清楚的一致标签。"""
    return 'uncapped' if cap is None else f'{cap:g} mrad'


def plot_metrics(summary: pd.DataFrame, output_path: Path) -> None:
    """绘制仅筛选与不同截断上限的 IoU、定位误差对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for count, group in summary.groupby('event_count', sort=True):
        x = group['cap_display_mrad'].to_numpy()
        # x=9 表示“无上限”；图上会被单独标记为 uncapped。
        axes[0].errorbar(x, group['iou_mean'], yerr=group['iou_std'], marker='o', capsize=3,
                         label=f'{count} input events')
        axes[1].errorbar(x, group['localization_error_mm_mean'],
                         yerr=group['localization_error_mm_std'], marker='o', capsize=3,
                         label=f'{count} input events')
    for axis, ylabel in zip(axes, ['Equal-area IoU (higher is better)', 'Localization error [mm] (lower is better)']):
        axis.set_xlabel('Scattering-angle cap [mrad]; rightmost = uncapped')
        axis.set_ylabel(ylabel)
        axis.set_xticks([2, 3, 4, 6, 8, 9])
        axis.set_xticklabels(['2', '3', '4', '6', '8', 'uncapped'])
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'PoCA with theta >= {THETA_MIN_MRAD:g} mrad and capped theta² weighting', fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_representative_images(images: dict[str, np.ndarray], edges, output_path: Path) -> None:
    """展示 3000 入射事件时截断上限对 U 型轮廓的直观影响。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (label, image) in zip(axes.ravel(), images.items()):
        axis.imshow(np.log1p(image).T, origin='lower', cmap='inferno',
                    extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], aspect='equal')
        axis.set_title(label)
        axis.set_xlabel('PoCA x [mm]')
        axis.set_ylabel('PoCA y [mm]')
    fig.suptitle('Representative images: 3000 input events, theta >= 2 mrad', fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    required = {'poca_x_mm', 'poca_y_mm', 'poca_z_mm', 'theta_mrad'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    x_centers, y_centers = centers()
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict] = []
    representative_images: dict[str, np.ndarray] = {}
    representative_edges = None

    for count in [n for n in EVENT_COUNTS if n <= len(data)]:
        for repeat in range(REPEATS):
            # 每个 scheme 共用该批入射事件，保证实验公平。
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)]

            # 额外记录“未筛选、未截断”基线，让本文件自身完整可读。
            schemes = [('baseline  theta >= 0  uncapped', 0.0, None)]
            schemes += [(f'theta >= {THETA_MIN_MRAD:g}  cap {cap_label(cap)}', THETA_MIN_MRAD, cap)
                        for cap in THETA_CAPS_MRAD]
            for scheme_name, theta_min, cap in schemes:
                selected = subset.loc[subset['theta_mrad'] >= theta_min].copy()
                start = time.perf_counter()
                volume, edges, used = voxelize_weighted(selected, cap)
                image = volume.sum(axis=2)
                runtime_ms = (time.perf_counter() - start) * 1000.0
                iou, localization_error, contrast_ratio, truth_pixels = evaluate_image(
                    image, x_centers, y_centers
                )
                records.append({
                    'event_count': count,
                    'repeat': repeat + 1,
                    'scheme': scheme_name,
                    'theta_min_mrad': theta_min,
                    'theta_cap_mrad': np.nan if cap is None else cap,
                    'cap_display_mrad': 9.0 if cap is None else cap,
                    'input_events': len(subset),
                    'selected_events': len(selected),
                    'retention_ratio': len(selected) / len(subset),
                    'poca_points_in_volume': len(used),
                    'iou_equal_area': iou,
                    'localization_error_mm': localization_error,
                    'target_background_contrast': contrast_ratio,
                    'voxelization_runtime_ms': runtime_ms,
                    'truth_u_pixels': truth_pixels,
                })
                if count == 3000 and repeat == 0 and theta_min == THETA_MIN_MRAD:
                    representative_images[cap_label(cap)] = image
                    representative_edges = edges

    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT_DIR / 'angle_weighting_metrics_by_repeat.csv', index=False)
    summary = metrics.groupby(['event_count', 'scheme', 'theta_min_mrad', 'theta_cap_mrad', 'cap_display_mrad'],
                              dropna=False, as_index=False).agg(
        repeats=('repeat', 'count'),
        input_events=('input_events', 'first'),
        selected_events_mean=('selected_events', 'mean'),
        retention_ratio_mean=('retention_ratio', 'mean'),
        poca_points_in_volume_mean=('poca_points_in_volume', 'mean'),
        iou_mean=('iou_equal_area', 'mean'),
        iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
        contrast_ratio_mean=('target_background_contrast', 'mean'),
        runtime_ms_mean=('voxelization_runtime_ms', 'mean'),
    )
    summary.to_csv(OUT_DIR / 'angle_weighting_metrics_summary.csv', index=False)

    # 绘图只使用 theta >= 2 mrad 的截断加权方案；基线保留在 CSV 表中。
    filtered_summary = summary[summary['theta_min_mrad'] == THETA_MIN_MRAD]
    plot_metrics(filtered_summary, OUT_DIR / 'angle_weighting_metrics_vs_cap.png')
    if representative_edges is not None:
        plot_representative_images(representative_images, representative_edges,
                                   OUT_DIR / 'angle_weighting_poca_xy_3000_events.png')

    config = {
        'source_csv': str(INPUT_CSV), 'source_rows': int(len(data)),
        'random_seed': RANDOM_SEED, 'event_counts': EVENT_COUNTS, 'repeats': REPEATS,
        'theta_min_mrad': THETA_MIN_MRAD, 'theta_caps_mrad': ['uncapped', 8.0, 6.0, 4.0, 3.0, 2.0],
        'weight': 'min(theta_mrad^2, theta_cap_mrad^2), after theta_mrad >= theta_min_mrad',
        'fair_comparison': 'Each scheme uses the same input-event subset within a repeat.',
    }
    (OUT_DIR / 'angle_weighting_experiment_config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    (OUT_DIR / 'angle_weighting_experiment_summary.md').write_text(
        '# 散射角截断加权 PoCA 实验\n\n'
        '本实验固定最小散射角为 2 mrad，并扫描散射角权重上限。所有方案在每次重复中使用相同入射事件子集。\n\n'
        '截断上限的作用是限制极端大角散射事件的 theta² 权重，不能只根据一张图判断其有效性；应以 IoU、定位误差和重复实验标准差共同判断。\n',
        encoding='utf-8'
    )
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'Output directory: {OUT_DIR}')


if __name__ == '__main__':
    main()
