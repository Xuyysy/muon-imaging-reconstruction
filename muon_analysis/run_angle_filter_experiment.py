"""散射角阈值筛选的低统计量 PoCA 对比实验。

本脚本对应论文 v0.3 的第一项改进：在固定入射缪子数量下，
仅保留散射角不低于阈值的事件，再沿用固定体素 PoCA 重建。

这样比较的是“相同采集时间（相同入射事件数）下”的成像质量，
而不是人为让每组保留相同数量的事件。0 mrad 阈值就是不筛选的
固定体素 PoCA 基线。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 复用基线中的体素化、真值掩膜、IoU 与定位误差定义。
# 因此本实验与 v0.2 的评价口径完全一致。
from run_baseline_experiment import (
    BINS,
    RANDOM_SEED,
    XY_LIMIT_MM,
    Z_LIMIT_MM,
    evaluate_image,
    voxelize_poca,
)


ROOT = Path('/Users/xuyaoyang/geant4-test')
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'angle_filter_results'

# 与基线相同的入射事件数；每组都从同一批抽样事件开始，保证公平。
EVENT_COUNTS = [1000, 3000, 5000]
REPEATS = 5

# 0 表示不筛选。其余候选值覆盖轻度、中度、严格筛选。
# 最终阈值只能根据 IoU/定位误差的验证结果选取，不能只凭图片判断。
ANGLE_THRESHOLDS_MRAD = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]


def filter_by_scattering_angle(data: pd.DataFrame, theta_min_mrad: float) -> pd.DataFrame:
    """保留散射角 theta_mrad 不小于指定阈值的事件。

    小角度事件通常接近直线传播，其 PoCA 点对目标定位信息较弱，
    大量累积时可能形成背景。注意：阈值过大也会丢掉有效事件，
    所以必须通过低统计量实验寻找平衡，而不是预设“越大越好”。
    """
    return data.loc[data['theta_mrad'] >= theta_min_mrad].copy()


def make_centers() -> tuple[np.ndarray, np.ndarray]:
    """返回与基线体素网格严格一致的 XY 体素中心坐标。"""
    x_edges = np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1)
    y_edges = np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, BINS + 1)
    return (x_edges[:-1] + x_edges[1:]) / 2.0, (y_edges[:-1] + y_edges[1:]) / 2.0


def plot_metrics(summary: pd.DataFrame, output_path: Path) -> None:
    """绘制阈值扫描曲线：IoU、定位误差和事件保留比例。"""
    figures = [
        ('iou_mean', 'iou_std', 'Equal-area IoU', 'higher is better'),
        ('localization_error_mm_mean', 'localization_error_mm_std', 'Localization error [mm]', 'lower is better'),
        ('retention_ratio_mean', 'retention_ratio_std', 'Retained-event ratio', 'reference only'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, (mean_col, std_col, title, direction) in zip(axes, figures):
        for count, group in summary.groupby('event_count', sort=True):
            axis.errorbar(
                group['theta_min_mrad'], group[mean_col], yerr=group[std_col].fillna(0),
                marker='o', capsize=3, linewidth=1.8, label=f'{count} input events',
            )
        axis.set_title(title)
        axis.set_xlabel('Minimum scattering angle [mrad]')
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        axis.text(0.02, 0.03, direction, transform=axis.transAxes, fontsize=9, color='#555555')
    axes[0].legend(fontsize=8)
    fig.suptitle('Low-statistics PoCA with scattering-angle filtering', fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_representative_images(images: dict[float, np.ndarray], edges, output_path: Path) -> None:
    """显示 3000 个入射事件下不同阈值的 XY 成像，供直观检查。"""
    thresholds = list(images)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, threshold in zip(axes.ravel(), thresholds):
        image = images[threshold]
        # 对数显示仅改变视觉展示；IoU 等指标仍使用原始强度图计算。
        shown = np.log1p(image).T
        axis.imshow(
            shown, origin='lower', cmap='inferno',
            extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]],
            aspect='equal',
        )
        axis.set_title(f'theta >= {threshold:g} mrad')
        axis.set_xlabel('PoCA x [mm]')
        axis.set_ylabel('PoCA y [mm]')
    fig.suptitle('Representative PoCA images: 3000 input events', fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    required = {'poca_x_mm', 'poca_y_mm', 'poca_z_mm', 'theta_mrad'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    counts = [n for n in EVENT_COUNTS if n <= len(data)]
    rng = np.random.default_rng(RANDOM_SEED)
    x_centers, y_centers = make_centers()
    records: list[dict] = []
    representative_images: dict[float, np.ndarray] = {}
    representative_edges = None

    for count in counts:
        # 先抽取一次入射事件子集；同一个 subset 会用于全部阈值。
        # 这保证阈值之间唯一变化的是“是否筛掉小散射角事件”。
        for repeat in range(REPEATS):
            indices = rng.choice(len(data), size=count, replace=False)
            subset = data.iloc[indices]

            for threshold in ANGLE_THRESHOLDS_MRAD:
                start = time.perf_counter()
                selected = filter_by_scattering_angle(subset, threshold)
                volume, edges, used = voxelize_poca(selected)
                image = volume.sum(axis=2)
                runtime_ms = (time.perf_counter() - start) * 1000.0
                iou, localization_error, contrast_ratio, truth_pixels = evaluate_image(
                    image, x_centers, y_centers
                )
                records.append({
                    'event_count': count,
                    'repeat': repeat + 1,
                    'theta_min_mrad': threshold,
                    'input_events': len(subset),
                    'selected_events': len(selected),
                    'retention_ratio': len(selected) / len(subset),
                    'poca_points_in_volume': len(used),
                    'iou_equal_area': iou,
                    'localization_error_mm': localization_error,
                    'target_background_contrast': contrast_ratio,
                    'filter_and_voxelization_runtime_ms': runtime_ms,
                    'truth_u_pixels': truth_pixels,
                })

                # 只保存 3000 事件、第 1 次抽样的代表图，避免输出大量重复图片。
                if count == 3000 and repeat == 0:
                    representative_images[threshold] = image
                    representative_edges = edges

    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT_DIR / 'angle_filter_metrics_by_repeat.csv', index=False)
    summary = metrics.groupby(['event_count', 'theta_min_mrad'], as_index=False).agg(
        repeats=('repeat', 'count'),
        input_events=('input_events', 'first'),
        selected_events_mean=('selected_events', 'mean'),
        selected_events_std=('selected_events', 'std'),
        retention_ratio_mean=('retention_ratio', 'mean'),
        retention_ratio_std=('retention_ratio', 'std'),
        poca_points_in_volume_mean=('poca_points_in_volume', 'mean'),
        iou_mean=('iou_equal_area', 'mean'),
        iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
        contrast_ratio_mean=('target_background_contrast', 'mean'),
        contrast_ratio_std=('target_background_contrast', 'std'),
        runtime_ms_mean=('filter_and_voxelization_runtime_ms', 'mean'),
        runtime_ms_std=('filter_and_voxelization_runtime_ms', 'std'),
    )
    summary.to_csv(OUT_DIR / 'angle_filter_metrics_summary.csv', index=False)
    plot_metrics(summary, OUT_DIR / 'angle_filter_metrics_vs_threshold.png')
    if representative_edges is not None:
        plot_representative_images(
            representative_images, representative_edges,
            OUT_DIR / 'angle_filter_poca_xy_3000_events.png',
        )

    config = {
        'source_csv': str(INPUT_CSV),
        'source_rows': int(len(data)),
        'source_mode': 'Ideal (existing baseline CSV)',
        'random_seed': RANDOM_SEED,
        'event_counts': counts,
        'repeats': REPEATS,
        'theta_min_mrad_candidates': ANGLE_THRESHOLDS_MRAD,
        'voxel_grid': [BINS, BINS, BINS],
        'weight': 'sum(theta_mrad^2) after theta_mrad >= theta_min_mrad',
        'fair_comparison': 'For each repeat, every threshold uses the same input-event subset.',
        'iou_definition': 'Top-k prediction with k equal to the U-shaped ground-truth pixel count.',
    }
    (OUT_DIR / 'angle_filter_experiment_config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )

    lines = [
        '# 散射角筛选 PoCA 实验', '',
        '## 实验目的', '',
        '在固定入射事件数量下扫描最小散射角阈值，验证筛除小角散射事件能否改善低统计量 PoCA 成像。', '',
        '## 公平比较原则', '',
        '每次随机抽样得到的一批入射事件，会被全部阈值共同使用。阈值只改变保留事件，不改变采集事件数。', '',
        '## 输出文件', '',
        '- `angle_filter_metrics_by_repeat.csv`：每次抽样、每个阈值的原始指标。',
        '- `angle_filter_metrics_summary.csv`：均值与标准差。',
        '- `angle_filter_metrics_vs_threshold.png`：IoU、定位误差、保留比例随阈值变化。',
        '- `angle_filter_poca_xy_3000_events.png`：3000 个入射事件的阈值扫描成像图。', '',
        '## 解释规则', '',
        '只有当某个阈值在多个低统计量水平下稳定提高 IoU 或降低定位误差，且保留事件比例合理时，才能把它称为有效改进。',
    ]
    (OUT_DIR / 'angle_filter_experiment_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'Output directory: {OUT_DIR}')


if __name__ == '__main__':
    main()
