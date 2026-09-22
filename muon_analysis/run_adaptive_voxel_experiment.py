"""低统计量条件下的统计量驱动自适应体素 PoCA 实验。

重建流程：
1. 保留 theta >= 2 mrad 的事件，并采用 theta_cap = 3 mrad 的截断权重；
2. 先将成像区划分为 16 x 16 x 16 个粗体素；
3. 仅根据粗体素内的散射加权强度，选择其中得分最高的一部分细分为 2 x 2 x 2；
4. 未细分体素保持粗分辨率，细分体素保留 32 x 32 x 32 的局部细节。

目标 U 型掩膜只用于 IoU 与定位误差评价，绝不参与体素细分决策。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_baseline_experiment import BINS, RANDOM_SEED, XY_LIMIT_MM, Z_LIMIT_MM, evaluate_image


ROOT = Path('/Users/xuyaoyang/geant4-test')
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'adaptive_voxel_results'
EVENT_COUNTS = [1000, 3000, 5000]
REPEATS = 5

# 这一组参数来自上一轮截断加权实验的候选范围，而非使用真值调参。
THETA_MIN_MRAD = 2.0
THETA_CAP_MRAD = 3.0
COARSE_BINS = 16
REFINE_FACTOR = 2
FINE_BINS = COARSE_BINS * REFINE_FACTOR

# 在所有非空粗体素中，按散射加权强度排名，选取前若干比例细分。
# 0 表示完全不细分；它应与 16³ 均匀粗体素一致。
REFINE_FRACTIONS = [0.0, 0.10, 0.20, 0.30, 0.40]
# 在低统计量下，单个候选事件也可能含有价值；是否细分仍由全局排名控制。
# 后续会在“局部置信度”实验中再比较 1、2、3 个事件门槛的影响。
MIN_EVENTS_TO_REFINE = 1


def imaging_edges(bins: int) -> list[np.ndarray]:
    """为给定网格数创建一致的 x、y、z 成像边界。"""
    return [
        np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, bins + 1),
        np.linspace(-XY_LIMIT_MM, XY_LIMIT_MM, bins + 1),
        np.linspace(-Z_LIMIT_MM, Z_LIMIT_MM, bins + 1),
    ]


def prepare_events(data: pd.DataFrame) -> pd.DataFrame:
    """完成上一阶段确定的事件筛选和截断权重准备。"""
    valid = data[
        (data['theta_mrad'] >= THETA_MIN_MRAD)
        & (data['poca_x_mm'].abs() <= XY_LIMIT_MM)
        & (data['poca_y_mm'].abs() <= XY_LIMIT_MM)
        & (data['poca_z_mm'].abs() <= Z_LIMIT_MM)
    ].copy()
    theta = valid['theta_mrad'].to_numpy()
    valid['capped_weight'] = np.minimum(theta ** 2, THETA_CAP_MRAD ** 2)
    return valid


def fixed_fine_volume(events: pd.DataFrame):
    """32³ 固定体素参考方法，使用与自适应方法完全相同的事件和权重。"""
    edges = imaging_edges(FINE_BINS)
    volume, edges = np.histogramdd(
        events[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy(),
        bins=edges, weights=events['capped_weight'].to_numpy(),
    )
    return volume, edges


def adaptive_volume(events: pd.DataFrame, refine_fraction: float):
    """构建一层 16³ -> 32³ 的自适应体素体积。

    输出仍映射到规则 32³ 数组，便于与固定 32³ 体素计算同一套 IoU。
    对未细分的粗体素，其总权重均匀分配到 8 个细网格中，表示该区域
    只能被粗粒度解析；被细分的区域按真实 PoCA 点落点累积权重。
    """
    coarse_edges = imaging_edges(COARSE_BINS)
    fine_edges = imaging_edges(FINE_BINS)
    points = events[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy()
    weights = events['capped_weight'].to_numpy()
    coarse_weight, _ = np.histogramdd(points, bins=coarse_edges, weights=weights)
    coarse_count, _ = np.histogramdd(points, bins=coarse_edges)

    # 非空且至少有两个事件的体素才有资格细分，避免一个偶然事件触发细化。
    eligible = np.flatnonzero((coarse_count.ravel() >= MIN_EVENTS_TO_REFINE))
    n_refine = int(np.floor(len(eligible) * refine_fraction))
    selected_flat: set[int] = set()
    if n_refine > 0:
        ranked = eligible[np.argsort(coarse_weight.ravel()[eligible])[::-1]]
        selected_flat = set(ranked[:n_refine].tolist())

    # 初始时所有粗体素按均匀分配映射到 32³。这样总权重守恒。
    output = np.zeros((FINE_BINS, FINE_BINS, FINE_BINS), dtype=float)
    for ix in range(COARSE_BINS):
        for iy in range(COARSE_BINS):
            for iz in range(COARSE_BINS):
                flat = np.ravel_multi_index((ix, iy, iz), coarse_weight.shape)
                if flat not in selected_flat and coarse_weight[ix, iy, iz] > 0:
                    output[2 * ix:2 * ix + 2, 2 * iy:2 * iy + 2, 2 * iz:2 * iz + 2] = (
                        coarse_weight[ix, iy, iz] / 8.0
                    )

    # 只对被细分的粗体素重新按 2³ 子网格累积真实事件，恢复局部细节。
    if selected_flat:
        selected_indices = np.array(np.unravel_index(list(selected_flat), coarse_weight.shape)).T
        coarse_width = np.array([2 * XY_LIMIT_MM / COARSE_BINS,
                                 2 * XY_LIMIT_MM / COARSE_BINS,
                                 2 * Z_LIMIT_MM / COARSE_BINS])
        lower = np.array([-XY_LIMIT_MM, -XY_LIMIT_MM, -Z_LIMIT_MM])
        point_indices = np.floor((points - lower) / coarse_width).astype(int)
        point_indices = np.clip(point_indices, 0, COARSE_BINS - 1)
        for ix, iy, iz in selected_indices:
            mask = np.all(point_indices == [ix, iy, iz], axis=1)
            if not mask.any():
                continue
            # 当前粗体素的一个边界区间，均分为两个子区间，因此需要 3 个边界值。
            child_edges = [
                np.linspace(coarse_edges[0][ix], coarse_edges[0][ix + 1], 3),
                np.linspace(coarse_edges[1][iy], coarse_edges[1][iy + 1], 3),
                np.linspace(coarse_edges[2][iz], coarse_edges[2][iz + 1], 3),
            ]
            child, _ = np.histogramdd(points[mask], bins=child_edges, weights=weights[mask])
            output[2 * ix:2 * ix + 2, 2 * iy:2 * iy + 2, 2 * iz:2 * iz + 2] = child

    leaf_voxels = COARSE_BINS ** 3 + len(selected_flat) * (REFINE_FACTOR ** 3 - 1)
    return output, fine_edges, len(selected_flat), leaf_voxels


def xy_centers() -> tuple[np.ndarray, np.ndarray]:
    edges = imaging_edges(FINE_BINS)
    return (edges[0][:-1] + edges[0][1:]) / 2, (edges[1][:-1] + edges[1][1:]) / 2


def plot_metrics(summary: pd.DataFrame, fixed: pd.DataFrame, output_path: Path) -> None:
    """显示细分比例对 IoU 与定位误差的影响，并给出固定 32³ 参考线。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for count, group in summary.groupby('event_count', sort=True):
        axes[0].errorbar(group['refine_fraction'], group['iou_mean'], yerr=group['iou_std'],
                         marker='o', capsize=3, label=f'{count} adaptive')
        axes[1].errorbar(group['refine_fraction'], group['localization_error_mm_mean'],
                         yerr=group['localization_error_mm_std'], marker='o', capsize=3,
                         label=f'{count} adaptive')
        baseline = fixed[fixed['event_count'] == count].iloc[0]
        axes[0].axhline(baseline['iou_mean'], linestyle='--', linewidth=1, alpha=0.7)
        axes[1].axhline(baseline['localization_error_mm_mean'], linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set(xlabel='Fraction of eligible coarse voxels refined', ylabel='Equal-area IoU (higher is better)')
    axes[1].set(xlabel='Fraction of eligible coarse voxels refined', ylabel='Localization error [mm] (lower is better)')
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Adaptive voxel PoCA: solid = adaptive, dashed = fixed 32³ reference', fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_images(images: dict[str, np.ndarray], edges, output_path: Path) -> None:
    """展示 3000 事件下固定体素和不同细分比例的二维投影图。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (label, image) in zip(axes.ravel(), images.items()):
        axis.imshow(np.log1p(image).T, origin='lower', cmap='inferno',
                    extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], aspect='equal')
        axis.set_title(label)
        axis.set_xlabel('PoCA x [mm]')
        axis.set_ylabel('PoCA y [mm]')
    for axis in axes.ravel()[len(images):]:
        axis.axis('off')
    fig.suptitle('Adaptive voxel PoCA: 3000 input events', fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    required = {'poca_x_mm', 'poca_y_mm', 'poca_z_mm', 'theta_mrad'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    rng = np.random.default_rng(RANDOM_SEED)
    x_centers, y_centers = xy_centers()
    records: list[dict] = []
    images: dict[str, np.ndarray] = {}
    image_edges = None

    for count in [n for n in EVENT_COUNTS if n <= len(data)]:
        for repeat in range(REPEATS):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)]
            events = prepare_events(subset)

            # 固定 32³ 作为同筛选、同权重条件下的直接参照。
            start = time.perf_counter()
            fixed_volume, edges = fixed_fine_volume(events)
            fixed_runtime_ms = (time.perf_counter() - start) * 1000.0
            fixed_image = fixed_volume.sum(axis=2)
            fixed_iou, fixed_error, fixed_contrast, truth_pixels = evaluate_image(
                fixed_image, x_centers, y_centers
            )
            records.append({
                'method': 'fixed_32x32x32', 'event_count': count, 'repeat': repeat + 1,
                'refine_fraction': np.nan, 'input_events': len(subset), 'selected_events': len(events),
                'refined_coarse_voxels': np.nan, 'leaf_voxels': FINE_BINS ** 3,
                'iou_equal_area': fixed_iou, 'localization_error_mm': fixed_error,
                'target_background_contrast': fixed_contrast, 'runtime_ms': fixed_runtime_ms,
                'truth_u_pixels': truth_pixels,
            })
            if count == 3000 and repeat == 0:
                images['fixed 32³'] = fixed_image
                image_edges = edges

            for fraction in REFINE_FRACTIONS:
                start = time.perf_counter()
                volume, edges, n_refined, leaf_voxels = adaptive_volume(events, fraction)
                runtime_ms = (time.perf_counter() - start) * 1000.0
                image = volume.sum(axis=2)
                iou, localization_error, contrast_ratio, truth_pixels = evaluate_image(
                    image, x_centers, y_centers
                )
                records.append({
                    'method': 'adaptive_16_to_32', 'event_count': count, 'repeat': repeat + 1,
                    'refine_fraction': fraction, 'input_events': len(subset), 'selected_events': len(events),
                    'refined_coarse_voxels': n_refined, 'leaf_voxels': leaf_voxels,
                    'iou_equal_area': iou, 'localization_error_mm': localization_error,
                    'target_background_contrast': contrast_ratio, 'runtime_ms': runtime_ms,
                    'truth_u_pixels': truth_pixels,
                })
                if count == 3000 and repeat == 0:
                    images[f'adaptive {fraction:.0%}'] = image

    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT_DIR / 'adaptive_voxel_metrics_by_repeat.csv', index=False)
    fixed = metrics[metrics['method'] == 'fixed_32x32x32'].groupby('event_count', as_index=False).agg(
        iou_mean=('iou_equal_area', 'mean'), iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
    )
    summary = metrics[metrics['method'] == 'adaptive_16_to_32'].groupby(
        ['event_count', 'refine_fraction'], as_index=False
    ).agg(
        repeats=('repeat', 'count'), selected_events_mean=('selected_events', 'mean'),
        refined_coarse_voxels_mean=('refined_coarse_voxels', 'mean'), leaf_voxels_mean=('leaf_voxels', 'mean'),
        iou_mean=('iou_equal_area', 'mean'), iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
        runtime_ms_mean=('runtime_ms', 'mean'), runtime_ms_std=('runtime_ms', 'std'),
    )
    summary.to_csv(OUT_DIR / 'adaptive_voxel_metrics_summary.csv', index=False)
    fixed.to_csv(OUT_DIR / 'adaptive_voxel_fixed_reference.csv', index=False)
    plot_metrics(summary, fixed, OUT_DIR / 'adaptive_voxel_metrics_vs_refinement.png')
    if image_edges is not None:
        plot_images(images, image_edges, OUT_DIR / 'adaptive_voxel_poca_xy_3000_events.png')

    config = {
        'source_csv': str(INPUT_CSV), 'random_seed': RANDOM_SEED,
        'event_counts': EVENT_COUNTS, 'repeats': REPEATS,
        'theta_min_mrad': THETA_MIN_MRAD, 'theta_cap_mrad': THETA_CAP_MRAD,
        'coarse_grid': [COARSE_BINS] * 3, 'refine_factor': REFINE_FACTOR,
        'fine_grid_equivalent': [FINE_BINS] * 3,
        'refine_fractions': REFINE_FRACTIONS, 'minimum_events_to_refine': MIN_EVENTS_TO_REFINE,
        'selection_rule': 'Rank eligible coarse voxels by capped scattering-angle weighted sum; no target truth used.',
    }
    (OUT_DIR / 'adaptive_voxel_experiment_config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    (OUT_DIR / 'adaptive_voxel_experiment_summary.md').write_text(
        '# 自适应体素 PoCA 实验\n\n'
        '先对 16³ 粗体素进行散射加权统计，再细分排名靠前且至少包含两个事件的粗体素。未细分区域保留粗分辨率。U 型真值只用于最终评价，不参与细分。\n',
        encoding='utf-8'
    )
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'Output directory: {OUT_DIR}')


if __name__ == '__main__':
    main()
