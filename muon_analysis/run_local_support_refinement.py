"""邻域辅助、局部事件支撑的 2.5D PoCA 细分开发实验。

候选区域由同一 z 组中的 3×3 XY 邻域统计量决定，但只有本地粗 z
切片至少含两条通过筛选的事件时才细分。另比较根据本地有效事件
数收缩细节幅度的版本。所有规则均不读取目标真值。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import (
    RANDOM_SEED, THETA_CAP_MRAD, adaptive_volume, imaging_edges,
)
from run_baseline_experiment import u_target_mask
from run_continuous_event_quality_experiment import evaluate
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm, theta_noise_floor_mrad,
)
from run_z_pooled_refinement import z_pooled_volume

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
Z_GROUP_SIZE = 4
LOCAL_MIN_EVENTS = 2
METHODS = ('coarse16', 'adaptive40', 'z_pool_4', 'local_hard', 'local_shrink')


def local_support_volumes(events: pd.DataFrame, sigma_mm: float,
                          spacing_mm: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """用 3×3 XY 邻域的 4 层统计量提议，再以单层本地计数决策。

    硬细分把本地事件全部按 XY 的 2×2 子格落点放置；收缩版混合
    “粗格均匀分配”和“细格实际落点”。权重完全保留在原 z 层。
    """
    coarse_edges, fine_edges = imaging_edges(16), imaging_edges(32)
    points = events[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy()
    weights = events['capped_weight'].to_numpy()
    coarse, _ = np.histogramdd(points, bins=coarse_edges, weights=weights)
    squared, _ = np.histogramdd(points, bins=coarse_edges, weights=weights ** 2)
    count, _ = np.histogramdd(points, bins=coarse_edges)
    fine_xy, _ = np.histogramdd(
        points, bins=(fine_edges[0], fine_edges[1], coarse_edges[2]), weights=weights)

    # 每组先合并相邻 4 个 z 粗层，再只在 XY 上加 3×3 邻域。
    grouped_weight = coarse.reshape(16, 16, 4, 4).sum(axis=3)
    grouped_sq = squared.reshape(16, 16, 4, 4).sum(axis=3)
    padded_weight = np.pad(grouped_weight, ((1, 1), (1, 1), (0, 0)))
    padded_sq = np.pad(grouped_sq, ((1, 1), (1, 1), (0, 0)))
    neighbor_weight = np.zeros_like(grouped_weight)
    neighbor_sq = np.zeros_like(grouped_sq)
    for dx in range(3):
        for dy in range(3):
            neighbor_weight += padded_weight[dx:dx + 16, dy:dy + 16]
            neighbor_sq += padded_sq[dx:dx + 16, dy:dy + 16]
    neighbor_neff = np.divide(neighbor_weight ** 2, neighbor_sq,
                              out=np.zeros_like(neighbor_weight), where=neighbor_sq > 0)
    theta_noise = theta_noise_floor_mrad(sigma_mm, spacing_mm)
    threshold = 2.0 * (1.0 + (theta_noise / THETA_CAP_MRAD) ** 2)
    proposal = neighbor_neff >= threshold
    proposed_slices = np.repeat(proposal, Z_GROUP_SIZE, axis=2)
    # 关键约束：邻居不能“借给”空切片事件。至少两条本地事件才细分。
    selected = proposed_slices & (count >= LOCAL_MIN_EVENTS)
    local_neff = np.divide(coarse ** 2, squared,
                           out=np.zeros_like(coarse), where=squared > 0)
    # 后验式经验收缩仅控制细节幅度，并不声称严格贝叶斯推断。
    alpha = np.where(selected, local_neff / (local_neff + threshold), 0.0)
    base = np.repeat(np.repeat(coarse / 4.0, 2, axis=0), 2, axis=1)
    select_fine = np.repeat(np.repeat(selected, 2, axis=0), 2, axis=1)
    alpha_fine = np.repeat(np.repeat(alpha, 2, axis=0), 2, axis=1)
    hard = np.where(select_fine, fine_xy, base)
    shrink = base + alpha_fine * (fine_xy - base)
    for volume in (hard, shrink):
        # 比较每一个 z 层，而不只比较总体；不会发生沿 z 的事件搬移。
        if not np.allclose(volume.sum(axis=(0, 1)), coarse.sum(axis=(0, 1)),
                           rtol=1e-10, atol=1e-8):
            raise AssertionError('Refinement changed per-z scattering weight')
    return {'local_hard': hard, 'local_shrink': shrink}, {
        'proposed_coarse_slices': int(proposed_slices.sum()),
        'refined_coarse_slices': int(selected.sum()),
        'empty_or_singleton_slices_blocked': int((proposed_slices & ~selected).sum()),
        'mean_shrink_alpha': float(alpha[selected].mean()) if selected.any() else 0.0,
        'max_neighbor_effective_events': float(neighbor_neff.max()),
        'threshold_effective_events': float(threshold),
    }


def run(input_csv: Path, output_dir: Path, counts: tuple[int, ...],
        sigmas: tuple[float, ...], repeats: int) -> None:
    data = pd.read_csv(input_csv)
    if len(data) < max(counts):
        raise ValueError('Not enough four-layer complete events')
    output_dir.mkdir(parents=True, exist_ok=True)
    spacing = same_side_layer_spacing_mm(data)
    axes = imaging_edges(32)
    cx, cy = [(axis[1:] + axis[:-1]) / 2 for axis in axes[:2]]
    truth = u_target_mask(cx, cy)
    rng = np.random.default_rng(RANDOM_SEED + 831)  # 和前两轮同事件/噪声。
    rows: list[dict] = []
    for count in counts:
        for repeat in range(1, repeats + 1):
            subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                     for prefix in ('up1', 'up2', 'down1', 'down2')
                     for axis in ('x', 'y')}
            for sigma in sigmas:
                measured = measured_hits(subset, sigma, draws)
                observed = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
                events = corrected_events(observed, sigma, spacing)
                # 耗时只包围重建，不把评分、写盘和作图计入。
                variants: dict[str, tuple[np.ndarray, dict, float]] = {}
                start = time.perf_counter()
                coarse, _, _, _ = adaptive_volume(events, 0.0)
                variants['coarse16'] = (coarse, {'refined_coarse_slices': 0},
                                        time.perf_counter() - start)
                start = time.perf_counter()
                adaptive, _, old_selected, _ = adaptive_volume(events, .40)
                variants['adaptive40'] = (adaptive,
                    {'refined_coarse_slices': old_selected}, time.perf_counter() - start)
                start = time.perf_counter()
                pooled, pool_info = z_pooled_volume(events, sigma, spacing, Z_GROUP_SIZE)
                variants['z_pool_4'] = (pooled, pool_info, time.perf_counter() - start)
                start = time.perf_counter()
                local, local_info = local_support_volumes(events, sigma, spacing)
                local_runtime = time.perf_counter() - start
                # 两个局部版本共享候选计算，耗时作为组合诊断，不能分别
                # 解读为独立部署运行时间。
                for method, volume in local.items():
                    variants[method] = (volume, local_info, local_runtime)
                for method, (volume, info, runtime) in variants.items():
                    metrics = evaluate(volume.sum(axis=2), truth, cx, cy)
                    rows.append({'method': method, 'event_count': count,
                                 'repeat': repeat, 'position_sigma_mm': sigma,
                                 'selected_events': len(events),
                                 'reconstruction_seconds': runtime,
                                 **info, **metrics})
            print(f'local support: N={count}, repeat={repeat}/{repeats}', flush=True)
    frame = pd.DataFrame(rows)
    key = ['method', 'event_count', 'repeat', 'position_sigma_mm']
    if frame.duplicated(key).any():
        raise AssertionError('Duplicate method-condition key')
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary = frame.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{col}_mean': (col, 'mean') for col in
           ('iou', 'global_error_mm', 'auto_error_mm', 'refined_coarse_slices',
            'proposed_coarse_slices', 'empty_or_singleton_slices_blocked',
            'mean_shrink_alpha', 'reconstruction_seconds')},
        **{f'{col}_std': (col, 'std') for col in
           ('iou', 'global_error_mm', 'auto_error_mm', 'refined_coarse_slices')})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    base = frame[frame.method == 'adaptive40'].set_index(key[1:])
    pairs = []
    for method in METHODS:
        if method == 'adaptive40':
            continue
        other = frame[frame.method == method].set_index(key[1:])
        if not base.index.equals(other.index):
            raise AssertionError('Paired method-condition rows are not aligned')
        for condition in base.index:
            pairs.append({'method': method, 'event_count': condition[0],
                          'repeat': condition[1], 'position_sigma_mm': condition[2],
                          'iou_gain': float(other.loc[condition, 'iou'] - base.loc[condition, 'iou']),
                          'global_error_reduction_mm': float(base.loc[condition, 'global_error_mm']
                                                              - other.loc[condition, 'global_error_mm']),
                          'auto_error_reduction_mm': float(base.loc[condition, 'auto_error_mm']
                                                            - other.loc[condition, 'auto_error_mm'])})
    paired = pd.DataFrame(pairs)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    noisy = paired[paired.position_sigma_mm > 0]
    paired_summary = noisy.groupby('method', as_index=False).agg(
        paired_n=('iou_gain', 'size'), iou_gain_mean=('iou_gain', 'mean'),
        iou_win_fraction=('iou_gain', lambda x: float((x > 0).mean())),
        global_error_reduction_mm_mean=('global_error_reduction_mm', 'mean'),
        auto_error_reduction_mm_mean=('auto_error_reduction_mm', 'mean'))
    paired_summary.to_csv(output_dir / 'paired_noisy_summary.csv', index=False)
    fig, plot_axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    part = summary[summary.event_count == max(counts)]
    for method in METHODS:
        piece = part[part.method == method].sort_values('position_sigma_mm')
        for ax, metric in zip(plot_axes, ('iou', 'auto_error_mm', 'refined_coarse_slices')):
            ax.errorbar(piece.position_sigma_mm, piece[f'{metric}_mean'],
                        yerr=piece[f'{metric}_std'], marker='o', capsize=2,
                        label=method)
    for ax, ylabel in zip(plot_axes, ('Equal-area IoU', 'Auto-window error [mm]',
                                       'Refined 16³ coarse slices')):
        ax.set(xlabel='Hit-position sigma [mm]', ylabel=ylabel)
        ax.grid(alpha=.25)
    plot_axes[0].legend(fontsize=7)
    fig.suptitle('U development: neighborhood proposal with local z-slice support')
    fig.savefig(output_dir / 'comparison_5000.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'input_csv': str(input_csv), 'input_rows': len(data),
        'scene_role': 'U development, not independent validation',
        'counts': counts, 'sigmas_mm': sigmas, 'repeats': repeats,
        'seed': RANDOM_SEED + 831, 'neighbor_xy': [3, 3], 'z_group_size': 4,
        'local_min_events': LOCAL_MIN_EVENTS,
        'proposal': '3x3 XY neighbor n_eff over 4 z slices >= 2*(1+(theta_noise/3mrad)^2)',
        'execution': 'refine only local 16^3 slices with >=2 selected events',
        'shrink_alpha': 'local_n_eff/(local_n_eff+proposal_threshold)',
        'truth_used_for_selection': False,
        'timing_note': 'local_hard and local_shrink share candidate computation; runtime is joint diagnostic.',
        'warning': 'L shape has informed rule design and is not an independent test.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(paired_summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'muon_analysis/muon_tracks.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/local_support_refinement_results')
    parser.add_argument('--counts', type=int, nargs='+', default=list(COUNTS))
    parser.add_argument('--sigmas', type=float, nargs='+', default=list(SIGMAS_MM))
    parser.add_argument('--repeats', type=int, default=REPEATS)
    args = parser.parse_args()
    run(args.input, args.output_dir, tuple(args.counts), tuple(args.sigmas), args.repeats)


if __name__ == '__main__':
    main()
