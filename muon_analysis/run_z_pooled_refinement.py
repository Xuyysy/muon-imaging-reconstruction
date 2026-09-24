"""沿 z 汇总局部统计量的自适应 XY 细分开发实验。

上轮 16³ 粗体素太稀疏。本轮只改变细分判据汇总的 z 层数：
1、4、16 层。无论汇总多少层，重建仍保留 16 层 z 网格；因此不能
把二维投影上的改善误称为三维 z 定位改善。U 型真值仅用于事后评分。
"""
from __future__ import annotations

import argparse
import json
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
from run_noise_aware_refinement import noise_aware_volume
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
GROUP_SIZES = (1, 4, 16)
METHODS = ('coarse16', 'adaptive40', 'noise_aware_3d',
           'z_pool_1', 'z_pool_4', 'z_pool_16')


def z_pooled_volume(events: pd.DataFrame, sigma_mm: float, spacing_mm: float,
                    z_group_size: int) -> tuple[np.ndarray, dict[str, float]]:
    """先在 z_group_size 个粗层共享有效计数，再决定其中各层的 XY 细分。

    返回网格是 32×32×16；细分只作用于 x/y，不细分 z。
    每个粗 z 层自身的散射权重仍保留，不将事件搬到其他 z 层。
    """
    if z_group_size not in GROUP_SIZES:
        raise ValueError('z_group_size must be one of 1, 4, 16')
    coarse_edges, fine_edges = imaging_edges(16), imaging_edges(32)
    points = events[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy()
    weight = events['capped_weight'].to_numpy()
    coarse, _ = np.histogramdd(points, bins=coarse_edges, weights=weight)
    squared, _ = np.histogramdd(points, bins=coarse_edges, weights=weight ** 2)
    fine_xy, _ = np.histogramdd(points,
                                bins=(fine_edges[0], fine_edges[1], coarse_edges[2]),
                                weights=weight)
    # 只对判据汇总 z；重建本身仍按 16 个物理 z 层分别累积。
    grouped_weight = coarse.reshape(16, 16, 16 // z_group_size, z_group_size).sum(axis=3)
    grouped_sq = squared.reshape(16, 16, 16 // z_group_size, z_group_size).sum(axis=3)
    neff = np.divide(grouped_weight ** 2, grouped_sq,
                     out=np.zeros_like(grouped_weight), where=grouped_sq > 0)
    theta_noise = theta_noise_floor_mrad(sigma_mm, spacing_mm)
    threshold = 2.0 * (1.0 + (theta_noise / THETA_CAP_MRAD) ** 2)
    select_group = neff >= threshold
    select_slice = np.repeat(select_group, z_group_size, axis=2)
    select_fine_xy = np.repeat(np.repeat(select_slice, 2, axis=0), 2, axis=1)
    uniform_xy = np.repeat(np.repeat(coarse / 4.0, 2, axis=0), 2, axis=1)
    output = np.where(select_fine_xy, fine_xy, uniform_xy)
    if not np.isclose(output.sum(), coarse.sum(), rtol=1e-10, atol=1e-8):
        raise AssertionError('XY refinement must conserve scattering weight')
    return output, {
        'eligible_nonempty_regions': int(np.sum(neff > 0)),
        'selected_regions': int(select_group.sum()),
        'refined_coarse_slices': int(select_slice.sum()),
        'max_effective_events': float(neff.max()),
        'threshold_effective_events': float(threshold),
    }


def run(input_csv: Path, output_dir: Path, counts: tuple[int, ...],
        sigmas: tuple[float, ...], repeats: int) -> None:
    data = pd.read_csv(input_csv)
    if len(data) < max(counts):
        raise ValueError('Not enough four-layer complete events')
    output_dir.mkdir(parents=True, exist_ok=True)
    spacing = same_side_layer_spacing_mm(data)
    edges = imaging_edges(32)
    cx, cy = [(axis[1:] + axis[:-1]) / 2 for axis in edges[:2]]
    truth = u_target_mask(cx, cy)
    rng = np.random.default_rng(RANDOM_SEED + 831)  # 和上一轮共用事件/噪声配对设计。
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
                coarse, _, _, _ = adaptive_volume(events, 0.0)
                adaptive, _, old_selected, _ = adaptive_volume(events, .40)
                three_d, selected_3d, _ = noise_aware_volume(events, sigma, spacing)
                variants = {
                    'coarse16': (coarse, {'selected_regions': 0,
                                          'refined_coarse_slices': 0}),
                    'adaptive40': (adaptive, {'selected_regions': old_selected,
                                               'refined_coarse_slices': old_selected}),
                    'noise_aware_3d': (three_d, {'selected_regions': selected_3d,
                                                 'refined_coarse_slices': selected_3d}),
                }
                for size in GROUP_SIZES:
                    variants[f'z_pool_{size}'] = z_pooled_volume(events, sigma, spacing, size)
                # 一层分组时，XY 投影必须和之前的 3D 门槛规则一致。
                if not np.allclose(variants['z_pool_1'][0].sum(axis=2),
                                   three_d.sum(axis=2), rtol=1e-12, atol=1e-9):
                    raise AssertionError('z_pool_1 does not match the prior 3D projection')
                for method, (volume, diagnostics) in variants.items():
                    metrics = evaluate(volume.sum(axis=2), truth, cx, cy)
                    rows.append({'method': method, 'event_count': count,
                                 'repeat': repeat, 'position_sigma_mm': sigma,
                                 'selected_events': len(events),
                                 'z_group_size': int(method.split('_')[-1])
                                 if method.startswith('z_pool_') else np.nan,
                                 'z_output_bin_width_mm': 12.5 if method.startswith('z_pool_')
                                 else 6.25 if method != 'coarse16' else 12.5,
                                 **diagnostics, **metrics})
            print(f'z pooling: N={count}, repeat={repeat}/{repeats}', flush=True)
    frame = pd.DataFrame(rows)
    if frame.duplicated(['method', 'event_count', 'repeat', 'position_sigma_mm']).any():
        raise AssertionError('Duplicated condition-method key')
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary = frame.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{col}_mean': (col, 'mean') for col in
           ('iou', 'global_error_mm', 'auto_error_mm', 'selected_regions',
            'refined_coarse_slices', 'eligible_nonempty_regions', 'max_effective_events')},
        **{f'{col}_std': (col, 'std') for col in
           ('iou', 'global_error_mm', 'auto_error_mm', 'selected_regions',
            'refined_coarse_slices')})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    base = frame[frame.method == 'adaptive40'].set_index(
        ['event_count', 'repeat', 'position_sigma_mm'])
    pairs = []
    for method in METHODS:
        if method == 'adaptive40':
            continue
        other = frame[frame.method == method].set_index(base.index.names)
        if not base.index.equals(other.index):
            raise AssertionError('Paired rows do not align')
        for key in base.index:
            pairs.append({'method': method, 'event_count': key[0], 'repeat': key[1],
                          'position_sigma_mm': key[2],
                          'iou_gain': float(other.loc[key, 'iou'] - base.loc[key, 'iou']),
                          'global_error_reduction_mm': float(base.loc[key, 'global_error_mm']
                                                             - other.loc[key, 'global_error_mm']),
                          'auto_error_reduction_mm': float(base.loc[key, 'auto_error_mm']
                                                           - other.loc[key, 'auto_error_mm'])})
    paired = pd.DataFrame(pairs)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    noisy = paired[paired.position_sigma_mm > 0]
    paired_summary = noisy.groupby('method', as_index=False).agg(
        paired_n=('iou_gain', 'size'), iou_gain_mean=('iou_gain', 'mean'),
        iou_win_fraction=('iou_gain', lambda x: float((x > 0).mean())),
        global_error_reduction_mm_mean=('global_error_reduction_mm', 'mean'),
        auto_error_reduction_mm_mean=('auto_error_reduction_mm', 'mean'))
    paired_summary.to_csv(output_dir / 'paired_noisy_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    part = summary[summary.event_count == max(counts)]
    for method in METHODS:
        section = part[part.method == method].sort_values('position_sigma_mm')
        for ax, col in zip(axes, ('iou', 'auto_error_mm', 'refined_coarse_slices')):
            ax.errorbar(section.position_sigma_mm, section[f'{col}_mean'],
                        yerr=section[f'{col}_std'], marker='o', capsize=2,
                        label=method)
    for ax, label in zip(axes, ('Equal-area IoU', 'Auto-window error [mm]',
                                'Refined 16³ coarse slices')):
        ax.set(xlabel='Hit-position sigma [mm]', ylabel=label)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle('U development: sharing statistics across z, 5000 input events')
    fig.savefig(output_dir / 'comparison_5000.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'input_csv': str(input_csv), 'input_rows': len(data),
        'scene_role': 'U development, not independent held-out',
        'counts': counts, 'sigmas_mm': sigmas, 'repeats': repeats,
        'seed': RANDOM_SEED + 831, 'z_group_sizes': GROUP_SIZES,
        'coarse_bins': [16, 16, 16], 'xy_child_bins': [2, 2],
        'rule': 'n_eff >= 2*(1+(theta_noise/3mrad)^2)',
        'truth_used_for_selection': False,
        'z_caveat': 'Pooled variants retain 16 z bins, 12.5 mm nominal z width.',
        'warning': 'L shape already informed this design and cannot validate it independently.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(paired_summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'muon_analysis/muon_tracks.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/z_pooled_refinement_results')
    parser.add_argument('--counts', type=int, nargs='+', default=list(COUNTS))
    parser.add_argument('--sigmas', type=float, nargs='+', default=list(SIGMAS_MM))
    parser.add_argument('--repeats', type=int, default=REPEATS)
    args = parser.parse_args()
    run(args.input, args.output_dir, tuple(args.counts), tuple(args.sigmas), args.repeats)


if __name__ == '__main__':
    main()
