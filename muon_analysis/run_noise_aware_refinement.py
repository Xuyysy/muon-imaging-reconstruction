"""开发实验：按局部有效事件数和已知击中误差决定是否细分体素。

只用中心 U 型的现有 Geant4 CSV 开发规则；真值仅用于事后评分。
L 型结果已提示固定 40% 细分的局限，所以不能再把 L 型当作本规则的独立测试。
脚本不修改原始 CSV，也不自动修改论文。
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
    RANDOM_SEED, THETA_CAP_MRAD, adaptive_volume, fixed_fine_volume, imaging_edges,
)
from run_baseline_experiment import u_target_mask
from run_continuous_event_quality_experiment import evaluate
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
METHODS = ('coarse16', 'fixed32', 'adaptive40', 'noise_aware')


def noise_aware_volume(events: pd.DataFrame, sigma_mm: float,
                       layer_spacing_mm: float) -> tuple[np.ndarray, int, float]:
    """根据粗体素有效样本数作无真值的一层 16³→32³ 细分。

    n_eff=(Σw)^2/Σw²，避免一个大权重事件伪装成多个独立事件。
    无噪声门槛设为 2 个有效事件：少于两条独立信息时，粗体素内
    无法作最基本的空间对比。这里并不声称两条事件足以可靠估计
    全部 8 个子体素；角度测量噪声相对 3 mrad 权重上限越大，
    门槛越高。门槛来自可辨识性的最低要求，并非按目标 IoU 调优。
    未达门槛的粗体素把总权重均分到 8 子体素；达门槛时使用实际
    落点。两种情况均保持总散射权重守恒。
    """
    coarse_edges, fine_edges = imaging_edges(16), imaging_edges(32)
    points = events[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy()
    weights = events['capped_weight'].to_numpy()
    coarse, _ = np.histogramdd(points, bins=coarse_edges, weights=weights)
    squared, _ = np.histogramdd(points, bins=coarse_edges, weights=weights ** 2)
    fine, _ = np.histogramdd(points, bins=fine_edges, weights=weights)
    neff = np.divide(coarse ** 2, squared, out=np.zeros_like(coarse), where=squared > 0)
    theta_noise = theta_noise_floor_mrad(sigma_mm, layer_spacing_mm)
    min_neff = 2.0 * (1.0 + (theta_noise / THETA_CAP_MRAD) ** 2)
    selected = neff >= min_neff
    # 对未细分的粗体素，写回均匀分配；被细分者保留 fine 的实际直方图。
    uniform = np.repeat(np.repeat(np.repeat(coarse / 8.0, 2, axis=0), 2, axis=1), 2, axis=2)
    selection_fine = np.repeat(np.repeat(np.repeat(selected, 2, axis=0), 2, axis=1), 2, axis=2)
    volume = np.where(selection_fine, fine, uniform)
    if not np.isclose(volume.sum(), coarse.sum(), rtol=1e-10, atol=1e-8):
        raise AssertionError('Refinement must conserve total scattering weight')
    return volume, int(selected.sum()), float(min_neff)


def run(input_csv: Path, out_dir: Path, counts: tuple[int, ...],
        sigmas: tuple[float, ...], repeats: int) -> None:
    data = pd.read_csv(input_csv)
    if len(data) < max(counts):
        raise ValueError('Complete-event CSV is smaller than requested sample')
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = same_side_layer_spacing_mm(data)
    edges = imaging_edges(32)
    cx, cy = [(axis[1:] + axis[:-1]) / 2 for axis in edges[:2]]
    truth = u_target_mask(cx, cy)
    rng = np.random.default_rng(RANDOM_SEED + 831)
    rows: list[dict] = []
    for count in counts:
        for repeat in range(1, repeats + 1):
            subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
            # 同一事件子集和同一组标准正态抽样用于全部 sigma，便于配对。
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                     for prefix in ('up1', 'up2', 'down1', 'down2')
                     for axis in ('x', 'y')}
            for sigma in sigmas:
                measured = measured_hits(subset, sigma, draws)
                observed = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
                events = corrected_events(observed, sigma, spacing)
                coarse, _, coarse_selected, _ = adaptive_volume(events, 0.0)
                fixed, _ = fixed_fine_volume(events)
                adaptive, _, old_selected, _ = adaptive_volume(events, .40)
                aware, selected, min_neff = noise_aware_volume(events, sigma, spacing)
                volumes = {'coarse16': coarse, 'fixed32': fixed,
                           'adaptive40': adaptive, 'noise_aware': aware}
                for name, volume in volumes.items():
                    metrics = evaluate(volume.sum(axis=2), truth, cx, cy)
                    rows.append({'method': name, 'event_count': count, 'repeat': repeat,
                                 'position_sigma_mm': sigma, 'selected_events': len(events),
                                 'refined_coarse_voxels': selected if name == 'noise_aware'
                                 else old_selected if name == 'adaptive40' else coarse_selected
                                 if name == 'coarse16' else 4096,
                                 'min_effective_events': min_neff if name == 'noise_aware' else np.nan,
                                 **metrics})
            print(f'U development: N={count}, repeat={repeat}/{repeats}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / 'by_repeat.csv', index=False)
    group = ['method', 'event_count', 'position_sigma_mm']
    metrics = ('iou', 'global_error_mm', 'auto_error_mm', 'refined_coarse_voxels')
    summary = frame.groupby(group, as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in metrics},
        **{f'{name}_std': (name, 'std') for name in metrics})
    summary.to_csv(out_dir / 'summary.csv', index=False)
    reference = frame[frame.method == 'adaptive40'].set_index(
        ['event_count', 'repeat', 'position_sigma_mm'])
    pairs = []
    for name in METHODS:
        if name == 'adaptive40':
            continue
        other = frame[frame.method == name].set_index(reference.index.names)
        if not reference.index.equals(other.index):
            raise AssertionError('Paired conditions are not aligned')
        for key in reference.index:
            pairs.append({'method': name, 'event_count': key[0], 'repeat': key[1],
                          'position_sigma_mm': key[2],
                          'iou_gain': float(other.loc[key, 'iou'] - reference.loc[key, 'iou']),
                          'global_error_reduction_mm': float(reference.loc[key, 'global_error_mm']
                                                              - other.loc[key, 'global_error_mm']),
                          'auto_error_reduction_mm': float(reference.loc[key, 'auto_error_mm']
                                                            - other.loc[key, 'auto_error_mm'])})
    paired = pd.DataFrame(pairs)
    paired.to_csv(out_dir / 'paired_by_repeat.csv', index=False)
    noisy = paired[paired.position_sigma_mm > 0]
    comparison = noisy.groupby('method', as_index=False).agg(
        paired_n=('iou_gain', 'size'), iou_gain_mean=('iou_gain', 'mean'),
        iou_win_fraction=('iou_gain', lambda x: float((x > 0).mean())),
        global_error_reduction_mm_mean=('global_error_reduction_mm', 'mean'),
        auto_error_reduction_mm_mean=('auto_error_reduction_mm', 'mean'))
    comparison.to_csv(out_dir / 'paired_noisy_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    part = summary[summary.event_count == max(counts)]
    for name in METHODS:
        block = part[part.method == name].sort_values('position_sigma_mm')
        for ax, metric in zip(axes, ('iou', 'global_error_mm', 'auto_error_mm')):
            ax.errorbar(block.position_sigma_mm, block[f'{metric}_mean'],
                        yerr=block[f'{metric}_std'], marker='o', capsize=2, label=name)
    for ax, label in zip(axes, ('Equal-area IoU', 'Global centroid error [mm]',
                                'Automatic-window error [mm]')):
        ax.set(xlabel='Hit-position sigma [mm]', ylabel=label)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('U-shape development: noise-aware refinement vs frozen baselines')
    fig.savefig(out_dir / 'comparison.png', dpi=180)
    plt.close(fig)
    (out_dir / 'config.json').write_text(json.dumps({
        'input_csv': str(input_csv), 'input_rows': len(data), 'scene_role': 'development U only',
        'counts': counts, 'sigmas_mm': sigmas, 'repeats': repeats,
        'seed': RANDOM_SEED + 831, 'angle_floor_mrad': THETA_CAP_MRAD,
        'rule': 'refine if effective count >= 2*(1+(theta_noise/3mrad)^2)',
        'truth_used_for_refinement': False,
        'warning': 'L shape has informed problem formulation; it is not independent validation for this rule.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(comparison.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'muon_analysis/muon_tracks.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/noise_aware_refinement_results')
    parser.add_argument('--counts', type=int, nargs='+', default=list(COUNTS))
    parser.add_argument('--sigmas', type=float, nargs='+', default=list(SIGMAS_MM))
    parser.add_argument('--repeats', type=int, default=REPEATS)
    args = parser.parse_args()
    run(args.input, args.output_dir, tuple(args.counts), tuple(args.sigmas), args.repeats)


if __name__ == '__main__':
    main()
