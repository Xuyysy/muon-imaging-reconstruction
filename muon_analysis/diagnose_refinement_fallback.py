"""事后分析局部细分失败，并冻结一个不使用目标真值的保守回退规则。

该脚本读取上轮逐重复 CSV；仅为局部区域位置归因重建相同 U 型
事件子样本。目标真值仅用于事后诊断和评分，不参与策略选择。
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

from run_adaptive_voxel_experiment import RANDOM_SEED, THETA_CAP_MRAD, imaging_edges
from run_baseline_experiment import u_target_mask
from run_continuous_event_quality_experiment import evaluate
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_local_support_refinement import local_support_volumes
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
NOISE_FRACTION_LIMIT = 0.25
MIN_SELECTED_EVENTS = 16 * 16  # 粗 XY 投影每格平均至少约 1 条 PoCA。
KEY = ['event_count', 'repeat', 'position_sigma_mm']


def choose_reconstruction(selected_events: int, sigma_mm: float,
                          spacing_mm: float) -> str:
    """不读取图像/真值/评价指标，只按几何规模和噪声尺度回退。"""
    theta_noise = theta_noise_floor_mrad(sigma_mm, spacing_mm)
    sufficient_events = selected_events >= MIN_SELECTED_EVENTS
    sufficiently_quiet = theta_noise < NOISE_FRACTION_LIMIT * THETA_CAP_MRAD
    return 'adaptive40' if sufficient_events and sufficiently_quiet else 'coarse16'


def policy_table(raw: pd.DataFrame, spacing_mm: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同一配对键的现成重建结果之间做选择，绝不重算或覆盖原表。"""
    if raw.duplicated(KEY + ['method']).any():
        raise ValueError('Duplicate condition-method keys in source results')
    by_method = {name: raw[raw.method == name].set_index(KEY).sort_index()
                 for name in ('coarse16', 'adaptive40', 'local_shrink')}
    if not by_method['coarse16'].index.equals(by_method['adaptive40'].index):
        raise ValueError('Baseline condition keys are not aligned')
    rows = []
    for condition in by_method['coarse16'].index:
        coarse = by_method['coarse16'].loc[condition]
        adaptive = by_method['adaptive40'].loc[condition]
        if int(coarse.selected_events) != int(adaptive.selected_events):
            raise AssertionError('Paired methods must share accepted events')
        choice = choose_reconstruction(int(coarse.selected_events), condition[2], spacing_mm)
        chosen = by_method[choice].loc[condition]
        rows.append({'event_count': condition[0], 'repeat': condition[1],
                     'position_sigma_mm': condition[2],
                     'selected_events': int(coarse.selected_events),
                     'theta_noise_mrad': theta_noise_floor_mrad(condition[2], spacing_mm),
                     'chosen_method': choice, 'iou': float(chosen.iou),
                     'global_error_mm': float(chosen.global_error_mm),
                     'auto_error_mm': float(chosen.auto_error_mm),
                     'iou_gain_vs_coarse': float(chosen.iou - coarse.iou),
                     'iou_gain_vs_adaptive40': float(chosen.iou - adaptive.iou),
                     'auto_error_reduction_vs_coarse_mm': float(coarse.auto_error_mm
                                                                 - chosen.auto_error_mm)})
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        adaptive40_chosen=('chosen_method', lambda x: int((x == 'adaptive40').sum())),
        selected_events_mean=('selected_events', 'mean'),
        iou_mean=('iou', 'mean'), iou_std=('iou', 'std'),
        auto_error_mm_mean=('auto_error_mm', 'mean'),
        iou_gain_vs_coarse_mean=('iou_gain_vs_coarse', 'mean'),
        auto_error_reduction_vs_coarse_mm_mean=(
            'auto_error_reduction_vs_coarse_mm', 'mean'))
    return frame, summary


def spatial_diagnostic(data: pd.DataFrame, source: pd.DataFrame,
                       spacing_mm: float) -> pd.DataFrame:
    """只在事后把选中体素与已知 U 形几何比较，不回写选择规则。"""
    fine_edges = imaging_edges(32)
    cx, cy = [(axis[1:] + axis[:-1]) / 2 for axis in fine_edges[:2]]
    truth_xy = u_target_mask(cx, cy)
    # 32² 真值折成 16² 粗 XY 网格，部分相交的格定义为边界。
    overlap = truth_xy.reshape(16, 2, 16, 2).mean(axis=(1, 3))
    coarse_z = imaging_edges(16)[2]
    cz = (coarse_z[1:] + coarse_z[:-1]) / 2
    in_target_z = (cz >= -50.0) & (cz <= 50.0)
    interior = (overlap == 1)[:, :, None] & in_target_z[None, None, :]
    boundary = (overlap > 0)[:, :, None] & (overlap < 1)[:, :, None] \
        & in_target_z[None, None, :]
    rng = np.random.default_rng(RANDOM_SEED + 831)
    lookup = source[source.method == 'local_shrink'].set_index(KEY)
    rows = []
    for count in COUNTS:
        for repeat in range(1, REPEATS + 1):
            subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                     for prefix in ('up1', 'up2', 'down1', 'down2')
                     for axis in ('x', 'y')}
            if count != 5000:
                continue  # 仍需消耗相同 RNG，才能与上一轮 5000 条子集配对。
            for sigma in SIGMAS_MM:
                measured = measured_hits(subset, sigma, draws)
                observed = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
                events = corrected_events(observed, sigma, spacing_mm)
                volumes, info, debug = local_support_volumes(
                    events, sigma, spacing_mm, return_debug=True)
                condition = (count, repeat, sigma)
                measured_metrics = evaluate(volumes['local_shrink'].sum(axis=2),
                                            truth_xy, cx, cy)
                if not np.isclose(measured_metrics['iou'], lookup.loc[condition, 'iou']):
                    raise AssertionError('Replayed event/noise subset differs from source result')
                selected = debug['selected']
                weight = debug['coarse_weight']
                n = int(selected.sum())
                interior_n = int(np.sum(selected & interior))
                boundary_n = int(np.sum(selected & boundary))
                outside_n = n - interior_n - boundary_n
                chosen_weight = float(weight[selected].sum())
                outside_weight = float(weight[selected & ~interior & ~boundary].sum())
                rows.append({'event_count': count, 'repeat': repeat,
                             'position_sigma_mm': sigma,
                             'selected_events': len(events),
                             'proposed_coarse_slices': info['proposed_coarse_slices'],
                             'refined_coarse_slices': n,
                             'interior_slices': interior_n,
                             'boundary_slices': boundary_n,
                             'outside_geometric_target_slices': outside_n,
                             'outside_selected_fraction': outside_n / n if n else np.nan,
                             'outside_selected_weight_fraction':
                             outside_weight / chosen_weight if chosen_weight > 0 else np.nan,
                             'iou_gain_vs_coarse': float(measured_metrics['iou'] -
                                 source[(source.method == 'coarse16')
                                        & (source.event_count == count)
                                        & (source.repeat == repeat)
                                        & (source.position_sigma_mm == sigma)].iloc[0].iou),
                             'auto_error_reduction_vs_coarse_mm': float(
                                 source[(source.method == 'coarse16')
                                        & (source.event_count == count)
                                        & (source.repeat == repeat)
                                        & (source.position_sigma_mm == sigma)].iloc[0].auto_error_mm
                                 - measured_metrics['auto_error_mm'])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'muon_analysis/muon_tracks.csv')
    parser.add_argument('--source-results', type=Path,
                        default=ROOT / 'muon_analysis/local_support_refinement_results/by_repeat.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/refinement_fallback_results')
    args = parser.parse_args()
    raw = pd.read_csv(args.source_results)
    data = pd.read_csv(args.input)
    spacing = same_side_layer_spacing_mm(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy, policy_summary = policy_table(raw, spacing)
    policy.to_csv(args.output_dir / 'policy_by_repeat.csv', index=False)
    policy_summary.to_csv(args.output_dir / 'policy_summary.csv', index=False)
    spatial = spatial_diagnostic(data, raw, spacing)
    spatial.to_csv(args.output_dir / 'spatial_diagnostic_by_repeat.csv', index=False)
    spatial_summary = spatial.groupby('position_sigma_mm', as_index=False).agg(
        repeats=('repeat', 'size'),
        refined_coarse_slices_mean=('refined_coarse_slices', 'mean'),
        outside_selected_fraction_mean=('outside_selected_fraction', 'mean'),
        outside_selected_weight_fraction_mean=('outside_selected_weight_fraction', 'mean'),
        iou_gain_vs_coarse_mean=('iou_gain_vs_coarse', 'mean'),
        auto_error_reduction_vs_coarse_mm_mean=(
            'auto_error_reduction_vs_coarse_mm', 'mean'))
    spatial_summary.to_csv(args.output_dir / 'spatial_diagnostic_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    part = policy_summary[policy_summary.event_count == 5000].sort_values('position_sigma_mm')
    for method in ('coarse16', 'adaptive40', 'local_shrink'):
        block = raw[(raw.method == method) & (raw.event_count == 5000)]
        values = block.groupby('position_sigma_mm', as_index=False).agg(iou=('iou', 'mean'))
        axes[0].plot(values.position_sigma_mm, values.iou, 'o-', label=method)
    axes[0].plot(part.position_sigma_mm, part.iou_mean, 'ko--', label='conservative policy')
    axes[0].set(xlabel='Hit-position sigma [mm]', ylabel='Equal-area IoU')
    axes[0].legend(fontsize=7)
    axes[1].plot(spatial_summary.position_sigma_mm,
                 spatial_summary.outside_selected_fraction_mean, 'o-')
    axes[1].set(xlabel='Hit-position sigma [mm]',
                ylabel='Selected coarse slices outside target envelope')
    axes[1].set_xlim(-0.01, 0.31)
    axes[1].text(0.98, 0.08, 'sigma=0.30: no slices refined',
                 transform=axes[1].transAxes, ha='right', fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle('U development: post-hoc failure attribution and conservative fallback')
    fig.savefig(args.output_dir / 'diagnostic_5000.png', dpi=180)
    plt.close(fig)
    (args.output_dir / 'config.json').write_text(json.dumps({
        'input_csv': str(args.input), 'source_results': str(args.source_results),
        'scene_role': 'U development only', 'counts': COUNTS,
        'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'policy': 'adaptive40 iff selected_events >= 16*16 and theta_noise < 0.25*theta_cap; else coarse16',
        'source_threshold': 'same-side spacing from input CSV, theta cap 3 mrad',
        'truth_usage': 'post-hoc diagnostic and scoring only',
        'validation_warning': 'L shape already informed design; new geometry is required for independent test.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(policy_summary.to_string(index=False))
    print(spatial_summary.to_string(index=False))


if __name__ == '__main__':
    main()
