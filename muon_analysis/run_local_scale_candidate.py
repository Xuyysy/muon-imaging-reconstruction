"""冻结局部尺度候选的开发消融；不是独立验证或三维叶体素创新。

先读B1/LOCAL_SCALE_CANDIDATE_PLAN.md。复用360条件，核宽和两个
因子的常数都在运行前固定。H/S旧得分逐条核对，不改已有结果。
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
from diagnose_split_confidence import ROOT, SCENES
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges, Z_LIMIT_MM
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_soft_z_gate_experiment import COUNTS, SIGMAS_MM, REPEATS, GRIDS, candidate_events, xy_image
from run_frame_z_gate_validation import smooth_hard_image, SMOOTH_SIGMA_MM
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm
from run_fixed_budget_refinement import fractional_pair_metrics, continuous_image_metrics
from run_shape_smoothing_control import KEYS, empty_region, shape_scores
from local_scale_reconstruction import (
    native_support, local_scale_image, SUPPORT_HALF_WIDTH_MM,
    SUPPORT_SCALE, LOG_CONTRAST_SCALE, GUIDE_FLOOR_RATIO,
)

METHODS = ('hard', 'smoothed', 'support_only', 'edge_only', 'combined')
METRICS = ('fractional_iou', 'target_mass_fraction', 'empty_mass_fraction',
           'empty_fractional_occupancy', 'footprint_proxy_distance')


def run(output_dir):
    edges = imaging_edges(32)
    cx, cy = [(a[:-1]+a[1:])/2 for a in edges[:2]]
    rows, pictures = [], {}
    for index, (scene, (filename, mask_fn)) in enumerate(SCENES.items()):
        data = pd.read_csv(ROOT/'muon_analysis'/filename)
        truth, empty = mask_fn(cx, cy), empty_region(scene, cx, cy)
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED+2107+index*100)
        for count in COUNTS:
            for repeat in range(1, REPEATS+1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                draws = {f'{plane}_{axis}': rng.normal(0, 1, count)
                         for plane in ('up1', 'up2', 'down1', 'down2') for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(measured, 0., np.random.default_rng(0))
                    ids, weight = candidate_events(observed, sigma, spacing)
                    weight *= (np.abs(observed.poca_z_mm.to_numpy()[ids]) <= Z_LIMIT_MM)
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    for bins in GRIDS:
                        base = xy_image(xy, weight, bins)
                        fixed = smooth_hard_image(base, bins)
                        # 此函数仅知道位置/权重/查询网格，不接受真值或形状。
                        neff, guide = native_support(xy, weight, bins)
                        images, diagnostics = {'hard': base, 'smoothed': fixed}, {}
                        for method, support, edge in (('support_only', True, False),
                                                      ('edge_only', False, True), ('combined', True, True)):
                            images[method], diagnostics[method] = local_scale_image(base, bins, neff, guide, support, edge)
                        for method in METHODS:
                            image = images[method]
                            rows.append({'scenario': scene, 'event_count': count, 'repeat': repeat,
                                         'position_sigma_mm': sigma, 'grid': bins, 'method': method,
                                         'total_image_weight': float(image.sum()),
                                         **fractional_pair_metrics(image, base, truth),
                                         **continuous_image_metrics(image, base, truth),
                                         **shape_scores(image, truth, empty, cx, scene),
                                         **diagnostics.get(method, {'alpha_mass_mean': float(method == 'smoothed'),
                                                                    'physical_neff_mass_mean': np.nan})})
                            if count == 5000 and repeat == 1 and sigma == .05 and bins == 32:
                                pictures[(scene, method)] = image
                print(f'Local-scale candidate: {scene}, N={count}, repeat={repeat}', flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 1800 or frame.duplicated(KEYS+['method']).any():
        raise AssertionError('Incomplete local-scale experiment')
    old = pd.read_csv(ROOT/'muon_analysis/shape_smoothing_control_results/by_repeat.csv')
    paired_old = frame[frame.method.isin(['hard', 'smoothed'])].merge(
        old, on=KEYS+['method'], suffixes=('', '_old'), validate='one_to_one')
    if len(paired_old) != 720:
        raise AssertionError('Old H/S pairing lost')
    for metric in (*METRICS, 'total_image_weight'):
        if not np.allclose(paired_old[metric], paired_old[metric+'_old'], atol=1e-12, rtol=1e-12):
            raise AssertionError(f'Frozen H/S metric mismatch: {metric}')
    views = {m: frame[frame.method == m].set_index(KEYS).sort_index() for m in METHODS}
    for method in METHODS:
        if not np.allclose(views[method].total_image_weight, views['hard'].total_image_weight, rtol=1e-12, atol=1e-9):
            raise AssertionError('Candidate mass differs from H')
    # 各列保留差值符号：IoU/目标内比例越高越好，空区/代理距离越低越好。
    paired = pd.DataFrame({
        **{f'{m}_{metric}': views[m][metric] for m in METHODS for metric in METRICS},
        **{f'{m}_minus_{reference}_{metric}': views[m][metric]-views[reference][metric]
           for m in ('support_only', 'edge_only', 'combined') for reference in ('hard', 'smoothed') for metric in METRICS},
    }).reset_index()
    group_keys = [k for k in KEYS if k != 'repeat']
    summary = paired.groupby(group_keys, as_index=False).mean(numeric_only=True).drop(columns='repeat')
    spread = paired.groupby(group_keys, as_index=False).agg(
        repeats=('repeat', 'size'), combined_iou_diff_s_std=('combined_minus_smoothed_fractional_iou', 'std'),
        combined_iou_wins_s=('combined_minus_smoothed_fractional_iou', lambda x: int((x > 1e-12).sum())),
        combined_empty_improvements_s=('combined_minus_smoothed_empty_mass_fraction', lambda x: int((x < -1e-12).sum())))
    summary = summary.merge(spread, on=group_keys, validate='one_to_one')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir/'by_repeat.csv', index=False)
    paired.to_csv(output_dir/'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir/'summary.csv', index=False)
    (output_dir/'config.json').write_text(json.dumps({
        'plan': 'B1/LOCAL_SCALE_CANDIDATE_PLAN.md', 'role': 'old-data development ablation, not independent validation',
        'methods': METHODS, 'paired_conditions': 360, 'old_H_S_rows_reproduced': len(paired_old),
        'spatial_sigma_mm': SMOOTH_SIGMA_MM, 'physical_support_half_width_mm': SUPPORT_HALF_WIDTH_MM,
        'support_scale': SUPPORT_SCALE, 'log_contrast_scale': LOG_CONTRAST_SCALE, 'guide_floor_ratio': GUIDE_FLOOR_RATIO,
        'truth_in_reconstruction': False, 'changes_3d_voxel_tree': False, 'parameter_search': False,
    }, indent=2)+'\n')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for j, scene in enumerate(SCENES):
        part = summary[(summary.scenario == scene)&(summary.event_count == 5000)&(summary.grid == 32)]
        for method in ('support_only', 'edge_only', 'combined'):
            axes[0, j].plot(part.position_sigma_mm, part[f'{method}_minus_smoothed_fractional_iou'], 'o-', label=method)
            axes[1, j].plot(part.position_sigma_mm, part[f'{method}_minus_smoothed_empty_mass_fraction'], 'o-', label=method)
        for ax in axes[:, j]:
            ax.axhline(0, color='black', lw=.7); ax.grid(alpha=.2)
            ax.set(title=scene, xlabel='Hit-position sigma [mm]')
    axes[0, 0].set_ylabel('Fractional IoU difference vs fixed S (higher better)')
    axes[1, 0].set_ylabel('Empty-region mass difference vs fixed S (lower better)')
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output_dir/'local_scale_ablation.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for i, (scene, (_, mask_fn)) in enumerate(SCENES.items()):
        vmax = max(np.log1p(pictures[(scene, m)]).max() for m in ('hard', 'smoothed', 'combined'))
        for j, method in enumerate(('hard', 'smoothed', 'combined')):
            ax = axes[i, j]
            im = ax.imshow(np.log1p(pictures[(scene, method)]).T, origin='lower', extent=[-90,90,-90,90],
                           vmin=0, vmax=vmax, cmap='inferno')
            ax.contour(cx, cy, mask_fn(cx, cy).T, levels=[.5], colors='cyan', linewidths=.8)
            ax.set(title=f'{scene}: {method}', xlabel='x [mm]', ylabel='y [mm]')
        fig.colorbar(im, ax=axes[i, :].tolist(), label='log(1 + scattering weight)')
    fig.suptitle('Local-scale development candidate; N=5000, sigma=.05 mm, repeat 1')
    fig.savefig(output_dir/'local_scale_images.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'muon_analysis/local_scale_candidate_results')
    run(parser.parse_args().output_dir)
