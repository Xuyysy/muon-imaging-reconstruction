"""旧U/L/双块开发场景的固定平滑强基线；不调核宽、不新增Notebook。

先读B1/SHAPE_SMOOTHING_CONTROL_PLAN.md。复用旧事件及扰动种子，
真值只进入评分；这不是新Geant4流或完整三维重建验证。
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
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_frame_z_gate_validation import SMOOTH_SIGMA_MM, smooth_hard_image
from run_simple_z_gate_baseline import simple_z_gate
from run_soft_z_gate_experiment import (
    COUNTS, SIGMAS_MM, REPEATS, GRIDS, candidate_events,
    xy_image, z_acceptance_weights, score_image,
)
from run_uncertainty_aware_angle_experiment import (
    same_side_layer_spacing_mm, theta_noise_floor_mrad,
)
from run_fixed_budget_refinement import fractional_pair_metrics, continuous_image_metrics

KEYS = ['scenario', 'event_count', 'repeat', 'position_sigma_mm', 'grid']


def fractional_membership(image, k):
    """并列像素平分剩余名额，不按坐标任意挑选；仅用于终点评分。"""
    if not 0 < k < image.size or not np.isfinite(image).all():
        raise ValueError('Invalid fractional membership image/area')
    boundary = np.partition(image.ravel(), -k)[-k]
    above, tied = image > boundary, image == boundary
    slots = k - int(above.sum())
    member = above.astype(float) + tied * (slots / int(tied.sum()))
    if not np.isclose(member.sum(), k):
        raise AssertionError('Fractional membership changed evaluated area')
    return member


def empty_region(scene, cx, cy):
    """根据仿真几何定义空区，只给评价用；三类区域含义不同。"""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    if scene == 'u_center':
        return (np.abs(x) < 30) & (y > -30) & (y <= 60)
    if scene == 'l_shape':
        return (x > -30) & (x <= 60) & (y > -30) & (y <= 60)
    if scene == 'double_block':
        return (np.abs(x) < 15) & (np.abs(y) <= 30)
    raise ValueError('Unknown development shape')


def shape_scores(image, truth, empty, cx, scene):
    """空域质量以全图总权重为分母，不混用旧gap的局部足迹分母。

    分数空域占用仍是等真值面积评价，不是真实部署时的检测概率。
    双块最低召回避免一块亮、另一块丢失被整体IoU掩盖。
    """
    if (truth & empty).any() or not empty.any() or image.sum() <= 0:
        raise ValueError('Invalid shape/empty region or zero image')
    member = fractional_membership(image, int(truth.sum()))
    intersection = float(member[truth].sum())
    iou = intersection / (2*truth.sum() - intersection)
    score = fractional_pair_metrics(image, image, truth)['fractional_iou']
    if not np.isclose(iou, score):
        raise AssertionError('Membership disagrees with fractional IoU')
    minimum = np.nan
    if scene == 'double_block':
        left = truth & (cx[:, None] < 0)
        right = truth & (cx[:, None] > 0)
        minimum = min(float(member[left].mean()), float(member[right].mean()))
    return {'empty_mass_fraction': float(image[empty].sum()/image.sum()),
            'empty_fractional_occupancy': float(member[empty].mean()),
            'fractional_min_block_recall': minimum}


def run(output_dir: Path):
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1]+axis[1:])/2 for axis in edges[:2]]
    rows, pictures = [], {}
    for index, (scene, (filename, mask_fn)) in enumerate(SCENES.items()):
        data = pd.read_csv(ROOT/'muon_analysis'/filename)
        if len(data) < max(COUNTS):
            raise ValueError('Insufficient complete tracks')
        truth, empty = mask_fn(cx, cy), empty_region(scene, cx, cy)
        spacing = same_side_layer_spacing_mm(data)
        # 与旧simple_z_gate实验完全相同；不能只匹配平均得分。
        rng = np.random.default_rng(RANDOM_SEED+2107+index*100)
        for count in COUNTS:
            for repeat in range(1, REPEATS+1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                draws = {f'{plane}_{axis}': rng.normal(0, 1, count)
                         for plane in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(measured, 0., np.random.default_rng(0))
                    ids, weight = candidate_events(observed, sigma, spacing)
                    z = observed.poca_z_mm.to_numpy()[ids]
                    theta = observed.theta_mrad.to_numpy()[ids]
                    corrected = np.sqrt(np.maximum(theta**2-theta_noise_floor_mrad(sigma, spacing)**2, 0.))
                    seed = RANDOM_SEED+300000*index+10000*count+100*repeat+int(round(sigma*1000))
                    samples = estimate_z_samples(measured.iloc[ids], sigma, seed, BOOTSTRAP_DRAWS)
                    hard, propagated = z_acceptance_weights(z, samples)
                    analytic = simple_z_gate(z, corrected, sigma)
                    if sigma == 0 and not (np.array_equal(hard, propagated) and np.array_equal(hard, analytic)):
                        raise AssertionError('Zero-error H/A/P mismatch')
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    for bins in GRIDS:
                        base = xy_image(xy, weight*hard, bins)
                        images = {'hard': base,
                                  'simple': xy_image(xy, weight*analytic, bins),
                                  'propagated': xy_image(xy, weight*propagated, bins),
                                  'smoothed': smooth_hard_image(base, bins)}
                        for method, image in images.items():
                            rows.append({'scenario': scene, 'event_count': count, 'repeat': repeat,
                                         'position_sigma_mm': sigma, 'grid': bins, 'method': method,
                                         'total_image_weight': float(image.sum()),
                                         **score_image(image, truth, cx, cy, scene),
                                         **fractional_pair_metrics(image, base, truth),
                                         **continuous_image_metrics(image, base, truth),
                                         **shape_scores(image, truth, empty, cx, scene)})
                            if count == 5000 and repeat == 1 and bins == 32 and sigma in (0., .05, .30):
                                pictures[(scene, sigma, method)] = image.copy()
                print(f'Shape smoothing: {scene}, N={count}, repeat={repeat}', flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 1440 or frame.duplicated(KEYS+['method']).any():
        raise AssertionError('Incomplete paired grid')
    # 逐条复现旧记录，包括旧硬Top-k以保留可追溯性；新主评分另列。
    prior = pd.read_csv(ROOT/'muon_analysis/simple_z_gate_results/by_repeat.csv')
    old = frame[frame.method != 'smoothed'].merge(prior, on=KEYS+['method'],
                                               suffixes=('', '_old'), validate='one_to_one')
    if len(old) != 1080:
        raise AssertionError('Archived pairing missing')
    for name in ('iou', 'global_error_mm', 'total_image_weight',
                 'min_block_recall', 'gap_false_positive_fraction'):
        if not np.allclose(old[name], old[name+'_old'], atol=1e-10, rtol=1e-12, equal_nan=True):
            raise AssertionError(f'Archive mismatch: {name}')
    views = {m: frame[frame.method == m].set_index(KEYS).sort_index() for m in images}
    h, p, s = (views[m] for m in ('hard', 'propagated', 'smoothed'))
    if not np.allclose(h.total_image_weight, s.total_image_weight, rtol=1e-12, atol=1e-9):
        raise AssertionError('S mass conservation failed')
    paired = pd.DataFrame({
        **{f'{m}_{metric}': views[m][metric] for m in views for metric in
           ('fractional_iou', 'target_mass_fraction', 'empty_mass_fraction',
            'empty_fractional_occupancy', 'fractional_min_block_recall')},
        **{f'{m}_gain_{metric}': views[m][metric]-h[metric] for m in ('simple', 'propagated', 'smoothed')
           for metric in ('fractional_iou', 'target_mass_fraction', 'empty_mass_fraction',
                          'empty_fractional_occupancy')},
        **{f'{m}_proxy_distance_difference': views[m].footprint_proxy_distance-h.footprint_proxy_distance
           for m in ('simple', 'propagated', 'smoothed')},
        'p_minus_s_iou': p.fractional_iou-s.fractional_iou,
        'p_minus_s_mass': p.target_mass_fraction-s.target_mass_fraction,
        'p_minus_s_empty': p.empty_mass_fraction-s.empty_mass_fraction,
    }).reset_index()
    # proxy_distance越小越好，此列命名为差值而不是正向收益，避免误解。
    summary = paired.groupby([k for k in KEYS if k != 'repeat'], as_index=False).mean(numeric_only=True)
    summary = summary.drop(columns='repeat')
    # 五次抽样离散度是描述统计，不冒称独立Geant4流的置信区间。
    group_keys = [k for k in KEYS if k != 'repeat']
    spread = paired.groupby(group_keys, as_index=False).agg(
        repeats=('repeat', 'size'),
        smoothed_iou_gain_std=('smoothed_gain_fractional_iou', 'std'),
        smoothed_empty_mass_gain_std=('smoothed_gain_empty_mass_fraction', 'std'),
        smoothed_iou_wins=('smoothed_gain_fractional_iou', lambda x: int((x > 1e-12).sum())),
        propagated_vs_smoothed_iou_wins=('p_minus_s_iou', lambda x: int((x > 1e-12).sum())),
    )
    summary = summary.merge(spread, on=group_keys, validate='one_to_one')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir/'by_repeat.csv', index=False)
    paired.to_csv(output_dir/'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir/'summary.csv', index=False)
    (output_dir/'config.json').write_text(json.dumps({
        'plan': 'B1/SHAPE_SMOOTHING_CONTROL_PLAN.md', 'role': 'old development scenes, not independent validation',
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'grids': GRIDS, 'repeats': REPEATS,
        'scenes': list(SCENES), 'sigma_mm_frozen': SMOOTH_SIGMA_MM,
        'archived_method_rows_reproduced': len(old), 'empty_mass_denominator': 'whole image total weight',
        'truth_usage': 'evaluation only; fractional area is truth dependent',
        'proxy_distance_difference_sign': 'negative is better',
    }, indent=2)+'\n')
    # 固定展示首次重复，不选择最好图；同场景/误差的三图共享色标。
    for sigma in (0., .05, .30):
        fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
        for i, (scene, (_, mask_fn)) in enumerate(SCENES.items()):
            vmax = max(np.log1p(pictures[(scene, sigma, m)]).max() for m in ('hard', 'propagated', 'smoothed'))
            for j, method in enumerate(('hard', 'propagated', 'smoothed')):
                ax = axes[i, j]
                im = ax.imshow(np.log1p(pictures[(scene, sigma, method)]).T, origin='lower',
                               extent=[-90, 90, -90, 90], cmap='inferno', vmin=0, vmax=vmax)
                ax.contour(cx, cy, mask_fn(cx, cy).T, levels=[.5], colors='cyan', linewidths=.8)
                ax.set(title=f'{scene}: {method}', xlabel='x [mm]', ylabel='y [mm]')
            fig.colorbar(im, ax=axes[i, :].tolist(), label='log(1 + scattering weight)')
        fig.suptitle(f'Fixed 5.625 mm smoothing; N=5000, sigma={sigma:g} mm; repeat 1')
        fig.savefig(output_dir/f'shape_images_sigma_{sigma:g}.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for j, scene in enumerate(SCENES):
        part = summary[(summary.scenario == scene) & (summary.event_count == 5000) & (summary.grid == 32)]
        for m in ('propagated', 'smoothed'):
            axes[0, j].plot(part.position_sigma_mm, part[f'{m}_gain_fractional_iou'], 'o-', label=m)
            axes[1, j].plot(part.position_sigma_mm, part[f'{m}_gain_empty_mass_fraction'], 'o-', label=m)
        for ax in axes[:, j]:
            ax.axhline(0, color='black', lw=.8)
            ax.grid(alpha=.2)
            ax.set(xlabel='Hit-position sigma [mm]', title=scene)
    axes[0, 0].set_ylabel('Fractional IoU difference vs H (higher better)')
    axes[1, 0].set_ylabel('Empty-region mass difference vs H (lower better)')
    axes[0, 0].legend()
    fig.savefig(output_dir/'shape_smoothing_metrics.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'muon_analysis/shape_smoothing_control_results')
    run(parser.parse_args().output_dir)
