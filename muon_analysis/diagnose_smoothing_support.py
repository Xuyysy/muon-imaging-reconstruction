"""固定平滑的观测支撑诊断：不调核、不选阈值、不拟合预测器。

输入复用旧事件/测量扰动。特征函数不接受目标真值；真值只给
独立的图像评分与旧记录核对。先读B1/SMOOTHING_SUPPORT_DIAGNOSIS_PLAN.md。
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

FEATURES = ('accepted_events', 'global_neff', 'occupied_fraction',
            'occupied_cell_neff_median', 'low_cell_support_mass_fraction',
            'occupied_neighborhood_neff_median', 'low_neighborhood_support_mass_fraction',
            'adjacent_total_variation', 'split_half_l1', 'smoothing_l1')
ENDPOINTS = ('smoothed_gain_fractional_iou', 'smoothed_gain_empty_mass_fraction')


def neighborhood_sum(image):
    """原生3x3窗相加；零填充，边缘不环绕且不是卷积后的独立事件。"""
    padded = np.pad(image, 1)
    return sum(padded[i:i+image.shape[0], j:j+image.shape[1]]
               for i in range(3) for j in range(3))


def observed_features(xy, weights, bins):
    """只有XY和权重，不读取真值掩膜、材料名或最终IoU。

    neff=(Σw)^2/Σw²是加权统计支撑，不是独立事件数或噪声方差。
    两网格的3x3窗物理大小不同，后续分层报告而非统一解释。
    """
    active = weights > 0
    xy, weights = xy[active], weights[active]
    if len(weights) < 2 or not np.isfinite(weights).all():
        raise ValueError('Insufficient/nonfinite observed support')
    edges = imaging_edges(bins)[:2]
    def histogram(w, points=xy):
        return np.histogram2d(points[:, 0], points[:, 1], bins=edges, weights=w)[0]
    mass, squared = histogram(weights), histogram(weights**2)
    neff = np.divide(mass**2, squared, out=np.zeros_like(mass), where=squared > 0)
    local, local_squared = neighborhood_sum(mass), neighborhood_sum(squared)
    local_neff = np.divide(local**2, local_squared, out=np.zeros_like(local), where=local_squared > 0)
    occupied = mass > 0
    p = mass/mass.sum()
    halves = []
    for offset in (0, 1):
        half = histogram(weights[offset::2], xy[offset::2])
        halves.append(half/half.sum())
    # L1是整图扰动量，不使用真值决定被改变像素是否正确。
    base = xy_image(xy, weights, bins)
    smooth = smooth_hard_image(base, bins)
    return {
        'accepted_events': len(weights), 'global_neff': float(weights.sum()**2/np.square(weights).sum()),
        'occupied_fraction': float(occupied.mean()),
        'occupied_cell_neff_median': float(np.median(neff[occupied])),
        'low_cell_support_mass_fraction': float(p[neff < 4].sum()),
        'occupied_neighborhood_neff_median': float(np.median(local_neff[occupied])),
        'low_neighborhood_support_mass_fraction': float(p[local_neff < 4].sum()),
        'adjacent_total_variation': float(np.abs(np.diff(p, axis=0)).sum()+np.abs(np.diff(p, axis=1)).sum()),
        'split_half_l1': float(np.abs(halves[0]-halves[1]).sum()),
        'smoothing_l1': float(np.abs(smooth/smooth.sum()-base/base.sum()).sum()),
    }


def rank_correlation(feature, outcome):
    """Spearman描述值，常量列返回NaN；不计算显著性或最优切点。"""
    if feature.nunique() < 2 or outcome.nunique() < 2:
        return np.nan
    return float(feature.rank().corr(outcome.rank()))


def run(output_dir):
    archive = pd.read_csv(ROOT/'muon_analysis/shape_smoothing_control_results/paired_by_repeat.csv')
    rows = []
    edges = imaging_edges(32)
    cx, cy = [(a[:-1]+a[1:])/2 for a in edges[:2]]
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
                    gate = (np.abs(observed.poca_z_mm.to_numpy()[ids]) <= Z_LIMIT_MM)
                    weighted = weight*gate
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    for bins in GRIDS:
                        base = xy_image(xy, weighted, bins)
                        smooth = smooth_hard_image(base, bins)
                        fractional = fractional_pair_metrics(smooth, base, truth)
                        mass = continuous_image_metrics(smooth, base, truth)
                        hs = shape_scores(base, truth, empty, cx, scene)
                        ss = shape_scores(smooth, truth, empty, cx, scene)
                        rows.append({'scenario': scene, 'event_count': count, 'repeat': repeat,
                                     'position_sigma_mm': sigma, 'grid': bins,
                                     **observed_features(xy, weighted, bins),
                                     'checked_iou_gain': fractional['fractional_gain'],
                                     'checked_mass_gain': mass['target_mass_gain'],
                                     'checked_empty_gain': ss['empty_mass_fraction']-hs['empty_mass_fraction']})
                print(f'Support diagnosis: {scene}, N={count}, repeat={repeat}', flush=True)
    frame = pd.DataFrame(rows).merge(archive, on=KEYS, validate='one_to_one')
    if len(frame) != 360:
        raise AssertionError('Incomplete support diagnostic')
    for checked, prior in [('checked_iou_gain', 'smoothed_gain_fractional_iou'),
                           ('checked_mass_gain', 'smoothed_gain_target_mass_fraction'),
                           ('checked_empty_gain', 'smoothed_gain_empty_mass_fraction')]:
        if not np.allclose(frame[checked], frame[prior], atol=1e-12, rtol=0):
            raise AssertionError(f'Frozen score mismatch: {checked}')
    frame['outcome'] = np.where(frame.smoothed_gain_fractional_iou > 1e-12, 'positive',
                               np.where(frame.smoothed_gain_fractional_iou < -1e-12, 'negative', 'zero'))
    counts = frame.groupby(['event_count', 'position_sigma_mm', 'grid'], as_index=False).agg(
        cases=('outcome', 'size'), positive=('outcome', lambda s: int((s == 'positive').sum())),
        negative=('outcome', lambda s: int((s == 'negative').sum())), zero=('outcome', lambda s: int((s == 'zero').sum())),
        iou_gain_mean=('smoothed_gain_fractional_iou', 'mean'), empty_gain_mean=('smoothed_gain_empty_mass_fraction', 'mean'))
    correlations = []
    # 池化结果只作对照；同N/sigma/grid结果控制设计层级，但仍不是独立样本。
    groups = [('pooled', None, frame)]
    groups += [('same_design', key, g) for key, g in frame.groupby(['event_count', 'position_sigma_mm', 'grid'])]
    for scope, key, group in groups:
        for feature in FEATURES:
            for endpoint in ENDPOINTS:
                correlations.append({'scope': scope, 'event_count': key[0] if key else np.nan,
                                     'position_sigma_mm': key[1] if key else np.nan,
                                     'grid': key[2] if key else np.nan, 'feature': feature,
                                     'endpoint': endpoint, 'cases': len(group),
                                     'spearman': rank_correlation(group[feature], group[endpoint])})
    ranges = []
    for (bins, outcome), group in frame.groupby(['grid', 'outcome']):
        for feature in FEATURES:
            ranges.append({'grid': bins, 'outcome': outcome, 'feature': feature,
                           'cases': len(group), 'min': group[feature].min(),
                           'median': group[feature].median(), 'max': group[feature].max()})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir/'by_repeat.csv', index=False)
    counts.to_csv(output_dir/'design_summary.csv', index=False)
    pd.DataFrame(correlations).to_csv(output_dir/'rank_correlations.csv', index=False)
    pd.DataFrame(ranges).to_csv(output_dir/'feature_ranges.csv', index=False)
    (output_dir/'config.json').write_text(json.dumps({
        'plan': 'B1/SMOOTHING_SUPPORT_DIAGNOSIS_PLAN.md', 'role': 'descriptive old-data diagnosis; no learned policy',
        'conditions': len(frame), 'sigma_mm_frozen': SMOOTH_SIGMA_MM,
        'local_window': 'native 3x3, differs in physical size across grids',
        'low_neff_cut_for_description_only': 4, 'truth_in_features': False,
        'split': 'alternating positive-weight observed order; not independent validation',
        'correlation_caveat': 'overlapping subsamples; no significance or fitted thresholds',
    }, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for i, bins in enumerate(GRIDS):
        part = frame[frame.grid == bins]
        for outcome, marker in [('positive', 'o'), ('negative', 'x'), ('zero', '+')]:
            g = part[part.outcome == outcome]
            axes[i, 0].scatter(g.low_neighborhood_support_mass_fraction, g.smoothed_gain_fractional_iou,
                               s=20, marker=marker, alpha=.65, label=outcome)
            axes[i, 1].scatter(g.smoothing_l1, g.smoothed_gain_empty_mass_fraction,
                               s=20, marker=marker, alpha=.65, label=outcome)
        axes[i, 0].set(xlabel='Mass fraction in native 3x3 neff < 4', ylabel='Fractional IoU difference S-H')
        axes[i, 1].set(xlabel='Normalized L1 change from smoothing', ylabel='Empty-region mass difference S-H')
        for ax in axes[i]:
            ax.axhline(0, color='black', lw=.7)
            ax.set_title(f'Native {bins} grid; old development conditions')
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.savefig(output_dir/'smoothing_support_diagnosis.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'muon_analysis/smoothing_support_diagnosis_results')
    run(parser.parse_args().output_dir)
