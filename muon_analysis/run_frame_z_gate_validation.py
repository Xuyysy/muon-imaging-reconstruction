"""按预注册方案，在全新方框目标的两份 Geant4 流上冻结验证 H/A/P。

先读 FRAME_Z_GATE_VALIDATION_PROTOCOL.md。此脚本不扫描参数，不用真值
决定门权重，也不根据当前结果回退到某个方法。真值只用于评价。
"""
from __future__ import annotations

import argparse
import json
import time
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges, Z_LIMIT_MM
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_simple_z_gate_baseline import simple_z_gate
from run_soft_z_gate_experiment import (
    COUNTS, GRIDS, REPEATS, ROOT, SIGMAS_MM, candidate_events, xy_image,
    z_acceptance_weights,
)
from run_uncertainty_aware_angle_experiment import (
    same_side_layer_spacing_mm, theta_noise_floor_mrad,
)
from run_fixed_budget_refinement import (
    EPS, fractional_pair_metrics, continuous_image_metrics,
    test_fractional_metrics, test_continuous_metrics,
)

STREAMS = {
    'seed_a': 'muon_tracks_frame_seed_a.csv',
    'seed_b': 'muon_tracks_frame_seed_b.csv',
}
KEYS = ['stream', 'event_count', 'repeat', 'position_sigma_mm', 'grid']
SMOOTH_SIGMA_MM = 5.625
TIMING_REPEATS = 3


def smooth_hard_image(image: np.ndarray, bins: int) -> np.ndarray:
    """固定物理核宽的质量守恒平滑；不接受真值或误差评分。

    列j是源像素j向目标像素i的质量分配，域边缘截断后按列归一化。
    16格先还原原生直方图，再平滑和映射32格，保留公平分辨率。
    """
    if (image.shape != (32, 32) or bins not in GRIDS
            or not np.isfinite(image).all() or (image < 0).any()):
        raise ValueError('Invalid smoothing image/grid')
    native = image if bins == 32 else image.reshape(16, 2, 16, 2).sum(axis=(1, 3))
    edges = imaging_edges(bins)[0]
    sigma_pixels = SMOOTH_SIGMA_MM / float(edges[1] - edges[0])
    radius = int(np.ceil(3*sigma_pixels))
    displacement = np.arange(bins)[:, None] - np.arange(bins)[None, :]
    kernel = np.exp(-.5 * np.square(displacement / sigma_pixels))
    kernel[np.abs(displacement) > radius] = 0.
    kernel /= kernel.sum(axis=0, keepdims=True)
    # 显式两轴求和，避免本机小矩阵BLAS路径的浮点警告；不隐藏警告。
    # 与K @ native @ K.T数学等价，核构造和两轴求和都计入S耗时。
    stage = np.einsum('ij,jk->ik', kernel, native, optimize=False)
    smooth = np.einsum('ij,kj->ik', stage, kernel, optimize=False)
    if bins == 16:
        smooth = np.repeat(np.repeat(smooth/4, 2, axis=0), 2, axis=1)
    if (not np.isfinite(smooth).all()
            or not np.isclose(smooth.sum(), image.sum(), atol=1e-9, rtol=1e-12)):
        raise AssertionError('Smoothing changed total weight')
    return smooth


def test_smoothing_control():
    for bins in GRIDS:
        zero = smooth_hard_image(np.zeros((32, 32)), bins)
        if not np.array_equal(zero, np.zeros((32, 32))):
            raise AssertionError('Zero image changed under smoothing')
        impulse = np.zeros((32, 32))
        impulse[:2, :2] = .25  # 成像域边缘的冲激也必须守恒。
        smooth = smooth_hard_image(impulse, bins)
        if (smooth < 0).any() or not np.isclose(smooth.sum(), 1.):
            raise AssertionError('Smoothing positivity/mass failure')
        if bins == 16 and not np.allclose(smooth[::2, ::2], smooth[1::2, 1::2]):
            raise AssertionError('Native 16-grid mapping changed')


def benchmark_reconstructions(measured, z, corrected, xy, weight, sigma, seed,
                              reference_images, rotation):
    """共享候选之后的方法增量计时；包括两个网格，不包括评分。

    时间不含共同径迹预处理或Geant4，不能当作总扫描/成像时间。
    核构造计入S成本，P包含重新生成32次传播，两网格只做一次传播。
    """
    methods = ('hard', 'simple', 'propagated', 'smoothed')

    def reconstruct(method):
        if method == 'simple':
            gate = simple_z_gate(z, corrected, sigma)
        elif method == 'propagated':
            samples = estimate_z_samples(measured, sigma, seed, BOOTSTRAP_DRAWS)
            _, gate = z_acceptance_weights(z, samples)
        else:
            gate = (np.abs(z) <= Z_LIMIT_MM).astype(float)
        pair = {bins: xy_image(xy, weight*gate, bins) for bins in GRIDS}
        if method == 'smoothed':
            pair = {bins: smooth_hard_image(im, bins) for bins, im in pair.items()}
        return pair

    for method in methods:
        warm = reconstruct(method)
        if not all(np.allclose(warm[bins], reference_images[(bins, method)],
                               rtol=1e-12, atol=1e-9) for bins in GRIDS):
            raise AssertionError('Benchmark reconstruction changed image')
    rows = []
    for trial in range(TIMING_REPEATS):
        order = (rotation + trial) % len(methods)
        for offset in range(len(methods)):
            method = methods[(offset + order) % len(methods)]
            start = time.perf_counter_ns()
            pair = reconstruct(method)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            # 图像检查在停止计时之后，评分完全不进计时区间。
            if not all(np.allclose(pair[bins], reference_images[(bins, method)],
                                   rtol=1e-12, atol=1e-9) for bins in GRIDS):
                raise AssertionError('Timed reconstruction changed image')
            rows.append({'method': method, 'timing_repeat': trial+1,
                         'order': offset+1, 'reconstruction_ms_both_grids': elapsed_ms})
    return rows


def frame_mask(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """四块 G4Box 的 XY 足迹；仅用于成像后计算真值指标。"""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    footprint = (np.abs(x) <= 60.) & (np.abs(y) <= 60.)
    hole = (np.abs(x) < 40.) & (np.abs(y) < 40.)
    return footprint & ~hole


def score_frame(image: np.ndarray, truth: np.ndarray,
                cx: np.ndarray, cy: np.ndarray) -> dict[str, float]:
    """等面积 Top-k 轮廓；孔洞误报以全部孔洞像素为分母。"""
    if image.shape != truth.shape or not np.isfinite(image).all():
        raise ValueError('Unpaired or nonfinite image')
    k = int(truth.sum())
    predicted = np.zeros_like(truth, dtype=bool)
    predicted.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    x, y = np.meshgrid(cx, cy, indexing='ij')
    hole = (np.abs(x) < 40.) & (np.abs(y) < 40.)
    intersection = int((predicted & truth).sum())
    return {
        'iou': float(intersection / (predicted | truth).sum()),
        'frame_recall': float(intersection / k),
        'hole_false_positive_fraction': float((predicted & hole).sum() / hole.sum()),
    }


def run(output_dir: Path, score_audit: bool = False, smoothing_control: bool = False) -> None:
    if smoothing_control:
        score_audit = True
        test_smoothing_control()
    if score_audit:
        test_fractional_metrics()
        test_continuous_metrics()
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = frame_mask(cx, cy)
    rows, timing_rows = [], []
    for stream_index, (stream, filename) in enumerate(STREAMS.items()):
        data = pd.read_csv(ROOT / 'muon_analysis' / filename)
        if len(data) < max(COUNTS):
            raise ValueError(f'{stream}: fewer than {max(COUNTS)} complete tracks')
        spacing = same_side_layer_spacing_mm(data)
        # 新分析 RNG 与开发脚本不同；每种方法共用子抽样和测量扰动。
        rng = np.random.default_rng(RANDOM_SEED + 5107 + 1000 * stream_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                normals = {f'{plane}_{axis}': rng.normal(0, 1, count)
                           for plane in ('up1', 'up2', 'down1', 'down2')
                           for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, normals)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    ids, angle_weight = candidate_events(observed, sigma, spacing)
                    z = observed.poca_z_mm.to_numpy()[ids]
                    theta = observed.theta_mrad.to_numpy()[ids]
                    floor = theta_noise_floor_mrad(sigma, spacing)
                    corrected = np.sqrt(np.maximum(theta**2 - floor**2, 0.))
                    analytic = simple_z_gate(z, corrected, sigma)
                    seed = (RANDOM_SEED + 600000 + 300000 * stream_index
                            + 10000 * count + 100 * repeat
                            + int(round(sigma * 1000)))
                    samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                 seed, BOOTSTRAP_DRAWS)
                    hard, propagated = z_acceptance_weights(z, samples)
                    if sigma == 0 and (not np.array_equal(hard, analytic)
                                       or not np.array_equal(hard, propagated)):
                        raise AssertionError('Zero-noise H/A/P control failed')
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    reference_images = {}
                    for bins in GRIDS:
                        # 新评分始终配对同条件硬图；不改变任何门权重。
                        hard_image = xy_image(xy, angle_weight * hard, bins)
                        for method, gate in (('hard', hard), ('simple', analytic),
                                             ('propagated', propagated)):
                            image = xy_image(xy, angle_weight * gate, bins)
                            reference_images[(bins, method)] = image
                            extra = {}
                            if score_audit:
                                if sigma == 0 and not np.array_equal(image, hard_image):
                                    raise AssertionError('Zero-noise image identity failed')
                                extra = {
                                    **fractional_pair_metrics(image, hard_image, truth),
                                    **continuous_image_metrics(image, hard_image, truth),
                                    'total_image_weight': float(image.sum()),
                                    'total_weight_ratio_vs_hard': float(image.sum() / hard_image.sum()),
                                }
                            rows.append({
                                'stream': stream, 'event_count': count,
                                'repeat': repeat, 'position_sigma_mm': sigma,
                                'grid': bins, 'method': method,
                                'candidates': len(ids),
                                'effective_z_accepted_events': float(gate.sum()),
                                **score_frame(image, truth, cx, cy),
                                **extra,
                            })
                        if smoothing_control:
                            image = smooth_hard_image(hard_image, bins)
                            reference_images[(bins, 'smoothed')] = image
                            rows.append({
                                'stream': stream, 'event_count': count, 'repeat': repeat,
                                'position_sigma_mm': sigma, 'grid': bins, 'method': 'smoothed',
                                'candidates': len(ids),
                                'effective_z_accepted_events': float(hard.sum()),
                                **score_frame(image, truth, cx, cy),
                                **fractional_pair_metrics(image, hard_image, truth),
                                **continuous_image_metrics(image, hard_image, truth),
                                'total_image_weight': float(image.sum()),
                                'total_weight_ratio_vs_hard': float(image.sum()/hard_image.sum()),
                            })
                    if smoothing_control:
                        timing_key = {'stream': stream, 'event_count': count, 'repeat': repeat,
                                      'position_sigma_mm': sigma}
                        timing_rows.extend({**timing_key, **row} for row in benchmark_reconstructions(
                            measured.iloc[ids], z, corrected, xy, angle_weight, sigma, seed,
                            reference_images, stream_index + count + repeat + int(sigma*100)))
                print(f'Frame {stream}: N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    expected = len(STREAMS) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRIDS) * (
        4 if smoothing_control else 3)
    if len(frame) != expected or frame.duplicated(KEYS + ['method']).any():
        raise AssertionError('Incomplete or duplicate paired validation')
    if score_audit:
        previous = pd.read_csv(ROOT / 'muon_analysis/frame_z_gate_results/by_repeat.csv')
        frozen_methods = frame[frame.method != 'smoothed']
        merged = frozen_methods.merge(previous, on=KEYS + ['method'],
                             suffixes=('', '_old'), validate='one_to_one')
        if len(merged) != 720:
            raise AssertionError('Original frame archive incomplete')
        for column in ('iou', 'frame_recall', 'hole_false_positive_fraction',
                       'candidates', 'effective_z_accepted_events'):
            if not np.allclose(merged[column], merged[column+'_old'], rtol=0, atol=EPS):
                raise AssertionError(f'Original frame result changed: {column}')
        if smoothing_control:
            audited = pd.read_csv(ROOT / 'muon_analysis/frame_gate_score_audit_results/by_repeat.csv')
            matched = frozen_methods.merge(audited, on=KEYS + ['method'],
                                           suffixes=('', '_prior'), validate='one_to_one')
            for column in ('fractional_iou', 'fractional_gain', 'target_mass_gain',
                           'footprint_proxy_gain', 'total_weight_ratio_vs_hard'):
                if len(matched) != 720 or not np.allclose(matched[column], matched[column+'_prior'],
                                                        atol=EPS, rtol=0):
                    raise AssertionError('Archived gate audit changed')
    views = {name: frame[frame.method == name].set_index(KEYS).sort_index()
             for name in ('hard', 'simple', 'propagated')}
    hard, simple, propagated = (views[name] for name in
                                ('hard', 'simple', 'propagated'))
    paired = pd.DataFrame({
        'hard_iou': hard.iou, 'simple_iou': simple.iou,
        'propagated_iou': propagated.iou,
        'simple_gain_vs_hard': simple.iou - hard.iou,
        'propagated_gain_vs_hard': propagated.iou - hard.iou,
        'propagated_gain_vs_simple': propagated.iou - simple.iou,
        'hard_frame_recall': hard.frame_recall,
        'simple_frame_recall': simple.frame_recall,
        'propagated_frame_recall': propagated.frame_recall,
        'hard_hole_fp': hard.hole_false_positive_fraction,
        'simple_hole_fp': simple.hole_false_positive_fraction,
        'propagated_hole_fp': propagated.hole_false_positive_fraction,
    }).reset_index()
    if score_audit:
        paired_extra = pd.DataFrame({
            'hard_fractional_iou': hard.fractional_iou,
            'hard_target_mass_fraction': hard.target_mass_fraction,
            'hard_footprint_proxy_distance': hard.footprint_proxy_distance,
            **{f'{method}_{column}': view[column]
               for method, view in (('simple', simple), ('propagated', propagated))
               for column in ('fractional_iou', 'fractional_gain', 'gain_bound_min',
                              'gain_bound_max', 'target_mass_gain', 'footprint_proxy_gain',
                              'concentration_penalty', 'target_mass_reward',
                              'total_weight_ratio_vs_hard')},
        }).reset_index()
        paired = paired.merge(paired_extra, on=KEYS, validate='one_to_one')
        if smoothing_control:
            smooth = frame[frame.method == 'smoothed'].set_index(KEYS).sort_index()
            control_extra = pd.DataFrame({
                'smoothed_iou': smooth.iou,
                'propagated_hard_iou_gain_vs_smoothed': propagated.iou - smooth.iou,
                **{f'smoothed_{column}': smooth[column] for column in
                   ('fractional_iou', 'fractional_gain', 'target_mass_gain',
                    'footprint_proxy_gain', 'total_weight_ratio_vs_hard')},
                'propagated_fractional_gain_vs_smoothed': propagated.fractional_iou - smooth.fractional_iou,
                'propagated_mass_gain_vs_smoothed': propagated.target_mass_fraction - smooth.target_mass_fraction,
                'propagated_proxy_gain_vs_smoothed': smooth.footprint_proxy_distance - propagated.footprint_proxy_distance,
            }).reset_index()
            paired = paired.merge(control_extra, on=KEYS, validate='one_to_one')
        for prefix in ('simple', 'propagated'):
            low, high = paired[f'{prefix}_gain_bound_min'], paired[f'{prefix}_gain_bound_max']
            gain = paired[f'{prefix}_gain_vs_hard']
            if not ((gain >= low-EPS).all() and (gain <= high+EPS).all()):
                raise AssertionError('Legacy gain outside legal tie bounds')
        zero = paired[paired.position_sigma_mm == 0]
        for prefix in ('simple', 'propagated'):
            for column in ('fractional_gain', 'target_mass_gain', 'footprint_proxy_gain',
                           'concentration_penalty', 'target_mass_reward'):
                if not np.allclose(zero[f'{prefix}_{column}'], 0, atol=EPS, rtol=0):
                    raise AssertionError('Zero-noise audit metric changed')
    if not np.allclose(paired.loc[paired.position_sigma_mm == 0,
                                  ['simple_gain_vs_hard',
                                   'propagated_gain_vs_hard']], 0):
        raise AssertionError('Zero-noise image identity failed')
    summary = paired.groupby(['stream', 'event_count', 'position_sigma_mm', 'grid'],
                             as_index=False).agg(
        repeats=('repeat', 'size'),
        hard_iou_mean=('hard_iou', 'mean'),
        simple_iou_mean=('simple_iou', 'mean'),
        propagated_iou_mean=('propagated_iou', 'mean'),
        simple_gain_mean=('simple_gain_vs_hard', 'mean'),
        propagated_gain_mean=('propagated_gain_vs_hard', 'mean'),
        extra_gain_mean=('propagated_gain_vs_simple', 'mean'),
        propagated_wins=('propagated_gain_vs_hard',
                         lambda x: int((x > 1e-12).sum())),
        propagated_losses=('propagated_gain_vs_hard',
                           lambda x: int((x < -1e-12).sum())),
        hard_hole_fp_mean=('hard_hole_fp', 'mean'),
        simple_hole_fp_mean=('simple_hole_fp', 'mean'),
        propagated_hole_fp_mean=('propagated_hole_fp', 'mean'),
        hard_frame_recall_mean=('hard_frame_recall', 'mean'),
        simple_frame_recall_mean=('simple_frame_recall', 'mean'),
        propagated_frame_recall_mean=('propagated_frame_recall', 'mean'))
    if score_audit:
        group = ['stream', 'event_count', 'position_sigma_mm', 'grid']
        audit_cols = [column for column in paired_extra if column not in KEYS]
        audited = paired.groupby(group, as_index=False).agg(
            **{column+'_mean': (column, 'mean') for column in audit_cols},
            propagated_fractional_wins=('propagated_fractional_gain', lambda x: int((x>EPS).sum())),
            propagated_mass_wins=('propagated_target_mass_gain', lambda x: int((x>EPS).sum())),
            propagated_proxy_wins=('propagated_footprint_proxy_gain', lambda x: int((x>EPS).sum())))
        summary = summary.merge(audited, on=group, validate='one_to_one')
        if smoothing_control:
            controls = paired.groupby(group, as_index=False).agg(
                **{column+'_mean': (column, 'mean') for column in control_extra if column not in KEYS},
                propagated_vs_smoothed_fractional_wins=(
                    'propagated_fractional_gain_vs_smoothed', lambda x: int((x>EPS).sum())))
            summary = summary.merge(controls, on=group, validate='one_to_one')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    if smoothing_control:
        timing = pd.DataFrame(timing_rows)
        if len(timing) != 120*4*TIMING_REPEATS:
            raise AssertionError('Incomplete timing matrix')
        timing.to_csv(output_dir / 'timing_by_repeat.csv', index=False)
        timing.groupby(['stream', 'event_count', 'position_sigma_mm', 'method'], as_index=False).agg(
            timed_runs=('timing_repeat', 'size'),
            median_ms=('reconstruction_ms_both_grids', 'median'),
            min_ms=('reconstruction_ms_both_grids', 'min'),
            max_ms=('reconstruction_ms_both_grids', 'max')).to_csv(output_dir / 'timing_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for stream, marker in (('seed_a', 'o'), ('seed_b', 's')):
        part = summary[(summary.stream == stream) & (summary.event_count == 5000)
                       & (summary.grid == 32)].sort_values('position_sigma_mm')
        ax.plot(part.position_sigma_mm, part.propagated_gain_mean,
                marker=marker, label=f'{stream}: 32-draw − hard')
        ax.plot(part.position_sigma_mm, part.simple_gain_mean,
                marker=marker, ls='--', label=f'{stream}: simple − hard')
    ax.axhline(0, color='black', lw=1)
    ax.set(xlabel='Hit-position sigma [mm]',
           ylabel='Paired equal-area IoU gain',
           title='Held-out closed-frame target, 5000 input muons, 32×32 XY')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / 'frame_gate_iou_5000_32.png', dpi=180)
    plt.close(fig)
    if score_audit:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for stream, marker in (('seed_a', 'o'), ('seed_b', 's')):
            part = summary[(summary.stream == stream) & (summary.event_count == 5000)
                           & (summary.grid == 32)].sort_values('position_sigma_mm')
            axes[0].plot(part.position_sigma_mm, part.propagated_gain_mean,
                         marker=marker, ls='--', label=stream+' legacy')
            axes[0].plot(part.position_sigma_mm, part.propagated_fractional_gain_mean,
                         marker=marker, label=stream+' fractional')
            axes[1].plot(part.position_sigma_mm, part.propagated_target_mass_gain_mean,
                         marker=marker, label=stream)
            axes[2].plot(part.position_sigma_mm, part.propagated_footprint_proxy_gain_mean,
                         marker=marker, label=stream)
        for ax, ylabel in zip(axes, ('IoU gain', 'Target mass fraction gain', 'Footprint proxy gain')):
            ax.axhline(0, color='black', lw=1)
            ax.set(xlabel='Hit sigma [mm]', ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle('Frozen frame P-H audit: N=5000, 32x32 XY; old streams, no retuning')
        fig.savefig(output_dir / 'frame_gate_score_audit.png', dpi=180)
        plt.close(fig)
    if smoothing_control:
        main = summary[(summary.event_count == 5000) & (summary.grid == 32)
                       & summary.position_sigma_mm.isin((.05, .10))].sort_values(
                           ['stream', 'position_sigma_mm'])
        labels = [f'{r.stream}/{r.position_sigma_mm:.2f}' for r in main.itertuples()]
        positions = np.arange(len(main))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for offset, method in zip((-.24, 0, .24), ('simple', 'propagated', 'smoothed')):
            axes[0].bar(positions+offset, main[f'{method}_fractional_gain_mean'], width=.23, label=method)
            proxy = ('propagated_footprint_proxy_gain_mean' if method == 'propagated'
                     else f'{method}_footprint_proxy_gain_mean')
            axes[1].bar(positions+offset, main[proxy], width=.23, label=method)
        for ax, label in zip(axes, ('Fractional IoU gain vs H', 'Footprint proxy gain vs H')):
            ax.axhline(0, color='black', lw=1)
            ax.set_xticks(positions, labels)
            ax.set(ylabel=label, xlabel='Old stream / hit sigma [mm]')
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('Fixed 5.625 mm smoothing S vs frozen A/P: N=5000, grid32')
        fig.savefig(output_dir / 'frame_smoothing_control.png', dpi=180)
        plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'protocol': ('B1/FRAME_SMOOTHING_CONTROL_PLAN.md' if smoothing_control else
                     'B1/FRAME_GATE_SCORE_AUDIT_PLAN.md' if score_audit
                     else 'FRAME_Z_GATE_VALIDATION_PROTOCOL.md'),
        'role': ('development-only fixed smoothing/cost control on old frame streams' if smoothing_control else
                 'retrospective frozen-frame multimetric score audit' if score_audit
                 else 'held-out target family with two independent Geant4 seeds'),
        'streams': STREAMS, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'grids': GRIDS, 'repeats': REPEATS,
        'bootstrap_draws': BOOTSTRAP_DRAWS,
        'primary': 'N=5000, grid=32, sigma=0.05/0.10, P-H IoU per stream',
        'truth_only_for_scoring': True,
        'smoothing_sigma_mm': SMOOTH_SIGMA_MM if smoothing_control else None,
        'smoothing_at_zero_noise': smoothing_control,
        'timing_repeats': TIMING_REPEATS if smoothing_control else 0,
        'timing_scope': 'method-only reconstruction of both native grids; excludes common preprocessing/scoring/I/O',
        'timing_environment': {'os': platform.system(), 'architecture': platform.machine(),
                               'python': platform.python_version(), 'numpy': np.__version__},
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    primary = summary[(summary.event_count == 5000) & (summary.grid == 32)]
    print(primary[['stream', 'position_sigma_mm', 'hard_iou_mean',
                   'simple_iou_mean', 'propagated_iou_mean',
                   'propagated_gain_mean', 'propagated_wins',
                   'propagated_losses']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=None)
    parser.add_argument('--score-audit', action='store_true', help='多指标复核，默认写入独立目录')
    parser.add_argument('--smoothing-control', action='store_true', help='固定核平滑对照与方法增量计时')
    args = parser.parse_args()
    run(args.output_dir or ROOT / 'muon_analysis' / (
        'frame_smoothing_control_results' if args.smoothing_control else
        'frame_gate_score_audit_results' if args.score_audit else 'frame_z_gate_results'),
        args.score_audit, args.smoothing_control)
