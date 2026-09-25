"""观测侧软 PoCA-Z 边界的同事件开发集消融。

仅把硬 Z 入选 0/1 替换为击中误差传播得到的通过比例 p_z。
散射角处理、名义 XY 落点和体素尺度都保持相同；这不是完整的
三维概率重建。先读 SOFT_Z_GATE_PLAN.md 再解释结果。
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

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from diagnose_split_confidence import ROOT, SCENES
from run_adaptive_voxel_experiment import (
    RANDOM_SEED, THETA_CAP_MRAD, THETA_MIN_MRAD, XY_LIMIT_MM, Z_LIMIT_MM,
    imaging_edges,
)
from run_continuous_event_quality_experiment import evaluate
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_double_block_validation import shape_metrics
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import (
    same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
GRIDS = (16, 32)


def candidate_events(observed: pd.DataFrame, sigma: float,
                     layer_spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """观测角度与 XY 门共同固定，返回事件索引和截断散射权重。

这里**不**检查名义 PoCA-Z；这样原本 Z 越界、但误差扰动候选可能
入界的事件，也能被软门自然纳入。真实/无误差 PoCA 均不使用。
    """
    theta = observed.theta_mrad.to_numpy(float)
    floor = theta_noise_floor_mrad(sigma, layer_spacing_mm)
    corrected = np.sqrt(np.maximum(theta ** 2 - floor ** 2, 0.0))
    xy = ((observed.poca_x_mm.abs() <= XY_LIMIT_MM)
          & (observed.poca_y_mm.abs() <= XY_LIMIT_MM)).to_numpy()
    selected = np.flatnonzero((corrected >= THETA_MIN_MRAD) & xy
                              & np.isfinite(corrected))
    weight = np.minimum(corrected[selected] ** 2, THETA_CAP_MRAD ** 2)
    return selected, weight


def z_acceptance_weights(observed_z: np.ndarray,
                         sampled_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同一候选事件的硬门 0/1 与软门通过比例；零噪声必须一致。"""
    observed_z = np.asarray(observed_z, float)
    sampled_z = np.asarray(sampled_z, float)
    if sampled_z.ndim != 2 or sampled_z.shape[0] != len(observed_z):
        raise ValueError('Unpaired observed and propagated PoCA-Z')
    if not np.isfinite(observed_z).all() or not np.isfinite(sampled_z).all():
        raise ValueError('Nonfinite PoCA-Z samples')
    hard = (np.abs(observed_z) <= Z_LIMIT_MM).astype(float)
    soft = np.mean(np.abs(sampled_z) <= Z_LIMIT_MM, axis=1)
    return hard, soft


def xy_image(xy: np.ndarray, weights: np.ndarray, bins: int) -> np.ndarray:
    """固定 XY 体素投影；16 格均匀映射到 32 格以公平计算 IoU。"""
    if bins not in GRIDS or xy.shape != (len(weights), 2):
        raise ValueError('Invalid XY point array or grid')
    edges = imaging_edges(bins)
    histogram, _, _ = np.histogram2d(xy[:, 0], xy[:, 1],
                                      bins=edges[:2], weights=weights)
    if bins == 16:
        histogram = np.repeat(np.repeat(histogram / 4.0, 2, axis=0),
                              2, axis=1)
    return histogram


def score_image(image: np.ndarray, truth: np.ndarray,
                cx: np.ndarray, cy: np.ndarray, scene: str) -> dict:
    """真值只进入最终评估；双目标另统计最弱目标和空隙伪影。"""
    scored = evaluate(image, truth, cx, cy)
    result = {'iou': scored['iou'], 'global_error_mm': scored['global_error_mm']}
    if scene == 'double_block':
        extra = shape_metrics(image, truth, cx, cy)
        result.update({key: extra[key] for key in
                       ('min_block_recall', 'gap_false_positive_fraction')})
    else:
        result.update({'min_block_recall': np.nan,
                       'gap_false_positive_fraction': np.nan})
    return result


def run(output_dir: Path) -> None:
    rows = []
    shown = {}
    edges32 = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges32[:2]]
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        data = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        truth = mask_fn(cx, cy)
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 2107 + scene_index * 100)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                noise_draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                               for prefix in ('up1', 'up2', 'down1', 'down2')
                               for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, noise_draws)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    ids, weight = candidate_events(observed, sigma, spacing)
                    seed = (RANDOM_SEED + 300000 * scene_index + 10000 * count
                            + 100 * repeat + int(round(sigma * 1000)))
                    start = time.perf_counter()
                    z_samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                   seed, BOOTSTRAP_DRAWS)
                    propagation_ms = 1000 * (time.perf_counter() - start)
                    hard_gate, soft_gate = z_acceptance_weights(
                        observed.poca_z_mm.to_numpy()[ids], z_samples)
                    if sigma == 0 and not np.array_equal(hard_gate, soft_gate):
                        raise AssertionError('Zero-noise hard/soft gates differ')
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    for bins in GRIDS:
                        for method, gate in (('hard', hard_gate), ('soft_z', soft_gate)):
                            image = xy_image(xy, weight * gate, bins)
                            scored = score_image(image, truth, cx, cy, scene)
                            rows.append({
                                'scenario': scene, 'event_count': count,
                                'repeat': repeat, 'position_sigma_mm': sigma,
                                'grid': bins, 'method': method,
                                'xy_angle_candidates': len(ids),
                                'effective_z_accepted_events': float(gate.sum()),
                                'total_image_weight': float(image.sum()),
                                'propagation_ms': propagation_ms if method == 'soft_z' else 0.,
                                **scored,
                            })
                            if (scene == 'u_center' and count == 5000
                                    and repeat == 1 and sigma == .05):
                                shown[(bins, method)] = image
                print(f'Soft Z gate: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    keys = ['scenario', 'event_count', 'repeat', 'position_sigma_mm', 'grid']
    if frame.duplicated(keys + ['method']).any() or len(frame) != (
            len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRIDS) * 2):
        raise AssertionError('Incomplete paired condition grid')
    # 对每个指标都按同一条件配对，不能把不同随机事件抽样的均值误当差值。
    hard = frame[frame.method == 'hard'].set_index(keys)
    soft = frame[frame.method == 'soft_z'].set_index(keys)
    paired = pd.DataFrame({
        'iou_hard': hard.iou, 'iou_soft': soft.iou,
        'iou_gain': soft.iou - hard.iou,
        'global_error_hard_mm': hard.global_error_mm,
        'global_error_soft_mm': soft.global_error_mm,
        'global_error_gain_mm': hard.global_error_mm - soft.global_error_mm,
        'effective_events_hard': hard.effective_z_accepted_events,
        'effective_events_soft': soft.effective_z_accepted_events,
        'soft_propagation_ms': soft.propagation_ms,
    }).reset_index()
    zero = paired[paired.position_sigma_mm == 0]
    if not np.allclose(zero.iou_gain, 0) or not np.allclose(
            zero.global_error_gain_mm, 0):
        raise AssertionError('Zero-noise imaging control changed')
    summary = paired.groupby(['scenario', 'event_count', 'position_sigma_mm', 'grid'],
                             as_index=False).agg(
        repeats=('repeat', 'size'), iou_hard_mean=('iou_hard', 'mean'),
        iou_soft_mean=('iou_soft', 'mean'), iou_gain_mean=('iou_gain', 'mean'),
        iou_gain_std=('iou_gain', 'std'),
        iou_wins=('iou_gain', lambda x: int((x > 1.e-12).sum())),
        iou_losses=('iou_gain', lambda x: int((x < -1.e-12).sum())),
        global_error_gain_mm_mean=('global_error_gain_mm', 'mean'),
        effective_events_hard_mean=('effective_events_hard', 'mean'),
        effective_events_soft_mean=('effective_events_soft', 'mean'),
        soft_propagation_ms_mean=('soft_propagation_ms', 'mean'))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True,
                             constrained_layout=True)
    for ax, bins in zip(axes, GRIDS):
        part = summary[(summary.event_count == 5000) & (summary.grid == bins)]
        for scene in SCENES:
            one = part[part.scenario == scene].sort_values('position_sigma_mm')
            ax.plot(one.position_sigma_mm, one.iou_gain_mean,
                    marker='o', label=scene)
        ax.axhline(0, color='black', lw=1)
        ax.set(xlabel='Hit-position sigma [mm]',
               title=f'Fixed {bins}×{bins} XY grid')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Paired equal-area IoU gain (soft Z - hard Z)')
    axes[1].legend()
    fig.savefig(output_dir / 'soft_z_iou_gain_5000.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    common_vmax = max(float(np.log1p(image).max()) for image in shown.values())
    for ax, ((bins, method), image) in zip(axes.ravel(), sorted(shown.items())):
        ax.imshow(np.log1p(image).T, origin='lower', cmap='inferno',
                  vmin=0, vmax=common_vmax,
                  extent=[-XY_LIMIT_MM, XY_LIMIT_MM, -XY_LIMIT_MM, XY_LIMIT_MM])
        ax.set_title(f'U: {bins} grid, {method}')
        ax.set(xlabel='PoCA x [mm]', ylabel='PoCA y [mm]')
    fig.savefig(output_dir / 'u_soft_z_images_5000_sigma005.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'SOFT_Z_GATE_PLAN.md', 'scenes': list(SCENES),
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'grids': GRIDS, 'z_limit_mm': Z_LIMIT_MM,
        'bootstrap_draws': BOOTSTRAP_DRAWS,
        'soft_gate': 'fraction of propagated measured-hit PoCA-z samples within ROI',
        'xy_location': 'observed nominal PoCA XY for both methods',
        'role': '2.5D development-set ablation; no target truth in reconstruction',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary[(summary.event_count == 5000)
                  & (summary.position_sigma_mm == .05)].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/soft_z_gate_results')
    run(parser.parse_args().output_dir)
