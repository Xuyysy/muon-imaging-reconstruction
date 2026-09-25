"""观测角度/深度的简易连续门，与 32 次扰动软门同事件比较。

先读 SIMPLE_Z_GATE_BASELINE_PLAN.md。本实验使用已观察过的开发场景，
只考察简易规则能否解释先前软门收益；不调尺度常数或宣称独立验证。
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
from run_adaptive_voxel_experiment import RANDOM_SEED, Z_LIMIT_MM
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_soft_z_gate_experiment import (
    COUNTS, GRIDS, REPEATS, ROOT, SCENES, SIGMAS_MM, candidate_events,
    score_image, xy_image, z_acceptance_weights,
)
from run_uncertainty_aware_angle_experiment import (
    same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

KEYS = ['scenario', 'event_count', 'repeat', 'position_sigma_mm', 'grid']


def simple_z_gate(observed_z_mm: np.ndarray,
                  corrected_theta_mrad: np.ndarray,
                  sigma_mm: float) -> np.ndarray:
    """观测侧无待调常数的连续边界启发式，**不是校准的后验概率**。

    近平行轨迹的小角度放大深度不稳定性，故用 sigma/theta 作为
    深度软化尺度。θ 已通过共同 2 mrad 门，确保分母不是零。
    """
    z = np.asarray(observed_z_mm, float)
    theta = np.asarray(corrected_theta_mrad, float)
    if z.shape != theta.shape or not np.isfinite(z).all() or not np.isfinite(theta).all():
        raise ValueError('Unpaired or nonfinite observed Z and corrected theta')
    if sigma_mm < 0 or (theta <= 0).any():
        raise ValueError('Sigma must be nonnegative and selected theta positive')
    hard = (np.abs(z) <= Z_LIMIT_MM).astype(float)
    if sigma_mm == 0:
        return hard
    scale_mm = sigma_mm / (theta / 1000.0)
    signed_margin = (Z_LIMIT_MM - np.abs(z)) / scale_mm
    # 数值截断只防止 exp 溢出，不改变常规边界附近的权重。
    signed_margin = np.clip(signed_margin, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-signed_margin))


def run(output_dir: Path) -> None:
    rows: list[dict] = []
    from run_adaptive_voxel_experiment import imaging_edges
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        data = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        truth = mask_fn(cx, cy)  # 真值只给评价函数，三种门完全不用。
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
                    z = observed.poca_z_mm.to_numpy()[ids]
                    theta_obs = observed.theta_mrad.to_numpy()[ids]
                    noise_floor = theta_noise_floor_mrad(sigma, spacing)
                    theta_corr = np.sqrt(np.maximum(theta_obs ** 2 - noise_floor ** 2, 0.))
                    start = time.perf_counter()
                    simple = simple_z_gate(z, theta_corr, sigma)
                    simple_ms = (time.perf_counter() - start) * 1000.0
                    seed = (RANDOM_SEED + 300000 * scene_index + 10000 * count
                            + 100 * repeat + int(round(sigma * 1000)))
                    start = time.perf_counter()
                    z_samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                   seed, BOOTSTRAP_DRAWS)
                    propagated_ms = (time.perf_counter() - start) * 1000.0
                    hard, propagated = z_acceptance_weights(z, z_samples)
                    if sigma == 0 and (not np.array_equal(simple, hard)
                                       or not np.array_equal(propagated, hard)):
                        raise AssertionError('Zero-noise gate control failed')
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    for bins in GRIDS:
                        for name, gate, estimator_ms in (
                            ('hard', hard, 0.), ('simple', simple, simple_ms),
                            ('propagated', propagated, propagated_ms)):
                            image = xy_image(xy, weight * gate, bins)
                            scored = score_image(image, truth, cx, cy, scene)
                            rows.append({
                                'scenario': scene, 'event_count': count,
                                'repeat': repeat, 'position_sigma_mm': sigma,
                                'grid': bins, 'method': name,
                                'xy_angle_candidates': len(ids),
                                'effective_z_accepted_events': float(gate.sum()),
                                'total_image_weight': float(image.sum()),
                                'estimator_ms': estimator_ms, **scored,
                            })
                print(f'Simple Z gate: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    expected = len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRIDS) * 3
    if len(frame) != expected or frame.duplicated(KEYS + ['method']).any():
        raise AssertionError('Incomplete or duplicate paired conditions')
    variants = {name: frame[frame.method == name].set_index(KEYS).sort_index()
                for name in ('hard', 'simple', 'propagated')}
    hard, simple, propagated = (variants[name] for name in
                                ('hard', 'simple', 'propagated'))
    # 同一 RNG 与前轮一致；不能用相近的均值冒充相同事件的比较。
    prior = pd.read_csv(ROOT / 'muon_analysis/soft_z_gate_results/by_repeat.csv')
    for old_name, current in (('hard', hard), ('soft_z', propagated)):
        old = prior[prior.method == old_name].set_index(KEYS).sort_index()
        if len(old) != len(current) or not np.allclose(old.iou, current.iou,
                                                     rtol=0, atol=1e-12):
            raise AssertionError(f'Previous {old_name} pairing could not be reproduced')
    paired = pd.DataFrame({
        'hard_iou': hard.iou, 'simple_iou': simple.iou,
        'propagated_iou': propagated.iou,
        'simple_gain_vs_hard': simple.iou - hard.iou,
        'propagated_gain_vs_hard': propagated.iou - hard.iou,
        'propagated_gain_vs_simple': propagated.iou - simple.iou,
        'hard_global_error_mm': hard.global_error_mm,
        'simple_global_error_mm': simple.global_error_mm,
        'propagated_global_error_mm': propagated.global_error_mm,
        'hard_min_block_recall': hard.min_block_recall,
        'simple_min_block_recall': simple.min_block_recall,
        'propagated_min_block_recall': propagated.min_block_recall,
        'hard_gap_false_positive': hard.gap_false_positive_fraction,
        'simple_gap_false_positive': simple.gap_false_positive_fraction,
        'propagated_gap_false_positive': propagated.gap_false_positive_fraction,
        'simple_estimator_ms': simple.estimator_ms,
        'propagated_estimator_ms': propagated.estimator_ms,
    }).reset_index()
    zero = paired[paired.position_sigma_mm == 0]
    if not np.allclose(zero.simple_gain_vs_hard, 0) or not np.allclose(
            zero.propagated_gain_vs_hard, 0):
        raise AssertionError('Zero-noise images changed')
    summary = paired.groupby(['scenario', 'event_count', 'position_sigma_mm', 'grid'],
                             as_index=False).agg(
        repeats=('repeat', 'size'), hard_iou_mean=('hard_iou', 'mean'),
        simple_iou_mean=('simple_iou', 'mean'),
        propagated_iou_mean=('propagated_iou', 'mean'),
        simple_gain_mean=('simple_gain_vs_hard', 'mean'),
        propagated_gain_mean=('propagated_gain_vs_hard', 'mean'),
        extra_gain_mean=('propagated_gain_vs_simple', 'mean'),
        extra_gain_std=('propagated_gain_vs_simple', 'std'),
        extra_wins=('propagated_gain_vs_simple', lambda x: int((x > 1e-12).sum())),
        extra_losses=('propagated_gain_vs_simple', lambda x: int((x < -1e-12).sum())),
        simple_estimator_ms_mean=('simple_estimator_ms', 'mean'),
        propagated_estimator_ms_mean=('propagated_estimator_ms', 'mean'))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True,
                             constrained_layout=True)
    for ax, scene in zip(axes, SCENES):
        part = summary[(summary.scenario == scene) & (summary.event_count == 5000)
                       & (summary.grid == 32)].sort_values('position_sigma_mm')
        ax.plot(part.position_sigma_mm, part.simple_gain_mean, 'o-', label='Simple vs hard')
        ax.plot(part.position_sigma_mm, part.propagated_gain_mean, 's-',
                label='32-draw vs hard')
        ax.axhline(0, color='black', lw=1)
        ax.set(title=scene, xlabel='Hit-position sigma [mm]')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Paired equal-area IoU gain')
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / 'simple_vs_propagated_5000_32.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'SIMPLE_Z_GATE_BASELINE_PLAN.md',
        'role': 'post-hoc development comparison, not independent validation',
        'scenes': list(SCENES), 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'grids': GRIDS, 'bootstrap_draws': BOOTSTRAP_DRAWS,
        'simple_scale_mm': 'sigma_mm / (theta_corrected_mrad/1000)',
        'simple_weight': 'sigmoid((100-abs(poca_z_observed_mm))/simple_scale_mm)',
        'sigma_zero': 'all methods equal to hard gate',
        'truth_used_only_for_evaluation': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary[(summary.event_count == 5000) & (summary.grid == 32)
                  & (summary.position_sigma_mm > 0)][
        ['scenario', 'position_sigma_mm', 'hard_iou_mean', 'simple_iou_mean',
         'propagated_iou_mean', 'simple_gain_mean', 'propagated_gain_mean',
         'extra_gain_mean', 'extra_wins', 'extra_losses']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/simple_z_gate_results')
    run(parser.parse_args().output_dir)
