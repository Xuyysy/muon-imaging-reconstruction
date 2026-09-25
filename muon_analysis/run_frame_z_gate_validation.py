"""按预注册方案，在全新方框目标的两份 Geant4 流上冻结验证 H/A/P。

先读 FRAME_Z_GATE_VALIDATION_PROTOCOL.md。此脚本不扫描参数，不用真值
决定门权重，也不根据当前结果回退到某个方法。真值只用于评价。
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

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
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

STREAMS = {
    'seed_a': 'muon_tracks_frame_seed_a.csv',
    'seed_b': 'muon_tracks_frame_seed_b.csv',
}
KEYS = ['stream', 'event_count', 'repeat', 'position_sigma_mm', 'grid']


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


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = frame_mask(cx, cy)
    rows = []
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
                    for bins in GRIDS:
                        for method, gate in (('hard', hard), ('simple', analytic),
                                             ('propagated', propagated)):
                            image = xy_image(xy, angle_weight * gate, bins)
                            rows.append({
                                'stream': stream, 'event_count': count,
                                'repeat': repeat, 'position_sigma_mm': sigma,
                                'grid': bins, 'method': method,
                                'candidates': len(ids),
                                'effective_z_accepted_events': float(gate.sum()),
                                **score_frame(image, truth, cx, cy),
                            })
                print(f'Frame {stream}: N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    expected = len(STREAMS) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRIDS) * 3
    if len(frame) != expected or frame.duplicated(KEYS + ['method']).any():
        raise AssertionError('Incomplete or duplicate paired validation')
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
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
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
    (output_dir / 'config.json').write_text(json.dumps({
        'protocol': 'FRAME_Z_GATE_VALIDATION_PROTOCOL.md',
        'role': 'held-out target family with two independent Geant4 seeds',
        'streams': STREAMS, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'grids': GRIDS, 'repeats': REPEATS,
        'bootstrap_draws': BOOTSTRAP_DRAWS,
        'primary': 'N=5000, grid=32, sigma=0.05/0.10, P-H IoU per stream',
        'truth_only_for_scoring': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    primary = summary[(summary.event_count == 5000) & (summary.grid == 32)]
    print(primary[['stream', 'position_sigma_mm', 'hard_iou_mean',
                   'simple_iou_mean', 'propagated_iou_mean',
                   'propagated_gain_mean', 'propagated_wins',
                   'propagated_losses']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/frame_z_gate_results')
    run(parser.parse_args().output_dir)
