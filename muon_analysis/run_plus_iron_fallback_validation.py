"""新铁制十字目标的冻结噪声底回退 H/P/F 配对检验。"""
from __future__ import annotations

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
from run_noise_floor_fallback_diagnostic import choose_propagated
from run_soft_z_gate_experiment import (
    COUNTS, GRIDS, REPEATS, ROOT, SIGMAS_MM, candidate_events, xy_image,
    z_acceptance_weights,
)
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

STREAMS = {'seed_a': 'muon_tracks_plus_iron_seed_a.csv',
           'seed_b': 'muon_tracks_plus_iron_seed_b.csv'}
KEYS = ['stream', 'event_count', 'repeat', 'position_sigma_mm', 'grid']


def plus_mask(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """与 Geant4 三块不重叠 G4Box 同足迹；只用于评价。"""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    stem = (np.abs(x) <= 15.) & (np.abs(y) <= 60.)
    arms = (np.abs(x) <= 60.) & (np.abs(y) <= 15.)
    return stem | arms


def score_plus(image: np.ndarray, truth: np.ndarray,
               cx: np.ndarray, cy: np.ndarray) -> dict[str, float]:
    """真值面积 Top-k IoU 与十字四角空白误报，均不反馈给算法。"""
    if image.shape != truth.shape or not np.isfinite(image).all():
        raise ValueError('Invalid image')
    k = int(truth.sum())
    predicted = np.zeros_like(truth, dtype=bool)
    predicted.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    x, y = np.meshgrid(cx, cy, indexing='ij')
    corners = ((np.abs(x) > 15.) & (np.abs(x) < 60.)
               & (np.abs(y) > 15.) & (np.abs(y) < 60.))
    intersection = int((predicted & truth).sum())
    return {'iou': float(intersection / (predicted | truth).sum()),
            'corner_false_positive_fraction': float((predicted & corners).sum()
                                                    / corners.sum())}


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = plus_mask(cx, cy)
    rows = []
    for stream_index, (stream, filename) in enumerate(STREAMS.items()):
        data = pd.read_csv(ROOT / 'muon_analysis' / filename)
        if len(data) < max(COUNTS):
            raise ValueError(f'{stream}: insufficient complete tracks')
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 8107 + 1000 * stream_index)
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
                    seed = (RANDOM_SEED + 900000 + 300000 * stream_index
                            + 10000 * count + 100 * repeat
                            + int(round(sigma * 1000)))
                    samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                 seed, BOOTSTRAP_DRAWS)
                    hard, propagated = z_acceptance_weights(z, samples)
                    use_p = choose_propagated(sigma, spacing)
                    fallback = propagated if use_p else hard
                    if sigma == 0 and not np.array_equal(hard, propagated):
                        raise AssertionError('Zero-noise H/P identity failed')
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    for bins in GRIDS:
                        for method, gate in (('hard', hard),
                                             ('propagated', propagated),
                                             ('fallback', fallback)):
                            image = xy_image(xy, angle_weight * gate, bins)
                            rows.append({
                                'stream': stream, 'event_count': count,
                                'repeat': repeat, 'position_sigma_mm': sigma,
                                'grid': bins, 'method': method,
                                'choose_propagated': use_p,
                                'candidates': len(ids),
                                **score_plus(image, truth, cx, cy),
                            })
                print(f'Plus iron {stream}: N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    expected = len(STREAMS) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRIDS) * 3
    if len(frame) != expected or frame.duplicated(KEYS + ['method']).any():
        raise AssertionError('Incomplete paired validation')
    views = {name: frame[frame.method == name].set_index(KEYS).sort_index()
             for name in ('hard', 'propagated', 'fallback')}
    hard, p, f = (views[name] for name in ('hard', 'propagated', 'fallback'))
    paired = pd.DataFrame({
        'hard_iou': hard.iou, 'propagated_iou': p.iou, 'fallback_iou': f.iou,
        'propagated_gain_vs_hard': p.iou - hard.iou,
        'fallback_gain_vs_hard': f.iou - hard.iou,
        'fallback_gain_vs_propagated': f.iou - p.iou,
        'hard_corner_fp': hard.corner_false_positive_fraction,
        'propagated_corner_fp': p.corner_false_positive_fraction,
        'fallback_corner_fp': f.corner_false_positive_fraction,
        'choose_propagated': f.choose_propagated,
    }).reset_index()
    if not np.array_equal(paired.fallback_iou,
                          np.where(paired.choose_propagated,
                                   paired.propagated_iou, paired.hard_iou)):
        raise AssertionError('Fallback diverged from chosen method')
    if not np.allclose(paired[paired.position_sigma_mm == 0]
                       [['propagated_gain_vs_hard', 'fallback_gain_vs_hard']], 0):
        raise AssertionError('Zero-noise image changed')
    summary = paired.groupby(['stream', 'event_count', 'position_sigma_mm', 'grid'],
                             as_index=False).agg(
        repeats=('repeat', 'size'),
        choose_propagated=('choose_propagated', 'first'),
        hard_iou_mean=('hard_iou', 'mean'),
        propagated_iou_mean=('propagated_iou', 'mean'),
        fallback_iou_mean=('fallback_iou', 'mean'),
        propagated_gain_mean=('propagated_gain_vs_hard', 'mean'),
        fallback_gain_mean=('fallback_gain_vs_hard', 'mean'),
        fallback_vs_propagated_mean=('fallback_gain_vs_propagated', 'mean'),
        hard_corner_fp_mean=('hard_corner_fp', 'mean'),
        propagated_corner_fp_mean=('propagated_corner_fp', 'mean'),
        fallback_corner_fp_mean=('fallback_corner_fp', 'mean'))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for stream, marker in (('seed_a', 'o'), ('seed_b', 's')):
        part = summary[(summary.stream == stream) & (summary.event_count == 5000)
                       & (summary.grid == 32)].sort_values('position_sigma_mm')
        ax.plot(part.position_sigma_mm, part.propagated_gain_mean,
                marker=marker, label=f'{stream}: P − H')
        ax.plot(part.position_sigma_mm, part.fallback_gain_mean,
                marker=marker, ls='--', label=f'{stream}: F − H')
    ax.axhline(0, color='black', lw=1)
    ax.set(xlabel='Hit-position sigma [mm]', ylabel='Paired equal-area IoU gain',
           title='Unseen plus-shaped iron target, 5000 input muons, 32×32 XY')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / 'plus_iron_fallback_5000_32.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'protocol': 'NOISE_FLOOR_FALLBACK_PROTOCOL.md',
        'role': 'previously unseen shape and material, two Geant4 seeds',
        'streams': STREAMS, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'grids': GRIDS, 'repeats': REPEATS, 'draws': BOOTSTRAP_DRAWS,
        'fallback': 'hard if noise_floor/theta_min>=2 else propagated',
        'truth_only_for_scoring': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary[(summary.event_count == 5000) & (summary.grid == 32)][
        ['stream', 'position_sigma_mm', 'choose_propagated',
         'hard_iou_mean', 'propagated_iou_mean', 'fallback_iou_mean',
         'propagated_gain_mean', 'fallback_gain_mean',
         'hard_corner_fp_mean', 'propagated_corner_fp_mean']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/plus_iron_fallback_results')
