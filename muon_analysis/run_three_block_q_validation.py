"""按生成数据前冻结的方案，在新三块阶梯形铁目标上检验 H/P/Q。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from local_support_soft_z_gate import local_support_gate
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_soft_z_gate_experiment import ROOT, candidate_events, xy_image, z_acceptance_weights
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

STREAMS = ('a', 'b')
COUNTS = (1000, 5000)
SIGMAS_MM = (0., .05, .30)
REPEATS = 5
GRID = 32
KEY = ['stream', 'event_count', 'repeat', 'position_sigma_mm']
BLOCK_CENTERS_MM = ((-50., -45.), (0., 0.), (50., 45.))
BLOCK_HALF_WIDTH_MM = 20.


def block_truth(cx: np.ndarray, cy: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """仅供终点评分；用 32×32 网格中心判定三个 40 mm 方柱足迹。"""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    blocks = [((np.abs(x - bx) <= BLOCK_HALF_WIDTH_MM)
               & (np.abs(y - by) <= BLOCK_HALF_WIDTH_MM))
              for bx, by in BLOCK_CENTERS_MM]
    if not all(block.any() for block in blocks):
        raise AssertionError('Target block has no scoring pixels')
    return np.logical_or.reduce(blocks), blocks


def score_image(image: np.ndarray, truth: np.ndarray,
                blocks: list[np.ndarray]) -> dict[str, float]:
    """等面积 Top-k IoU 与各块召回；评分不反馈到事件或参数选择。"""
    if image.shape != truth.shape or not np.isfinite(image).all():
        raise ValueError('Invalid image')
    predicted = np.zeros_like(truth, dtype=bool)
    k = int(truth.sum())
    predicted.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    intersection = int((predicted & truth).sum())
    recalls = [float((predicted & block).sum() / block.sum()) for block in blocks]
    return {'iou': float(intersection / (predicted | truth).sum()),
            'min_block_recall': min(recalls),
            **{f'block_{i + 1}_recall': value for i, value in enumerate(recalls)}}


def verdict(summary: pd.DataFrame) -> dict:
    """严格使用事先规定的两流低统计量主判据和轻误差安全判据。"""
    primary = summary[(summary.event_count == 1000)
                      & (summary.position_sigma_mm == .30)]
    safety = summary[(summary.event_count == 1000)
                     & (summary.position_sigma_mm == .05)]
    if (list(primary.stream) != list(STREAMS)
            or list(safety.stream) != list(STREAMS)):
        raise AssertionError('Missing preregistered streams or conditions')
    primary_pass = bool((primary.q_minus_p >= -1e-12).all()
                        and (primary.q_minus_h >= -1e-12).all()
                        and (primary.q_minus_p_min_block_recall >= -.10 - 1e-12).all())
    safety_pass = bool((safety.q_minus_p >= -.02 - 1e-12).all())
    return {
        'primary_low_count_high_error_pass': primary_pass,
        'low_error_safety_pass': safety_pass,
        'overall_pass': primary_pass and safety_pass,
        'criterion': ('N=1000 σ=0.30 mm: both streams mean Q-P>=0, Q-H>=0, '
                      'min-block-recall(Q-P)>=-0.10; '
                      'N=1000 σ=0.05 mm: both streams mean Q-P>=-0.02'),
        'scope': 'new three-block iron shape only; prior 75 mm plus Q failure remains',
    }


def run(output_dir: Path) -> None:
    edges = imaging_edges(GRID)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth, blocks = block_truth(cx, cy)
    rows = []
    sources = []
    for si, stream in enumerate(STREAMS):
        source = ROOT / 'muon_analysis' / f'muon_tracks_three_block_iron_seed_{stream}.csv'
        tracks = pd.read_csv(source)
        if len(tracks) < max(COUNTS):
            raise ValueError(f'Insufficient complete tracks in {source}')
        sources.append({'stream': stream, 'complete_tracks': len(tracks),
                        'csv': source.name})
        spacing = same_side_layer_spacing_mm(tracks)
        rng = np.random.default_rng(RANDOM_SEED + 29011 + 1000 * si)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = tracks.iloc[rng.choice(len(tracks), count, replace=False)].copy()
                normals = {f'{plane}_{axis}': rng.normal(size=count)
                           for plane in ('up1', 'up2', 'down1', 'down2')
                           for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, normals)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    ids, angle_weight = candidate_events(observed, sigma, spacing)
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    z = observed.poca_z_mm.to_numpy()[ids]
                    seed = (RANDOM_SEED + 2900000 + 100000 * si
                            + 10000 * count + 100 * repeat + int(round(sigma * 1000)))
                    samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                 seed, BOOTSTRAP_DRAWS)
                    h, p = z_acceptance_weights(z, samples)
                    q = local_support_gate(xy, angle_weight, h, p)
                    if sigma == 0 and not (np.array_equal(h, p)
                                           and np.array_equal(h, q)):
                        raise AssertionError('Zero-noise H/P/Q differ')
                    for method, gate in (('hard', h), ('propagated', p),
                                         ('local_support', q)):
                        image = xy_image(xy, angle_weight * gate, GRID)
                        rows.append({
                            'stream': stream, 'event_count': count,
                            'repeat': repeat, 'position_sigma_mm': sigma,
                            'method': method, 'candidate_count': len(ids),
                            'hard_z_count': int(h.sum()),
                            **score_image(image, truth, blocks),
                        })
                print(f'Three-block Q validation: {stream}, N={count}, '
                      f'repeat {repeat}/{REPEATS}', flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 2 * len(COUNTS) * REPEATS * len(SIGMAS_MM) * 3:
        raise AssertionError('Incomplete validation matrix')
    if frame.duplicated(KEY + ['method']).any():
        raise AssertionError('Duplicated validation rows')
    methods = {name: frame[frame.method == name].set_index(KEY).sort_index()
               for name in ('hard', 'propagated', 'local_support')}
    h, p, q = (methods[name] for name in ('hard', 'propagated', 'local_support'))
    if not (h.index.equals(p.index) and h.index.equals(q.index)):
        raise AssertionError('H/P/Q results are not paired')
    paired = pd.DataFrame({
        'candidate_count': h.candidate_count,
        'hard_z_count': h.hard_z_count,
        'h_iou': h.iou, 'p_iou': p.iou, 'q_iou': q.iou,
        'q_minus_p': q.iou - p.iou,
        'q_minus_h': q.iou - h.iou,
        'p_minus_h': p.iou - h.iou,
        'h_min_block_recall': h.min_block_recall,
        'p_min_block_recall': p.min_block_recall,
        'q_min_block_recall': q.min_block_recall,
        'q_minus_p_min_block_recall': q.min_block_recall - p.min_block_recall,
        **{f'{method}_block_{i}_recall': methods[method].loc[:, f'block_{i}_recall']
           for method in ('hard', 'propagated', 'local_support')
           for i in (1, 2, 3)},
    }).reset_index()
    zero = paired[paired.position_sigma_mm == 0]
    if not np.array_equal(zero[['q_minus_p', 'q_minus_h', 'p_minus_h']].to_numpy(),
                          np.zeros((len(zero), 3))):
        raise AssertionError('Zero-noise scores differ')
    summary = paired.groupby(['stream', 'event_count', 'position_sigma_mm'],
                             as_index=False).agg(
        repeats=('repeat', 'size'),
        **{col: (col, 'mean') for col in paired.columns if col not in KEY})
    result = verdict(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_method_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    pd.DataFrame(sources).to_csv(output_dir / 'source_counts.csv', index=False)
    (output_dir / 'verdict.json').write_text(json.dumps(
        result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'config.json').write_text(json.dumps({
        'protocol': 'THREE_BLOCK_Q_VALIDATION_PROTOCOL.md',
        'role': 'new-shape frozen Q test, not amendment of prior failed primary test',
        'streams': STREAMS, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'grid': GRID, 'draws': BOOTSTRAP_DRAWS,
        'block_centers_mm': BLOCK_CENTERS_MM,
        'block_half_width_mm': BLOCK_HALF_WIDTH_MM,
        'truth_only_for_final_scoring': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for stream, color in (('a', 'tab:blue'), ('b', 'tab:orange')):
        part = summary[(summary.stream == stream) & (summary.event_count == 1000)]
        part = part.sort_values('position_sigma_mm')
        for ax, metric in zip(axes, ('q_minus_p', 'q_minus_h')):
            ax.plot(part.position_sigma_mm, part[metric], color=color,
                    marker='o', label=f'stream {stream.upper()}')
    for ax in axes:
        ax.axhline(0, color='black', lw=1)
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Mean equal-area IoU: Q − P')
    axes[1].set_ylabel('Mean equal-area IoU: Q − H')
    axes[0].legend(fontsize=8)
    fig.suptitle('Frozen Q on new three-block iron shape, N=1000')
    fig.savefig(output_dir / 'three_block_q_validation.png', dpi=180)
    plt.close(fig)
    print(summary[['stream', 'event_count', 'position_sigma_mm',
                   'h_iou', 'p_iou', 'q_iou', 'q_minus_p', 'q_minus_h',
                   'q_minus_p_min_block_recall']].to_string(index=False))
    print(result)


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/three_block_q_validation_results')
