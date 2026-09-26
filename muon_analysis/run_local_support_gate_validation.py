"""冻结 Q 门在新 75 mm 铝/铁十字目标上的预定检验。"""
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
from run_plus_iron_fallback_validation import plus_mask, score_plus
from run_soft_z_gate_experiment import ROOT, candidate_events, xy_image, z_acceptance_weights
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

MATERIALS = ('al', 'iron')
STREAMS = ('a', 'b')
COUNTS = (1000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.30)
REPEATS = 5
GRID = 32
KEY = ['material', 'stream', 'event_count', 'repeat', 'position_sigma_mm']


def source_csv(material: str, stream: str) -> Path:
    if material not in MATERIALS or stream not in STREAMS:
        raise ValueError('Unknown material or stream')
    return ROOT / 'muon_analysis' / f'muon_tracks_plus_{material}_mid_seed_{stream}.csv'


def decide_success(summary: pd.DataFrame) -> dict:
    """预先固定的条件判断；绝不按测试数据改动判据。"""
    high = summary[(summary.event_count == 5000)
                   & (summary.position_sigma_mm == .30)]
    low = summary[(summary.event_count == 5000)
                  & (summary.position_sigma_mm == .05)]
    if len(high) != 4 or len(low) != 4:
        raise AssertionError('Missing preregistered primary or safety conditions')
    high_pass = bool((high.q_minus_h >= -1e-12).all()
                     and (high.q_minus_p >= -1e-12).all())
    low_pass = bool((low.q_minus_p >= -.02 - 1e-12).all())
    return {
        'primary_high_noise_pass': high_pass,
        'low_noise_safety_pass': low_pass,
        'overall_pass': high_pass and low_pass,
        'criterion': ('N=5000 σ=0.30 mm: all four Q-H>=0 and Q-P>=0; '
                      'N=5000 σ=0.05 mm: all four Q-P>=-0.02'),
    }


def run(output_dir: Path) -> None:
    edges = imaging_edges(GRID)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = plus_mask(cx, cy)
    rows = []
    source_counts = []
    for material_index, material in enumerate(MATERIALS):
        for stream_index, stream in enumerate(STREAMS):
            tracks = pd.read_csv(source_csv(material, stream))
            if len(tracks) < max(COUNTS):
                raise ValueError('Not enough complete tracks')
            source_counts.append({'material': material, 'stream': stream,
                                  'complete_tracks': len(tracks)})
            spacing = same_side_layer_spacing_mm(tracks)
            rng = np.random.default_rng(RANDOM_SEED + 12013
                                        + 1000 * stream_index + 100 * material_index)
            for count in COUNTS:
                for repeat in range(1, REPEATS + 1):
                    subset = tracks.iloc[rng.choice(len(tracks), count,
                                                   replace=False)].copy()
                    normals = {f'{plane}_{axis}': rng.normal(0, 1, count)
                               for plane in ('up1', 'up2', 'down1', 'down2')
                               for axis in ('x', 'y')}
                    for sigma in SIGMAS_MM:
                        measured = measured_hits(subset, sigma, normals)
                        observed = recompute_from_noisy_hits(
                            measured, 0., np.random.default_rng(0))
                        ids, angle_weight = candidate_events(observed, sigma, spacing)
                        xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                        z = observed.poca_z_mm.to_numpy()[ids]
                        seed = (RANDOM_SEED + 1200000 + 300000 * material_index
                                + 100000 * stream_index + 10000 * count
                                + 100 * repeat + int(round(sigma * 1000)))
                        samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                     seed, BOOTSTRAP_DRAWS)
                        h, p = z_acceptance_weights(z, samples)
                        q = local_support_gate(xy, angle_weight, h, p)
                        if sigma == 0 and not (np.array_equal(h, p)
                                               and np.array_equal(h, q)):
                            raise AssertionError('Zero-noise H/P/Q should match')
                        for method, gate in (('hard', h), ('propagated', p),
                                             ('local_support', q)):
                            image = xy_image(xy, angle_weight * gate, GRID)
                            rows.append({
                                'material': material, 'stream': stream,
                                'event_count': count, 'repeat': repeat,
                                'position_sigma_mm': sigma, 'method': method,
                                'candidate_count': len(ids),
                                'hard_z_count': int(h.sum()),
                                **score_plus(image, truth, cx, cy),
                            })
                    print(f'75 mm validation: {material}, {stream}, '
                          f'N={count}, repeat {repeat}/{REPEATS}', flush=True)
    frame = pd.DataFrame(rows)
    expected = len(MATERIALS) * len(STREAMS) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * 3
    if len(frame) != expected or frame.duplicated(KEY + ['method']).any():
        raise AssertionError('Incomplete or duplicated validation matrix')
    views = {method: frame[frame.method == method].set_index(KEY).sort_index()
             for method in ('hard', 'propagated', 'local_support')}
    h, p, q = (views[method] for method in
               ('hard', 'propagated', 'local_support'))
    if not (h.index.equals(p.index) and h.index.equals(q.index)):
        raise AssertionError('H/P/Q not paired')
    paired = pd.DataFrame({
        'candidate_count': h.candidate_count,
        'hard_z_count': h.hard_z_count,
        'h_iou': h.iou, 'p_iou': p.iou, 'q_iou': q.iou,
        'p_minus_h': p.iou - h.iou,
        'q_minus_h': q.iou - h.iou,
        'q_minus_p': q.iou - p.iou,
        'h_corner_fp': h.corner_false_positive_fraction,
        'p_corner_fp': p.corner_false_positive_fraction,
        'q_corner_fp': q.corner_false_positive_fraction,
    }).reset_index()
    if not np.array_equal(paired.loc[paired.position_sigma_mm == 0,
                                     ['p_minus_h', 'q_minus_h', 'q_minus_p']],
                          np.zeros((len(paired[paired.position_sigma_mm == 0]), 3))):
        raise AssertionError('Zero-noise scores differ')
    summary = paired.groupby(['material', 'stream', 'event_count',
                              'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{name: (name, 'mean') for name in paired.columns if name not in KEY})
    verdict = decide_success(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    pd.DataFrame(source_counts).to_csv(output_dir / 'source_counts.csv', index=False)
    (output_dir / 'verdict.json').write_text(json.dumps(verdict, ensure_ascii=False,
                                                        indent=2) + '\n', encoding='utf-8')
    (output_dir / 'config.json').write_text(json.dumps({
        'protocol': 'LOCAL_SUPPORT_GATE_VALIDATION_PROTOCOL.md',
        'role': 'frozen Q validation on unseen 75 mm thickness, same plus XY footprint',
        'materials': MATERIALS, 'streams': STREAMS,
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'grid': GRID, 'draws': BOOTSTRAP_DRAWS,
        'target_truth_only_for_scoring': True,
        'event_level_material_pairing_assumed': False,
        'formula': 'Q = h*p + (1-h)*p*n_eff/(n_eff+2) on 16x16 hard-gate grid',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True,
                             sharey=True)
    for material, color in (('al', 'tab:blue'), ('iron', 'tab:red')):
        for stream, marker in (('a', 'o'), ('b', 's')):
            part = summary[(summary.material == material) & (summary.stream == stream)
                           & (summary.event_count == 5000)].sort_values('position_sigma_mm')
            for ax, column in zip(axes, ('q_minus_h', 'q_minus_p')):
                ax.plot(part.position_sigma_mm, part[column],
                        color=color, marker=marker, label=f'{material} {stream}')
    for ax, title in zip(axes, ('Q − H', 'Q − P')):
        ax.axhline(0, color='black', lw=1)
        ax.set(xlabel='Hit-position sigma [mm]', title=title)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Mean equal-area IoU difference')
    axes[0].legend(fontsize=8)
    fig.suptitle('Frozen local-support Q on unseen 75 mm plus target, N=5000')
    fig.savefig(output_dir / 'local_support_gate_75mm.png', dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(verdict)


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/local_support_gate_validation_results')
