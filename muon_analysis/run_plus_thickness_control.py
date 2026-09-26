"""同十字 XY 足迹的 50/100 mm 厚度对照；算法参数不按结果调整。

阅读 PLUS_THICKNESS_CONTROL_PLAN.md；Geant4 同种子不等于同一条缪子。
"""
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
from run_plus_iron_fallback_validation import plus_mask, score_plus
from run_soft_z_gate_experiment import ROOT, candidate_events, xy_image, z_acceptance_weights
from run_uncertainty_aware_angle_experiment import (
    same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

MATERIALS = ('al', 'iron')
THICKNESSES_MM = (50, 100)
STREAMS = ('a', 'b')
SIGMAS_MM = (0.0, 0.05, 0.30)
REPEATS = 5
COUNT = 5000
GRID = 32
KEY = ['material', 'thickness_mm', 'stream', 'repeat', 'position_sigma_mm']


def csv_for(material: str, thickness_mm: int, stream: str) -> Path:
    """厚度 100 mm 使用旧文件；50 mm 使用本轮新文件。"""
    if material not in MATERIALS or thickness_mm not in THICKNESSES_MM or stream not in STREAMS:
        raise ValueError('Unknown material/thickness/stream')
    thin = '_thin' if thickness_mm == 50 else ''
    return ROOT / 'muon_analysis' / f'muon_tracks_plus_{material}{thin}_seed_{stream}.csv'


def advance_to_count_5000(rng: np.random.Generator, total: int) -> None:
    """保持旧全档位脚本的随机数位置，确保原 100 mm 数据逐格可复现。"""
    for count in (1000, 3000):
        for _ in range(REPEATS):
            rng.choice(total, count, replace=False)
            for _ in range(8):
                rng.normal(0, 1, count)


def run(output_dir: Path) -> None:
    edges = imaging_edges(GRID)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = plus_mask(cx, cy)
    rows = []
    source_rows = []
    for material in MATERIALS:
        for thickness in THICKNESSES_MM:
            for stream_index, stream in enumerate(STREAMS):
                data = pd.read_csv(csv_for(material, thickness, stream))
                if len(data) < COUNT:
                    raise ValueError('Fewer complete tracks than requested events')
                source_rows.append({'material': material, 'thickness_mm': thickness,
                                    'stream': stream, 'complete_tracks': len(data)})
                spacing = same_side_layer_spacing_mm(data)
                rng = np.random.default_rng(RANDOM_SEED + 8107 + 1000 * stream_index)
                advance_to_count_5000(rng, len(data))
                for repeat in range(1, REPEATS + 1):
                    subset = data.iloc[rng.choice(len(data), COUNT, replace=False)].copy()
                    normals = {f'{plane}_{axis}': rng.normal(0, 1, COUNT)
                               for plane in ('up1', 'up2', 'down1', 'down2')
                               for axis in ('x', 'y')}
                    for sigma in SIGMAS_MM:
                        measured = measured_hits(subset, sigma, normals)
                        observed = recompute_from_noisy_hits(
                            measured, 0., np.random.default_rng(0))
                        ids, angle_weight = candidate_events(observed, sigma, spacing)
                        z = observed.poca_z_mm.to_numpy()[ids]
                        seed = (RANDOM_SEED + 900000 + 300000 * stream_index
                                + 10000 * COUNT + 100 * repeat
                                + int(round(sigma * 1000)))
                        samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                     seed, BOOTSTRAP_DRAWS)
                        hard, propagated = z_acceptance_weights(z, samples)
                        if sigma == 0 and not np.array_equal(hard, propagated):
                            raise AssertionError('Zero-noise H/P diverged')
                        theta = observed.theta_mrad.to_numpy(float)[ids]
                        floor = theta_noise_floor_mrad(sigma, spacing)
                        corrected = np.sqrt(np.maximum(theta**2 - floor**2, 0.))
                        xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                        for method, gate in (('hard', hard), ('propagated', propagated)):
                            image = xy_image(xy, angle_weight * gate, GRID)
                            rows.append({
                                'material': material, 'thickness_mm': thickness,
                                'stream': stream, 'repeat': repeat,
                                'position_sigma_mm': sigma, 'method': method,
                                'complete_tracks': len(data), 'candidate_count': len(ids),
                                'theta_corrected_median_mrad': float(np.median(corrected)),
                                **score_plus(image, truth, cx, cy),
                            })
                    print(f'Thickness control: {material}, {thickness} mm, '
                          f'{stream}, repeat {repeat}/{REPEATS}', flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != len(MATERIALS) * len(THICKNESSES_MM) * len(STREAMS) * REPEATS * len(SIGMAS_MM) * 2:
        raise AssertionError('Incomplete condition matrix')
    h = frame[frame.method == 'hard'].set_index(KEY).sort_index()
    p = frame[frame.method == 'propagated'].set_index(KEY).sort_index()
    if not h.index.equals(p.index) or frame.duplicated(KEY + ['method']).any():
        raise AssertionError('H/P condition pairing failed')
    paired = h[['complete_tracks', 'candidate_count', 'theta_corrected_median_mrad',
                'iou', 'corner_false_positive_fraction']].rename(columns={
                    'iou': 'hard_iou',
                    'corner_false_positive_fraction': 'hard_corner_fp',
                })
    paired['propagated_iou'] = p.iou
    paired['propagated_corner_fp'] = p.corner_false_positive_fraction
    paired['iou_gain'] = p.iou - h.iou
    paired = paired.reset_index()
    if not np.array_equal(paired.loc[paired.position_sigma_mm == 0, 'iou_gain'],
                          np.zeros(len(paired[paired.position_sigma_mm == 0]))):
        raise AssertionError('Zero-noise score identity failed')
    # 厚目标的选定档位必须逐次等于上一轮保存的铝/铁结果。
    old = pd.read_csv(ROOT / 'muon_analysis/plus_material_control_results/by_repeat.csv')
    for _, current in frame[frame.thickness_mm == 100].iterrows():
        matched = old[(old.material == current.material)
                      & (old.stream == current.stream)
                      & (old.event_count == COUNT)
                      & (old['repeat'] == current['repeat'])
                      & (old.position_sigma_mm == current.position_sigma_mm)
                      & (old.grid == GRID)
                      & (old.method == current.method)]
        if len(matched) != 1 or not np.isclose(matched.iloc[0].iou, current.iou, rtol=0, atol=1e-12) \
                or matched.iloc[0].candidate_count != current.candidate_count:
            raise AssertionError(f'Old 100 mm result not reproduced: {current.to_dict()}')
    summary = paired.groupby(['material', 'thickness_mm', 'stream', 'position_sigma_mm'],
                             as_index=False).agg(
        repeats=('repeat', 'size'), complete_tracks=('complete_tracks', 'first'),
        candidate_count=('candidate_count', 'mean'),
        theta_corrected_median_mrad=('theta_corrected_median_mrad', 'mean'),
        hard_iou=('hard_iou', 'mean'), propagated_iou=('propagated_iou', 'mean'),
        iou_gain=('iou_gain', 'mean'),
        hard_corner_fp=('hard_corner_fp', 'mean'),
        propagated_corner_fp=('propagated_corner_fp', 'mean'))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    pd.DataFrame(source_rows).to_csv(output_dir / 'source_counts.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for material, color in (('al', 'tab:blue'), ('iron', 'tab:red')):
        for stream, marker in (('a', 'o'), ('b', 's')):
            part = summary[(summary.material == material) & (summary.stream == stream)
                           & (summary.position_sigma_mm == 0.30)].sort_values('thickness_mm')
            axes[0].plot(part.thickness_mm, part.theta_corrected_median_mrad,
                         color=color, marker=marker, label=f'{material}, seed {stream}')
            axes[1].plot(part.thickness_mm, part.iou_gain,
                         color=color, marker=marker, label=f'{material}, seed {stream}')
    axes[0].set(ylabel='Median corrected angle [mrad]')
    axes[1].set(ylabel='Mean equal-area IoU gain (P − H)')
    axes[1].axhline(0, color='black', lw=1)
    for ax in axes:
        ax.set_xlabel('Target thickness [mm]')
        ax.set_xticks(THICKNESSES_MM)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Plus target thickness control: N=5000, 32×32, hit σ=0.30 mm')
    fig.savefig(output_dir / 'plus_thickness_control_5000_32.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'PLUS_THICKNESS_CONTROL_PLAN.md',
        'role': 'material-by-thickness descriptive control, not new selector validation',
        'materials': MATERIALS, 'thicknesses_mm': THICKNESSES_MM,
        'streams': STREAMS, 'count': COUNT, 'grid': GRID,
        'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS, 'draws': BOOTSTRAP_DRAWS,
        'old_100_mm_results_reproduced': True,
        'same_muon_event_pairing_assumed': False,
        'truth_only_for_scoring': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/plus_thickness_control_results')
