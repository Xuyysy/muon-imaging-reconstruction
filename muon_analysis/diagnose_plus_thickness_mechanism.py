"""已见十字厚度数据的 H→S→P 顺序机制诊断；不训练新门控规则。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from diagnose_soft_z_mechanism import decompose_gates
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_plus_iron_fallback_validation import plus_mask, score_plus
from run_plus_thickness_control import (
    COUNT, GRID, KEY, MATERIALS, REPEATS, ROOT, SIGMAS_MM, STREAMS,
    THICKNESSES_MM, advance_to_count_5000, csv_for,
)
from run_soft_z_gate_experiment import candidate_events, xy_image, z_acceptance_weights
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm


def gate_features(hard: np.ndarray, soft: np.ndarray, weight: np.ndarray,
                  z_mm: np.ndarray, input_count: int) -> dict[str, float]:
    """全部来自观测事件及预定误差传播；无目标真值。"""
    if not (len(hard) == len(soft) == len(weight) == len(z_mm)):
        raise ValueError('Unaligned candidate arrays')
    if len(hard) == 0:
        raise ValueError('No candidate events')
    inside = hard == 1
    outside = ~inside
    hard_mass = float(np.dot(weight, hard))
    suppressed_mass = float(np.dot(weight, hard * soft))
    rescued_mass = float(np.dot(weight, (1 - hard) * soft))
    full_mass = float(np.dot(weight, soft))
    return {
        'candidate_count': len(hard),
        'candidate_fraction': len(hard) / input_count,
        'hard_z_count': int(inside.sum()),
        'outside_z_count': int(outside.sum()),
        'abs_z_median_mm': float(np.median(np.abs(z_mm))),
        'abs_z_p90_mm': float(np.quantile(np.abs(z_mm), .9)),
        'rescued_event_count': int(np.sum(outside & (soft > 0))),
        'suppressed_event_count': int(np.sum(inside & (soft < 1))),
        'fractional_gate_count': int(np.sum((soft > 0) & (soft < 1))),
        'hard_mass': hard_mass,
        'suppression_mass_loss': hard_mass - suppressed_mass,
        'rescue_mass': rescued_mass,
        'soft_mass': full_mass,
        'suppression_fraction_of_hard_mass':
            (hard_mass - suppressed_mass) / hard_mass if hard_mass else np.nan,
        'rescue_fraction_of_soft_mass': rescued_mass / full_mass if full_mass else np.nan,
    }


def run(output_dir: Path) -> None:
    edges = imaging_edges(GRID)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = plus_mask(cx, cy)
    # 仅在最后评分阶段使用真值和四角空白区，绝不反馈给 h/p/w。
    mesh_x, mesh_y = np.meshgrid(cx, cy, indexing='ij')
    corners = ((np.abs(mesh_x) > 15.) & (np.abs(mesh_x) < 60.)
               & (np.abs(mesh_y) > 15.) & (np.abs(mesh_y) < 60.))
    saved = pd.read_csv(ROOT / 'muon_analysis/plus_thickness_control_results/by_repeat.csv')
    saved = saved.set_index(KEY + ['method']).sort_index()
    rows = []
    for material in MATERIALS:
        for thickness in THICKNESSES_MM:
            for stream_index, stream in enumerate(STREAMS):
                data = pd.read_csv(csv_for(material, thickness, stream))
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
                        ids, weight = candidate_events(observed, sigma, spacing)
                        z = observed.poca_z_mm.to_numpy()[ids]
                        seed = (RANDOM_SEED + 900000 + 300000 * stream_index
                                + 10000 * COUNT + 100 * repeat
                                + int(round(sigma * 1000)))
                        sampled = estimate_z_samples(measured.iloc[ids], sigma,
                                                     seed, BOOTSTRAP_DRAWS)
                        hard, soft = z_acceptance_weights(z, sampled)
                        gates = decompose_gates(hard, soft)
                        features = gate_features(hard, soft, weight, z, COUNT)
                        xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                        images = {method: xy_image(xy, weight * gate, GRID)
                                  for method, gate in gates.items()}
                        if not np.allclose(images['full_soft'],
                                           images['suppress_only'] + images['rescue_only'],
                                           rtol=1e-12, atol=1e-10):
                            raise AssertionError('P != S + R image')
                        rescue_image = images['rescue_only']
                        lost_image = images['hard'] - images['suppress_only']
                        if np.min(lost_image) < -1e-10:
                            raise AssertionError('Suppression image has negative lost mass')
                        rescue_total = float(rescue_image.sum())
                        lost_total = float(lost_image.sum())
                        # 真值区域质量占比只帮助解释已见数据，不可作为可部署的风险特征。
                        spatial_scores = {
                            'rescue_mass_in_truth_fraction':
                                float(rescue_image[truth].sum() / rescue_total)
                                if rescue_total else np.nan,
                            'rescue_mass_in_corners_fraction':
                                float(rescue_image[corners].sum() / rescue_total)
                                if rescue_total else np.nan,
                            'lost_mass_in_truth_fraction':
                                float(lost_image[truth].sum() / lost_total)
                                if lost_total else np.nan,
                        }
                        scores = {method: score_plus(images[method], truth, cx, cy)
                                  for method in ('hard', 'suppress_only', 'full_soft')}
                        index = (material, thickness, stream, repeat, sigma)
                        for current_method, saved_method in (('hard', 'hard'),
                                                             ('full_soft', 'propagated')):
                            old = saved.loc[index + (saved_method,)]
                            if (int(old.candidate_count) != features['candidate_count']
                                    or not np.isclose(old.iou, scores[current_method]['iou'],
                                                      rtol=0, atol=1e-12)):
                                raise AssertionError(f'Cannot reproduce saved {index} {saved_method}')
                        h, s, p = (scores[name] for name in
                                   ('hard', 'suppress_only', 'full_soft'))
                        if sigma == 0 and not (h['iou'] == s['iou'] == p['iou']
                                               and features['rescue_mass'] == 0
                                               and features['suppression_mass_loss'] == 0):
                            raise AssertionError('Zero-noise H=S=P failed')
                        rows.append({
                            'material': material, 'thickness_mm': thickness,
                            'stream': stream, 'repeat': repeat,
                            'position_sigma_mm': sigma, **features, **spatial_scores,
                            'hard_iou': h['iou'], 'suppress_only_iou': s['iou'],
                            'propagated_iou': p['iou'],
                            'suppression_step_iou': s['iou'] - h['iou'],
                            'rescue_step_iou': p['iou'] - s['iou'],
                            'total_iou_gain': p['iou'] - h['iou'],
                            'hard_corner_fp': h['corner_false_positive_fraction'],
                            'suppress_corner_fp': s['corner_false_positive_fraction'],
                            'propagated_corner_fp': p['corner_false_positive_fraction'],
                        })
                    print(f'Plus mechanism: {material}, {thickness} mm, '
                          f'{stream}, repeat {repeat}/{REPEATS}', flush=True)
    frame = pd.DataFrame(rows)
    expected = len(MATERIALS) * len(THICKNESSES_MM) * len(STREAMS) * REPEATS * len(SIGMAS_MM)
    if len(frame) != expected or frame.duplicated(KEY).any():
        raise AssertionError('Incomplete condition matrix')
    if not np.allclose(frame.suppression_step_iou + frame.rescue_step_iou,
                       frame.total_iou_gain, rtol=0, atol=1e-12):
        raise AssertionError('Sequential IoU arithmetic failed')
    summary = frame.groupby(['material', 'thickness_mm', 'stream',
                             'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{name: (name, 'mean') for name in frame.columns
           if name not in KEY})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True,
                             sharex=True)
    labels = [f'{material} {thickness}mm {stream}'
              for material in MATERIALS for thickness in THICKNESSES_MM
              for stream in STREAMS]
    for ax, sigma in zip(axes, (.05, .30)):
        focused = summary[summary.position_sigma_mm == sigma].set_index(
            ['material', 'thickness_mm', 'stream'])
        records = [focused.loc[(material, thickness, stream)]
                   for material in MATERIALS for thickness in THICKNESSES_MM
                   for stream in STREAMS]
        x = np.arange(len(records))
        ax.bar(x - .19, [r.suppression_step_iou for r in records], .36,
               label='H→S: downweight nominally inside')
        ax.bar(x + .19, [r.rescue_step_iou for r in records], .36,
               label='S→P: add nominally outside')
        ax.axhline(0, color='black', lw=1)
        ax.set_ylabel(f'IoU step, σ={sigma:g} mm')
        ax.grid(axis='y', alpha=.25)
    axes[0].legend(fontsize=8)
    axes[-1].set_xticks(np.arange(len(labels)), labels, rotation=25, ha='right')
    fig.suptitle('Post-hoc plus target soft-Z decomposition: N=5000, 32×32')
    fig.savefig(output_dir / 'plus_thickness_mechanism_steps.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'PLUS_THICKNESS_MECHANISM_PLAN.md',
        'role': 'post-hoc explanation of already observed H/P contrast',
        'input_count': COUNT, 'grid': GRID, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'draws': BOOTSTRAP_DRAWS,
        'prior_h_p_results_reproduced': True,
        'truth_only_for_scoring': True,
        'event_level_physical_pairing_across_conditions': False,
        'sequence': 'H -> S -> P, P = S + R in image intensity',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary[summary.position_sigma_mm == .30][
        ['material', 'thickness_mm', 'stream', 'candidate_count',
         'hard_z_count', 'outside_z_count', 'rescued_event_count',
         'suppressed_event_count', 'suppression_mass_loss', 'rescue_mass',
         'suppression_step_iou', 'rescue_step_iou', 'total_iou_gain']
    ].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/plus_thickness_mechanism_results')
