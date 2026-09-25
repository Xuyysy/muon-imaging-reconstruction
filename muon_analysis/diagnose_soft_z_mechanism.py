"""事后拆分软 PoCA-Z 边界的降权作用与越界回流作用。

详见 SOFT_Z_MECHANISM_PLAN.md。只做开发集机制解释；不把顺序的
IoU 差值误称为物理因果贡献，也不调软门的参数。
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
from run_adaptive_voxel_experiment import RANDOM_SEED, XY_LIMIT_MM, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_soft_z_gate_experiment import (
    COUNTS, GRIDS, REPEATS, ROOT, SCENES, SIGMAS_MM, candidate_events,
    score_image, xy_image, z_acceptance_weights,
)
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

KEYS = ['scenario', 'event_count', 'repeat', 'position_sigma_mm', 'grid']


def decompose_gates(hard: np.ndarray, soft: np.ndarray) -> dict[str, np.ndarray]:
    """返回 H、S、F、R 四类事件系数，逐事件 F=S+R。"""
    hard, soft = np.asarray(hard, float), np.asarray(soft, float)
    if hard.shape != soft.shape or not np.isfinite(soft).all():
        raise ValueError('Unpaired or nonfinite gate weights')
    if not np.isin(hard, [0., 1.]).all() or ((soft < 0) | (soft > 1)).any():
        raise ValueError('Hard gate must be binary and soft gate in [0,1]')
    suppress = hard * soft
    rescue = (1.0 - hard) * soft
    if not np.allclose(suppress + rescue, soft):
        raise AssertionError('Event-level soft-gate decomposition failed')
    return {'hard': hard, 'suppress_only': suppress,
            'full_soft': soft, 'rescue_only': rescue}


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows: list[dict] = []
    representative = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        data = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        truth = mask_fn(cx, cy)  # 真值只交给评分函数。
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 2107 + scene_index * 100)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                         for prefix in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    ids, angle_weight = candidate_events(observed, sigma, spacing)
                    seed = (RANDOM_SEED + 300000 * scene_index + 10000 * count
                            + 100 * repeat + int(round(sigma * 1000)))
                    z_samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                   seed, BOOTSTRAP_DRAWS)
                    hard, soft = z_acceptance_weights(
                        observed.poca_z_mm.to_numpy()[ids], z_samples)
                    gates = decompose_gates(hard, soft)
                    xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    masses = {name: float(np.dot(angle_weight, gate))
                              for name, gate in gates.items()}
                    for bins in GRIDS:
                        images = {name: xy_image(xy, angle_weight * gate, bins)
                                  for name, gate in gates.items()}
                        if not np.allclose(images['full_soft'],
                                           images['suppress_only'] + images['rescue_only'],
                                           rtol=1e-12, atol=1e-10):
                            raise AssertionError('Full soft image != suppression + rescue')
                        for name in ('hard', 'suppress_only', 'full_soft'):
                            scored = score_image(images[name], truth, cx, cy, scene)
                            rows.append({
                                'scenario': scene, 'event_count': count,
                                'repeat': repeat, 'position_sigma_mm': sigma,
                                'grid': bins, 'method': name,
                                'xy_angle_candidates': len(ids),
                                'angle_weighted_image_mass': masses[name],
                                'suppression_mass_loss': masses['hard'] - masses['suppress_only'],
                                'rescue_mass': masses['rescue_only'],
                                **scored,
                            })
                            if (scene == 'u_center' and count == 5000
                                    and repeat == 1 and sigma == .05 and bins == 32):
                                representative[name] = images[name]
                print(f'Soft Z mechanism: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    if frame.duplicated(KEYS + ['method']).any() or len(frame) != (
            len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRIDS) * 3):
        raise AssertionError('Incomplete mechanism-condition grid')
    variants = {name: frame[frame.method == name].set_index(KEYS)
                for name in ('hard', 'suppress_only', 'full_soft')}
    hard, suppress, full = (variants[name] for name in
                            ('hard', 'suppress_only', 'full_soft'))
    paired = pd.DataFrame({
        'iou_hard': hard.iou, 'iou_suppress_only': suppress.iou,
        'iou_full_soft': full.iou,
        'iou_suppression_step': suppress.iou - hard.iou,
        'iou_rescue_step': full.iou - suppress.iou,
        'iou_total_gain': full.iou - hard.iou,
        'global_error_hard_mm': hard.global_error_mm,
        'global_error_suppress_mm': suppress.global_error_mm,
        'global_error_full_mm': full.global_error_mm,
        'hard_mass': hard.angle_weighted_image_mass,
        'suppressed_mass': suppress.angle_weighted_image_mass,
        'full_mass': full.angle_weighted_image_mass,
        'suppression_mass_loss': full.suppression_mass_loss,
        'rescue_mass': full.rescue_mass,
        'min_block_recall_hard': hard.min_block_recall,
        'min_block_recall_suppress': suppress.min_block_recall,
        'min_block_recall_full': full.min_block_recall,
        'gap_false_positive_hard': hard.gap_false_positive_fraction,
        'gap_false_positive_suppress': suppress.gap_false_positive_fraction,
        'gap_false_positive_full': full.gap_false_positive_fraction,
    }).reset_index()
    previous = pd.read_csv(ROOT / 'muon_analysis/soft_z_gate_results/by_repeat.csv')
    for method, current in (('hard', hard), ('soft_z', full)):
        old = previous[previous.method == method].set_index(KEYS).sort_index()
        current = current.sort_index()
        if len(old) != len(current) or not np.allclose(old.iou, current.iou,
                                                     rtol=0, atol=1e-12):
            raise AssertionError(f'Cannot reproduce prior {method} images')
    zero = paired[paired.position_sigma_mm == 0]
    if not np.allclose(zero.iou_suppression_step, 0) or not np.allclose(
            zero.iou_rescue_step, 0):
        raise AssertionError('Zero-noise H=S=F control failed')
    summary = paired.groupby(['scenario', 'event_count', 'position_sigma_mm', 'grid'],
                             as_index=False).agg(
        repeats=('repeat', 'size'),
        hard_iou_mean=('iou_hard', 'mean'),
        suppress_iou_mean=('iou_suppress_only', 'mean'),
        full_iou_mean=('iou_full_soft', 'mean'),
        suppression_iou_step_mean=('iou_suppression_step', 'mean'),
        rescue_iou_step_mean=('iou_rescue_step', 'mean'),
        total_iou_gain_mean=('iou_total_gain', 'mean'),
        suppression_step_wins=('iou_suppression_step',
                               lambda x: int((x > 1e-12).sum())),
        rescue_step_wins=('iou_rescue_step',
                          lambda x: int((x > 1e-12).sum())),
        suppression_mass_loss_mean=('suppression_mass_loss', 'mean'),
        rescue_mass_mean=('rescue_mass', 'mean'),
        hard_mass_mean=('hard_mass', 'mean'))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    focused = summary[(summary.event_count == 5000) & (summary.grid == 32)
                      & (summary.position_sigma_mm > 0)].sort_values(
                          ['scenario', 'position_sigma_mm'])
    xpos = np.arange(len(focused))
    width = .38
    ax.bar(xpos - width / 2, focused.suppression_iou_step_mean, width,
           label='Suppress nominally inside')
    ax.bar(xpos + width / 2, focused.rescue_iou_step_mean, width,
           label='Add nominally outside')
    ax.set_xticks(xpos, [f'{row.scenario}\n{row.position_sigma_mm:g} mm'
                           for row in focused.itertuples()])
    ax.axhline(0, color='black', lw=1)
    ax.set(ylabel='Sequential equal-area IoU difference',
           title='Post-hoc soft-Z mechanism decomposition, N=5000, 32×32 XY')
    ax.legend()
    ax.grid(axis='y', alpha=.25)
    fig.savefig(output_dir / 'mechanism_steps_5000_32.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    vmax = max(float(np.log1p(image).max()) for image in representative.values())
    for ax, name in zip(axes, ('hard', 'suppress_only', 'full_soft')):
        ax.imshow(np.log1p(representative[name]).T, origin='lower',
                  cmap='inferno', vmin=0, vmax=vmax,
                  extent=[-XY_LIMIT_MM, XY_LIMIT_MM, -XY_LIMIT_MM, XY_LIMIT_MM])
        ax.set(title=name, xlabel='PoCA x [mm]', ylabel='PoCA y [mm]')
    fig.savefig(output_dir / 'u_mechanism_images_5000_sigma005.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'SOFT_Z_MECHANISM_PLAN.md', 'analysis_type': 'post-hoc development ablation',
        'source_experiment': 'soft_z_gate_results',
        'scenes': list(SCENES), 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'grids': GRIDS, 'bootstrap_draws': BOOTSTRAP_DRAWS,
        'no_new_thresholds': True, 'truth_used_only_for_evaluation': True,
        'sequence': 'hard -> suppress nominally inside -> full with outside rescue',
        'caution': 'IoU sequential differences are not independent causal shares.',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(focused[['scenario', 'position_sigma_mm', 'hard_iou_mean',
                   'suppress_iou_mean', 'full_iou_mean',
                   'suppression_iou_step_mean', 'rescue_iou_step_mean',
                   'suppression_mass_loss_mean', 'rescue_mass_mean']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/soft_z_mechanism_results')
    run(parser.parse_args().output_dir)
