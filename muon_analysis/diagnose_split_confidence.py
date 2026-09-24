"""Diagnose the frozen split-half rule on *development* scenes only.

U, L and double-block events are allowed to inform a future revision. T-shaped
held-out events are intentionally absent. This script never changes the
pre-registered confidence_refinement.py and does not sweep its parameters.
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

from confidence_refinement import (
    MIN_HALF_EFFECTIVE_EVENTS, _child_weights, _effective_count,
    confidence_refined_volume, split_half_consistency,
)
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges
from run_baseline_experiment import u_target_mask
from run_continuous_event_quality_experiment import evaluate
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_double_block_validation import double_block_mask
from run_event_uncertainty_experiment import measured_hits
from run_heldout_l_shape_validation import l_target_mask
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    'u_center': ('muon_tracks.csv', u_target_mask),
    'l_shape': ('muon_tracks_l_shape.csv', l_target_mask),
    'double_block': ('muon_tracks_double_block.csv', double_block_mask),
}
COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5


def topk_mask(image: np.ndarray, k: int) -> np.ndarray:
    """Existing equal-area evaluation mask; never supplied to reconstruction."""
    mask = np.zeros_like(image, dtype=bool)
    mask.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    return mask


def diagnose_events(events: pd.DataFrame, sigma: float, spacing: float,
                    truth: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> dict:
    """Decompose support, consistency, image change and Top-k stability."""
    points_xy = events[['poca_x_mm', 'poca_y_mm']].to_numpy(dtype=float)
    weights = events['capped_weight'].to_numpy(dtype=float)
    child = _child_weights(points_xy, weights)
    child_squared = _child_weights(points_xy, weights ** 2)
    n_eff = _effective_count(child, child_squared)
    weight_xy = child.sum(axis=2)
    consistency, supported_splits = split_half_consistency(points_xy, weights)
    coarse_volume, _, _, _ = adaptive_volume(events, 0.0)
    refined_volume, info = confidence_refined_volume(events, sigma, spacing)
    coarse_image = coarse_volume.sum(axis=2)
    refined_image = refined_volume.sum(axis=2)
    if not np.isclose(coarse_image.sum(), refined_image.sum()):
        raise AssertionError('Total reconstruction weight changed')
    k = int(truth.sum())
    coarse_top = topk_mask(coarse_image, k)
    refined_top = topk_mask(refined_image, k)
    active = consistency > 0
    total = float(weight_xy.sum())
    # Recover alpha from the output's published diagnostic formula without
    # changing the frozen implementation. Weighted alpha answers whether the
    # few active cells carry meaningful scattering mass, not just cell count.
    from run_adaptive_voxel_experiment import THETA_CAP_MRAD
    from run_uncertainty_aware_angle_experiment import theta_noise_floor_mrad
    theta_noise = theta_noise_floor_mrad(sigma, spacing)
    alpha = consistency * n_eff / (n_eff + 4.0) / (1.0 + (theta_noise / THETA_CAP_MRAD) ** 2)
    return {
        'selected_events': len(events),
        'occupied_xy_cells': int(np.count_nonzero(weight_xy)),
        'full_neff_ge_8_cells': int(np.count_nonzero(n_eff >= 2 * MIN_HALF_EFFECTIVE_EVENTS)),
        'split_supported_xy_cells': int(np.count_nonzero(supported_splits)),
        'positive_consistency_xy_cells': int(np.count_nonzero(active)),
        'active_refined_xy_cells': info['active_refined_xy_cells'],
        'mean_alpha_nonzero': info['mean_alpha_nonzero'],
        'active_weight_fraction': float(weight_xy[active].sum() / total) if total else 0.0,
        'effective_detail_fraction': float(np.sum(weight_xy * alpha) / total) if total else 0.0,
        'relative_image_l1': float(np.abs(refined_image - coarse_image).sum() / total)
            if total else 0.0,
        'topk_changed_pixels': int(np.logical_xor(coarse_top, refined_top).sum()),
        'coarse_iou': evaluate(coarse_image, truth, cx, cy)['iou'],
        'confidence_iou': evaluate(refined_image, truth, cx, cy)['iou'],
    }


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows: list[dict] = []
    inputs = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        path = ROOT / 'muon_analysis' / csv_name
        data = pd.read_csv(path)
        if len(data) < max(COUNTS):
            raise ValueError(f'Insufficient complete events: {scene}')
        inputs[scene] = {'path': str(path), 'rows': len(data)}
        truth = mask_fn(cx, cy)
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                         for prefix in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(measured, 0.0,
                                                         np.random.default_rng(0))
                    events = corrected_events(observed, sigma, spacing)
                    result = diagnose_events(events, sigma, spacing, truth, cx, cy)
                    rows.append({'scenario': scene, 'event_count': count,
                                 'repeat': repeat, 'position_sigma_mm': sigma,
                                 **result, 'iou_gain_vs_coarse':
                                     result['confidence_iou'] - result['coarse_iou']})
                print(f'development diagnosis: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    key = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
    if len(frame) != len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM):
        raise AssertionError('Missing diagnostic conditions')
    if frame.duplicated(key).any():
        raise AssertionError('Duplicate diagnostic condition')
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    metrics = ('selected_events', 'occupied_xy_cells', 'full_neff_ge_8_cells',
               'split_supported_xy_cells', 'positive_consistency_xy_cells',
               'mean_alpha_nonzero', 'active_weight_fraction',
               'effective_detail_fraction', 'relative_image_l1',
               'topk_changed_pixels', 'iou_gain_vs_coarse')
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in metrics},
        **{f'{name}_std': (name, 'std') for name in
           ('split_supported_xy_cells', 'positive_consistency_xy_cells',
            'relative_image_l1', 'topk_changed_pixels', 'iou_gain_vs_coarse')})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'scene_role': 'U/L/double development diagnosis only; T held-out data excluded',
        'inputs': inputs, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'seed_base': RANDOM_SEED + 1523,
        'rule_status': 'frozen implementation, no threshold scan',
        'truth_usage': 'Top-k and IoU evaluation only; no reconstruction decisions',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for scene in SCENES:
        part = summary[(summary.scenario == scene) &
                       (summary.event_count == 5000)].sort_values('position_sigma_mm')
        for ax, metric in zip(axes, ('split_supported_xy_cells',
                                     'positive_consistency_xy_cells',
                                     'relative_image_l1')):
            ax.plot(part.position_sigma_mm, part[f'{metric}_mean'], 'o-', label=scene)
    for ax, label in zip(axes, ('XY cells passing split-half count gate',
                                'XY cells with positive consistency',
                                'Relative image L1 change vs coarse16')):
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / 'gate_diagnosis_5000.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/split_confidence_diagnosis_results')
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == '__main__':
    main()
