"""Predeclared two-factor ablation on U/L/double development events only.

See SPLIT_GATE_ABLATION_PLAN.md. The frozen implementation is imported only
for an equality check; this file does not change that independent-test rule.
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

from confidence_refinement import (COARSE_BINS, FINE_BINS, SPLIT_REPEATS,
                                   SPLIT_SEED, _child_weights, _effective_count,
                                   confidence_refined_volume)
from diagnose_split_confidence import (COUNTS, REPEATS, ROOT, SCENES,
                                       SIGMAS_MM, topk_mask)
from run_adaptive_voxel_experiment import RANDOM_SEED, THETA_CAP_MRAD, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import (corrected_events,
    same_side_layer_spacing_mm, theta_noise_floor_mrad)

# This 2x2 design was documented before examining its output. It is not a
# hyperparameter search over the held-out T scene.
VARIANTS = {
    'frozen4_shrink': (4.0, True),
    'gate2_shrink': (2.0, True),
    'gate4_no_shrink': (4.0, False),
    'gate2_no_shrink': (2.0, False),
}


def consistency_at_gate(points_xy: np.ndarray, weights: np.ndarray,
                        minimum_half_neff: float) -> tuple[np.ndarray, np.ndarray]:
    """Recompute frozen positive cosine agreement with only the count gate varied."""
    agreement = np.zeros((COARSE_BINS, COARSE_BINS), dtype=float)
    supported = np.zeros_like(agreement, dtype=int)
    rng = np.random.default_rng(SPLIT_SEED)
    for _ in range(SPLIT_REPEATS):
        order = rng.permutation(len(weights))
        for_first, for_second = order[:len(order) // 2], order[len(order) // 2:]
        child_a = _child_weights(points_xy[for_first], weights[for_first])
        child_b = _child_weights(points_xy[for_second], weights[for_second])
        sq_a = _child_weights(points_xy[for_first], weights[for_first] ** 2)
        sq_b = _child_weights(points_xy[for_second], weights[for_second] ** 2)
        valid = (_effective_count(child_a, sq_a) >= minimum_half_neff) & (
            _effective_count(child_b, sq_b) >= minimum_half_neff)
        total_a, total_b = child_a.sum(axis=2), child_b.sum(axis=2)
        pa = np.divide(child_a, total_a[:, :, None], out=np.zeros_like(child_a),
                       where=total_a[:, :, None] > 0) - .25
        pb = np.divide(child_b, total_b[:, :, None], out=np.zeros_like(child_b),
                       where=total_b[:, :, None] > 0) - .25
        dot = np.sum(pa * pb, axis=2)
        norm = np.sqrt(np.sum(pa ** 2, axis=2) * np.sum(pb ** 2, axis=2))
        cosine = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
        agreement += np.where(valid, np.maximum(cosine, 0), 0)
        supported += valid
    return agreement / SPLIT_REPEATS, supported


def ablated_volume(events: pd.DataFrame, sigma: float, spacing: float,
                   half_gate: float, shrink: bool) -> tuple[np.ndarray, dict]:
    """Same per-z weight-conserving mixture as frozen, with two explicit factors."""
    points = events[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy(dtype=float)
    weights = events['capped_weight'].to_numpy(dtype=float)
    coarse, _ = np.histogramdd(points, bins=imaging_edges(COARSE_BINS), weights=weights)
    child = _child_weights(points[:, :2], weights)
    child_sq = _child_weights(points[:, :2], weights ** 2)
    total_xy = child.sum(axis=2)
    pattern = np.divide(child, total_xy[:, :, None], out=np.zeros_like(child),
                        where=total_xy[:, :, None] > 0)
    consistency, supported = consistency_at_gate(points[:, :2], weights, half_gate)
    neff = _effective_count(child, child_sq)
    theta_noise = theta_noise_floor_mrad(sigma, spacing)
    noise_factor = 1 / (1 + (theta_noise / THETA_CAP_MRAD) ** 2)
    shrink_factor = neff / (neff + 4) if shrink else np.ones_like(neff)
    alpha = consistency * shrink_factor * noise_factor
    base = coarse[:, :, :, None] / 4
    detailed = coarse[:, :, :, None] * pattern[:, :, None, :]
    mixed = (1 - alpha[:, :, None, None]) * base + alpha[:, :, None, None] * detailed
    volume = mixed.reshape(16, 16, 16, 2, 2).transpose(0, 3, 1, 4, 2).reshape(32, 32, 16)
    if not np.allclose(volume.sum(axis=(0, 1)), coarse.sum(axis=(0, 1)),
                       rtol=1e-10, atol=1e-8):
        raise AssertionError('Ablation changed per-z weight')
    return volume, {'supported_xy_cells': int(np.count_nonzero(supported)),
                    'positive_xy_cells': int(np.count_nonzero(consistency)),
                    'active_xy_cells': int(np.count_nonzero(alpha))}


def score(events: pd.DataFrame, sigma: float, spacing: float,
          truth: np.ndarray) -> list[dict]:
    """Score all methods on the *same* selected events; truth is evaluation-only."""
    points = events[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy(dtype=float)
    weights = events['capped_weight'].to_numpy(dtype=float)
    coarse, _ = np.histogramdd(points, bins=imaging_edges(COARSE_BINS), weights=weights)
    coarse_image = np.repeat(np.repeat(coarse.sum(axis=2), 2, axis=0), 2, axis=1) / 4
    coarse_mask = topk_mask(coarse_image, int(truth.sum()))
    total = float(coarse_image.sum())
    results = []
    for name, (gate, shrink) in VARIANTS.items():
        volume, info = ablated_volume(events, sigma, spacing, gate, shrink)
        if name == 'frozen4_shrink':
            frozen, _ = confidence_refined_volume(events, sigma, spacing)
            if not np.allclose(volume, frozen, rtol=1e-13, atol=1e-12):
                raise AssertionError('Factorial frozen arm differs from pre-registered code')
        image = volume.sum(axis=2)
        mask = topk_mask(image, int(truth.sum()))
        intersection = int(np.logical_and(mask, truth).sum())
        union = int(np.logical_or(mask, truth).sum())
        results.append({'method': name, 'half_gate': gate, 'shrink': shrink,
                        'selected_events': len(events), **info,
                        'relative_image_l1': float(np.abs(image - coarse_image).sum() / total)
                            if total else 0.0,
                        'topk_changed_pixels': int(np.logical_xor(mask, coarse_mask).sum()),
                        'iou': intersection / union if union else np.nan})
    return results


def run(output_dir: Path) -> None:
    edges = imaging_edges(FINE_BINS)
    cx, cy = [(edge[:-1] + edge[1:]) / 2 for edge in edges[:2]]
    rows = []
    input_info = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        path = ROOT / 'muon_analysis' / csv_name
        data = pd.read_csv(path)
        input_info[scene] = {'csv': str(path), 'rows': len(data)}
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
                    for item in score(events, sigma, spacing, truth):
                        rows.append({'scenario': scene, 'event_count': count,
                                     'repeat': repeat, 'position_sigma_mm': sigma,
                                     **item})
                print(f'gate ablation: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(VARIANTS):
        raise AssertionError('Missing ablation condition')
    key = ['scenario', 'event_count', 'repeat', 'position_sigma_mm', 'method']
    if frame.duplicated(key).any():
        raise AssertionError('Duplicate ablation condition')
    reference = pd.read_csv(ROOT / 'muon_analysis/split_confidence_diagnosis_results/by_repeat.csv')
    merged = frame.merge(reference[key[:-1] + ['coarse_iou', 'confidence_iou']],
                         on=key[:-1], validate='many_to_one')
    frozen = merged[merged.method == 'frozen4_shrink']
    if not np.allclose(frozen.iou, frozen.confidence_iou, rtol=1e-12, atol=1e-12):
        raise AssertionError('Paired frozen IoU differs from previous diagnosis')
    merged['iou_gain_vs_coarse'] = merged.iou - merged.coarse_iou
    merged['win'] = (merged.iou_gain_vs_coarse > 1e-12).astype(int)
    merged['loss'] = (merged.iou_gain_vs_coarse < -1e-12).astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary = merged.groupby(['scenario', 'event_count', 'position_sigma_mm', 'method'],
                             as_index=False).agg(
        repeats=('repeat', 'size'), iou_mean=('iou', 'mean'),
        iou_gain_mean=('iou_gain_vs_coarse', 'mean'),
        iou_gain_std=('iou_gain_vs_coarse', 'std'),
        wins=('win', 'sum'), losses=('loss', 'sum'),
        changed_pixels_mean=('topk_changed_pixels', 'mean'),
        image_l1_mean=('relative_image_l1', 'mean'),
        supported_xy_mean=('supported_xy_cells', 'mean'),
        positive_xy_mean=('positive_xy_cells', 'mean'))
    summary['ties'] = summary.repeats - summary.wins - summary.losses
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'role': 'development-set diagnostic, not independent validation',
        'plan': 'SPLIT_GATE_ABLATION_PLAN.md', 'scenes': input_info,
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'seed_base': RANDOM_SEED + 1523, 'variants': VARIANTS,
        't_scene_used': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(summary, output_dir / 'gate_ablation_5000.png')


def plot(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True, constrained_layout=True)
    for row, scene in enumerate(SCENES):
        for method in VARIANTS:
            part = summary[(summary.scenario == scene) & (summary.event_count == 5000) &
                           (summary.method == method)].sort_values('position_sigma_mm')
            axes[row, 0].plot(part.position_sigma_mm, part.iou_gain_mean, 'o-',
                              label=method)
            axes[row, 1].plot(part.position_sigma_mm, part.image_l1_mean, 'o-',
                              label=method)
        axes[row, 0].axhline(0, color='k', linewidth=.7)
        axes[row, 0].set_ylabel(f'{scene}\nIoU gain vs coarse16')
        axes[row, 1].set_ylabel(f'{scene}\nRelative image L1 vs coarse16')
        for ax in axes[row]:
            ax.grid(alpha=.25)
    for ax in axes[-1]:
        ax.set_xlabel('Hit-position sigma [mm]')
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/split_gate_ablation_results')
    run(parser.parse_args().output_dir)
