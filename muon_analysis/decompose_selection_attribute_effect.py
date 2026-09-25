"""Separate selection-set and event-attribute changes on development scenes.

See SELECTION_ATTRIBUTE_DECOMPOSITION_PLAN.md. This is a paired descriptive
factorial decomposition, not an adaptive reconstruction or a causal percentage.
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

from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM, topk_mask
from run_adaptive_voxel_experiment import RANDOM_SEED, THETA_CAP_MRAD, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm, theta_noise_floor_mrad)

METHODS = {'baseline': ('ideal', 'ideal'),
           'selection_only': ('noisy', 'ideal'),
           'attribute_only': ('ideal', 'noisy'),
           'full': ('noisy', 'noisy')}


def event_attributes(recomputed: pd.DataFrame, sigma: float,
                     spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Return paired XY and capped corrected-theta weight for *all* input rows."""
    xy = recomputed[['poca_x_mm', 'poca_y_mm']].to_numpy(dtype=float)
    theta = recomputed.theta_mrad.to_numpy(dtype=float)
    floor = theta_noise_floor_mrad(sigma, spacing)
    corrected_squared = np.maximum(theta ** 2 - floor ** 2, 0.0)
    weights = np.minimum(corrected_squared, THETA_CAP_MRAD ** 2)
    if not np.isfinite(xy).all() or not np.isfinite(weights).all():
        raise ValueError('Nonfinite event attributes')
    return xy, weights


def xy_image(mask: np.ndarray, xy: np.ndarray, weights: np.ndarray,
             edges: list[np.ndarray]) -> np.ndarray:
    """Make one counterfactual without filtering the chosen membership again."""
    if len(mask) != len(xy) or len(mask) != len(weights):
        raise ValueError('Input event arrays are not paired')
    image, _, _ = np.histogram2d(xy[mask, 0], xy[mask, 1],
                                 bins=edges[:2], weights=weights[mask])
    return image


def counterfactual_images(reference: pd.DataFrame, noisy: pd.DataFrame,
                          reference_selected: pd.DataFrame,
                          noisy_selected: pd.DataFrame, sigma: float,
                          spacing: float, edges: list[np.ndarray]) -> tuple[dict, dict]:
    """Build the exact 2x2 membership x attributes factorial images."""
    if len(reference) != len(noisy):
        raise ValueError('Reference/noisy event count differs')
    expected_index = pd.RangeIndex(len(reference))
    if not reference.index.equals(expected_index) or not noisy.index.equals(expected_index):
        raise ValueError('Recomputed rows must retain original input order')
    ideal_mask = np.zeros(len(reference), dtype=bool)
    noisy_mask = np.zeros(len(reference), dtype=bool)
    ideal_mask[reference_selected.index.to_numpy()] = True
    noisy_mask[noisy_selected.index.to_numpy()] = True
    ideal_xy, ideal_weight = event_attributes(reference, 0.0, spacing)
    noisy_xy, noisy_weight = event_attributes(noisy, sigma, spacing)
    masks = {'ideal': ideal_mask, 'noisy': noisy_mask}
    attrs = {'ideal': (ideal_xy, ideal_weight),
             'noisy': (noisy_xy, noisy_weight)}
    images = {name: xy_image(masks[membership], *attrs[attribute], edges)
              for name, (membership, attribute) in METHODS.items()}
    counts = {'baseline_selected': int(ideal_mask.sum()),
              'noisy_selected': int(noisy_mask.sum()),
              'kept': int(np.count_nonzero(ideal_mask & noisy_mask)),
              'lost': int(np.count_nonzero(ideal_mask & ~noisy_mask)),
              'gained': int(np.count_nonzero(~ideal_mask & noisy_mask))}
    return images, counts


def score_images(images: dict, truth: np.ndarray) -> dict:
    """Use truth only for four *post hoc* Top-k IoUs, never for images."""
    base = images['baseline']
    membership = images['selection_only']
    attributes = images['attribute_only']
    full = images['full']
    interaction = full - membership - attributes + base
    if not np.allclose(base + (membership - base) + (attributes - base) + interaction,
                       full, rtol=1e-12, atol=1e-10):
        raise AssertionError('Factorial vector identity failed')
    baseline_mass = float(base.sum())
    if baseline_mass <= 0:
        raise ValueError('Baseline image has no selected scattering weight')
    diffs = {'selection': membership - base, 'attribute': attributes - base,
             'interaction': interaction, 'full': full - base}
    result = {f'{key}_raw_l1_over_baseline_mass':
              float(np.abs(value).sum() / baseline_mass) for key, value in diffs.items()}
    normalized = {key: image / image.sum() if image.sum() > 0 else
                  np.full_like(image, np.nan) for key, image in images.items()}
    result.update({f'{key}_shape_l1':
                   float(np.abs(normalized[name] - normalized['baseline']).sum())
                   for key, name in [('selection', 'selection_only'),
                                     ('attribute', 'attribute_only'), ('full', 'full')]})
    for name, image in images.items():
        if image.sum() > 0:
            mask = topk_mask(image, int(truth.sum()))
            intersection = np.logical_and(mask, truth).sum()
            union = np.logical_or(mask, truth).sum()
            result[f'{name}_iou'] = float(intersection / union)
        else:
            result[f'{name}_iou'] = np.nan
        result[f'{name}_mass'] = float(image.sum())
    return result


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    centers = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows = []
    inputs = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        path = ROOT / 'muon_analysis' / csv_name
        data = pd.read_csv(path)
        if len(data) < max(COUNTS):
            raise ValueError(f'Not enough complete events in {scene}')
        inputs[scene] = {'path': str(path), 'rows': len(data)}
        truth = mask_fn(*centers)
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                         for prefix in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                reference = recompute_from_noisy_hits(
                    measured_hits(subset, 0.0, draws), 0.0, np.random.default_rng(0))
                reference_selected = corrected_events(reference, 0.0, spacing)
                for sigma in SIGMAS_MM:
                    noisy = recompute_from_noisy_hits(
                        measured_hits(subset, sigma, draws), 0.0, np.random.default_rng(0))
                    noisy_selected = corrected_events(noisy, sigma, spacing)
                    images, counts = counterfactual_images(reference, noisy,
                        reference_selected, noisy_selected, sigma, spacing, edges)
                    metrics = score_images(images, truth)
                    if sigma == 0 and (not np.array_equal(images['baseline'], images['full'])
                                       or counts['lost'] or counts['gained']):
                        raise AssertionError('Sigma=0 paired control changed')
                    rows.append({'scenario': scene, 'event_count': count,
                                 'repeat': repeat, 'position_sigma_mm': sigma,
                                 **counts, **metrics})
                print(f'selection/attribute: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    keys = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
    if len(frame) != len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM):
        raise AssertionError('Missing counterfactual condition')
    if frame.duplicated(keys).any():
        raise AssertionError('Duplicate condition')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    metrics = [column for column in frame if column not in keys]
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(repeats=('repeat', 'size'),
        **{f'{metric}_mean': (metric, 'mean') for metric in metrics})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'role': 'paired development-scene counterfactual description only',
        'plan': 'SELECTION_ATTRIBUTE_DECOMPOSITION_PLAN.md',
        'inputs': inputs, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'seed_base': RANDOM_SEED + 1523,
        'images': METHODS, 't_scene_used': False,
        'truth_usage': 'Top-k IoU only, not counterfactual image construction',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(summary, output_dir / 'selection_attribute_5000.png')


def plot(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True, constrained_layout=True)
    colors = {'selection': '#1f77b4', 'attribute': '#ff7f0e', 'full': '#2ca02c',
              'interaction': '#d62728'}
    for row, scene in enumerate(SCENES):
        part = summary[(summary.scenario == scene) &
                       (summary.event_count == 5000)].sort_values('position_sigma_mm')
        for name in ('selection', 'attribute', 'interaction', 'full'):
            axes[row, 0].plot(part.position_sigma_mm,
                part[f'{name}_raw_l1_over_baseline_mass_mean'], 'o-',
                label=name, color=colors[name])
        for name in ('selection', 'attribute', 'full'):
            axes[row, 1].plot(part.position_sigma_mm,
                part[f'{name}_shape_l1_mean'], 'o-',
                label=name, color=colors[name])
        axes[row, 0].set_ylabel(f'{scene}\nRaw image L1 / baseline mass')
        axes[row, 1].set_ylabel(f'{scene}\nNormalized image shape L1')
        for ax in axes[row]:
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Hit-position sigma [mm]')
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/selection_attribute_results')
    run(parser.parse_args().output_dir)
