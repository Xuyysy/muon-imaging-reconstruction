"""Explain paired event entry/exit by fixed angle and POCA-geometry gates.

The gate rules and paired samples are frozen in SELECTION_GATE_TRANSITIONS_PLAN.md.
No target mask or truth labels are used by this diagnostic.
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

from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM
from run_adaptive_voxel_experiment import (RANDOM_SEED, THETA_MIN_MRAD,
                                           XY_LIMIT_MM, Z_LIMIT_MM)
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm, theta_noise_floor_mrad)


def gate_masks(events: pd.DataFrame, sigma: float, spacing: float) -> dict[str, np.ndarray]:
    """Reproduce the existing corrected-theta and 3D POCA acceptance gates.

    Keep x/y and z components separate so that a geometry failure can be
    diagnosed, while the actual reconstruction still uses their conjunction.
    Comparisons with NaN are false, just as in the existing prepare_events.
    """
    theta = events.theta_mrad.to_numpy(dtype=float)
    floor = theta_noise_floor_mrad(sigma, spacing)
    corrected = np.sqrt(np.maximum(theta ** 2 - floor ** 2, 0.0))
    angle = corrected >= THETA_MIN_MRAD
    xy = ((events.poca_x_mm.abs() <= XY_LIMIT_MM)
          & (events.poca_y_mm.abs() <= XY_LIMIT_MM)).to_numpy(dtype=bool)
    z = (events.poca_z_mm.abs() <= Z_LIMIT_MM).to_numpy(dtype=bool)
    return {'angle': angle, 'xy': xy, 'z': z, 'geometry': xy & z,
            'selected': angle & xy & z}


def _failure_partition(changed: np.ndarray, gates: dict[str, np.ndarray],
                       prefix: str) -> dict[str, int]:
    """Assign each selected-set change to one exclusive failed-gate state."""
    fail_angle = ~gates['angle']
    fail_geometry = ~gates['geometry']
    pieces = {
        f'{prefix}_angle_only': changed & fail_angle & ~fail_geometry,
        f'{prefix}_geometry_only': changed & ~fail_angle & fail_geometry,
        f'{prefix}_both': changed & fail_angle & fail_geometry,
    }
    if np.any(changed & ~np.logical_or.reduce(list(pieces.values()))):
        raise AssertionError('Changed selection has no failed-gate category')
    if sum(int(mask.sum()) for mask in pieces.values()) != int(changed.sum()):
        raise AssertionError('Failed-gate categories overlap')
    geometry_failed = changed & fail_geometry
    pieces.update({
        f'{prefix}_geometry_xy_only': geometry_failed & ~gates['xy'] & gates['z'],
        f'{prefix}_geometry_z_only': geometry_failed & gates['xy'] & ~gates['z'],
        f'{prefix}_geometry_xy_and_z': geometry_failed & ~gates['xy'] & ~gates['z'],
    })
    if sum(int(pieces[f'{prefix}_geometry_{kind}'].sum())
           for kind in ('xy_only', 'z_only', 'xy_and_z')) != int(geometry_failed.sum()):
        raise AssertionError('Geometry failure categories do not partition')
    return {name: int(mask.sum()) for name, mask in pieces.items()}


def classify_transitions(reference: pd.DataFrame, noisy: pd.DataFrame,
                         sigma: float, spacing: float) -> dict[str, int | float]:
    """Count changed selection membership and diagnose the failed gate.

    The reference/noisy arrays must preserve the exact same original row order.
    Lost events are categorized by the noisy failed gates; gained events by
    the reference failed gates. We make no causal percentage attribution.
    """
    if len(reference) != len(noisy):
        raise ValueError('Unpaired event arrays')
    expected = pd.RangeIndex(len(reference))
    if not reference.index.equals(expected) or not noisy.index.equals(expected):
        raise ValueError('Event row order/index was not retained')
    ideal = gate_masks(reference, 0.0, spacing)
    measured = gate_masks(noisy, sigma, spacing)
    selected0, selected1 = ideal['selected'], measured['selected']
    lost = selected0 & ~selected1
    gained = ~selected0 & selected1
    kept = selected0 & selected1
    baseline_count = int(selected0.sum())
    output = {
        'baseline_selected': baseline_count,
        'noisy_selected': int(selected1.sum()),
        'kept': int(kept.sum()),
        'lost': int(lost.sum()),
        'gained': int(gained.sum()),
        'retained_fraction': float(kept.sum() / baseline_count) if baseline_count else np.nan,
    }
    output.update(_failure_partition(lost, measured, 'lost'))
    output.update(_failure_partition(gained, ideal, 'gained'))
    # This checks the algebra *and* exact compatibility with the production
    # selection function, including the noise-floor correction and all bounds.
    for state, data, selected, state_sigma in (
        ('ideal', reference, selected0, 0.0), ('noisy', noisy, selected1, sigma)):
        actual = corrected_events(data, state_sigma, spacing).index.to_numpy()
        if not np.array_equal(np.flatnonzero(selected), actual):
            raise AssertionError(f'{state} gate mask differs from corrected_events')
    if output['lost'] != sum(output[f'lost_{kind}'] for kind in
                             ('angle_only', 'geometry_only', 'both')):
        raise AssertionError('Lost event partition mismatch')
    if output['gained'] != sum(output[f'gained_{kind}'] for kind in
                               ('angle_only', 'geometry_only', 'both')):
        raise AssertionError('Gained event partition mismatch')
    return output


def run(output_dir: Path) -> None:
    previous_path = ROOT / 'muon_analysis/selection_attribute_results/by_repeat.csv'
    previous = pd.read_csv(previous_path)
    rows = []
    inputs = {}
    for scene_index, (scene, (csv_name, _)) in enumerate(SCENES.items()):
        path = ROOT / 'muon_analysis' / csv_name
        data = pd.read_csv(path)
        if len(data) < max(COUNTS):
            raise ValueError(f'Not enough complete events in {scene}')
        inputs[scene] = {'path': str(path), 'rows': len(data)}
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
                for sigma in SIGMAS_MM:
                    noisy = recompute_from_noisy_hits(
                        measured_hits(subset, sigma, draws), 0.0, np.random.default_rng(0))
                    metrics = classify_transitions(reference, noisy, sigma, spacing)
                    if sigma == 0 and (metrics['lost'] or metrics['gained']):
                        raise AssertionError('Sigma=0 control has selection changes')
                    rows.append({'scenario': scene, 'event_count': count,
                                 'repeat': repeat, 'position_sigma_mm': sigma, **metrics})
                print(f'gate transitions: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    keys = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
    if len(frame) != len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM):
        raise AssertionError('Incomplete condition grid')
    if frame.duplicated(keys).any():
        raise AssertionError('Duplicate condition keys')
    paired = frame.merge(previous[keys + ['baseline_selected', 'noisy_selected',
                                           'kept', 'lost', 'gained']], on=keys,
                         suffixes=('', '_previous'), validate='one_to_one')
    if len(paired) != len(frame):
        raise AssertionError('Could not match all previous paired conditions')
    for field in ('baseline_selected', 'noisy_selected', 'kept', 'lost', 'gained'):
        if not np.array_equal(paired[field], paired[f'{field}_previous']):
            raise AssertionError(f'Paired selection mismatch: {field}')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    metrics = [column for column in frame if column not in keys]
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(repeats=('repeat', 'size'),
        **{f'{metric}_mean': (metric, 'mean') for metric in metrics})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'role': 'development-scene gate-state diagnosis, no method tuning',
        'plan': 'SELECTION_GATE_TRANSITIONS_PLAN.md',
        'paired_reference': str(previous_path), 'inputs': inputs,
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'seed_base': RANDOM_SEED + 1523, 't_scene_used': False,
        'truth_used': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(summary, output_dir / 'selection_gate_transitions_5000.png')


def plot(summary: pd.DataFrame, path: Path) -> None:
    """Plot mutually exclusive failed-gate states; labels are mean events."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    palette = {'angle_only': '#1f77b4', 'geometry_only': '#ff7f0e',
               'both': '#9467bd'}
    for row, scene in enumerate(SCENES):
        part = summary[(summary.scenario == scene) &
                       (summary.event_count == 5000) &
                       (summary.position_sigma_mm > 0)].sort_values('position_sigma_mm')
        for col, direction in enumerate(('lost', 'gained')):
            ax = axes[row, col]
            bottom = np.zeros(len(part))
            for cause, color in palette.items():
                values = part[f'{direction}_{cause}_mean'].to_numpy()
                ax.bar([str(s) for s in part.position_sigma_mm], values, bottom=bottom,
                       label=cause, color=color)
                bottom += values
            ax.set_title(f'{scene}: {direction}')
            ax.set_ylabel('Mean events / 5000 input')
            ax.grid(axis='y', alpha=.2)
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Hit-position sigma [mm]')
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/selection_gate_results')
    run(parser.parse_args().output_dir)
