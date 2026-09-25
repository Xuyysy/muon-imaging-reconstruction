"""Paired POCA-Z movement, parallelism and fixed-boundary stability diagnostic.

Read POCA_Z_STABILITY_PLAN.md first. Uses only development scenes and never
changes the reconstruction or selects thresholds from target truth.
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

from diagnose_selection_gate_transitions import gate_masks
from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM
from run_adaptive_voxel_experiment import RANDOM_SEED, Z_LIMIT_MM
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

STRATA = {
    'z_margin_mm': ((0.0, 10.0), (10.0, 30.0), (30.0, np.inf)),
    'theta0_mrad': ((2.0, 3.0), (3.0, 5.0), (5.0, np.inf)),
}


def parallelism_denominator(hits: pd.DataFrame) -> np.ndarray:
    """Return 1-(u_in dot u_out)^2 using the same four hit layers as POCA.

    This is the denominator of the line--line closest-approach equations for
    unit directions. It approaches zero for parallel or antiparallel tracks.
    """
    def vector(first: str, second: str) -> np.ndarray:
        a = hits[[f'{first}_{axis}_mm' for axis in ('x', 'y', 'z')]].to_numpy(float)
        b = hits[[f'{second}_{axis}_mm' for axis in ('x', 'y', 'z')]].to_numpy(float)
        direction = b - a
        return direction / np.linalg.norm(direction, axis=1, keepdims=True)
    incoming = vector('up1', 'up2')
    outgoing = vector('down1', 'down2')
    dot = np.einsum('ij,ij->i', incoming, outgoing)
    return np.maximum(1.0 - dot ** 2, 0.0)


def _stat(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else np.nan


def analyze_pair(reference: pd.DataFrame, noisy: pd.DataFrame,
                 reference_hits: pd.DataFrame, sigma: float,
                 spacing: float) -> tuple[dict, list[dict]]:
    """Measure POCA-Z flips among reference-selected events, no target truth."""
    if len(reference) != len(noisy) or len(reference) != len(reference_hits):
        raise ValueError('Unpaired hit and reconstruction rows')
    ideal_gate = gate_masks(reference, 0.0, spacing)
    noisy_gate = gate_masks(noisy, sigma, spacing)
    baseline = ideal_gate['selected']
    z_exit = baseline & ~noisy_gate['z']
    dz = np.abs(noisy.poca_z_mm.to_numpy() - reference.poca_z_mm.to_numpy())
    dxy = np.hypot(noisy.poca_x_mm.to_numpy() - reference.poca_x_mm.to_numpy(),
                   noisy.poca_y_mm.to_numpy() - reference.poca_y_mm.to_numpy())
    theta0 = reference.theta_mrad.to_numpy()
    denominator0 = parallelism_denominator(reference_hits)
    margin0 = Z_LIMIT_MM - np.abs(reference.poca_z_mm.to_numpy())
    if not np.isfinite(dz).all() or not np.isfinite(denominator0).all():
        raise ValueError('Nonfinite POCA-Z movement or track parallelism')
    n = int(baseline.sum())
    result = {
        'baseline_selected': n,
        'z_exit': int(z_exit.sum()),
        'z_exit_fraction': float(z_exit.sum() / n) if n else np.nan,
        'abs_dz_selected_median_mm': _stat(dz[baseline], .5),
        'abs_dz_selected_p90_mm': _stat(dz[baseline], .9),
        'abs_dz_z_exit_median_mm': _stat(dz[z_exit], .5),
        'abs_dz_z_exit_p90_mm': _stat(dz[z_exit], .9),
        'abs_dxy_selected_p90_mm': _stat(dxy[baseline], .9),
        'theta0_selected_median_mrad': _stat(theta0[baseline], .5),
        'theta0_z_exit_median_mrad': _stat(theta0[z_exit], .5),
        'denominator0_selected_median': _stat(denominator0[baseline], .5),
        'denominator0_z_exit_median': _stat(denominator0[z_exit], .5),
        'z_margin0_selected_median_mm': _stat(margin0[baseline], .5),
        'z_margin0_z_exit_median_mm': _stat(margin0[z_exit], .5),
    }
    strata_rows = []
    for feature, bins in STRATA.items():
        values = margin0 if feature == 'z_margin_mm' else theta0
        for bin_index, (low, high) in enumerate(bins):
            group = baseline & (values >= low) & (values < high)
            count = int(group.sum())
            exits = int(np.count_nonzero(group & z_exit))
            strata_rows.append({'feature': feature, 'bin_index': bin_index,
                                'lower': low, 'upper': high, 'events': count,
                                'z_exit': exits,
                                'z_exit_fraction': exits / count if count else np.nan})
        if sum(row['events'] for row in strata_rows if row['feature'] == feature) != n:
            raise AssertionError(f'{feature} bins fail to cover baseline-selected rows')
    if sigma == 0 and (np.any(dz != 0) or result['z_exit']):
        raise AssertionError('Sigma=0 POCA-Z control changed')
    return result, strata_rows


def run(output_dir: Path) -> None:
    gate_path = ROOT / 'muon_analysis/selection_gate_results/by_repeat.csv'
    previous = pd.read_csv(gate_path)
    rows, strata = [], []
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
                reference_hits = measured_hits(subset, 0.0, draws)
                reference = recompute_from_noisy_hits(
                    reference_hits, 0.0, np.random.default_rng(0))
                for sigma in SIGMAS_MM:
                    noisy_hits = measured_hits(subset, sigma, draws)
                    noisy = recompute_from_noisy_hits(
                        noisy_hits, 0.0, np.random.default_rng(0))
                    metrics, bins = analyze_pair(reference, noisy, reference_hits,
                                                 sigma, spacing)
                    key = {'scenario': scene, 'event_count': count,
                           'repeat': repeat, 'position_sigma_mm': sigma}
                    rows.append({**key, **metrics})
                    strata.extend({**key, **item} for item in bins)
                print(f'POCA-Z stability: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame, stratified = pd.DataFrame(rows), pd.DataFrame(strata)
    keys = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
    if len(frame) != len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM):
        raise AssertionError('Incomplete paired condition grid')
    if frame.duplicated(keys).any() or len(stratified) != 6 * len(frame):
        raise AssertionError('Duplicate conditions or missing strata')
    matched = frame.merge(previous[keys + ['baseline_selected',
                                           'lost_geometry_z_only',
                                           'lost_geometry_xy_and_z']],
                          on=keys, validate='one_to_one', suffixes=('', '_previous'))
    if len(matched) != len(frame):
        raise AssertionError('Previous gate conditions do not match')
    if not np.array_equal(matched.baseline_selected,
                          matched.baseline_selected_previous):
        raise AssertionError('Baseline event count differs from gate diagnosis')
    expected_z_exit = matched.lost_geometry_z_only + matched.lost_geometry_xy_and_z
    if not np.array_equal(matched.z_exit, expected_z_exit):
        raise AssertionError('POCA-Z exits differ from previous gate diagnosis')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    stratified.to_csv(output_dir / 'strata_by_repeat.csv', index=False)
    metrics = [column for column in frame if column not in keys]
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(repeats=('repeat', 'size'),
        **{f'{metric}_mean': (metric, 'mean') for metric in metrics})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    strata_summary = stratified.groupby(['scenario', 'event_count',
        'position_sigma_mm', 'feature', 'bin_index', 'lower', 'upper'],
        as_index=False, dropna=False).agg(repeats=('repeat', 'size'),
        events_mean=('events', 'mean'), z_exit_mean=('z_exit', 'mean'),
        z_exit_fraction_mean=('z_exit_fraction', 'mean'))
    strata_summary.to_csv(output_dir / 'strata_summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'role': 'POCA-Z stability diagnostic on development scenes only',
        'plan': 'POCA_Z_STABILITY_PLAN.md', 'paired_reference': str(gate_path),
        'inputs': inputs, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'seed_base': RANDOM_SEED + 1523,
        'strata': {key: [(float(a), None if np.isinf(b) else float(b))
                         for a, b in bins]
                   for key, bins in STRATA.items()},
        'truth_used': False, 't_scene_used': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(strata_summary, output_dir / 'poca_z_strata_5000.png')


def plot(summary: pd.DataFrame, path: Path) -> None:
    """Display predeclared stratum-specific boundary-exit rates at N=5000."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    colors = {0.05: '#1f77b4', 0.10: '#ff7f0e', 0.30: '#2ca02c'}
    labels = {'z_margin_mm': ['0-10', '10-30', '30+'],
              'theta0_mrad': ['2-3', '3-5', '5+']}
    for row, scene in enumerate(SCENES):
        for col, feature in enumerate(STRATA):
            ax = axes[row, col]
            for offset, sigma in enumerate((0.05, 0.10, 0.30)):
                part = summary[(summary.scenario == scene) &
                               (summary.event_count == 5000) &
                               (summary.position_sigma_mm == sigma) &
                               (summary.feature == feature)].sort_values('bin_index')
                if len(part) != 3:
                    raise AssertionError('Missing plot stratum')
                x = np.arange(3) + (offset - 1) * .24
                ax.bar(x, part.z_exit_fraction_mean, width=.23,
                       color=colors[sigma], label=f'{sigma:g} mm')
            ax.set_xticks(np.arange(3), labels[feature])
            ax.set_ylim(0, 1)
            ax.set_ylabel(f'{scene}\nPOCA-Z exit fraction')
            ax.set_xlabel('Reference Z margin [mm]' if col == 0
                          else 'Reference theta [mrad]')
            ax.grid(axis='y', alpha=.2)
    axes[0, 1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/poca_z_stability_results')
    run(parser.parse_args().output_dir)
