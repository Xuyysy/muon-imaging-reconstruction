"""Truth-free split-half subcell score, evaluated only on development scenes.

Read HELDOUT_SUBCELL_SCORE_PLAN.md before interpreting the output. The mask is
used *after* scores are computed, solely for diagnostic labels and metrics.
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

from confidence_refinement import SPLIT_REPEATS, SPLIT_SEED, _child_weights, _effective_count
from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

MIN_EVENTS_PER_HALF = 2
MIN_VALID_SPLITS = 2
PRIOR_PER_CHILD = 1.0


def heldout_subcell_score(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-half count log-likelihood gain over uniform, without truth/weights.

    Each split trains on one half and scores the other, then swaps them. All
    arrays are coarse 16x16 XY cells; NaN means insufficient split support.
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError('Expected (n,2) XY positions')
    values = np.zeros((16, 16), dtype=float)
    support = np.zeros((16, 16), dtype=int)
    rng = np.random.default_rng(SPLIT_SEED)
    for _ in range(SPLIT_REPEATS):
        order = rng.permutation(len(points_xy))
        first, second = order[:len(order) // 2], order[len(order) // 2:]
        count_a = _child_weights(points_xy[first], np.ones(len(first)))
        count_b = _child_weights(points_xy[second], np.ones(len(second)))
        total_a, total_b = count_a.sum(axis=2), count_b.sum(axis=2)
        valid = (total_a >= MIN_EVENTS_PER_HALF) & (total_b >= MIN_EVENTS_PER_HALF)
        # A symmetric held-out log score: neither half evaluates the pattern it
        # itself estimated. Unit pseudocounts prevent log(0) for sparse cells.
        pa = (count_a + PRIOR_PER_CHILD) / (total_a[:, :, None] + 4 * PRIOR_PER_CHILD)
        pb = (count_b + PRIOR_PER_CHILD) / (total_b[:, :, None] + 4 * PRIOR_PER_CHILD)
        numerator = (count_b * np.log(4 * pa)).sum(axis=2) + (
            count_a * np.log(4 * pb)).sum(axis=2)
        denominator = total_a + total_b
        split_value = np.divide(numerator, denominator,
                                out=np.zeros_like(numerator), where=denominator > 0)
        values += np.where(valid, split_value, 0)
        support += valid
    mean = np.divide(values, support, out=np.full_like(values, np.nan),
                     where=support >= MIN_VALID_SPLITS)
    return mean, support


def evaluate_cells(events: pd.DataFrame, truth: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Apply predeclared truth-only diagnostics after score construction."""
    target = truth.reshape(16, 2, 16, 2).transpose(0, 2, 1, 3).reshape(16, 16, 4)
    target_count = target.sum(axis=2)
    boundary = (target_count > 0) & (target_count < 4)
    background = target_count == 0
    points = events[['poca_x_mm', 'poca_y_mm']].to_numpy(dtype=float)
    weights = events.capped_weight.to_numpy(dtype=float)
    score, valid_splits = heldout_subcell_score(points)
    child = _child_weights(points, weights)
    child_squared = _child_weights(points, weights ** 2)
    neff = _effective_count(child, child_squared)
    mass = child.sum(axis=2)
    pattern = np.divide(child, mass[:, :, None], out=np.zeros_like(child),
                        where=mass[:, :, None] > 0)
    q = target / np.maximum(target_count, 1)[:, :, None]
    gain = np.abs(.25 - q).sum(axis=2) - np.abs(pattern - q).sum(axis=2)
    eligible = boundary & (neff >= 4)
    scored = eligible & np.isfinite(score)
    positive = scored & (score > 0)
    nonpositive = scored & (score <= 0)
    scored_background = background & np.isfinite(score)

    def mean_or_nan(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else np.nan

    metrics = {
        'selected_events': len(events),
        'boundary_eligible_cells': int(eligible.sum()),
        'boundary_scored_cells': int(scored.sum()),
        'boundary_positive_score_cells': int(positive.sum()),
        'boundary_nonpositive_score_cells': int(nonpositive.sum()),
        'boundary_positive_gain_mean': mean_or_nan(gain[positive]),
        'boundary_nonpositive_gain_mean': mean_or_nan(gain[nonpositive]),
        'boundary_positive_gain_fraction': mean_or_nan((gain[positive] > 0).astype(float)),
        'boundary_nonpositive_gain_fraction': mean_or_nan((gain[nonpositive] > 0).astype(float)),
        'background_scored_cells': int(scored_background.sum()),
        'background_positive_score_cells': int(np.count_nonzero(scored_background & (score > 0))),
    }
    relevant = eligible | scored_background
    ix, iy = np.where(relevant)
    cells = pd.DataFrame({
        'ix': ix, 'iy': iy, 'target_child_count': target_count[ix, iy],
        'is_boundary': boundary[ix, iy], 'is_background': background[ix, iy],
        'effective_events': neff[ix, iy], 'valid_splits': valid_splits[ix, iy],
        'heldout_log_gain': score[ix, iy],
        'truth_proxy_l1_improvement': np.where(boundary[ix, iy], gain[ix, iy], np.nan),
    })
    return metrics, cells


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    centers = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows, cell_frames = [], []
    inputs = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        path = ROOT / 'muon_analysis' / csv_name
        data = pd.read_csv(path)
        if len(data) < max(COUNTS):
            raise ValueError(f'Insufficient complete events: {scene}')
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
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(measured, 0.0,
                                                         np.random.default_rng(0))
                    events = corrected_events(observed, sigma, spacing)
                    metrics, cells = evaluate_cells(events, truth)
                    key = {'scenario': scene, 'event_count': count,
                           'repeat': repeat, 'position_sigma_mm': sigma}
                    rows.append({**key, **metrics})
                    cell_frames.append(cells.assign(**key))
                print(f'held-out subcell score: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    key_columns = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
    if len(frame) != len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM):
        raise AssertionError('Missing diagnostic conditions')
    if frame.duplicated(key_columns).any():
        raise AssertionError('Duplicate condition')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    pd.concat(cell_frames, ignore_index=True).to_csv(output_dir / 'cells.csv', index=False)
    metric_columns = [c for c in frame.columns if c not in key_columns]
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(repeats=('repeat', 'size'),
        **{f'{c}_mean': (c, 'mean') for c in metric_columns})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'role': 'development-only score diagnosis, no revised imaging method',
        'plan': 'HELDOUT_SUBCELL_SCORE_PLAN.md', 'inputs': inputs,
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'seed_base': RANDOM_SEED + 1523, 'split_seed': SPLIT_SEED,
        'split_repeats': SPLIT_REPEATS, 'min_events_per_half': MIN_EVENTS_PER_HALF,
        'min_valid_splits': MIN_VALID_SPLITS, 'prior_per_child': PRIOR_PER_CHILD,
        't_scene_used': False, 'truth_usage': 'diagnostic labels and metrics only',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(summary, output_dir / 'heldout_subcell_score_5000.png')


def plot(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
    for scene in SCENES:
        p = summary[(summary.scenario == scene) &
                    (summary.event_count == 5000)].sort_values('position_sigma_mm')
        axes[0].plot(p.position_sigma_mm, p.boundary_eligible_cells_mean, 'o-', label=scene)
        axes[0].plot(p.position_sigma_mm, p.boundary_scored_cells_mean, 'x--',
                     color=axes[0].lines[-1].get_color())
        axes[1].plot(p.position_sigma_mm, p.boundary_positive_score_cells_mean,
                     'o-', label=scene)
        axes[2].plot(p.position_sigma_mm, p.background_positive_score_cells_mean,
                     'o-', label=scene)
    for ax, title, ylabel in zip(axes,
        ('Eligible vs scored boundary cells (solid vs dashed)',
         'Boundary cells with positive held-out score',
         'Background cells with positive held-out score'),
        ('XY cells', 'XY cells', 'XY cells')):
        ax.set(xlabel='Hit-position sigma [mm]', ylabel=ylabel, title=title)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/heldout_subcell_score_results')
    run(parser.parse_args().output_dir)
