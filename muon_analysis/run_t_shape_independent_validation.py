"""Run the pre-registered T-target test of the frozen split-half rule.

The rule's SHA-256 is checked before data are read. No parameter grid search or
per-condition tuning is performed. Target geometry enters evaluation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_refinement import confidence_refined_volume
from diagnose_refinement_fallback import choose_reconstruction
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, fixed_fine_volume, imaging_edges
from run_continuous_event_quality_experiment import evaluate
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_local_support_refinement import local_support_volumes
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = '2e11bff1e4172da2bc8d0b98459760f72dc32aa9065f9ff1a4726841fc1259e5'
COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
METHODS = ('fixed32', 'coarse16', 'adaptive40', 'local_shrink',
           'frozen_fallback', 'split_confidence')


def assert_frozen_rule() -> None:
    """Detect accidental edits to the pre-registered reconstruction code."""
    path = Path(__file__).with_name('confidence_refinement.py')
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != FROZEN_SHA256:
        raise RuntimeError(f'Frozen rule hash mismatch: {observed}; do not run held-out test')


def t_mask(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Exact XY footprint of the two non-overlapping Geant4 boxes; scoring only."""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    bar = (x >= -60) & (x <= 60) & (y >= 20) & (y <= 50)
    stem = (x >= -15) & (x <= 15) & (y >= -50) & (y <= 20)
    return bar | stem


def preregistered_verdict(paired_summary: pd.DataFrame) -> dict:
    """Apply the written N=5000 success criterion without changing it."""
    part = paired_summary[(paired_summary.method == 'split_confidence') &
                          (paired_summary.event_count == 5000)].set_index('position_sigma_mm')
    if set(part.index) != set(SIGMAS_MM):
        raise AssertionError('Missing a pre-registered sigma condition')
    eligible = [sigma for sigma in (0.0, 0.05)
                if part.loc[sigma, 'iou_gain_mean'] >= 0.02
                and part.loc[sigma, 'iou_win_count'] >= 4]
    valid = [sigma for sigma in eligible
             if all(part.loc[other, 'iou_gain_mean'] >= -0.02
                    for other in SIGMAS_MM if other != sigma)]
    return {
        'criterion_met': bool(valid),
        'eligible_gain_sigmas_mm': eligible,
        'qualifying_sigmas_mm': valid,
        'required_gain': 0.02,
        'required_paired_wins_out_of_5': 4,
        'max_mean_loss_at_other_sigmas': 0.02,
        'interpretation': 'descriptive practical criterion, not independent-simulation significance',
    }


def run(input_csv: Path, output_dir: Path) -> None:
    assert_frozen_rule()  # Check before any T-target data are inspected.
    data = pd.read_csv(input_csv)
    if len(data) < max(COUNTS):
        raise ValueError('At least 5000 complete four-layer tracks are required')
    spacing = same_side_layer_spacing_mm(data)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = t_mask(cx, cy)
    rng = np.random.default_rng(RANDOM_SEED + 1129)
    rows: list[dict] = []
    pictures: dict[tuple[float, str], np.ndarray] = {}
    for count in COUNTS:
        for repeat in range(1, REPEATS + 1):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
            # One set of unit normal draws per subset; sigma only rescales it.
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                     for prefix in ('up1', 'up2', 'down1', 'down2')
                     for axis in ('x', 'y')}
            for sigma in SIGMAS_MM:
                measured = measured_hits(subset, sigma, draws)
                observed = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
                events = corrected_events(observed, sigma, spacing)
                variants: dict[str, tuple[np.ndarray, dict, float]] = {}
                start = time.perf_counter()
                fixed, _ = fixed_fine_volume(events)
                variants['fixed32'] = (fixed, {}, time.perf_counter() - start)
                start = time.perf_counter()
                coarse, _, _, _ = adaptive_volume(events, 0.0)
                variants['coarse16'] = (coarse, {}, time.perf_counter() - start)
                start = time.perf_counter()
                adaptive, _, selected, _ = adaptive_volume(events, 0.40)
                variants['adaptive40'] = (adaptive, {'refined_coarse_voxels': selected},
                                          time.perf_counter() - start)
                start = time.perf_counter()
                local, local_info = local_support_volumes(events, sigma, spacing)
                variants['local_shrink'] = (local['local_shrink'], local_info,
                                            time.perf_counter() - start)
                choice = choose_reconstruction(len(events), sigma, spacing)
                variants['frozen_fallback'] = (variants[choice][0],
                                                {'chosen_base_method': choice}, variants[choice][2])
                start = time.perf_counter()
                new_volume, new_info = confidence_refined_volume(events, sigma, spacing)
                variants['split_confidence'] = (new_volume, new_info,
                                                time.perf_counter() - start)
                for method in METHODS:
                    volume, info, runtime = variants[method]
                    image = volume.sum(axis=2)
                    if count == 5000 and repeat == 1 and sigma in (0.0, 0.10):
                        pictures[(sigma, method)] = image.copy()
                    rows.append({'scenario': 't_shape', 'method': method,
                                 'event_count': count, 'repeat': repeat,
                                 'position_sigma_mm': sigma,
                                 'selected_events': len(events),
                                 'reconstruction_seconds': runtime,
                                 'active_refined_xy_cells': info.get('active_refined_xy_cells', np.nan),
                                 'mean_alpha_nonzero': info.get('mean_alpha_nonzero', np.nan),
                                 'chosen_base_method': info.get('chosen_base_method', ''),
                                 **evaluate(image, truth, cx, cy)})
            print(f'T target: N={count}, repeat={repeat}/{REPEATS}', flush=True)

    frame = pd.DataFrame(rows)
    key = ['event_count', 'repeat', 'position_sigma_mm']
    if len(frame) != len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(METHODS):
        raise AssertionError('Unexpected result count')
    if frame.duplicated(['method', *key]).any():
        raise AssertionError('Duplicate method/condition result')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary = frame.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{col}_mean': (col, 'mean') for col in
           ('iou', 'global_error_mm', 'auto_error_mm', 'selected_events',
            'active_refined_xy_cells', 'mean_alpha_nonzero', 'reconstruction_seconds')},
        **{f'{col}_std': (col, 'std') for col in
           ('iou', 'global_error_mm', 'auto_error_mm')})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    base = frame[frame.method == 'coarse16'].set_index(key).sort_index()
    paired = []
    for method in METHODS:
        if method == 'coarse16':
            continue
        other = frame[frame.method == method].set_index(key).sort_index()
        if not base.index.equals(other.index):
            raise AssertionError('Paired condition keys differ')
        for condition in base.index:
            reference, candidate = base.loc[condition], other.loc[condition]
            paired.append({'method': method, **dict(zip(key, condition)),
                           'iou_gain_vs_coarse16': float(candidate.iou - reference.iou),
                           'global_error_reduction_mm': float(reference.global_error_mm -
                                                              candidate.global_error_mm)})
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    paired_summary = paired_frame.groupby(
        ['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
            repeats=('repeat', 'size'),
            iou_gain_mean=('iou_gain_vs_coarse16', 'mean'),
            iou_gain_std=('iou_gain_vs_coarse16', 'std'),
            iou_win_count=('iou_gain_vs_coarse16', lambda x: int((x > 0).sum())),
            global_error_reduction_mm_mean=('global_error_reduction_mm', 'mean'))
    paired_summary.to_csv(output_dir / 'paired_summary.csv', index=False)
    verdict = preregistered_verdict(paired_summary)
    (output_dir / 'verdict.json').write_text(json.dumps(verdict, indent=2,
                                                       ensure_ascii=False), encoding='utf-8')
    (output_dir / 'config.json').write_text(json.dumps({
        'input_csv': str(input_csv), 'input_rows': len(data),
        'scene_role': 'previously unseen T shape for frozen split-half rule',
        'source': 'Ideal parallel 4 GeV muons', 'primary_muons': 10000,
        'frozen_rule_sha256': FROZEN_SHA256,
        'event_counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'sampling_seed': RANDOM_SEED + 1129,
        'truth_usage': 'evaluation and figure overlay only',
        'criterion': 'CONFIDENCE_REFINEMENT_PROTOCOL.md; evaluated without retuning',
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    displayed = summary[summary.event_count == 5000]
    for method in METHODS:
        part = displayed[displayed.method == method].sort_values('position_sigma_mm')
        for ax, metric in zip(axes, ('iou', 'global_error_mm')):
            ax.errorbar(part.position_sigma_mm, part[f'{metric}_mean'],
                        yerr=part[f'{metric}_std'], marker='o', capsize=2, label=method)
    axes[0].set_ylabel('Equal-area IoU (higher better)')
    axes[1].set_ylabel('Global centroid error [mm] (lower better)')
    for ax in axes:
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.savefig(output_dir / 'metrics_5000.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(METHODS), figsize=(17, 6),
                             sharex=True, sharey=True, constrained_layout=True)
    for row, sigma in enumerate((0.0, 0.10)):
        for col, method in enumerate(METHODS):
            image = pictures[(sigma, method)]
            ax = axes[row, col]
            ax.imshow(np.sqrt(image.T / max(float(image.max()), 1e-12)),
                      origin='lower', extent=(-90, 90, -90, 90), cmap='magma',
                      vmin=0, vmax=1, interpolation='nearest')
            ax.contour(cx, cy, truth.T.astype(float), levels=[.5], colors=['cyan'])
            ax.set_title(method, fontsize=8)
            if col == 0:
                ax.set_ylabel(f'sigma={sigma:.2f} mm\ny [mm]')
            if row == 1:
                ax.set_xlabel('x [mm]')
    fig.savefig(output_dir / 'reconstruction_5000.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT / 'muon_analysis/muon_tracks_t_shape.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/t_shape_independent_results')
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == '__main__':
    main()
