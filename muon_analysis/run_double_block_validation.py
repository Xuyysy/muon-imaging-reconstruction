"""Frozen-method validation on two separate aluminium blocks.

This scene was not used to tune any method. Ground truth is passed only to
the scoring function. All methods use the same event subsets and noise draws.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_refinement_fallback import choose_reconstruction
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, fixed_fine_volume, imaging_edges
from run_continuous_event_quality_experiment import evaluate
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_local_support_refinement import local_support_volumes
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
METHODS = ('fixed32', 'coarse16', 'adaptive40', 'local_shrink', 'frozen_fallback')


def double_block_mask(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Same XY footprint as the Geant4 boxes; used only for evaluation."""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    return (((x >= -65) & (x <= -15)) | ((x >= 15) & (x <= 65))) & (y >= -30) & (y <= 30)


def shape_metrics(image: np.ndarray, truth: np.ndarray,
                  cx: np.ndarray, cy: np.ndarray) -> dict[str, float]:
    """Assess both separated components and false signal in the empty gap.

    The existing auto-window error assumes one connected target. We retain it
    for comparability but must not use it as the main localization score here.
    """
    x, y = np.meshgrid(cx, cy, indexing='ij')
    k = int(truth.sum())
    predicted = np.zeros_like(truth, dtype=bool)
    predicted.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    left = truth & (x < 0)
    right = truth & (x > 0)
    gap = (x > -15) & (x < 15) & (y >= -30) & (y <= 30)
    footprint = (x >= -65) & (x <= 65) & (y >= -30) & (y <= 30)
    footprint_weight = float(image[footprint].sum())
    return {
        'left_recall': float((predicted & left).sum() / left.sum()),
        'right_recall': float((predicted & right).sum() / right.sum()),
        'min_block_recall': float(min((predicted & left).sum() / left.sum(),
                                      (predicted & right).sum() / right.sum())),
        'gap_false_positive_fraction': float((predicted & gap).sum() / gap.sum()),
        'gap_mass_fraction': float(image[gap].sum() / footprint_weight)
        if footprint_weight > 0 else float('nan'),
    }


def run(input_csv: Path, output_dir: Path) -> None:
    data = pd.read_csv(input_csv)
    if len(data) < max(COUNTS):
        raise ValueError('At least 5000 complete four-layer tracks are required')
    spacing = same_side_layer_spacing_mm(data)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = double_block_mask(cx, cy)
    rng = np.random.default_rng(RANDOM_SEED + 947)
    rows = []
    pictures = {}
    for count in COUNTS:
        for repeat in range(1, REPEATS + 1):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
            # Reuse the same unit-Gaussian draws across all sigma values.
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                     for prefix in ('up1', 'up2', 'down1', 'down2')
                     for axis in ('x', 'y')}
            for sigma in SIGMAS_MM:
                measured = measured_hits(subset, sigma, draws)
                observed = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
                events = corrected_events(observed, sigma, spacing)
                timed = {}
                start = time.perf_counter()
                fixed, _ = fixed_fine_volume(events)
                timed['fixed32'] = (fixed, time.perf_counter() - start, 32768)
                start = time.perf_counter()
                coarse, _, _, coarse_leaves = adaptive_volume(events, 0.0)
                timed['coarse16'] = (coarse, time.perf_counter() - start, coarse_leaves)
                start = time.perf_counter()
                adaptive, _, n_refined, adaptive_leaves = adaptive_volume(events, 0.40)
                timed['adaptive40'] = (adaptive, time.perf_counter() - start, adaptive_leaves)
                start = time.perf_counter()
                local, info = local_support_volumes(events, sigma, spacing)
                timed['local_shrink'] = (local['local_shrink'], time.perf_counter() - start,
                                         float('nan'))
                # The policy is frozen. It sees selected count and known hit resolution,
                # never the target mask or the evaluation metrics.
                choice = choose_reconstruction(len(events), sigma, spacing)
                chosen_volume, _, chosen_leaves = timed[choice]
                timed['frozen_fallback'] = (chosen_volume, 0.0, chosen_leaves)
                for method in METHODS:
                    volume, runtime, leaves = timed[method]
                    image = volume.sum(axis=2)
                    if count == 5000 and repeat == 1 and sigma in (0.0, 0.10):
                        pictures[(sigma, method)] = image.copy()
                    rows.append({'scenario': 'double_block', 'method': method,
                                 'event_count': count, 'repeat': repeat,
                                 'position_sigma_mm': sigma, 'selected_events': len(events),
                                 'chosen_base_method': choice if method == 'frozen_fallback' else '',
                                 'refined_coarse_slices': n_refined if method == 'adaptive40' else
                                     info['refined_coarse_slices'] if method == 'local_shrink' else 0,
                                 'leaf_voxels': leaves,
                                 'reconstruction_seconds': runtime if method != 'frozen_fallback' else
                                     timed[choice][1],
                                 **evaluate(image, truth, cx, cy),
                                 **shape_metrics(image, truth, cx, cy)})
            print(f'double block: N={count}, repeat={repeat}/{REPEATS}', flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    key = ['event_count', 'repeat', 'position_sigma_mm']
    if frame.duplicated(['method', *key]).any():
        raise AssertionError('Duplicate method/condition rows')
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary = frame.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{col}_mean': (col, 'mean') for col in
           ('iou', 'global_error_mm', 'auto_error_mm', 'min_block_recall',
            'gap_false_positive_fraction', 'gap_mass_fraction', 'selected_events',
            'refined_coarse_slices', 'leaf_voxels', 'reconstruction_seconds')},
        **{f'{col}_std': (col, 'std') for col in
           ('iou', 'global_error_mm', 'auto_error_mm', 'min_block_recall')})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    base = frame[frame.method == 'coarse16'].set_index(key).sort_index()
    paired = []
    for method in METHODS:
        if method == 'coarse16':
            continue
        other = frame[frame.method == method].set_index(key).sort_index()
        if not base.index.equals(other.index):
            raise AssertionError('Methods were not compared on paired conditions')
        for condition in base.index:
            a, b = base.loc[condition], other.loc[condition]
            paired.append({'method': method, **dict(zip(key, condition)),
                           'iou_gain_vs_coarse16': float(b.iou - a.iou),
                           'auto_error_reduction_mm': float(a.auto_error_mm - b.auto_error_mm)})
    paired = pd.DataFrame(paired)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    paired.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        iou_gain_mean=('iou_gain_vs_coarse16', 'mean'),
        iou_win_fraction=('iou_gain_vs_coarse16', lambda x: float((x > 0).mean())),
        auto_error_reduction_mm_mean=('auto_error_reduction_mm', 'mean'),
    ).to_csv(output_dir / 'paired_summary.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    part = summary[summary.event_count == 5000]
    for method in METHODS:
        series = part[part.method == method].sort_values('position_sigma_mm')
        for ax, metric in zip(axes, ('iou', 'min_block_recall')):
            ax.errorbar(series.position_sigma_mm, series[f'{metric}_mean'],
                        yerr=series[f'{metric}_std'], marker='o', capsize=2, label=method)
    axes[0].set_ylabel('Equal-area IoU (higher better)')
    axes[1].set_ylabel('Lower recall of the two blocks (higher better)')
    for ax in axes:
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / 'metrics_5000.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(METHODS), figsize=(15, 6),
                             sharex=True, sharey=True, constrained_layout=True)
    for row, sigma in enumerate((0.0, 0.10)):
        for col, method in enumerate(METHODS):
            image = pictures[(sigma, method)]
            ax = axes[row, col]
            ax.imshow(np.sqrt(image.T / max(float(image.max()), 1e-12)),
                      origin='lower', extent=(-90, 90, -90, 90), cmap='magma',
                      vmin=0, vmax=1, interpolation='nearest')
            ax.contour(cx, cy, truth.T.astype(float), levels=[.5], colors=['cyan'])
            ax.set_title(method, fontsize=9)
            if col == 0:
                ax.set_ylabel(f'sigma={sigma:.2f} mm\ny [mm]')
            if row == 1:
                ax.set_xlabel('x [mm]')
    fig.savefig(output_dir / 'reconstruction_5000.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'input_csv': str(input_csv), 'input_rows': len(data),
        'scene_role': 'new shape, independent test of frozen U/L-developed rules',
        'geometry': 'two aluminium 50x60x100 mm boxes, x centers -40 and +40 mm',
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'seed': RANDOM_SEED + 947, 'truth_usage': 'evaluation and display only',
        'policy': 'imported unchanged from diagnose_refinement_fallback.py',
        'timing': 'per-method reconstruction only; fallback reports chosen base method runtime',
        'leaf_voxels_note': 'not comparable for 2.5D local_shrink; recorded as NaN',
        'localization_note': 'auto-window error assumes one connected target; min-block recall and gap leakage are primary shape checks',
    }, indent=2, ensure_ascii=False), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT / 'muon_analysis/muon_tracks_double_block.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/double_block_validation_results')
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == '__main__':
    main()
