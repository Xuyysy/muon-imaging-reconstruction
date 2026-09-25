"""Development-only check of boundary subcell signal and POCA perturbation.

Read BOUNDARY_POCA_DIAGNOSIS_PLAN.md before interpreting output. The target
mask enters only diagnostic scoring; it never changes event selection or image.
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

from confidence_refinement import _child_weights, _effective_count
from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

MIN_NEFF = (4.0, 8.0)


def boundary_metrics(events: pd.DataFrame, truth: np.ndarray) -> dict:
    """Compare observed child-weight pattern to binary truth *after* imaging.

    This is a truth-dependent explanatory score, not a feasible rule for an
    unknown target. One score per mixed coarse XY cell prevents high-count
    cells from silently dominating the mean.
    """
    if truth.shape != (32, 32):
        raise ValueError('Expected a 32x32 evaluation mask')
    target = truth.reshape(16, 2, 16, 2).transpose(0, 2, 1, 3).reshape(16, 16, 4)
    target_count = target.sum(axis=2)
    boundary = (target_count > 0) & (target_count < 4)
    points = events[['poca_x_mm', 'poca_y_mm']].to_numpy(dtype=float)
    weights = events.capped_weight.to_numpy(dtype=float)
    child = _child_weights(points, weights)
    child_sq = _child_weights(points, weights ** 2)
    neff = _effective_count(child, child_sq)
    mass = child.sum(axis=2)
    pattern = np.divide(child, mass[:, :, None], out=np.zeros_like(child),
                        where=mass[:, :, None] > 0)
    target_pattern = target / target_count.clip(min=1)[:, :, None]
    uniform = np.full(4, .25)
    uniform_l1 = np.abs(uniform - target_pattern).sum(axis=2)
    observed_l1 = np.abs(pattern - target_pattern).sum(axis=2)
    inside_mass = (pattern * target).sum(axis=2)
    area_fraction = target_count / 4
    result = {'mixed_boundary_cells': int(boundary.sum())}
    for gate in MIN_NEFF:
        suffix = str(int(gate))
        eligible = boundary & (mass > 0) & (neff >= gate)
        improvement = uniform_l1[eligible] - observed_l1[eligible]
        enrichment = inside_mass[eligible] - area_fraction[eligible]
        result.update({
            f'boundary_cells_neff_ge_{suffix}': int(eligible.sum()),
            f'boundary_l1_improvement_neff_ge_{suffix}': float(improvement.mean())
                if len(improvement) else np.nan,
            f'boundary_positive_fraction_neff_ge_{suffix}':
                float(np.mean(improvement > 0)) if len(improvement) else np.nan,
            f'boundary_inside_enrichment_neff_ge_{suffix}': float(enrichment.mean())
                if len(enrichment) else np.nan,
        })
    return result


def displacement_metrics(reference: pd.DataFrame, noisy: pd.DataFrame,
                         ref_selected: pd.DataFrame, noisy_selected: pd.DataFrame,
                         fine_xy_mm: float) -> dict:
    """Pair by original input-row position, never by nearest reconstructed point."""
    if len(reference) != len(noisy):
        raise ValueError('Unpaired input event counts')
    columns = ['poca_x_mm', 'poca_y_mm', 'poca_z_mm']
    ref = reference[columns].to_numpy(dtype=float)
    obs = noisy[columns].to_numpy(dtype=float)
    xy = np.linalg.norm(obs[:, :2] - ref[:, :2], axis=1)
    xyz = np.linalg.norm(obs - ref, axis=1)
    finite = np.isfinite(xy) & np.isfinite(xyz)
    # recompute_from_noisy_hits returns a RangeIndex in original input order;
    # corrected_events filters rows without resetting that index.
    paired_selected = ref_selected.index.intersection(noisy_selected.index).to_numpy()
    paired_selected = paired_selected[finite[paired_selected]]

    def statistics(values: np.ndarray, prefix: str) -> dict:
        if not len(values):
            return {f'{prefix}_median_mm': np.nan, f'{prefix}_p90_mm': np.nan,
                    f'{prefix}_p95_mm': np.nan}
        return {f'{prefix}_median_mm': float(np.median(values)),
                f'{prefix}_p90_mm': float(np.percentile(values, 90)),
                f'{prefix}_p95_mm': float(np.percentile(values, 95))}

    return {'all_finite_pairs': int(finite.sum()),
            'both_selected_pairs': len(paired_selected),
            'both_selected_fraction_input': len(paired_selected) / len(reference),
            'reference_selected_events': len(ref_selected),
            'retained_reference_fraction': len(paired_selected) / len(ref_selected)
                if len(ref_selected) else np.nan,
            'selected_xy_ge_one_fine_cell_fraction':
                float(np.mean(xy[paired_selected] >= fine_xy_mm))
                if len(paired_selected) else np.nan,
            **statistics(xy[finite], 'all_xy'),
            **statistics(xy[paired_selected], 'selected_xy'),
            **statistics(xyz[paired_selected], 'selected_xyz')}


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    fine_xy_mm = float(edges[0][1] - edges[0][0])
    rows = []
    input_info = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        path = ROOT / 'muon_analysis' / csv_name
        data = pd.read_csv(path)
        if len(data) < max(COUNTS):
            raise ValueError(f'Insufficient data for {scene}')
        input_info[scene] = {'path': str(path), 'rows': len(data)}
        truth = mask_fn(cx, cy)
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                         for prefix in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                ideal_hits = measured_hits(subset, 0.0, draws)
                reference = recompute_from_noisy_hits(ideal_hits, 0.0,
                                                      np.random.default_rng(0))
                ref_selected = corrected_events(reference, 0.0, spacing)
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, draws)
                    noisy = recompute_from_noisy_hits(measured, 0.0,
                                                      np.random.default_rng(0))
                    selected = corrected_events(noisy, sigma, spacing)
                    rows.append({'scenario': scene, 'event_count': count,
                                 'repeat': repeat, 'position_sigma_mm': sigma,
                                 'selected_events': len(selected),
                                 **boundary_metrics(selected, truth),
                                 **displacement_metrics(reference, noisy,
                                                        ref_selected, selected, fine_xy_mm)})
                print(f'boundary/POCA: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    key = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
    if len(frame) != len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM):
        raise AssertionError('Missing development diagnostic condition')
    if frame.duplicated(key).any():
        raise AssertionError('Duplicate condition')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    metrics = ['selected_events', 'boundary_cells_neff_ge_4',
               'boundary_cells_neff_ge_8', 'boundary_l1_improvement_neff_ge_4',
               'boundary_l1_improvement_neff_ge_8',
               'boundary_positive_fraction_neff_ge_4',
               'boundary_inside_enrichment_neff_ge_4', 'both_selected_pairs',
               'both_selected_fraction_input', 'retained_reference_fraction',
               'selected_xy_median_mm',
               'selected_xy_p90_mm', 'selected_xy_p95_mm',
               'selected_xy_ge_one_fine_cell_fraction', 'all_xy_p90_mm']
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in metrics})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'role': 'development diagnostic only; truth used only for scoring',
        'inputs': input_info, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'seed_base': RANDOM_SEED + 1523,
        'fine_xy_cell_mm': fine_xy_mm, 'minimum_effective_events': MIN_NEFF,
        't_scene_used': False, 'reference': 'same-event sigma=0 reconstructed POCA',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(summary, fine_xy_mm, output_dir / 'boundary_poca_scale_5000.png')


def plot(summary: pd.DataFrame, fine_xy_mm: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for scene in SCENES:
        part = summary[(summary.scenario == scene) & (summary.event_count == 5000)]
        part = part.sort_values('position_sigma_mm')
        axes[0].plot(part.position_sigma_mm,
                     part.boundary_cells_neff_ge_4_mean, 'o-', label=scene)
        axes[1].plot(part.position_sigma_mm,
                     part.boundary_l1_improvement_neff_ge_4_mean, 'o-', label=scene)
        axes[2].plot(part.position_sigma_mm, part.selected_xy_p90_mm_mean,
                     'o-', label=scene)
    axes[1].axhline(0, color='k', linewidth=.8)
    axes[2].axhline(fine_xy_mm, color='k', linestyle='--', linewidth=.8,
                    label=f'fine XY cell = {fine_xy_mm:.2f} mm')
    for ax, label in zip(axes, ('Mixed-boundary XY cells with n_eff>=4',
                                'L1 improvement vs uniform (higher is better)',
                                'Paired selected POCA XY displacement P90 [mm]')):
        ax.set(xlabel='Hit-position sigma [mm]', ylabel=label)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/boundary_poca_diagnosis_results')
    run(parser.parse_args().output_dir)
