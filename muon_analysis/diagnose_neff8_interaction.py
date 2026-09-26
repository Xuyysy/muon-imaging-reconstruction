"""诊断已归档 neff8 选格：单格收益、按原排名加入的边际收益和非加性。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_single_cell_refinement_signal import KEY, topk_iou
from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_fixed_budget_refinement import (
    BUDGET, FOCUS_COUNTS, FOCUS_SIGMAS_MM, apply_selection, images_and_ranks,
)
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

TOL = 1e-12


def run(output_dir: Path) -> None:
    """严格重播原随机流；先核对存档，再作事后诊断。"""
    archive_dir = ROOT / 'muon_analysis/fixed_budget_refinement_results'
    old = pd.read_csv(archive_dir / 'by_repeat.csv')
    old = old[old.method.isin(('coarse', 'neff8'))].set_index(KEY + ['method'])
    picks = pd.read_csv(archive_dir / 'selected_cells.csv')
    picks = picks[picks.method == 'neff8'].copy()
    if len(picks) != len(SCENES) * len(FOCUS_COUNTS) * len(FOCUS_SIGMAS_MM) * REPEATS * BUDGET:
        raise AssertionError('Archived neff8 selection is incomplete')
    picks['rank'] = picks.groupby(KEY).cumcount() + 1
    picks = picks.set_index(KEY).sort_index()
    single = pd.read_csv(ROOT / 'muon_analysis/single_cell_refinement_signal_results/cells.csv')
    single = single.set_index(KEY + ['ix', 'iy'])
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows, per_cell = [], []
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        tracks = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        truth = mask_fn(cx, cy)
        spacing = same_side_layer_spacing_mm(tracks)
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                # 原实验即使不评价 N=3000，也消耗其抽样和噪声随机数。
                subset = tracks.iloc[rng.choice(len(tracks), count, replace=False)].copy()
                draws = {f'{plane}_{axis}': rng.normal(size=count)
                         for plane in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                if count not in FOCUS_COUNTS:
                    continue
                for sigma in SIGMAS_MM:
                    if sigma not in FOCUS_SIGMAS_MM:
                        continue
                    key = (scene, count, repeat, sigma)
                    observed = recompute_from_noisy_hits(
                        measured_hits(subset, sigma, draws),
                        0., np.random.default_rng(0))
                    events = corrected_events(observed, sigma, spacing)
                    base, fine, neff, _, occupied = images_and_ranks(events)
                    base_iou = topk_iou(base, truth)
                    expected = old.loc[(*key, 'coarse')]
                    if not np.isclose(base_iou, expected.iou, atol=TOL, rtol=0):
                        raise AssertionError(f'Coarse image mismatch: {key}')
                    chosen = picks.loc[key].sort_values('rank')
                    cells = list(zip(chosen.ix.astype(int), chosen.iy.astype(int)))
                    if len(cells) != BUDGET or len(set(cells)) != BUDGET:
                        raise AssertionError(f'Duplicate/incomplete selection: {key}')
                    expected_order = sorted(occupied,
                                            key=lambda cell: (-neff[cell], *cell))[:BUDGET]
                    if cells != expected_order:
                        raise AssertionError(f'neff ranking mismatch: {key}')
                    current = base.copy()
                    previous_iou = base_iou
                    isolated_gains = []
                    marginal_gains = []
                    selected_mass = 0.
                    for rank, (ix, iy) in enumerate(cells, 1):
                        one = single.loc[(*key, ix, iy)]
                        isolated_gain = float(one.single_cell_iou_gain)
                        isolated_image = apply_selection(base, fine, [(ix, iy)])
                        if not np.isclose(topk_iou(isolated_image, truth) - base_iou,
                                          isolated_gain, atol=TOL, rtol=0):
                            raise AssertionError(f'Archived single-cell mismatch: {key, ix, iy}')
                        # 排名路径固定；重算的是整图 Top-k，而非只看本地子格。
                        current = apply_selection(current, fine, [(ix, iy)])
                        current_iou = topk_iou(current, truth)
                        marginal_gain = current_iou - previous_iou
                        previous_iou = current_iou
                        isolated_gains.append(isolated_gain)
                        marginal_gains.append(marginal_gain)
                        selected_mass += float(one.scattering_mass)
                        per_cell.append({**dict(zip(KEY, key)), 'rank': rank,
                                         'ix': ix, 'iy': iy,
                                         'effective_events': float(one.effective_events),
                                         'scattering_mass': float(one.scattering_mass),
                                         'target_child_count': int(one.target_child_count),
                                         'isolated_iou_gain': isolated_gain,
                                         'sequential_marginal_iou_gain': marginal_gain})
                    actual_gain = current_iou - base_iou
                    archived = old.loc[(*key, 'neff8')]
                    if not (np.isclose(actual_gain, archived.iou_gain, atol=TOL, rtol=0)
                            and np.isclose(sum(marginal_gains), actual_gain, atol=TOL, rtol=0)):
                        raise AssertionError(f'Final image mismatch: {key}')
                    row = {**dict(zip(KEY, key)), 'coarse_iou': base_iou,
                           'actual_iou_gain': actual_gain,
                           'isolated_gain_sum': float(sum(isolated_gains)),
                           'nonadditivity_residual': float(actual_gain - sum(isolated_gains)),
                           'isolated_positive_cells': int(sum(x > TOL for x in isolated_gains)),
                           'isolated_negative_cells': int(sum(x < -TOL for x in isolated_gains)),
                           'isolated_zero_cells': int(sum(abs(x) <= TOL for x in isolated_gains)),
                           'sequential_positive_cells': int(sum(x > TOL for x in marginal_gains)),
                           'sequential_negative_cells': int(sum(x < -TOL for x in marginal_gains)),
                           'sequential_zero_cells': int(sum(abs(x) <= TOL for x in marginal_gains)),
                           'target_free_cells': int(sum(
                               single.loc[(*key, ix, iy)].target_child_count == 0
                               for ix, iy in cells)),
                           'selected_mass_fraction': float(selected_mass / base.sum())}
                    if (row['isolated_positive_cells'] + row['isolated_negative_cells']
                            + row['isolated_zero_cells'] != BUDGET):
                        raise AssertionError('Isolated labels do not partition budget')
                    rows.append(row)
                print(f'neff8 interaction: {scene}, N={count}, repeat {repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    cells_frame = pd.DataFrame(per_cell)
    expected_n = len(SCENES) * len(FOCUS_COUNTS) * len(FOCUS_SIGMAS_MM) * REPEATS
    if (len(frame) != expected_n or len(cells_frame) != expected_n * BUDGET
            or frame.duplicated(KEY).any()):
        raise AssertionError('Incomplete diagnostic matrix')
    measures = [col for col in frame if col not in KEY]
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{col}_mean': (col, 'mean') for col in measures})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    cells_frame.to_csv(output_dir / 'selected_cells.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'B1/NEFF8_INTERACTION_PLAN.md',
        'role': 'development-only neff8 selection failure analysis',
        'source': 'fixed_budget_refinement_results/selected_cells.csv',
        'counts': FOCUS_COUNTS, 'sigmas_mm': FOCUS_SIGMAS_MM,
        'repeats': REPEATS, 'budget': BUDGET,
        'truth_usage': 'IoU and target-child diagnostic label only',
        'interaction_definition': 'actual gain minus sum of isolated one-cell gains',
        'image_type': '2.5D XY projection, not full 3D density',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, (count, sigma) in zip(axes, ((1000, .30), (5000, .05))):
        part = summary[(summary.event_count == count)
                       & (summary.position_sigma_mm == sigma)].set_index('scenario')
        labels = list(SCENES)
        x = np.arange(len(labels))
        part = part.loc[labels]
        ax.bar(x - .2, part.isolated_gain_sum_mean, width=.4,
               label='Sum of isolated gains', color='tab:orange')
        ax.bar(x + .2, part.actual_iou_gain_mean, width=.4,
               label='Actual 8-cell gain', color='tab:blue')
        ax.axhline(0, color='black', lw=.9)
        ax.set_xticks(x, labels)
        ax.set_title(f'N={count}, hit σ={sigma:.2f} mm')
        ax.set_ylabel('Equal-area Top-k IoU gain')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / 'neff8_interaction.png', dpi=180)
    plt.close(fig)
    print(summary[['scenario', 'event_count', 'position_sigma_mm',
                   'actual_iou_gain_mean', 'isolated_gain_sum_mean',
                   'nonadditivity_residual_mean', 'isolated_positive_cells_mean',
                   'isolated_negative_cells_mean', 'target_free_cells_mean']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/neff8_interaction_results')
