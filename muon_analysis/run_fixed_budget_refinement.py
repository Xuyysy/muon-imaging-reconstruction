"""已见开发集上的固定预算 XY 细分排序诊断；真值贪心不可部署。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_refinement import _child_weights, _effective_count
from diagnose_heldout_subcell_score import heldout_subcell_score
from diagnose_single_cell_refinement_signal import topk_iou
from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM, topk_mask
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

BUDGET = 8
RANDOM_DRAWS = 16
FOCUS_COUNTS = (1000, 5000)
FOCUS_SIGMAS_MM = (.05, .30)
KEY = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
METHODS = ('coarse', 'neff8', 'heldout_positive', 'random8_mean', 'truth_greedy8')
EPS = 1e-12


def images_and_ranks(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """从同一入选事件构造粗图、细子格和两个无真值排序特征。"""
    points = events[['poca_x_mm', 'poca_y_mm']].to_numpy(float)
    weight = events.capped_weight.to_numpy(float)
    child = _child_weights(points, weight)
    neff = _effective_count(child, _child_weights(points, weight**2))
    counts = _child_weights(points, np.ones(len(points))).sum(axis=2)
    mass = child.sum(axis=2)
    base = np.repeat(np.repeat(mass / 4., 2, axis=0), 2, axis=1)
    fine = child.reshape(16, 16, 2, 2).transpose(0, 2, 1, 3).reshape(32, 32)
    score, _ = heldout_subcell_score(points)
    cells = [tuple(map(int, pair)) for pair in np.argwhere(counts > 0)]
    if len(cells) < BUDGET or not np.isclose(base.sum(), fine.sum(), atol=1e-9):
        raise AssertionError('Insufficient occupied cells or nonconserved image')
    return base, fine, neff, score, cells


def apply_selection(base: np.ndarray, fine: np.ndarray,
                    selected: list[tuple[int, int]]) -> np.ndarray:
    """替换选中粗格的四子格；其他像素严格沿用粗图。"""
    result = base.copy()
    for ix, iy in selected:
        sl = np.s_[2*ix:2*ix+2, 2*iy:2*iy+2]
        result[sl] = fine[sl]
    if not np.isclose(result.sum(), base.sum(), rtol=1e-12, atol=1e-9):
        raise AssertionError('Selection changed image total weight')
    return result


def greedy_with_truth(base: np.ndarray, fine: np.ndarray, truth: np.ndarray,
                      cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """只作诊断参照：真值直接参与逐步最大边际 IoU 选择，无法部署。"""
    selected: list[tuple[int, int]] = []
    remaining = set(cells)
    current = base.copy()
    current_iou = topk_iou(current, truth)
    for _ in range(BUDGET):
        best, best_gain = None, EPS
        for cell in sorted(remaining):
            ix, iy = cell
            sl = np.s_[2*ix:2*ix+2, 2*iy:2*iy+2]
            candidate = current.copy()
            candidate[sl] = fine[sl]
            gain = topk_iou(candidate, truth) - current_iou
            if gain > best_gain:
                best, best_gain = cell, gain
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        ix, iy = best
        current[2*ix:2*ix+2, 2*iy:2*iy+2] = fine[2*ix:2*ix+2, 2*iy:2*iy+2]
        current_iou += best_gain
    return selected


def image_metrics(image: np.ndarray, base: np.ndarray,
                  truth: np.ndarray, selected_count: int) -> dict:
    """Top-k 看轮廓，L1 只看连续图像改变量而非成像质量。"""
    k = int(truth.sum())
    return {
        'selected_cells': selected_count,
        'iou': topk_iou(image, truth),
        'topk_changed_pixels': int(np.logical_xor(
            topk_mask(image, k), topk_mask(base, k)).sum()),
        'relative_image_l1': (float(np.abs(image - base).sum() / base.sum())
                              if base.sum() > 0 else 0.),
    }


def run(output_dir: Path) -> None:
    archive = pd.read_csv(ROOT / 'muon_analysis/single_cell_refinement_signal_results'
                          / 'by_repeat.csv').set_index(KEY).sort_index()
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows, random_rows, selected_rows = [], [], []
    inputs = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        tracks = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        spacing = same_side_layer_spacing_mm(tracks)
        truth = mask_fn(cx, cy)
        inputs[scene] = {'csv': csv_name, 'complete_tracks': len(tracks)}
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                # 3000 组虽然不评分，仍必须抽样并消耗噪声 RNG，才能让
                # 后续 5000 组逐次复现上一轮已归档的抽样顺序。
                subset = tracks.iloc[rng.choice(len(tracks), count, replace=False)].copy()
                draws = {f'{plane}_{axis}': rng.normal(size=count)
                         for plane in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                if count not in FOCUS_COUNTS:
                    continue
                for sigma in SIGMAS_MM:
                    if sigma not in FOCUS_SIGMAS_MM:
                        continue
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    events = corrected_events(observed, sigma, spacing)
                    key = {'scenario': scene, 'event_count': count,
                           'repeat': repeat, 'position_sigma_mm': sigma}
                    old = archive.loc[(scene, count, repeat, sigma)]
                    base, fine, neff, score, cells = images_and_ranks(events)
                    base_iou = topk_iou(base, truth)
                    if not (len(events) == int(old.selected_events)
                            and np.isclose(base_iou, old.coarse_iou,
                                           rtol=0, atol=1e-12)):
                        raise AssertionError(f'Prior event/image mismatch: {key}')
                    # 并列分数用 ix,iy 升序；正分缺失者不可混作零分。
                    neff_selection = sorted(cells,
                                            key=lambda cell: (-neff[cell], *cell))[:BUDGET]
                    scored = [cell for cell in cells
                              if np.isfinite(score[cell]) and score[cell] > 0]
                    heldout_selection = sorted(
                        scored, key=lambda cell: (-score[cell], *cell))[:BUDGET]
                    oracle_selection = greedy_with_truth(base, fine, truth, cells)
                    selections = {
                        'coarse': [], 'neff8': neff_selection,
                        'heldout_positive': heldout_selection,
                        'truth_greedy8': oracle_selection,
                    }
                    for method, selected in selections.items():
                        image = apply_selection(base, fine, selected)
                        metrics = image_metrics(image, base, truth, len(selected))
                        rows.append({**key, 'method': method,
                                     'occupied_cells': len(cells),
                                     'scored_cells': int(np.isfinite(score).sum()),
                                     **metrics, 'iou_gain': metrics['iou'] - base_iou})
                        selected_rows.extend({**key, 'method': method,
                                              'ix': ix, 'iy': iy}
                                             for ix, iy in selected)
                    random_rng = np.random.default_rng(
                        RANDOM_SEED + 730001 + scene_index * 100000
                        + count * 10 + repeat * 1000 + int(round(sigma * 1000)))
                    trial_metrics = []
                    for draw_index in range(RANDOM_DRAWS):
                        choice = random_rng.choice(len(cells), BUDGET, replace=False)
                        selection = [cells[index] for index in choice]
                        image = apply_selection(base, fine, selection)
                        metrics = image_metrics(image, base, truth, BUDGET)
                        random_rows.append({**key, 'draw': draw_index,
                                            **metrics, 'iou_gain': metrics['iou'] - base_iou})
                        trial_metrics.append(metrics)
                    rows.append({**key, 'method': 'random8_mean',
                                 'occupied_cells': len(cells),
                                 'scored_cells': int(np.isfinite(score).sum()),
                                 **{name: float(np.mean([m[name] for m in trial_metrics]))
                                    for name in ('selected_cells', 'iou',
                                                 'topk_changed_pixels', 'relative_image_l1')},
                                 'iou_gain': float(np.mean([m['iou'] - base_iou
                                                            for m in trial_metrics]))})
                print(f'Fixed budget: {scene}, N={count}, repeat {repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    if (len(frame) != len(SCENES) * len(FOCUS_COUNTS) * len(FOCUS_SIGMAS_MM)
            * REPEATS * len(METHODS) or frame.duplicated(KEY + ['method']).any()):
        raise AssertionError('Incomplete fixed-budget comparison')
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm',
                             'method'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in
           ('selected_cells', 'occupied_cells', 'scored_cells', 'iou',
            'iou_gain', 'topk_changed_pixels', 'relative_image_l1')},
        **{f'{name}_std': (name, 'std') for name in ('iou_gain', 'selected_cells')})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    pd.DataFrame(random_rows).to_csv(output_dir / 'random_draws.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / 'selected_cells.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'FIXED_BUDGET_REFINEMENT_PLAN.md',
        'role': 'development-only fixed-budget ranking diagnostic',
        'inputs': inputs, 'counts': FOCUS_COUNTS, 'sigmas_mm': FOCUS_SIGMAS_MM,
        'repeats': REPEATS, 'budget': BUDGET, 'random_draws': RANDOM_DRAWS,
        'truth_usage': 'IoU scoring and nondeployable truth_greedy8 only',
        'greedy_is_global_upper_bound': False,
        'image_type': '2.5D XY projection, not full 3D density',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    focus = summary[(summary.event_count == 1000)
                    & (summary.position_sigma_mm == .30)]
    labels = list(SCENES)
    positions = np.arange(len(labels))
    colors = {'neff8': 'tab:blue', 'heldout_positive': 'tab:orange',
              'random8_mean': 'tab:gray', 'truth_greedy8': 'tab:green'}
    for offset, method in zip((-.27, -.09, .09, .27), colors):
        part = focus[focus.method == method].set_index('scenario').loc[labels]
        ax.bar(positions + offset, part.iou_gain_mean, width=.17,
               color=colors[method], label=method)
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel('Mean equal-area IoU gain vs coarse')
    ax.set_title('Fixed budget ≤8 XY cells; N=1000, hit σ=0.30 mm')
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / 'fixed_budget_refinement.png', dpi=180)
    plt.close(fig)
    print(focus[['scenario', 'method', 'selected_cells_mean', 'iou_gain_mean',
                 'topk_changed_pixels_mean', 'relative_image_l1_mean']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/fixed_budget_refinement_results')
