"""逐格替换粗图：比较实际 Top-k IoU 影响与无真值局部信号。"""
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
from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

KEY = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
EPS = 1e-12


def topk_iou(image: np.ndarray, truth: np.ndarray) -> float:
    """沿用项目中的等面积 Top-k 评分；真值仅在诊断函数中出现。"""
    if image.shape != (32, 32) or truth.shape != image.shape:
        raise ValueError('Expected aligned 32x32 arrays')
    k = int(truth.sum())
    if k <= 0 or k >= image.size:
        raise ValueError('Degenerate truth mask')
    chosen = np.zeros(image.shape, dtype=bool)
    chosen.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    overlap = int((chosen & truth).sum())
    return overlap / int((chosen | truth).sum())


def diagnose_cells(events: pd.DataFrame, truth: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """一次只细化一个占用粗格；不改变其他格或该格总散射权重。"""
    points = events[['poca_x_mm', 'poca_y_mm']].to_numpy(dtype=float)
    weights = events.capped_weight.to_numpy(dtype=float)
    child = _child_weights(points, weights)
    child_sq = _child_weights(points, weights**2)
    count = _child_weights(points, np.ones(len(points))).sum(axis=2)
    neff = _effective_count(child, child_sq)
    mass = child.sum(axis=2)
    score, valid_splits = heldout_subcell_score(points)
    # 粗图每个 16x16 格均匀摊到四个子格，与原 16 网格的 32 网格
    # 评价映射一致；细图仅在一个格使用观测 POCA 子格分配。
    coarse = np.repeat(np.repeat(mass / 4., 2, axis=0), 2, axis=1)
    fine = child.reshape(16, 16, 2, 2).transpose(0, 2, 1, 3).reshape(32, 32)
    if not np.isclose(coarse.sum(), fine.sum(), rtol=1e-12, atol=1e-9):
        raise AssertionError('Child detail changed total scattering weight')
    base_iou = topk_iou(coarse, truth)
    rows = []
    for ix, iy in np.argwhere(count > 0):
        replacement = coarse.copy()
        sl = np.s_[2*ix:2*ix+2, 2*iy:2*iy+2]
        replacement[sl] = fine[sl]
        if not np.isclose(replacement.sum(), coarse.sum(), rtol=1e-12, atol=1e-9):
            raise AssertionError('One-cell refinement changed image mass')
        gain = topk_iou(replacement, truth) - base_iou
        # 仅作为终点诊断标签；模型特征与细化运算均不读取目标真值。
        target_patch = truth[sl]
        rows.append({'ix': int(ix), 'iy': int(iy),
                     'cell_event_count': int(count[ix, iy]),
                     'effective_events': float(neff[ix, iy]),
                     'scattering_mass': float(mass[ix, iy]),
                     'heldout_log_gain': float(score[ix, iy]),
                     'valid_splits': int(valid_splits[ix, iy]),
                     'target_child_count': int(target_patch.sum()),
                     'single_cell_iou_gain': float(gain),
                     'positive_gain': bool(gain > EPS),
                     'negative_gain': bool(gain < -EPS)})
    cells = pd.DataFrame(rows)
    if cells.empty:
        raise AssertionError('No occupied XY cells for diagnostic')

    def stats(mask: pd.Series, name: str) -> dict:
        part = cells[mask]
        return {f'{name}_cells': len(part),
                f'{name}_positive_fraction': float(part.positive_gain.mean())
                if len(part) else np.nan,
                f'{name}_negative_fraction': float(part.negative_gain.mean())
                if len(part) else np.nan}

    scored = cells.heldout_log_gain.notna()
    report = {
        'selected_events': len(events),
        'coarse_iou': base_iou,
        'occupied_cells': len(cells),
        'positive_gain_cells': int(cells.positive_gain.sum()),
        'negative_gain_cells': int(cells.negative_gain.sum()),
        'unchanged_cells': int((~cells.positive_gain & ~cells.negative_gain).sum()),
        'scored_cells': int(scored.sum()),
        'scored_background_cells': int((scored & (cells.target_child_count == 0)).sum()),
        **stats(cells.effective_events < 2, 'neff_lt2'),
        **stats((cells.effective_events >= 2) & (cells.effective_events < 4), 'neff_2to4'),
        **stats(cells.effective_events >= 4, 'neff_ge4'),
        **stats(scored & (cells.heldout_log_gain > 0), 'heldout_positive'),
        **stats(scored & (cells.heldout_log_gain <= 0), 'heldout_nonpositive'),
    }
    if report['positive_gain_cells'] + report['negative_gain_cells'] + report['unchanged_cells'] != len(cells):
        raise AssertionError('Single-cell outcomes do not partition occupied cells')
    return report, cells


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows, cell_frames = [], []
    inputs = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        path = ROOT / 'muon_analysis' / csv_name
        tracks = pd.read_csv(path)
        if len(tracks) < max(COUNTS):
            raise ValueError(f'Insufficient tracks for {scene}')
        inputs[scene] = {'csv': csv_name, 'complete_tracks': len(tracks)}
        truth = mask_fn(cx, cy)
        spacing = same_side_layer_spacing_mm(tracks)
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = tracks.iloc[rng.choice(len(tracks), count, replace=False)].copy()
                draws = {f'{plane}_{axis}': rng.normal(0, 1, count)
                         for plane in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    selected = corrected_events(observed, sigma, spacing)
                    metrics, cells = diagnose_cells(selected, truth)
                    key = {'scenario': scene, 'event_count': count,
                           'repeat': repeat, 'position_sigma_mm': sigma}
                    rows.append({**key, **metrics})
                    cell_frames.append(cells.assign(**key))
                print(f'Single-cell diagnosis: {scene}, N={count}, '
                      f'repeat {repeat}/{REPEATS}', flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != len(SCENES) * len(COUNTS) * REPEATS * len(SIGMAS_MM):
        raise AssertionError('Incomplete diagnostic matrix')
    if frame.duplicated(KEY).any():
        raise AssertionError('Duplicated conditions')
    cells = pd.concat(cell_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    cells.to_csv(output_dir / 'cells.csv', index=False)
    metrics = [col for col in frame.columns if col not in KEY]
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in metrics})
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'SINGLE_CELL_REFINEMENT_SIGNAL_PLAN.md',
        'role': 'development-only per-cell counterfactual, no new reconstruction rule',
        'inputs': inputs, 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'seed_base': RANDOM_SEED + 1523,
        'coarse_grid': 16, 'fine_grid': 32,
        'truth_usage': 'single-cell IoU outcome and diagnostic target-child label only',
        'score_missing_is_zero': False,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for scene in SCENES:
        part = summary[(summary.scenario == scene) & (summary.event_count == 5000)]
        part = part.sort_values('position_sigma_mm')
        occupied = part.occupied_cells_mean
        axes[0].plot(part.position_sigma_mm,
                     (part.positive_gain_cells_mean + part.negative_gain_cells_mean)
                     / occupied, marker='o', label=scene)
        axes[1].plot(part.position_sigma_mm, part.scored_cells_mean / occupied,
                     marker='o', label=scene)
    axes[0].set_ylabel('Fraction changing Top-k IoU')
    axes[1].set_ylabel('Fraction with held-out score')
    for ax in axes:
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Development-only single-cell refinement signal, N=5000')
    fig.savefig(output_dir / 'single_cell_refinement_signal.png', dpi=180)
    plt.close(fig)
    print(summary[(summary.event_count == 5000)
                  & (summary.position_sigma_mm.isin((0., .05, .30)))][
        ['scenario', 'position_sigma_mm', 'occupied_cells_mean',
         'positive_gain_cells_mean', 'negative_gain_cells_mean',
         'scored_cells_mean', 'neff_ge4_cells_mean']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/single_cell_refinement_signal_results')
