"""对从未参与调参的 L 型铝目标，冻结既有 PoCA 重建与事件质量规则。

输入是 Geant4 四层击中 CSV；真值 L 掩膜仅传给 evaluate() 评分。
所有方法在相同事件子样本、相同探测器噪声和相同网格上配对比较。
这是一种新形状的外部场景验证，不是新的探测器/材料/自然缪子验证。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 批处理时只保存 PNG，不打开 macOS 图形窗口。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from localize_target_region import localize_sliding_window
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, fixed_fine_volume, imaging_edges
from run_baseline_experiment import voxelize_poca
from run_continuous_event_quality_experiment import estimate_quality, evaluate
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import REFINE_FRACTION, measured_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

EVENT_COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
METHODS = ('raw_fixed32', 'hard_fixed32', 'hard_coarse16', 'hard_adaptive40',
           'majority_adaptive40', 'pass_prob_adaptive40', 'pass_spread_adaptive40')
ROOT = Path(__file__).resolve().parents[1]


def l_target_mask(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """与 DetectorConstruction.cc 中两块无重叠铝盒严格对应，仅用于评价。"""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    bottom = (x >= -60) & (x <= 60) & (y >= -60) & (y <= -30)
    arm = (x >= -60) & (x <= -30) & (y >= -30) & (y <= 60)
    return bottom | arm


def score(image: np.ndarray, truth: np.ndarray, cx: np.ndarray,
          cy: np.ndarray) -> dict[str, float]:
    """轮廓、全图中心和两种预先固定窗口；不修改任何定位器参数。"""
    metrics = evaluate(image, truth, cx, cy)
    xg, yg = np.meshgrid(cx, cy, indexing='ij')
    tx, ty = float(xg[truth].mean()), float(yg[truth].mean())
    fixed = localize_sliding_window(image, cx, cy)
    metrics['fixed24_error_mm'] = float(np.hypot(fixed['x_mm'] - tx,
                                                  fixed['y_mm'] - ty))
    return metrics


def reconstruct_variants(observed: pd.DataFrame, nominal: pd.DataFrame,
                         probabilities: np.ndarray, spreads: np.ndarray,
                         fine_width: float) -> dict[str, tuple[np.ndarray, int]]:
    """同一名义事件集的硬筛选、多数通过和连续权重消融。"""
    ids = nominal.index.to_numpy(dtype=int)
    p, s = probabilities[ids], spreads[ids]
    majority = nominal.loc[p >= 0.5].copy()
    valid = p > 0
    pass_prob = nominal.loc[valid].copy()
    pass_prob['capped_weight'] *= p[valid]
    pass_spread = nominal.loc[valid].copy()
    pass_spread['capped_weight'] *= p[valid] / (1 + (s[valid] / fine_width) ** 2)

    raw_volume, _, raw_valid = voxelize_poca(observed)
    fixed_volume, _ = fixed_fine_volume(nominal)
    coarse_volume, _, _, _ = adaptive_volume(nominal, 0.0)
    hard_volume, _, _, _ = adaptive_volume(nominal, REFINE_FRACTION)
    majority_volume, _, _, _ = adaptive_volume(majority, REFINE_FRACTION)
    probability_volume, _, _, _ = adaptive_volume(pass_prob, REFINE_FRACTION)
    spread_volume, _, _, _ = adaptive_volume(pass_spread, REFINE_FRACTION)
    return {
        'raw_fixed32': (raw_volume.sum(axis=2), len(raw_valid)),
        'hard_fixed32': (fixed_volume.sum(axis=2), len(nominal)),
        'hard_coarse16': (coarse_volume.sum(axis=2), len(nominal)),
        'hard_adaptive40': (hard_volume.sum(axis=2), len(nominal)),
        'majority_adaptive40': (majority_volume.sum(axis=2), len(majority)),
        'pass_prob_adaptive40': (probability_volume.sum(axis=2), len(pass_prob)),
        'pass_spread_adaptive40': (spread_volume.sum(axis=2), len(pass_spread)),
    }


def make_figure(summary: pd.DataFrame, out_path: Path) -> None:
    """5000 条输入事件下，分别画轮廓和局部定位，防止只看 IoU。"""
    part = summary[summary.event_count == 5000]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method in METHODS:
        frame = part[part.method == method].sort_values('position_sigma_mm')
        for ax, metric in zip(axes, ('iou', 'global_error_mm', 'auto_error_mm')):
            ax.errorbar(frame.position_sigma_mm, frame[f'{metric}_mean'],
                        yerr=frame[f'{metric}_std'], marker='o', capsize=2,
                        label=method)
    for ax, label in zip(axes, ('Equal-area IoU (higher better)',
                                'Global centroid error [mm]',
                                'Multiscale-window error [mm]')):
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle('Held-out L shape, frozen rules, 5000 complete events')
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def make_reconstruction_figure(images: dict[tuple[float, str], np.ndarray],
                               truth: np.ndarray, cx: np.ndarray,
                               cy: np.ndarray, out_path: Path) -> None:
    """展示单个固定重复的 PoCA 投影；青色真值只作事后叠加。"""
    displayed = ('raw_fixed32', 'hard_fixed32', 'hard_coarse16',
                 'hard_adaptive40', 'majority_adaptive40')
    fig, axes = plt.subplots(2, len(displayed), figsize=(15, 6),
                             sharex=True, sharey=True, constrained_layout=True)
    for row, sigma in enumerate((0.0, 0.10)):
        for col, method in enumerate(displayed):
            image = images[(sigma, method)]
            # 每幅图单独归一化，便于观察形状；不能用色彩比较绝对散射强度。
            scaled = np.sqrt(image / max(float(image.max()), 1e-12))
            ax = axes[row, col]
            ax.imshow(scaled.T, origin='lower', extent=(-90, 90, -90, 90),
                      cmap='magma', vmin=0, vmax=1, interpolation='nearest')
            ax.contour(cx, cy, truth.T.astype(float), levels=[0.5],
                       colors=['cyan'], linewidths=1.2)
            ax.set_title(method, fontsize=9)
            if col == 0:
                ax.set_ylabel(f'sigma={sigma:.2f} mm\ny [mm]')
            if row == 1:
                ax.set_xlabel('x [mm]')
    fig.suptitle('Held-out L shape, 5000 events, repeat 1; cyan = truth for display only')
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT / 'muon_analysis' / 'muon_tracks_l_shape.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis' / 'heldout_l_shape_results')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if len(data) < max(EVENT_COUNTS):
        raise ValueError('Not enough four-layer complete events for frozen event counts')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = l_target_mask(cx, cy)
    fine_width = float(edges[0][1] - edges[0][0])
    spacing = same_side_layer_spacing_mm(data)
    rng = np.random.default_rng(RANDOM_SEED + 714)
    rows: list[dict] = []
    representative: dict[tuple[float, str], np.ndarray] = {}
    for count in EVENT_COUNTS:
        for repeat in range(1, REPEATS + 1):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
            noise_draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                           for prefix in ('up1', 'up2', 'down1', 'down2')
                           for axis in ('x', 'y')}
            for sigma in SIGMAS_MM:
                measured = measured_hits(subset, sigma, noise_draws)
                observed = recompute_from_noisy_hits(measured, 0.0,
                                                     np.random.default_rng(0))
                nominal = corrected_events(observed, sigma, spacing)
                quality_rng = np.random.default_rng(
                    RANDOM_SEED + 714000 + 100000 * count + 1000 * repeat
                    + int(sigma * 1000))
                p, s = estimate_quality(measured, sigma, spacing, quality_rng)
                variants = reconstruct_variants(observed, nominal, p, s, fine_width)
                for name, (image, contributing) in variants.items():
                    if count == 5000 and repeat == 1 and sigma in (0.0, 0.10):
                        representative[(sigma, name)] = image.copy()
                    rows.append({
                        'scenario': 'l_shape', 'event_count': count,
                        'repeat': repeat, 'position_sigma_mm': sigma,
                        'method': name, 'complete_input_events': count,
                        'contributing_events': contributing,
                        'truth_pixels': int(truth.sum()),
                        **score(image, truth, cx, cy),
                    })
            print(f'L shape: N={count}, repeat={repeat}/{REPEATS}', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / 'heldout_l_shape_by_repeat.csv', index=False)
    group = ['scenario', 'method', 'event_count', 'position_sigma_mm']
    metrics = ('iou', 'global_error_mm', 'auto_error_mm', 'fixed24_error_mm',
               'contributing_events')
    summary = result.groupby(group, as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in metrics},
        **{f'{name}_std': (name, 'std') for name in metrics})
    summary.to_csv(args.output_dir / 'heldout_l_shape_summary.csv', index=False)
    hard = result[result.method == 'hard_adaptive40'].set_index(
        ['event_count', 'repeat', 'position_sigma_mm'])
    paired = []
    for method in ('hard_coarse16', 'hard_fixed32', 'majority_adaptive40',
                   'pass_prob_adaptive40', 'pass_spread_adaptive40'):
        other = result[result.method == method].set_index(
            ['event_count', 'repeat', 'position_sigma_mm'])
        if not hard.index.equals(other.index):
            raise AssertionError('Paired rows differ; results would not be comparable')
        for key in hard.index:
            if key[2] == 0:
                continue
            paired.append({
                'method': method, 'event_count': key[0], 'repeat': key[1],
                'position_sigma_mm': key[2],
                'iou_gain_vs_hard_adaptive40': float(other.loc[key, 'iou'] - hard.loc[key, 'iou']),
                'global_error_reduction_mm': float(hard.loc[key, 'global_error_mm']
                                                  - other.loc[key, 'global_error_mm']),
                'auto_error_reduction_mm': float(hard.loc[key, 'auto_error_mm']
                                                - other.loc[key, 'auto_error_mm']),
            })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(args.output_dir / 'heldout_l_shape_paired.csv', index=False)
    paired_summary = paired_frame.groupby('method', as_index=False).agg(
        paired_n=('repeat', 'size'),
        iou_gain_mean=('iou_gain_vs_hard_adaptive40', 'mean'),
        iou_win_fraction=('iou_gain_vs_hard_adaptive40', lambda x: float((x > 0).mean())),
        global_error_reduction_mm_mean=('global_error_reduction_mm', 'mean'),
        auto_error_reduction_mm_mean=('auto_error_reduction_mm', 'mean'))
    paired_summary.to_csv(args.output_dir / 'heldout_l_shape_paired_summary.csv', index=False)
    make_figure(summary, args.output_dir / 'heldout_l_shape_5000.png')
    make_reconstruction_figure(representative, truth, cx, cy,
                               args.output_dir / 'heldout_l_shape_reconstruction.png')
    (args.output_dir / 'heldout_l_shape_config.json').write_text(json.dumps({
        'input_csv': str(args.input), 'input_rows': len(data),
        'target': '120 mm aluminium L, 30 mm legs, 100 mm z thickness',
        'target_role': 'held-out shape; not used in prior threshold/refinement/quality design',
        'event_counts': EVENT_COUNTS, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'source': 'ideal parallel 4 GeV muons', 'fine_bins': 32, 'coarse_bins': 16,
        'theta_min_mrad': 2, 'theta_cap_mrad': 3,
        'refinement_fraction': REFINE_FRACTION, 'quality_draws': 12,
        'quality_majority_fraction': 0.5,
        'truth_usage': 'evaluation only', 'rng_seed': RANDOM_SEED + 714,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(paired_summary.to_string(index=False))


if __name__ == '__main__':
    main()
