"""用测量击中的重复扰动，比较连续事件可靠性权重与简单多数门槛。

所有方案的起点都是同一批 nominal hard 修正后通过筛选的事件。
只使用四层测得的击中位置和已知的探测器位置分辨率 sigma：
  hard       : 每条通过事件的现有截断散射角权重；
  majority   : 12 次重复扰动中至少 6 次仍通过，保留原权重；
  pass_prob  : 权重再乘以通过概率 p；
  pass_spread: 权重再乘以 p / [1 + (s / voxel_width)^2]。
其中 s 是有效 PoCA 点的 XY RMS 空间扩散，voxel_width 为 32 格细体素边长。
这个物理尺度来自网格，而不是目标位置/真值参数拟合。

与旧版 propagated_events 不同，这四种方案都以“同一 nominal 事件集合”为
基础，而且每次重复扰动后使用相同的 hard 角度噪声底修正，使对照更直接。
目标 U 型真值掩膜仅用于最后的 IoU 与定位误差评价。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from localize_target_region import localize_multiscale_window
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import REFINE_FRACTION, measured_hits
from run_robust_localization_experiment import SCENARIOS
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'muon_analysis' / 'continuous_event_quality_results'
EVENT_COUNTS = (1000, 3000, 5000)
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
POSTERIOR_DRAWS = 12
MAJORITY_FRACTION = .50
SCENE_NAMES = ('center', 'offcenter', 'small', 'medium_offset', 'small_offset')
METHODS = ('hard', 'majority', 'pass_prob', 'pass_spread')


def estimate_quality(measured: pd.DataFrame, sigma: float, spacing: float,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """每条事件的通过概率 p 和通过时的 XY PoCA RMS 扩散 s。

    绝不读取理想 Geant4 击中、目标真值或最终图像指标。
    只有一次有效抽样时 s=0，但 p 只有 1/12，故这类事件权重仍很低。
    """
    n = len(measured)
    if sigma == 0:
        nominal = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
        valid = corrected_events(nominal, sigma, spacing)
        p = np.zeros(n)
        p[valid.index.to_numpy(dtype=int)] = 1.0
        return p, np.zeros(n)

    counts = np.zeros(n, dtype=int)
    sum_xy = np.zeros((n, 2))
    sum_sq = np.zeros(n)
    for _ in range(POSTERIOR_DRAWS):
        candidate = recompute_from_noisy_hits(measured, sigma, rng)
        passing = corrected_events(candidate, sigma, spacing)
        ids = passing.index.to_numpy(dtype=int)
        xy = passing[['poca_x_mm', 'poca_y_mm']].to_numpy()
        counts[ids] += 1
        sum_xy[ids] += xy
        sum_sq[ids] += np.sum(xy * xy, axis=1)
    probability = counts / POSTERIOR_DRAWS
    spread = np.zeros(n)
    multiple = counts >= 2
    mean = sum_xy[multiple] / counts[multiple, None]
    variance = sum_sq[multiple] / counts[multiple] - np.sum(mean * mean, axis=1)
    spread[multiple] = np.sqrt(np.maximum(variance, 0))
    return probability, spread


def evaluate(image: np.ndarray, truth: np.ndarray,
             cx: np.ndarray, cy: np.ndarray) -> dict:
    """同一张 XY 投影的轮廓 IoU、全图质心和自动窗口定位误差。"""
    xgrid, ygrid = np.meshgrid(cx, cy, indexing='ij')
    true_x, true_y = float(xgrid[truth].mean()), float(ygrid[truth].mean())
    if image.sum() <= 0:
        return {'iou': np.nan, 'global_error_mm': np.nan, 'auto_error_mm': np.nan}
    k = int(truth.sum())
    prediction = np.zeros_like(truth, dtype=bool)
    prediction.ravel()[np.argpartition(image.ravel(), -k)[-k:]] = True
    intersection = np.logical_and(prediction, truth).sum()
    union = np.logical_or(prediction, truth).sum()
    global_x = float(np.sum(image * xgrid) / image.sum())
    global_y = float(np.sum(image * ygrid) / image.sum())
    auto = localize_multiscale_window(image, cx, cy)
    return {
        'iou': float(intersection / union),
        'global_error_mm': float(np.hypot(global_x - true_x, global_y - true_y)),
        'auto_error_mm': float(np.hypot(auto['x_mm'] - true_x,
                                        auto['y_mm'] - true_y)),
    }


def plot_5000(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(SCENE_NAMES), 2, figsize=(11, 14),
                             sharex=True, constrained_layout=True)
    for row, scene in enumerate(SCENE_NAMES):
        for method in METHODS:
            part = summary[(summary.scenario == scene) &
                           (summary.event_count == 5000) &
                           (summary.method == method)]
            for col, (metric, label) in enumerate((('iou', 'Equal-area IoU'),
                                                    ('auto_error_mm', 'Auto-window error [mm]'))):
                axes[row, col].errorbar(part.position_sigma_mm,
                                        part[f'{metric}_mean'],
                                        yerr=part[f'{metric}_std'], marker='o',
                                        capsize=2, label=method)
                axes[row, col].set_ylabel(f'{scene}\n{label}')
                axes[row, col].grid(alpha=.25)
    for ax in axes[-1]:
        ax.set_xlabel('Hit-position sigma [mm]')
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle('Continuous event quality versus hard and majority: 5000 events')
    fig.savefig(OUT_DIR / 'continuous_event_quality_5000.png', dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    fine_width = float(edges[0][1] - edges[0][0])
    rows = []
    for scene in SCENE_NAMES:
        csv_path, mask_fn = SCENARIOS[scene]
        data = pd.read_csv(csv_path)
        spacing = same_side_layer_spacing_mm(data)
        truth = mask_fn(cx, cy)  # 仅传入 evaluate，不传入 quality。
        rng = np.random.default_rng(RANDOM_SEED + 23)
        for count in EVENT_COUNTS:
            if count > len(data):
                continue
            for repeat in range(REPEATS):
                subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
                draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                         for prefix in ('up1', 'up2', 'down1', 'down2') for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(
                        measured, 0.0, np.random.default_rng(0)
                    )
                    nominal = corrected_events(observed, sigma, spacing)
                    # 此种子仅影响重复扰动；四种方法使用同一套 p、s。
                    posterior_rng = np.random.default_rng(
                        RANDOM_SEED + 100000 * count + 1000 * repeat + int(sigma * 1000)
                    )
                    probability, spread = estimate_quality(
                        measured, sigma, spacing, posterior_rng
                    )
                    ids = nominal.index.to_numpy(dtype=int)
                    p = probability[ids]
                    s = spread[ids]
                    confidence = 1.0 / (1.0 + (s / fine_width) ** 2)
                    variants = {
                        'hard': nominal,
                        'majority': nominal.loc[p >= MAJORITY_FRACTION].copy(),
                        'pass_prob': nominal.loc[p > 0].copy(),
                        'pass_spread': nominal.loc[p > 0].copy(),
                    }
                    variants['pass_prob']['capped_weight'] *= p[p > 0]
                    variants['pass_spread']['capped_weight'] *= (
                        p[p > 0] * confidence[p > 0]
                    )
                    baseline_weight = float(nominal['capped_weight'].sum())
                    for method, events in variants.items():
                        volume, _, _, _ = adaptive_volume(events, REFINE_FRACTION)
                        metrics = evaluate(volume.sum(axis=2), truth, cx, cy)
                        rows.append({
                            'scenario': scene, 'event_count': count, 'repeat': repeat + 1,
                            'position_sigma_mm': sigma, 'method': method,
                            'nominal_events': len(nominal),
                            'contributing_events': len(events),
                            'weight_fraction_of_hard': float(
                                events['capped_weight'].sum() / baseline_weight
                            ) if baseline_weight else np.nan,
                            'nominal_pass_probability': float(p.mean())
                            if p.size else np.nan,
                            'nominal_poca_spread_median_mm': float(np.median(s))
                            if s.size else np.nan,
                            **metrics,
                        })
        print(f'Finished {scene}', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / 'continuous_event_quality_by_repeat.csv', index=False)
    group = ['scenario', 'method', 'event_count', 'position_sigma_mm']
    value_columns = ('iou', 'global_error_mm', 'auto_error_mm',
                     'contributing_events', 'weight_fraction_of_hard',
                     'nominal_pass_probability', 'nominal_poca_spread_median_mm')
    summary = results.groupby(group, as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in value_columns},
        **{f'{name}_std': (name, 'std') for name in value_columns},
    )
    summary.to_csv(OUT_DIR / 'continuous_event_quality_summary.csv', index=False)
    plot_5000(summary)
    (OUT_DIR / 'continuous_event_quality_config.json').write_text(json.dumps({
        'scenarios': SCENE_NAMES, 'event_counts': EVENT_COUNTS,
        'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'posterior_draws': POSTERIOR_DRAWS,
        'majority_fraction': MAJORITY_FRACTION,
        'spatial_scale_mm': fine_width,
        'weight': 'p_pass / [1+(poca_xy_rms/fine_voxel_width)^2]',
        'seed': RANDOM_SEED + 23,
        'truth_usage': 'evaluation only; not used for event quality or reconstruction',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[(summary.event_count == 5000) &
                  summary.position_sigma_mm.isin((.10, .30))][[
        'scenario', 'method', 'position_sigma_mm', 'iou_mean',
        'global_error_mm_mean', 'auto_error_mm_mean',
        'contributing_events_mean']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
