"""探测器误差下的软权重 PoCA 对照实验。

三组方法使用完全相同的事件抽样与击中误差：
observed = 观测角 2 mrad 硬筛选；
hard = 扣除角度噪声底后再做 2 mrad 硬筛选；
soft = 观测角 2 mrad 筛选，但根据测量不确定度降低每条事件权重。

软权重 q=theta_obs^2/(theta_obs^2+theta_noise^2) 在 [0,1] 内。
它是物理启发式置信度，并非精确的后验概率；sigma=0 时退化为原方法。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import (
    RANDOM_SEED, THETA_CAP_MRAD, THETA_MIN_MRAD, XY_LIMIT_MM, Z_LIMIT_MM,
    adaptive_volume, fixed_fine_volume, imaging_edges, prepare_events,
)
from run_baseline_experiment import evaluate_image
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm, shared_noise_data, theta_noise_floor_mrad,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'soft_uncertainty_results'
EVENT_COUNTS = [1000, 3000, 5000]
POSITION_SIGMAS_MM = [0.0, 0.05, 0.10, 0.30, 1.0]
REPEATS = 5
METHODS = ('observed', 'hard', 'soft')


def soft_events(noisy: pd.DataFrame, sigma_mm: float, spacing_mm: float) -> pd.DataFrame:
    """保留原事件选择，按角度噪声占比连续降低每条事件的贡献。

    先复用上一轮的成像范围、2 mrad 阈值和 3 mrad 权重上限；
    再乘上 q。不要改变 theta 列，否则不能与原方法公平对照。
    """
    events = prepare_events(noisy)
    theta_sq = events['theta_mrad'].to_numpy() ** 2
    noise_sq = theta_noise_floor_mrad(sigma_mm, spacing_mm) ** 2
    q = theta_sq / (theta_sq + noise_sq)
    events['reliability'] = q
    events['capped_weight'] = events['capped_weight'].to_numpy() * q
    return events


def plot_summary(summary: pd.DataFrame) -> None:
    """分事件数绘制 IoU 和定位误差，颜色表示方法，避免重叠曲线误读。"""
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10), constrained_layout=True)
    labels = {'observed': 'Observed angle', 'hard': 'Hard correction', 'soft': 'Soft weight'}
    colors = {'observed': '#7f7f7f', 'hard': '#1f77b4', 'soft': '#d62728'}
    for row, count in enumerate(EVENT_COUNTS):
        for method in METHODS:
            part = summary[(summary.event_count == count) & (summary.method == method)]
            if part.empty:
                continue
            x = part.position_sigma_mm.to_numpy()
            axes[row, 0].errorbar(x, part.iou_mean, yerr=part.iou_std,
                                  marker='o', capsize=2, label=labels[method], color=colors[method])
            axes[row, 1].errorbar(x, part.localization_error_mm_mean,
                                  yerr=part.localization_error_mm_std, marker='o', capsize=2,
                                  label=labels[method], color=colors[method])
        axes[row, 0].set_ylabel(f'{count} events\nIoU (higher better)')
        axes[row, 1].set_ylabel('Localization error [mm]\n(lower better)')
        for ax in axes[row]:
            ax.set_xlabel('Hit-position sigma [mm]')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle('Adaptive PoCA: observed angle vs hard correction vs soft reliability weight')
    fig.savefig(OUT_DIR / 'soft_uncertainty_metrics.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    spacing = same_side_layer_spacing_mm(data)
    edges = imaging_edges(32)
    centers = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rng = np.random.default_rng(RANDOM_SEED + 23)
    records = []
    for count in EVENT_COUNTS:
        if count > len(data):
            continue
        for repeat in range(REPEATS):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                     for prefix in ('up1', 'up2', 'down1', 'down2') for axis in ('x', 'y')}
            for sigma in POSITION_SIGMAS_MM:
                noisy = shared_noise_data(subset, sigma, draws)
                variants = {
                    'observed': prepare_events(noisy),
                    'hard': corrected_events(noisy, sigma, spacing),
                    'soft': soft_events(noisy, sigma, spacing),
                }
                for method, events in variants.items():
                    volume, _, refined, leaves = adaptive_volume(events, .40)
                    image = volume.sum(axis=2)
                    iou, error, contrast, truth_pixels = evaluate_image(image, *centers)
                    records.append({
                        'method': method, 'event_count': count, 'repeat': repeat + 1,
                        'position_sigma_mm': sigma, 'theta_noise_floor_mrad': theta_noise_floor_mrad(sigma, spacing),
                        'selected_events': len(events), 'refined_voxels': refined, 'leaf_voxels': leaves,
                        'mean_reliability': float(events.reliability.mean()) if method == 'soft' and len(events) else np.nan,
                        'iou_equal_area': iou, 'localization_error_mm': error,
                        'target_background_contrast': contrast, 'truth_u_pixels': truth_pixels,
                    })
    results = pd.DataFrame.from_records(records)
    results.to_csv(OUT_DIR / 'soft_uncertainty_metrics_by_repeat.csv', index=False)
    summary = results.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'), selected_events_mean=('selected_events', 'mean'),
        mean_reliability=('mean_reliability', 'mean'), iou_mean=('iou_equal_area', 'mean'),
        iou_std=('iou_equal_area', 'std'), localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
    )
    summary.to_csv(OUT_DIR / 'soft_uncertainty_metrics_summary.csv', index=False)
    plot_summary(summary)
    config = {
        'seed': RANDOM_SEED + 23, 'repeats': REPEATS, 'event_counts': EVENT_COUNTS,
        'position_sigmas_mm': POSITION_SIGMAS_MM, 'layer_spacing_mm': spacing,
        'theta_min_mrad': THETA_MIN_MRAD, 'theta_cap_mrad': THETA_CAP_MRAD,
        'refinement_fraction': .40,
        'soft_weight': 'min(theta_observed^2, theta_cap^2) * theta_observed^2/(theta_observed^2+theta_noise^2)',
        'soft_selection': 'theta_observed >= theta_min and PoCA inside imaging region',
        'note': 'Same random subsets and normal draws as prior uncertainty-aware experiment; target truth only in evaluation.',
    }
    (OUT_DIR / 'soft_uncertainty_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary[(summary.event_count == 5000) & summary.position_sigma_mm.isin([0.05, 0.10, 0.30])].to_string(index=False))


if __name__ == '__main__':
    main()
