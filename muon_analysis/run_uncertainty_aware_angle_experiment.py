"""不确定度感知散射角筛选实验。

本实验在上一轮“探测器击中位置误差”测试基础上，比较两种事件处理方式：

1. observed：直接使用含测量误差后的观测散射角；
2. corrected：扣除由四层探测器位置分辨率估计的散射角噪声底。

对于同侧两层探测器间距 d、每层 x/y 击中独立标准差 sigma，有
单个方向角差的方差近似为 4*sigma^2/d^2。因此二维散射角噪声为：

    theta_noise = sqrt(8) * sigma / d

采用近似修正：

    theta_corrected = sqrt(max(theta_observed^2 - theta_noise^2, 0))

它不是完整的轨迹协方差传播，而是一个可解释、可验证的第一步物理修正。
真值 U 型掩膜只用于最终评价，绝不参与事件筛选或体素细分。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import (
    RANDOM_SEED, THETA_CAP_MRAD, THETA_MIN_MRAD,
    adaptive_volume, fixed_fine_volume, imaging_edges, prepare_events,
)
from run_baseline_experiment import evaluate_image
from run_detector_resolution_robustness import recompute_from_noisy_hits


ROOT = Path('/Users/xuyaoyang/geant4-test')
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'uncertainty_aware_angle_results'
EVENT_COUNTS = [1000, 3000, 5000]
REPEATS = 5
POSITION_SIGMAS_MM = [0.0, 0.05, 0.10, 0.30, 1.0]
METHODS = {
    'fixed_observed': None,
    'fixed_corrected': None,
    'adaptive40_observed': 0.40,
    'adaptive40_corrected': 0.40,
}


def same_side_layer_spacing_mm(data: pd.DataFrame) -> float:
    """从 CSV 的实际层位置信息读取同侧两层探测器间距。"""
    up = np.abs(data['up2_z_mm'].to_numpy() - data['up1_z_mm'].to_numpy())
    down = np.abs(data['down2_z_mm'].to_numpy() - data['down1_z_mm'].to_numpy())
    return float(np.mean(np.concatenate([up, down])))


def theta_noise_floor_mrad(position_sigma_mm: float, layer_spacing_mm: float) -> float:
    """估计二维观测散射角的 RMS 噪声底（单位：mrad）。"""
    return float(np.sqrt(8.0) * position_sigma_mm / layer_spacing_mm * 1000.0)


def corrected_events(noisy: pd.DataFrame, position_sigma_mm: float, layer_spacing_mm: float) -> pd.DataFrame:
    """对观测 theta 扣除误差噪声底，再复用既有筛选与截断权重流程。"""
    corrected = noisy.copy()
    theta_noise = theta_noise_floor_mrad(position_sigma_mm, layer_spacing_mm)
    theta_observed = corrected['theta_mrad'].to_numpy()
    corrected['theta_mrad'] = np.sqrt(np.maximum(theta_observed ** 2 - theta_noise ** 2, 0.0))
    return prepare_events(corrected)


def shared_noise_data(subset: pd.DataFrame, sigma_mm: float, normal_draws: dict[str, np.ndarray]) -> pd.DataFrame:
    """为不同 sigma 使用同一组标准正态随机数，使横向比较更公平。"""
    noisy_input = subset.copy()
    for prefix in ['up1', 'up2', 'down1', 'down2']:
        noisy_input[f'{prefix}_x_mm'] += sigma_mm * normal_draws[f'{prefix}_x']
        noisy_input[f'{prefix}_y_mm'] += sigma_mm * normal_draws[f'{prefix}_y']
    # recompute_from_noisy_hits 负责完整轨迹与 PoCA 计算；这里传 0 避免二次加噪。
    return recompute_from_noisy_hits(noisy_input, 0.0, np.random.default_rng(0))


def plot_results(summary: pd.DataFrame, output_path: Path) -> None:
    """画出未修正与修正后方案在不同位置误差下的 IoU 和定位误差。"""
    labels = {
        'fixed_observed': 'fixed voxel, observed theta',
        'fixed_corrected': 'fixed voxel, corrected theta',
        'adaptive40_observed': 'adaptive 40%, observed theta',
        'adaptive40_corrected': 'adaptive 40%, corrected theta',
    }
    styles = {'observed': '--', 'corrected': '-'}
    colors = {1000: '#1f77b4', 3000: '#ff7f0e', 5000: '#2ca02c'}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    for method, group in summary.groupby('method'):
        row = 0 if method.startswith('fixed') else 1
        kind = 'corrected' if method.endswith('corrected') else 'observed'
        for count, part in group.groupby('event_count'):
            axes[row, 0].errorbar(part['position_sigma_mm'], part['iou_mean'], yerr=part['iou_std'],
                                  marker='o', linestyle=styles[kind], color=colors[count], capsize=3,
                                  label=f'{labels[method]}, {count} events')
            axes[row, 1].errorbar(part['position_sigma_mm'], part['localization_error_mm_mean'],
                                  yerr=part['localization_error_mm_std'], marker='o', linestyle=styles[kind],
                                  color=colors[count], capsize=3, label=f'{labels[method]}, {count} events')
        axes[row, 0].set_title('Fixed 32³ voxel' if row == 0 else 'Adaptive 16³ to 32³ (40% refined)')
    for row in range(2):
        axes[row, 0].set(ylabel='IoU (higher is better)')
        axes[row, 1].set(ylabel='Localization error [mm] (lower is better)')
        for col in range(2):
            axes[row, col].set(xlabel='Hit-position uncertainty sigma [mm]')
            axes[row, col].grid(alpha=0.25)
            axes[row, col].legend(fontsize=6.3, ncol=2)
    fig.suptitle('Uncertainty-aware scattering-angle correction under detector hit-position error')
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    layer_spacing_mm = same_side_layer_spacing_mm(data)
    edges = imaging_edges(32)
    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    rng = np.random.default_rng(RANDOM_SEED + 23)
    rows = []

    for count in [n for n in EVENT_COUNTS if n <= len(data)]:
        for repeat in range(REPEATS):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
            normal_draws = {
                f'{prefix}_{axis}': rng.normal(0.0, 1.0, len(subset))
                for prefix in ['up1', 'up2', 'down1', 'down2'] for axis in ['x', 'y']
            }
            for sigma in POSITION_SIGMAS_MM:
                noisy = shared_noise_data(subset, sigma, normal_draws)
                observed = prepare_events(noisy)
                corrected = corrected_events(noisy, sigma, layer_spacing_mm)
                for method, fraction in METHODS.items():
                    events = corrected if method.endswith('corrected') else observed
                    if fraction is None:
                        volume, _ = fixed_fine_volume(events)
                        refined, leaves = np.nan, 32 ** 3
                    else:
                        volume, _, refined, leaves = adaptive_volume(events, fraction)
                    image = volume.sum(axis=2)
                    iou, error, contrast, truth_pixels = evaluate_image(image, x_centers, y_centers)
                    rows.append({
                        'method': method, 'event_count': count, 'repeat': repeat + 1,
                        'position_sigma_mm': sigma,
                        'theta_noise_floor_mrad': theta_noise_floor_mrad(sigma, layer_spacing_mm),
                        'selected_events': len(events), 'refined_coarse_voxels': refined,
                        'leaf_voxels': leaves, 'iou_equal_area': iou,
                        'localization_error_mm': error, 'target_background_contrast': contrast,
                        'truth_u_pixels': truth_pixels,
                    })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / 'uncertainty_aware_metrics_by_repeat.csv', index=False)
    summary = metrics.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'count'), theta_noise_floor_mrad=('theta_noise_floor_mrad', 'mean'),
        selected_events_mean=('selected_events', 'mean'), selected_events_std=('selected_events', 'std'),
        iou_mean=('iou_equal_area', 'mean'), iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
    )
    summary.to_csv(OUT_DIR / 'uncertainty_aware_metrics_summary.csv', index=False)
    plot_results(summary, OUT_DIR / 'uncertainty_aware_angle_metrics.png')
    config = {
        'input_csv': str(INPUT_CSV), 'event_counts': EVENT_COUNTS, 'repeats': REPEATS,
        'position_sigmas_mm': POSITION_SIGMAS_MM, 'theta_min_mrad': THETA_MIN_MRAD,
        'theta_cap_mrad': THETA_CAP_MRAD, 'layer_spacing_mm': layer_spacing_mm,
        'theta_noise_formula_mrad': 'sqrt(8) * sigma_mm / layer_spacing_mm * 1000',
        'theta_correction': 'sqrt(max(theta_observed_mrad^2 - theta_noise_mrad^2, 0))',
        'note': 'Truth is used only by evaluation metrics, never by filtering or refinement.',
    }
    (OUT_DIR / 'uncertainty_aware_experiment_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'同侧层间距：{layer_spacing_mm:.3f} mm')
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()
