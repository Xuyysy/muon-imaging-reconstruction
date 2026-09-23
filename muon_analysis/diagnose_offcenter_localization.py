"""诊断偏心 U 目标在探测器误差下的定位偏差。

本脚本只分析已有重建图的强度质心，不更改筛选和重建规则。
目标掩膜仅用于事后把图像强度分成“目标区域内/外”，绝不参与重建。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges, prepare_events
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits, propagated_events
from run_offcenter_generalization import shifted_u_mask
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks_offcenter.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'offcenter_localization_diagnosis'
POSITION_SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
EVENT_COUNTS = (1000, 3000, 5000)
TARGET_COUNT = 5000
REPEATS = 5
METHODS = ('observed', 'hard', 'majority')


def weighted_centroid(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """计算给定二维强度图在 x/y 上的加权质心。"""
    total = float(image.sum())
    if total <= 0:
        return np.nan, np.nan
    return float((image * x).sum() / total), float((image * y).sum() / total)


def summarize_image(image: np.ndarray, truth: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """将全图质心拆成目标区域内、外强度的贡献。"""
    true_x = float(x[truth].mean())
    true_y = float(y[truth].mean())
    rec_x, rec_y = weighted_centroid(image, x, y)
    target_image = np.where(truth, image, 0.0)
    background_image = np.where(~truth, image, 0.0)
    in_x, in_y = weighted_centroid(target_image, x, y)
    out_x, out_y = weighted_centroid(background_image, x, y)
    target_fraction = float(target_image.sum() / image.sum()) if image.sum() else np.nan
    # 用非零图像像素的中位数估计散射背景强度；只使用重建图本身，
    # 不使用 U 型掩膜，也不使用预设目标面积。中位数对少数热点不敏感。
    positive = image[image > 0]
    background_level = float(np.median(positive)) if positive.size else 0.0
    excess = np.maximum(image - background_level, 0.0)
    corrected_x, corrected_y = weighted_centroid(excess, x, y)
    # 全图质心 = f×区域内质心 + (1−f)×区域外质心；验证该恒等式，防止方向搞错。
    if np.isfinite(target_fraction) and np.isfinite(in_x) and np.isfinite(out_x):
        assert np.allclose([rec_x, rec_y],
                           [target_fraction * in_x + (1 - target_fraction) * out_x,
                            target_fraction * in_y + (1 - target_fraction) * out_y])
    return {
        'true_x_mm': true_x, 'true_y_mm': true_y,
        'reconstructed_x_mm': rec_x, 'reconstructed_y_mm': rec_y,
        'bias_x_mm': rec_x - true_x, 'bias_y_mm': rec_y - true_y,
        'localization_error_mm': float(np.hypot(rec_x - true_x, rec_y - true_y)),
        'target_region_weight_fraction': target_fraction,
        'inside_centroid_x_mm': in_x, 'inside_centroid_y_mm': in_y,
        'outside_centroid_x_mm': out_x, 'outside_centroid_y_mm': out_y,
        'outside_pull_x_mm': (1 - target_fraction) * (out_x - in_x),
        'outside_pull_y_mm': (1 - target_fraction) * (out_y - in_y),
        'background_level': background_level,
        'background_corrected_x_mm': corrected_x,
        'background_corrected_y_mm': corrected_y,
        'background_corrected_error_mm': float(np.hypot(corrected_x - true_x, corrected_y - true_y)),
    }


def plot_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    colors = {'observed': '#7f7f7f', 'hard': '#1f77b4', 'majority': '#2ca02c'}
    for method in METHODS:
        part = summary[summary.method == method]
        x = part.position_sigma_mm
        axes[0, 0].errorbar(x, part.reconstructed_x_mm_mean, yerr=part.reconstructed_x_mm_std,
                            marker='o', color=colors[method], label=method)
        axes[0, 1].errorbar(x, part.reconstructed_y_mm_mean, yerr=part.reconstructed_y_mm_std,
                            marker='o', color=colors[method], label=method)
        axes[1, 0].errorbar(x, part.target_region_weight_fraction_mean,
                            yerr=part.target_region_weight_fraction_std, marker='o',
                            color=colors[method], label=method)
        axes[1, 1].errorbar(x, part.localization_error_mm_mean,
                            yerr=part.localization_error_mm_std, marker='o',
                            color=colors[method], label=method)
    true_x = float(summary.true_x_mm_mean.iloc[0])
    true_y = float(summary.true_y_mm_mean.iloc[0])
    axes[0, 0].axhline(true_x, linestyle='--', color='black', label='target truth')
    axes[0, 1].axhline(true_y, linestyle='--', color='black', label='target truth')
    axes[0, 0].set_ylabel('Reconstructed centroid x [mm]')
    axes[0, 1].set_ylabel('Reconstructed centroid y [mm]')
    axes[1, 0].set_ylabel('Weight fraction inside target mask')
    axes[1, 1].set_ylabel('Localization error [mm]')
    for axis in axes.ravel():
        axis.set_xlabel('Hit-position sigma [mm]')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle('Off-centre target: centroid bias and target/background weight')
    fig.savefig(OUT_DIR / 'offcenter_centroid_diagnosis.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    # 用同一批图像配对比较原质心和背景扣除后的质心，避免靠单张图判断。
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for method in METHODS:
        part = summary[summary.method == method]
        axes[0].errorbar(part.position_sigma_mm, part.localization_error_mm_mean,
                         yerr=part.localization_error_mm_std, marker='o', linestyle='--',
                         color=colors[method], label=f'{method}: original')
        axes[0].errorbar(part.position_sigma_mm, part.background_corrected_error_mm_mean,
                         yerr=part.background_corrected_error_mm_std, marker='o', linestyle='-',
                         color=colors[method], label=f'{method}: background subtracted')
        axes[1].plot(part.position_sigma_mm, part.background_level_mean,
                     marker='o', color=colors[method], label=method)
    axes[0].set(xlabel='Hit-position sigma [mm]', ylabel='Localization error [mm] (lower better)')
    axes[1].set(xlabel='Hit-position sigma [mm]', ylabel='Median positive image intensity')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    fig.suptitle('Off-centre localization: empirical image background subtraction')
    fig.savefig(OUT_DIR / 'offcenter_background_subtraction.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    spacing = same_side_layer_spacing_mm(data)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    x, y = np.meshgrid(cx, cy, indexing='ij')
    truth = shifted_u_mask(cx, cy)
    rng = np.random.default_rng(RANDOM_SEED + 23)
    rows = []
    for count in EVENT_COUNTS:
        for repeat in range(REPEATS):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                     for prefix in ('up1', 'up2', 'down1', 'down2') for axis in ('x', 'y')}
            if count != TARGET_COUNT:
                continue  # 先走过随机抽样，确保与跨场景实验的 5000 条样本完全相同。
            for sigma in POSITION_SIGMAS_MM:
                hits = measured_hits(subset, sigma, draws)
                observed = recompute_from_noisy_hits(hits, 0.0, np.random.default_rng(0))
                posterior_rng = np.random.default_rng(
                    RANDOM_SEED + 100000 * count + 1000 * repeat + int(sigma * 1000)
                )
                _, majority, _, _, _ = propagated_events(hits, sigma, posterior_rng)
                variants = {
                    'observed': prepare_events(observed),
                    'hard': corrected_events(observed, sigma, spacing),
                    'majority': majority,
                }
                for method, events in variants.items():
                    volume, _, _, _ = adaptive_volume(events, .40)
                    record = summarize_image(volume.sum(axis=2), truth, x, y)
                    record.update({'method': method, 'repeat': repeat + 1,
                                   'event_count': count, 'position_sigma_mm': sigma})
                    rows.append(record)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / 'offcenter_centroid_by_repeat.csv', index=False)
    numeric = [column for column in results.select_dtypes('number')
               if column not in ('repeat', 'event_count', 'position_sigma_mm')]
    summary = results.groupby(['method', 'position_sigma_mm'], as_index=False)[numeric].agg(['mean', 'std'])
    summary.columns = [a if not b else f'{a}_{b}' for a, b in summary.columns.to_flat_index()]
    summary.to_csv(OUT_DIR / 'offcenter_centroid_summary.csv', index=False)
    plot_summary(summary)
    (OUT_DIR / 'offcenter_centroid_config.json').write_text(json.dumps({
        'input_csv': str(INPUT_CSV), 'event_count': TARGET_COUNT,
        'repeats': REPEATS, 'position_sigmas_mm': POSITION_SIGMAS_MM,
        'seed': RANDOM_SEED + 23, 'methods': METHODS,
        'note': 'Ground-truth target mask is used only for retrospective diagnosis.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[['method', 'position_sigma_mm', 'true_x_mm_mean', 'true_y_mm_mean',
                   'reconstructed_x_mm_mean', 'reconstructed_y_mm_mean',
                   'target_region_weight_fraction_mean', 'localization_error_mm_mean',
                   'background_corrected_error_mm_mean',
                   'outside_pull_x_mm_mean', 'outside_pull_y_mm_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
