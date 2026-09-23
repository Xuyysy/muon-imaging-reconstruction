"""逐事件轨迹/PoCA 不确定度传播：与原方法、硬修正对照。

把每条*已测得*的四层击中坐标当作高斯位置测量结果；围绕该测量值
重复抽样可能的真实击中坐标，分别拟合轨迹并计算散射角和 PoCA。
每个抽样只承担原事件的 1/K 权重，因此没有把一条缪子当成 K 条缪子。

这是简化的测量误差传播，不是完整贝叶斯后验：忽略了探测层对齐、
多重散射的轨迹过程噪声、动量分布和非高斯探测器响应。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import (
    RANDOM_SEED, THETA_CAP_MRAD, THETA_MIN_MRAD, XY_LIMIT_MM, Z_LIMIT_MM,
    adaptive_volume, imaging_edges, prepare_events,
)
from run_baseline_experiment import evaluate_image
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'event_uncertainty_results'
EVENT_COUNTS = (1000, 3000, 5000)
POSITION_SIGMAS_MM = (0.0, 0.05, 0.10, 0.30, 1.0)
REPEATS = 5
POSTERIOR_DRAWS = 12
MIN_PASS_FRACTION = .50
REFINE_FRACTION = .40


def measured_hits(subset: pd.DataFrame, sigma_mm: float, draws: dict[str, np.ndarray]) -> pd.DataFrame:
    """从理想 Geant4 击中生成一次探测器真正测到的四层击中。"""
    measured = subset.copy()
    for prefix in ('up1', 'up2', 'down1', 'down2'):
        for axis in ('x', 'y'):
            measured[f'{prefix}_{axis}_mm'] += sigma_mm * draws[f'{prefix}_{axis}']
    return measured


def propagated_events(measured: pd.DataFrame, sigma_mm: float, rng: np.random.Generator):
    """对每条测量轨迹生成 K 个候选散射点，并估计其通过筛选的概率。

    不用真值位置进行任何筛选；PoCA 点若落在成像范围外，只作为失败抽样。
    通过筛选的抽样权重除以 K，表示这条事件对三维图的期望贡献。
    返回的 p_pass 和 xy_spread 仅用于诊断、解释，不参与重建决策。
    """
    n = len(measured)
    if sigma_mm == 0:
        nominal = recompute_from_noisy_hits(measured, 0.0, rng)
        selected = prepare_events(nominal)
        passed = selected.index.to_numpy()
        return selected, selected.copy(), len(passed) / n, 0.0, len(passed)

    candidates = []
    locations = np.full((POSTERIOR_DRAWS, n, 2), np.nan)
    passing = np.zeros((POSTERIOR_DRAWS, n), dtype=bool)
    for draw_index in range(POSTERIOR_DRAWS):
        sample = recompute_from_noisy_hits(measured, sigma_mm, rng)
        # prepare_events 完全复用上一阶段的 ROI、散射角门槛及 3 mrad 截断。
        valid = prepare_events(sample)
        ids = valid.index.to_numpy()
        passing[draw_index, ids] = True
        locations[draw_index, ids, 0] = valid['poca_x_mm'].to_numpy()
        locations[draw_index, ids, 1] = valid['poca_y_mm'].to_numpy()
        valid['capped_weight'] /= POSTERIOR_DRAWS
        valid['event_id'] = ids
        candidates.append(valid)

    event_rows = pd.concat(candidates, ignore_index=True)
    valid_counts = passing.sum(axis=0)
    # 多数通过门槛仅使用事件自身的重复测量结果，不能读取目标真值。
    majority_ids = np.flatnonzero(valid_counts / POSTERIOR_DRAWS >= MIN_PASS_FRACTION)
    majority_rows = event_rows[event_rows['event_id'].isin(majority_ids)].copy()
    # 对至少两次有效抽样的事件，计算 PoCA xy 位置围绕其均值的 RMS 扩散。
    # 这里用普通 numpy 运算并显式处理空样本，避免运行时警告。
    stable = valid_counts >= 2
    spread = np.nan
    if stable.any():
        loc = locations[:, stable, :]
        mask = passing[:, stable, None]
        count = valid_counts[stable][None, :, None]
        center = np.nansum(loc, axis=0, keepdims=True) / count
        radius_sq = np.where(mask, (loc - center) ** 2, 0.0).sum(axis=2)
        rms = np.sqrt(radius_sq.sum(axis=0) / valid_counts[stable])
        spread = float(np.median(rms))
    return event_rows, majority_rows, float(passing.mean()), spread, int((valid_counts > 0).sum())


def plot_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.5), constrained_layout=True)
    names = {'observed': 'Observed angle', 'hard': 'Hard correction',
             'propagated': 'All propagated candidates',
             'majority': 'Propagation + majority gate'}
    colors = {'observed': '#7f7f7f', 'hard': '#1f77b4',
              'propagated': '#d62728', 'majority': '#2ca02c'}
    for row, count in enumerate(EVENT_COUNTS):
        for method in names:
            part = summary[(summary.event_count == count) & (summary.method == method)]
            if part.empty:
                continue
            axes[row, 0].errorbar(part.position_sigma_mm, part.iou_mean, yerr=part.iou_std,
                                  marker='o', capsize=2, label=names[method], color=colors[method])
            axes[row, 1].errorbar(part.position_sigma_mm, part.localization_error_mm_mean,
                                  yerr=part.localization_error_mm_std, marker='o', capsize=2,
                                  label=names[method], color=colors[method])
        axes[row, 0].set_ylabel(f'{count} events\nIoU (higher better)')
        axes[row, 1].set_ylabel('Localization error [mm]\n(lower better)')
        for axis in axes[row]:
            axis.set_xlabel('Hit-position sigma [mm]')
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
    fig.suptitle('Adaptive PoCA: event-level detector uncertainty propagation')
    fig.savefig(OUT_DIR / 'event_uncertainty_metrics.png', dpi=170, bbox_inches='tight')
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
            # 与前两轮实验保持相同的随机种子、事件抽样和探测器扰动。
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                     for prefix in ('up1', 'up2', 'down1', 'down2') for axis in ('x', 'y')}
            for sigma in POSITION_SIGMAS_MM:
                hits = measured_hits(subset, sigma, draws)
                observed_data = recompute_from_noisy_hits(hits, 0.0, np.random.default_rng(0))
                # 后验抽样 RNG 单独固定，避免改变下一组原始事件抽样。
                posterior_rng = np.random.default_rng(RANDOM_SEED + 100000 * count + 1000 * repeat + int(sigma * 1000))
                propagated, majority, pass_probability, xy_spread, covered_events = propagated_events(
                    hits, sigma, posterior_rng
                )
                variants = {
                    'observed': prepare_events(observed_data),
                    'hard': corrected_events(observed_data, sigma, spacing),
                    'propagated': propagated,
                    'majority': majority,
                }
                for method, events in variants.items():
                    volume, _, refined, leaves = adaptive_volume(events, REFINE_FRACTION)
                    iou, error, contrast, truth_pixels = evaluate_image(volume.sum(axis=2), *centers)
                    # 抽样法会生成一条原事件的多个候选点，因此同时记录“候选行数”
                    # 和“真正贡献过权重的独立缪子数”，避免误把 K 次抽样当成 K 条缪子。
                    unique_events = (int(events.event_id.nunique())
                                     if method in ('propagated', 'majority') and sigma > 0
                                     else len(events))
                    records.append({
                        'method': method, 'event_count': count, 'repeat': repeat + 1,
                        'position_sigma_mm': sigma, 'selected_rows': len(events),
                        'contributing_unique_events': unique_events,
                        'event_pass_probability_mean': pass_probability if method in ('propagated', 'majority') else np.nan,
                        'poca_xy_spread_median_mm': xy_spread if method in ('propagated', 'majority') else np.nan,
                        'events_with_valid_candidate': covered_events if method in ('propagated', 'majority') else np.nan,
                        'refined_coarse_voxels': refined, 'leaf_voxels': leaves,
                        'iou_equal_area': iou, 'localization_error_mm': error,
                        'target_background_contrast': contrast, 'truth_u_pixels': truth_pixels,
                    })
    results = pd.DataFrame(records)
    results.to_csv(OUT_DIR / 'event_uncertainty_metrics_by_repeat.csv', index=False)
    summary = results.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'), iou_mean=('iou_equal_area', 'mean'),
        iou_std=('iou_equal_area', 'std'), localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
        contributing_unique_events_mean=('contributing_unique_events', 'mean'),
        event_pass_probability_mean=('event_pass_probability_mean', 'mean'),
        poca_xy_spread_median_mm=('poca_xy_spread_median_mm', 'mean'),
        events_with_valid_candidate_mean=('events_with_valid_candidate', 'mean'),
    )
    summary.to_csv(OUT_DIR / 'event_uncertainty_metrics_summary.csv', index=False)
    plot_summary(summary)
    config = {
        'input_csv': str(INPUT_CSV), 'seed': RANDOM_SEED + 23,
        'event_counts': EVENT_COUNTS, 'repeats': REPEATS,
        'position_sigmas_mm': POSITION_SIGMAS_MM, 'posterior_draws_per_event': POSTERIOR_DRAWS,
        'theta_min_mrad': THETA_MIN_MRAD, 'theta_cap_mrad': THETA_CAP_MRAD,
        'refine_fraction': REFINE_FRACTION, 'layer_spacing_mm': spacing,
        'majority_min_pass_fraction': MIN_PASS_FRACTION,
        'weight_per_candidate': 'min(theta_candidate^2, theta_cap^2) / posterior_draws',
        'note': 'Each candidate is one measurement-error sample; target truth is used only for metrics.',
    }
    (OUT_DIR / 'event_uncertainty_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[(summary.event_count == 5000) & summary.position_sigma_mm.isin([.05, .10, .30])].to_string(index=False))


if __name__ == '__main__':
    main()
