"""固定事件质量规则后，比较同一图像上的四种定位器。

目的：区分“事件筛选改变了图像”和“定位器从一个热点跳到另一个热点”。
不调参；四种定位器全部来自已有代码：全图质心、固定24格窗口、
多尺度窗口、95% 正像素截断后多尺度窗口。所有定位器只见重建图，
真值 U 型掩膜仅用于最后算误差和 IoU。

每个组合先与 continuous_event_quality_by_repeat.csv 严格校验。
重放不一致时停止运行，不得用不同事件子集做配对比较。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from localize_target_region import (
    localize_multiscale_window, localize_robust_multiscale_window,
    localize_sliding_window,
)
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges
from run_continuous_event_quality_experiment import (
    EVENT_COUNTS, REFINE_FRACTION, REPEATS, SCENE_NAMES, SCENARIOS,
    SIGMAS_MM, estimate_quality, evaluate,
)
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'muon_analysis/continuous_event_quality_results/continuous_event_quality_by_repeat.csv'
OUTPUT = ROOT / 'muon_analysis/event_quality_localizer_ablation_results'
METHODS = ('hard', 'majority')
LOCATORS = ('global', 'fixed24', 'multiscale', 'robust_multiscale')


def centroid(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """全图散射强度质心，不读取目标真值。"""
    xgrid, ygrid = np.meshgrid(x, y, indexing='ij')
    total = float(image.sum())
    return float((image * xgrid).sum() / total), float((image * ygrid).sum() / total)


def locate_all(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """同一张图同时交给四种固定定位器，不在验证集上选最佳。"""
    gx, gy = centroid(image, x, y)
    fixed = localize_sliding_window(image, x, y)
    multi = localize_multiscale_window(image, x, y)
    robust = localize_robust_multiscale_window(image, x, y)
    return {'global': (gx, gy),
            'fixed24': (fixed['x_mm'], fixed['y_mm']),
            'multiscale': (multi['x_mm'], multi['y_mm']),
            'robust_multiscale': (robust['x_mm'], robust['y_mm']),
            'multi_window_bins': multi['window_bins'],
            'robust_window_bins': robust['window_bins'],
            'multi_window_start': (multi['window_start_x'], multi['window_start_y']),
            'robust_window_start': (robust['window_start_x'], robust['window_start_y'])}


def validate_original(row: dict, original: pd.DataFrame) -> None:
    """IoU、全图和多尺度窗口误差均应逐行匹配旧实验。"""
    key = ((original.scenario == row['scenario']) &
           (original.event_count == row['event_count']) &
           (original.repeat == row['repeat']) &
           (np.isclose(original.position_sigma_mm, row['position_sigma_mm'])) &
           (original.method == row['method']))
    old = original.loc[key]
    if len(old) != 1:
        raise AssertionError(f'参考结果键不唯一或缺失: {row}')
    for new_field, old_field in (('iou', 'iou'),
                                 ('global_error_mm', 'global_error_mm'),
                                 ('multiscale_error_mm', 'auto_error_mm')):
        if not np.isclose(row[new_field], old.iloc[0][old_field], rtol=1e-9, atol=1e-9):
            raise AssertionError(f'{row["scenario"]}/{row["method"]}: {new_field} 不匹配')


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """平均误差之外保留 90% 分位数，观察窗口跳变的长尾风险。"""
    keys = ['scenario', 'event_count', 'position_sigma_mm', 'method']
    agg = {'repeats': ('repeat', 'size'), 'iou_mean': ('iou', 'mean')}
    for locator in LOCATORS:
        field = f'{locator}_error_mm'
        agg[f'{locator}_error_mean_mm'] = (field, 'mean')
        agg[f'{locator}_error_p90_mm'] = (field, lambda s: float(np.quantile(s, .9)))
    return raw.groupby(keys, as_index=False).agg(**agg)


def plot_5000(summary: pd.DataFrame) -> None:
    """五场景平均误差；异常场景仍须看原始 CSV，不能靠平均值下结论。"""
    part = summary[summary.event_count == 5000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        current = part[part.method == method]
        for locator in LOCATORS:
            line = current.groupby('position_sigma_mm', as_index=False)[
                f'{locator}_error_mean_mm'].mean()
            ax.plot(line.position_sigma_mm, line[f'{locator}_error_mean_mm'],
                    marker='o', label=locator)
        ax.set_title(f'{method}: 5000 input events')
        ax.set(xlabel='Hit-position sigma [mm]', ylabel='Mean localization error [mm]')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Same reconstructed images, four fixed localization rules')
    fig.savefig(OUTPUT / 'localizer_ablation_5000.png', dpi=180)
    plt.close(fig)


def main() -> None:
    reference = pd.read_csv(REFERENCE)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    xgrid, ygrid = np.meshgrid(cx, cy, indexing='ij')
    records = []
    for scene in SCENE_NAMES:
        csv_path, mask_fn = SCENARIOS[scene]
        data = pd.read_csv(csv_path)
        truth = mask_fn(cx, cy)  # 只用于下面的指标计算。
        true_xy = (float(xgrid[truth].mean()), float(ygrid[truth].mean()))
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 23)
        for count in EVENT_COUNTS:
            if count > len(data):
                continue
            for repeat in range(REPEATS):
                subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
                noise_draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                               for prefix in ('up1', 'up2', 'down1', 'down2')
                               for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, noise_draws)
                    observed = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
                    nominal = corrected_events(observed, sigma, spacing)
                    posterior_rng = np.random.default_rng(
                        RANDOM_SEED + 100000 * count + 1000 * repeat + int(sigma * 1000))
                    probability, _ = estimate_quality(measured, sigma, spacing, posterior_rng)
                    passing = probability[nominal.index.to_numpy(dtype=int)]
                    variants = {'hard': nominal,
                                'majority': nominal.loc[passing >= .50].copy()}
                    for method, events in variants.items():
                        volume, _, _, _ = adaptive_volume(events, REFINE_FRACTION)
                        image = volume.sum(axis=2)
                        locations = locate_all(image, cx, cy)
                        metrics = evaluate(image, truth, cx, cy)
                        row = {'scenario': scene, 'event_count': count,
                               'repeat': repeat + 1, 'position_sigma_mm': sigma,
                               'method': method, 'iou': metrics['iou'],
                               'multi_window_bins': locations['multi_window_bins'],
                               'robust_window_bins': locations['robust_window_bins'],
                               'multi_window_start_x': locations['multi_window_start'][0],
                               'multi_window_start_y': locations['multi_window_start'][1]}
                        for locator in LOCATORS:
                            point = locations[locator]
                            row[f'{locator}_x_mm'] = point[0]
                            row[f'{locator}_y_mm'] = point[1]
                            row[f'{locator}_error_mm'] = float(np.hypot(
                                point[0] - true_xy[0], point[1] - true_xy[1]))
                        validate_original(row, reference)
                        records.append(row)
        print(f'Completed and verified {scene}', flush=True)
    raw = pd.DataFrame(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTPUT / 'localizer_ablation_by_repeat.csv', index=False)
    summary = summarize(raw)
    summary.to_csv(OUTPUT / 'localizer_ablation_summary.csv', index=False)
    plot_5000(summary)
    (OUTPUT / 'localizer_ablation_config.json').write_text(json.dumps({
        'source_csv': str(REFERENCE), 'scenarios': SCENE_NAMES,
        'event_counts': EVENT_COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'methods': METHODS, 'locators': LOCATORS,
        'parameters_changed_for_this_test': False,
        'truth_usage': 'evaluation only; none of the locators reads target truth',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(raw[raw.position_sigma_mm > 0].groupby('method')[[
        f'{name}_error_mm' for name in LOCATORS]].mean().round(3).to_string())


if __name__ == '__main__':
    main()
