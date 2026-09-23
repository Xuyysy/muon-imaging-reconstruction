"""重放事件质量实验中“IoU 变好而自动定位变差”的样例。

这些案例从已有结果中事后挑选，仅用于查找机制，不用于证明新规则泛化。
重放时复用原实验的事件抽样顺序、探测器噪声和重复扰动种子；
若与原始逐重复 CSV 的 IoU/误差对不上，脚本会报错，而不会输出误导性图片。

目标真值只用于最后绘制轮廓和评价误差，不进入事件选择、体素重建或窗口定位。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from localize_target_region import localize_multiscale_window, smooth_3x3, MULTISCALE_WINDOW_BINS
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges
from run_continuous_event_quality_experiment import (
    EVENT_COUNTS, REFINE_FRACTION, SCENARIOS, estimate_quality, evaluate,
)
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'muon_analysis/continuous_event_quality_results/continuous_event_quality_by_repeat.csv'
OUTPUT = ROOT / 'muon_analysis/event_quality_failure_diagnosis'
METHODS = ('hard', 'majority')
# 仅用于诊断的已观察失败样例；不从这里选择未来算法参数。
CASES = (
    ('offcenter', 5000, 1, .10),
    ('offcenter', 5000, 2, .30),
    ('small', 3000, 4, .30),
)


def replay_subset(scene: str, count: int, repeat: int) -> tuple[pd.DataFrame, dict, float]:
    """严格走过先前抽样，以获得与原实验相同的目标重复。"""
    path, _ = SCENARIOS[scene]
    data = pd.read_csv(path)
    rng = np.random.default_rng(RANDOM_SEED + 23)
    for current_count in EVENT_COUNTS:
        if current_count > len(data):
            continue
        for current_repeat in range(1, 6):
            subset = data.iloc[rng.choice(len(data), size=current_count, replace=False)].copy()
            draws = {f'{prefix}_{axis}': rng.normal(0, 1, current_count)
                     for prefix in ('up1', 'up2', 'down1', 'down2')
                     for axis in ('x', 'y')}
            if current_count == count and current_repeat == repeat:
                return subset, draws, same_side_layer_spacing_mm(data)
    raise ValueError('目标事件数或重复编号不在原实验范围内')


def window_candidates(image: np.ndarray) -> dict[int, dict]:
    """记录每个候选尺度的最高分窗口，用于辨别尺度/位置跳变。"""
    smooth = smooth_3x3(image)
    background = float(smooth.mean())
    candidates = {}
    for size in MULTISCALE_WINDOW_BINS:
        windows = np.lib.stride_tricks.sliding_window_view(smooth, (size, size))
        score_map = (windows.mean(axis=(-2, -1)) - background) * size
        ix, iy = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
        candidates[size] = {'start_x': int(ix), 'start_y': int(iy),
                            'score': float(score_map[ix, iy])}
    return candidates


def reconstruct(scene: str, count: int, repeat: int, sigma: float) -> tuple[list[dict], dict]:
    """只从已测四层击中构造两种图，最后才接触真值掩膜。"""
    subset, noise_draws, spacing = replay_subset(scene, count, repeat)
    measured = measured_hits(subset, sigma, noise_draws)
    observed = recompute_from_noisy_hits(measured, 0.0, np.random.default_rng(0))
    nominal = corrected_events(observed, sigma, spacing)
    posterior_rng = np.random.default_rng(
        RANDOM_SEED + 100000 * count + 1000 * (repeat - 1) + int(sigma * 1000)
    )
    probability, _ = estimate_quality(measured, sigma, spacing, posterior_rng)
    passing = probability[nominal.index.to_numpy(dtype=int)]
    variants = {'hard': nominal, 'majority': nominal.loc[passing >= .50].copy()}

    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    images = {}
    outputs = []
    # 评价掩膜只在此处生成，不传给任何重建或窗口选择函数。
    _, mask_fn = SCENARIOS[scene]
    truth = mask_fn(cx, cy)
    xgrid, ygrid = np.meshgrid(cx, cy, indexing='ij')
    true_x, true_y = float(xgrid[truth].mean()), float(ygrid[truth].mean())
    for method in METHODS:
        events = variants[method]
        volume, _, _, _ = adaptive_volume(events, REFINE_FRACTION)
        image = volume.sum(axis=2)
        images[method] = image
        localization = localize_multiscale_window(image, cx, cy)
        metrics = evaluate(image, truth, cx, cy)
        target_weight_fraction = float(image[truth].sum() / image.sum())
        candidates = window_candidates(image)
        outputs.append({
            'scenario': scene, 'event_count': count, 'repeat': repeat,
            'position_sigma_mm': sigma, 'method': method,
            'contributing_events': len(events),
            'iou': metrics['iou'],
            'global_error_mm': metrics['global_error_mm'],
            'auto_error_mm': metrics['auto_error_mm'],
            'target_weight_fraction': target_weight_fraction,
            'true_x_mm': true_x, 'true_y_mm': true_y,
            'window_x_mm': localization['x_mm'],
            'window_y_mm': localization['y_mm'],
            'window_bins': localization['window_bins'],
            'window_start_x': localization['window_start_x'],
            'window_start_y': localization['window_start_y'],
            'window_score': localization['score'],
            'window_weight_fraction': localization['window_weight_fraction'],
            'candidate_scores': json.dumps(candidates),
        })
    return outputs, {'images': images, 'truth': truth, 'cx': cx, 'cy': cy, 'edges': edges}


def verify_replay(rows: list[dict], reference: pd.DataFrame) -> None:
    """重放必须数值匹配旧实验，才能解释对应的失败案例。"""
    for row in rows:
        selected = reference[
            (reference.scenario == row['scenario']) &
            (reference.event_count == row['event_count']) &
            (reference.repeat == row['repeat']) &
            (np.isclose(reference.position_sigma_mm, row['position_sigma_mm'])) &
            (reference.method == row['method'])
        ]
        if len(selected) != 1:
            raise AssertionError(f'原始逐重复结果缺少唯一记录: {row}')
        old = selected.iloc[0]
        for metric in ('iou', 'global_error_mm', 'auto_error_mm'):
            if not np.isclose(row[metric], old[metric], rtol=1e-9, atol=1e-9):
                raise AssertionError(
                    f'{row["scenario"]}/{row["method"]}: {metric} '
                    f'重放值 {row[metric]} 与原值 {old[metric]} 不符'
                )


def plot_case(rows: list[dict], state: dict) -> None:
    """同一色标展示图像、真值轮廓和自动选中的窗口。"""
    images, truth = state['images'], state['truth']
    cx, cy, edges = state['cx'], state['cy'], state['edges']
    upper = max(float(np.quantile(im, .995)) for im in images.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    for ax, row in zip(axes, rows):
        image = images[row['method']]
        ax.imshow(image.T, origin='lower', cmap='inferno', vmin=0, vmax=upper,
                  extent=(edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]))
        ax.contour(cx, cy, truth.T, levels=[.5], colors=['cyan'], linewidths=1.2)
        size = row['window_bins']
        start_x, start_y = row['window_start_x'], row['window_start_y']
        ax.add_patch(Rectangle((edges[0][start_x], edges[1][start_y]),
                               edges[0][start_x + size] - edges[0][start_x],
                               edges[1][start_y + size] - edges[1][start_y],
                               fill=False, edgecolor='lime', linewidth=2))
        ax.plot(row['true_x_mm'], row['true_y_mm'], marker='+', color='cyan', markersize=12)
        ax.plot(row['window_x_mm'], row['window_y_mm'], marker='x', color='lime', markersize=9)
        ax.set_title(f"{row['method']}: IoU {row['iou']:.3f}; auto error {row['auto_error_mm']:.1f} mm")
        ax.set(xlabel='x [mm]', ylabel='y [mm]', xlim=(-90, 90), ylim=(-90, 90))
        ax.set_aspect('equal')
    case = rows[0]
    fig.suptitle(f"{case['scenario']}: {case['event_count']} events, repeat {case['repeat']}, "
                 f"sigma={case['position_sigma_mm']:.2f} mm\n"
                 'cyan = truth boundary/centroid; green = selected window/centroid')
    filename = (f"{case['scenario']}_{case['event_count']}_r{case['repeat']}_"
                f"sigma{case['position_sigma_mm']:.2f}.png")
    fig.savefig(OUTPUT / filename, dpi=170)
    plt.close(fig)


def main() -> None:
    reference = pd.read_csv(REFERENCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for scene, count, repeat, sigma in CASES:
        rows, state = reconstruct(scene, count, repeat, sigma)
        verify_replay(rows, reference)
        plot_case(rows, state)
        records.extend(rows)
        print(f'Replay verified: {scene}, N={count}, repeat={repeat}, sigma={sigma}', flush=True)
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT / 'diagnostic_cases.csv', index=False)
    (OUTPUT / 'diagnostic_config.json').write_text(json.dumps({
        'cases': CASES, 'reference': str(REFERENCE), 'methods': METHODS,
        'truth_usage': 'retrospective scoring and plot overlays only',
        'case_selection': 'post-hoc failures; diagnosis only, not validation',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(table[['scenario', 'event_count', 'repeat', 'position_sigma_mm',
                 'method', 'iou', 'global_error_mm', 'auto_error_mm',
                 'window_bins', 'window_start_x', 'window_start_y',
                 'target_weight_fraction']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
