"""诊断高噪声时多尺度定位为何失败；不改变已有重建和定位算法。

问题拆成两类：
1. 少数异常亮点/事件是否占据大部分背景散射权重？
2. 还是大量错误 PoCA 点构成分散背景，使窗口误选别处？

真实 U 型掩膜只在事后计算“目标内/外”诊断量，不传给筛选、重建或定位。
同一重复的不同 sigma 复用同一批事件和标准正态噪声。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from localize_target_region import localize_multiscale_window, smooth_3x3
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import REFINE_FRACTION, measured_hits
from run_robust_localization_experiment import SCENARIOS
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'muon_analysis' / 'hotspot_diagnosis_results'
SCENE_NAMES = ('center', 'offcenter', 'small', 'medium_offset', 'small_offset')
EVENT_COUNT = 5000
REPEATS = 5
SIGMAS_MM = (0.0, 0.10, 0.30)


def top_share(weights: np.ndarray, fraction: float) -> float:
    """最高若干项占其所属总体权重的比例；0 表示没有权重。"""
    positive = weights[weights > 0]
    if positive.size == 0:
        return 0.0
    n_top = max(1, int(np.ceil(positive.size * fraction)))
    return float(np.partition(positive, -n_top)[-n_top:].sum() / positive.sum())


def event_target_share(events: pd.DataFrame, truth: np.ndarray,
                       x_edges: np.ndarray, y_edges: np.ndarray) -> float:
    """按 PoCA 的 xy 位置，把筛选后事件权重分配到真值内外作事后诊断。"""
    if events.empty:
        return np.nan
    x_bin = np.searchsorted(x_edges, events['poca_x_mm'].to_numpy(), side='right') - 1
    y_bin = np.searchsorted(y_edges, events['poca_y_mm'].to_numpy(), side='right') - 1
    x_bin = np.clip(x_bin, 0, truth.shape[0] - 1)
    y_bin = np.clip(y_bin, 0, truth.shape[1] - 1)
    weight = events['capped_weight'].to_numpy()
    return float(weight[truth[x_bin, y_bin]].sum() / weight.sum())


def diagnose(image: np.ndarray, events: pd.DataFrame, truth: np.ndarray,
             x_edges: np.ndarray, y_edges: np.ndarray) -> dict:
    """返回互相独立的集中度、扩散度和位置错误指标。"""
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    true_x = float(x_grid[truth].mean())
    true_y = float(y_grid[truth].mean())
    auto = localize_multiscale_window(image, x_centers, y_centers)
    size = auto['window_bins']
    start_x, start_y = auto['window_start_x'], auto['window_start_y']
    selected = np.zeros_like(truth)
    selected[start_x:start_x + size, start_y:start_y + size] = True

    background = image[~truth]
    positive_background = background[background > 0]
    # 这个比例由真值定义，仅用于解释错误原因；重建时绝对不可使用。
    target_fraction = float(image[truth].sum() / image.sum())
    background_image = np.where(~truth, image, 0.0)
    peak_background_3x3 = float(smooth_3x3(background_image).max() * 9)
    window_total = float(image[selected].sum())
    weights = events['capped_weight'].to_numpy()
    return {
        'selected_events': len(events),
        'event_max_weight_fraction': float(weights.max() / weights.sum()),
        'event_top5_weight_fraction': top_share(weights, .05),
        'event_target_weight_fraction': event_target_share(events, truth, x_edges, y_edges),
        'target_image_weight_fraction': target_fraction,
        'background_occupied_pixels': int(positive_background.size),
        'background_occupied_fraction': float(positive_background.size / background.size),
        'background_top5_pixel_weight_fraction': top_share(positive_background, .05),
        'background_peak3x3_weight_fraction': float(
            peak_background_3x3 / background_image.sum()
        ) if background_image.sum() > 0 else 0.0,
        'auto_window_bins': size,
        'auto_error_mm': float(np.hypot(auto['x_mm'] - true_x, auto['y_mm'] - true_y)),
        'auto_window_background_fraction': float(
            image[selected & ~truth].sum() / window_total
        ) if window_total > 0 else np.nan,
        'auto_window_x_mm': auto['x_mm'],
        'auto_window_y_mm': auto['y_mm'],
        'true_x_mm': true_x, 'true_y_mm': true_y,
    }


def plot_examples(examples: dict, edges: list[np.ndarray]) -> None:
    """同一批事件的三张图：白色真值轮廓与蓝色自动窗口只用于可视化。"""
    fig, axes = plt.subplots(1, len(SIGMAS_MM), figsize=(13, 4.2),
                             constrained_layout=True)
    x_edges, y_edges = edges[:2]
    for ax, sigma in zip(axes, SIGMAS_MM):
        image, truth, row = examples[sigma]
        ax.imshow(np.log1p(image).T, origin='lower', cmap='inferno',
                  extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                  aspect='equal')
        ax.contour((x_edges[:-1] + x_edges[1:]) / 2,
                   (y_edges[:-1] + y_edges[1:]) / 2, truth.T,
                   levels=[.5], colors='white', linewidths=1)
        size = row['auto_window_bins']
        ix = row['window_start_x']
        iy = row['window_start_y']
        ax.add_patch(Rectangle((x_edges[ix], y_edges[iy]),
                               x_edges[ix + size] - x_edges[ix],
                               y_edges[iy + size] - y_edges[iy],
                               fill=False, edgecolor='#49b8ff', linewidth=2))
        ax.scatter([row['auto_window_x_mm']], [row['auto_window_y_mm']],
                   color='#49b8ff', marker='x', s=50)
        ax.set_title(f'sigma={sigma:g} mm; error={row["auto_error_mm"]:.1f} mm')
        ax.set_xlabel('PoCA x [mm]')
        ax.set_ylabel('PoCA y [mm]')
    fig.suptitle('Small offset: truth outline (white), selected window (blue); repeat 5')
    fig.savefig(OUT_DIR / 'small_offset_same_events_noise_progression.png', dpi=180)
    plt.close(fig)


def plot_metrics(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    metrics = [
        ('target_image_weight_fraction_mean', 'Weight inside target'),
        ('background_occupied_fraction_mean', 'Occupied background pixels'),
        ('background_top5_pixel_weight_fraction_mean',
         'Top 5% background-pixel weight share'),
        ('auto_error_mm_mean', 'Auto-window error [mm]'),
    ]
    for ax, (column, ylabel) in zip(axes.ravel(), metrics):
        for scene, part in summary.groupby('scenario'):
            ax.errorbar(part.position_sigma_mm, part[column],
                        yerr=part[column.replace('_mean', '_std')],
                        marker='o', capsize=2, label=scene)
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Does high-noise failure come from isolated peaks or diffuse background?')
    fig.savefig(OUT_DIR / 'hotspot_failure_diagnostics.png', dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = imaging_edges(32)
    x_edges, y_edges = edges[:2]
    cx = (x_edges[:-1] + x_edges[1:]) / 2
    cy = (y_edges[:-1] + y_edges[1:]) / 2
    rows = []
    examples = {}
    for scene in SCENE_NAMES:
        csv_path, mask_fn = SCENARIOS[scene]
        data = pd.read_csv(csv_path)
        spacing = same_side_layer_spacing_mm(data)
        truth = mask_fn(cx, cy)
        rng = np.random.default_rng(RANDOM_SEED + 23)
        for count in (1000, 3000, EVENT_COUNT):
            for repeat in range(REPEATS):
                subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
                draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                         for prefix in ('up1', 'up2', 'down1', 'down2') for axis in ('x', 'y')}
                if count != EVENT_COUNT:
                    continue  # 保持前一轮 5000 事件的随机子样本完全一致。
                for sigma in SIGMAS_MM:
                    hits = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(hits, 0.0, np.random.default_rng(0))
                    events = corrected_events(observed, sigma, spacing)
                    volume, _, _, _ = adaptive_volume(events, REFINE_FRACTION)
                    image = volume.sum(axis=2)
                    record = diagnose(image, events, truth, x_edges, y_edges)
                    record.update({'scenario': scene, 'repeat': repeat + 1,
                                   'event_count': count, 'position_sigma_mm': sigma})
                    rows.append(record)
                    # 第 5 次重复的高噪声误选最明显，三张图仍使用同一批事件。
                    if scene == 'small_offset' and repeat == 4:
                        auto = localize_multiscale_window(image, cx, cy)
                        examples[sigma] = (image, truth, {
                            **record,
                            'window_start_x': auto['window_start_x'],
                            'window_start_y': auto['window_start_y'],
                        })
        print(f'Finished {scene}', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / 'hotspot_diagnosis_by_repeat.csv', index=False)
    metric_columns = [name for name in results.select_dtypes('number')
                      if name not in ('repeat', 'event_count', 'position_sigma_mm')]
    summary = results.groupby(['scenario', 'position_sigma_mm'], as_index=False)[
        metric_columns].agg(['mean', 'std'])
    summary.columns = [first if not second else f'{first}_{second}'
                       for first, second in summary.columns.to_flat_index()]
    summary.to_csv(OUT_DIR / 'hotspot_diagnosis_summary.csv', index=False)
    plot_examples(examples, edges)
    plot_metrics(summary)
    (OUT_DIR / 'hotspot_diagnosis_config.json').write_text(json.dumps({
        'scenes': SCENE_NAMES, 'event_count': EVENT_COUNT, 'repeats': REPEATS,
        'sigmas_mm': SIGMAS_MM, 'seed': RANDOM_SEED + 23,
        'refine_fraction': REFINE_FRACTION,
        'angle_method': 'hard correction, 2 mrad threshold, 3 mrad weight cap',
        'truth_usage': 'retrospective diagnosis and plot outline only',
        'top_share_definition': 'share among positive pixels/events, not all zero pixels',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[['scenario', 'position_sigma_mm', 'selected_events_mean',
                   'target_image_weight_fraction_mean',
                   'background_occupied_fraction_mean',
                   'background_top5_pixel_weight_fraction_mean',
                   'auto_error_mm_mean']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
