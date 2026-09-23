"""固定规则的无真值局部定位，对照全图强度质心。

使用上一轮完全相同的抽样、探测器误差、重建和目标场景。
局部目标选择器只接收二维 PoCA 图像；目标几何掩膜仅用于最后评价。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from localize_target_region import (
    SMOOTHING_WIDTH, TOP_PIXEL_FRACTION, WINDOW_BINS,
    localize_sliding_window, localize_target_region,
)
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges, prepare_events
from run_baseline_experiment import u_target_mask
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import REFINE_FRACTION, measured_hits, propagated_events
from run_offcenter_generalization import shifted_u_mask
from run_small_target_generalization import small_u_mask
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'muon_analysis' / 'local_region_results'
SCENARIOS = {
    'center': (ROOT / 'muon_analysis' / 'muon_tracks.csv', u_target_mask),
    'offcenter': (ROOT / 'muon_analysis' / 'muon_tracks_offcenter.csv', shifted_u_mask),
    'small': (ROOT / 'muon_analysis' / 'muon_tracks_small.csv', small_u_mask),
}
EVENT_COUNTS = (1000, 3000, 5000)
POSITION_SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
METHODS = ('observed', 'hard', 'majority')


def global_centroid(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    total = float(image.sum())
    if total <= 0:
        return np.nan, np.nan
    return float((image * x).sum() / total), float((image * y).sum() / total)


def plot_5000(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    colors = {'observed': '#7f7f7f', 'hard': '#1f77b4', 'majority': '#2ca02c'}
    for row, scenario in enumerate(SCENARIOS):
        for method in METHODS:
            part = summary[(summary.scenario == scenario) &
                           (summary.event_count == 5000) & (summary.method == method)]
            axes[row].errorbar(part.position_sigma_mm, part.global_error_mm_mean,
                               yerr=part.global_error_mm_std, color=colors[method],
                               linestyle='--', marker='o', capsize=2, label=f'{method}: global')
            axes[row].errorbar(part.position_sigma_mm, part.local_error_mm_mean,
                               yerr=part.local_error_mm_std, color=colors[method],
                               linestyle='-', marker='s', capsize=2, label=f'{method}: local')
        axes[row].set_ylabel(f'{scenario}\nLocalization error [mm]')
        axes[row].grid(alpha=.25)
        axes[row].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel('Hit-position sigma [mm]')
    fig.suptitle('Global vs unsupervised local-region centroid: 5000 events')
    fig.savefig(OUT_DIR / 'local_region_vs_global_5000.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for row, scenario in enumerate(SCENARIOS):
        for method in METHODS:
            part = summary[(summary.scenario == scenario) &
                           (summary.event_count == 5000) & (summary.method == method)]
            axes[row].errorbar(part.position_sigma_mm, part.global_error_mm_mean,
                               yerr=part.global_error_mm_std, color=colors[method],
                               linestyle='--', marker='o', capsize=2, label=f'{method}: global')
            axes[row].errorbar(part.position_sigma_mm, part.window_error_mm_mean,
                               yerr=part.window_error_mm_std, color=colors[method],
                               linestyle='-', marker='s', capsize=2, label=f'{method}: sliding window')
        axes[row].set_ylabel(f'{scenario}\nLocalization error [mm]')
        axes[row].grid(alpha=.25)
        axes[row].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel('Hit-position sigma [mm]')
    fig.suptitle('Global vs fixed-scale sliding-window centroid: 5000 events')
    fig.savefig(OUT_DIR / 'sliding_window_vs_global_5000.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    xgrid, ygrid = np.meshgrid(cx, cy, indexing='ij')
    rows = []
    for scenario, (csv_path, mask_function) in SCENARIOS.items():
        data = pd.read_csv(csv_path)
        spacing = same_side_layer_spacing_mm(data)
        # 目标掩膜在这里仅计算评价用的真值中心，绝不传入 localize_target_region。
        truth = mask_function(cx, cy)
        true_x, true_y = float(xgrid[truth].mean()), float(ygrid[truth].mean())
        rng = np.random.default_rng(RANDOM_SEED + 23)
        for count in EVENT_COUNTS:
            if count > len(data):
                continue
            for repeat in range(REPEATS):
                subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
                draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                         for prefix in ('up1', 'up2', 'down1', 'down2') for axis in ('x', 'y')}
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
                        volume, _, _, _ = adaptive_volume(events, REFINE_FRACTION)
                        image = volume.sum(axis=2)
                        gx, gy = global_centroid(image, xgrid, ygrid)
                        local = localize_target_region(image, cx, cy)
                        window = localize_sliding_window(image, cx, cy)
                        rows.append({
                            'scenario': scenario, 'method': method, 'event_count': count,
                            'repeat': repeat + 1, 'position_sigma_mm': sigma,
                            'true_x_mm': true_x, 'true_y_mm': true_y,
                            'global_x_mm': gx, 'global_y_mm': gy,
                            'local_x_mm': local['x_mm'], 'local_y_mm': local['y_mm'],
                            'window_x_mm': window['x_mm'], 'window_y_mm': window['y_mm'],
                            'global_error_mm': float(np.hypot(gx - true_x, gy - true_y)),
                            'local_error_mm': float(np.hypot(local['x_mm'] - true_x,
                                                             local['y_mm'] - true_y)),
                            'window_error_mm': float(np.hypot(window['x_mm'] - true_x,
                                                              window['y_mm'] - true_y)),
                            'window_weight_fraction': window['window_weight_fraction'],
                            'selected_pixels': local['selected_pixels'],
                            'component_pixels': local['component_pixels'],
                            'touches_boundary': local['touches_boundary'],
                        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / 'local_region_by_repeat.csv', index=False)
    summary = results.groupby(['scenario', 'method', 'event_count', 'position_sigma_mm'],
                              as_index=False).agg(
        repeats=('repeat', 'size'),
        global_error_mm_mean=('global_error_mm', 'mean'),
        global_error_mm_std=('global_error_mm', 'std'),
        local_error_mm_mean=('local_error_mm', 'mean'),
        local_error_mm_std=('local_error_mm', 'std'),
        window_error_mm_mean=('window_error_mm', 'mean'),
        window_error_mm_std=('window_error_mm', 'std'),
        window_weight_fraction_mean=('window_weight_fraction', 'mean'),
        component_pixels_mean=('component_pixels', 'mean'),
        touches_boundary_rate=('touches_boundary', 'mean'),
    )
    summary.to_csv(OUT_DIR / 'local_region_summary.csv', index=False)
    plot_5000(summary)
    (OUT_DIR / 'local_region_config.json').write_text(json.dumps({
        'scenarios': {name: str(path) for name, (path, _) in SCENARIOS.items()},
        'seed': RANDOM_SEED + 23, 'event_counts': EVENT_COUNTS,
        'position_sigmas_mm': POSITION_SIGMAS_MM, 'repeats': REPEATS,
        'top_pixel_fraction': TOP_PIXEL_FRACTION,
        'smoothing_width': SMOOTHING_WIDTH, 'connectivity': 8,
        'sliding_window_bins': WINDOW_BINS,
        'sliding_window_prior': 'maximum target extent is at most about 135 mm',
        'component_rule': 'largest integrated smoothed image intensity',
        'note': 'Localization rule fixed across all scenarios; truth used only for error calculation.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[(summary.event_count == 5000) &
                  summary.position_sigma_mm.isin([.05, .10, .30])]
          [['scenario', 'method', 'position_sigma_mm', 'global_error_mm_mean',
            'local_error_mm_mean', 'window_error_mm_mean',
            'component_pixels_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
