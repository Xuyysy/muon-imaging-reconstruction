"""验证亮点截断是否改善多尺度定位，包含从未参与设计的偏心小目标。

基线为：全图质心、固定 24×24 窗口、原多尺度窗口。
新方法只在原多尺度窗口前加入“正像素 95% 分位数”强度上限。
所有方法对同一张图计算，保证事件抽样和探测器噪声完全配对。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from localize_target_region import (
    MULTISCALE_WINDOW_BINS, ROBUST_POSITIVE_QUANTILE,
    localize_multiscale_window, localize_robust_multiscale_window,
    localize_sliding_window,
)
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import REFINE_FRACTION, measured_hits
from run_local_region_validation import global_centroid
from run_multiscale_localization import (
    EVENT_COUNTS, POSITION_SIGMAS_MM, REPEATS, ROOT,
    SCENARIOS as PREVIOUS_SCENARIOS,
)
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

OUT_DIR = ROOT / 'muon_analysis' / 'robust_localization_results'


def small_offset_mask(x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
    """与新 Geant4 几何一致；只用于最后的误差评价。"""
    x, y = np.meshgrid(x_centers, y_centers, indexing='ij')
    x, y = x - 30, y - 20
    bottom = (np.abs(x) <= 40) & (y >= -40) & (y <= -20)
    left = (x >= -40) & (x <= -20) & (y >= -20) & (y <= 40)
    right = (x >= 20) & (x <= 40) & (y >= -20) & (y <= 40)
    return bottom | left | right


SCENARIOS = dict(PREVIOUS_SCENARIOS)
SCENARIOS['small_offset'] = (
    ROOT / 'muon_analysis' / 'muon_tracks_small_offset.csv', small_offset_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS,
                        default=list(SCENARIOS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    xgrid, ygrid = np.meshgrid(cx, cy, indexing='ij')
    rows = []
    for scene in args.scenarios:
        csv_path, mask_fn = SCENARIOS[scene]
        data = pd.read_csv(csv_path)
        spacing = same_side_layer_spacing_mm(data)
        truth = mask_fn(cx, cy)
        true_x, true_y = float(xgrid[truth].mean()), float(ygrid[truth].mean())
        # 保持上一轮的种子与抽样顺序，便于逐行核对旧方法是否重现。
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
                    events = corrected_events(observed, sigma, spacing)
                    volume, _, _, _ = adaptive_volume(events, REFINE_FRACTION)
                    image = volume.sum(axis=2)
                    gx, gy = global_centroid(image, xgrid, ygrid)
                    fixed = localize_sliding_window(image, cx, cy)
                    multiscale = localize_multiscale_window(image, cx, cy)
                    robust = localize_robust_multiscale_window(image, cx, cy)
                    rows.append({
                        'scenario': scene, 'event_count': count,
                        'repeat': repeat + 1, 'position_sigma_mm': sigma,
                        'global_error_mm': float(np.hypot(gx - true_x, gy - true_y)),
                        'fixed24_error_mm': float(np.hypot(fixed['x_mm'] - true_x,
                                                          fixed['y_mm'] - true_y)),
                        'multiscale_error_mm': float(np.hypot(multiscale['x_mm'] - true_x,
                                                             multiscale['y_mm'] - true_y)),
                        'robust_error_mm': float(np.hypot(robust['x_mm'] - true_x,
                                                         robust['y_mm'] - true_y)),
                        'multiscale_window_bins': multiscale['window_bins'],
                        'robust_window_bins': robust['window_bins'],
                        'robust_x_mm': robust['x_mm'], 'robust_y_mm': robust['y_mm'],
                        'clip_value': robust['clip_value'],
                    })
        print(f'Finished {scene}', flush=True)
    result = pd.DataFrame(rows)
    suffix = '_'.join(args.scenarios)
    result.to_csv(OUT_DIR / f'robust_by_repeat_{suffix}.csv', index=False)
    value_columns = ['global_error_mm', 'fixed24_error_mm',
                     'multiscale_error_mm', 'robust_error_mm']
    summary = result.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                             as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in value_columns},
        **{f'{name}_std': (name, 'std') for name in value_columns},
        robust_window_bins_mean=('robust_window_bins', 'mean'),
    )
    summary.to_csv(OUT_DIR / f'robust_summary_{suffix}.csv', index=False)
    (OUT_DIR / f'robust_config_{suffix}.json').write_text(json.dumps({
        'scenarios': args.scenarios, 'event_counts': EVENT_COUNTS,
        'sigmas_mm': POSITION_SIGMAS_MM, 'repeats': REPEATS,
        'seed': RANDOM_SEED + 23, 'refine_fraction': REFINE_FRACTION,
        'candidate_window_bins': MULTISCALE_WINDOW_BINS,
        'positive_intensity_cap_quantile': ROBUST_POSITIVE_QUANTILE,
        'new_scene_not_used_to_select_parameters': 'small_offset',
        'truth_used_in_localizer': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(len(args.scenarios), 1,
                             figsize=(8, 3.0 * len(args.scenarios)),
                             sharex=True, squeeze=False, constrained_layout=True)
    labels = [('global_error_mm', 'Global'), ('fixed24_error_mm', 'Fixed 24'),
              ('multiscale_error_mm', 'Multiscale'), ('robust_error_mm', 'Robust')]
    for ax, scene in zip(axes[:, 0], args.scenarios):
        part = summary[(summary.scenario == scene) & (summary.event_count == 5000)]
        for name, label in labels:
            ax.errorbar(part.position_sigma_mm, part[f'{name}_mean'],
                        yerr=part[f'{name}_std'], marker='o', capsize=2, label=label)
        ax.set_title(scene)
        ax.set_ylabel('Localization error [mm]')
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel('Hit-position sigma [mm]')
    fig.savefig(OUT_DIR / f'robust_comparison_{suffix}.png', dpi=170)
    plt.close(fig)
    print(summary[summary.event_count == 5000][[
        'scenario', 'position_sigma_mm', 'global_error_mm_mean',
        'fixed24_error_mm_mean', 'multiscale_error_mm_mean',
        'robust_error_mm_mean', 'robust_window_bins_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
