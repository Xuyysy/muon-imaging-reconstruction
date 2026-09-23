"""多尺度无真值定位：先固定图像评分规则，再比较旧场景与新 Geant4 场景。

这里仅研究 hard 散射角修正 + 40% 自适应体素的定位误差。
输入图像完全不包含目标位置或尺寸；真值掩膜仅用于评价。
可用 --scenarios center offcenter small 先运行旧场景，规则冻结后再运行新场景。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from localize_target_region import MULTISCALE_WINDOW_BINS, localize_multiscale_window, localize_sliding_window
from run_adaptive_voxel_experiment import RANDOM_SEED, adaptive_volume, imaging_edges
from run_baseline_experiment import u_target_mask
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import REFINE_FRACTION, measured_hits
from run_local_region_validation import global_centroid
from run_offcenter_generalization import shifted_u_mask
from run_small_target_generalization import small_u_mask
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'muon_analysis' / 'multiscale_localization_results'
EVENT_COUNTS = (1000, 3000, 5000)
POSITION_SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5


def medium_offset_mask(x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
    """独立验证场景的真值：中心 (-20,+15)、外宽 100、壁厚 25 mm。"""
    x, y = np.meshgrid(x_centers, y_centers, indexing='ij')
    x, y = x + 20, y - 15  # 转换成以新目标中心为原点的局部坐标。
    bottom = (np.abs(x) <= 50) & (y >= -50) & (y <= -25)
    left = (x >= -50) & (x <= -25) & (y >= -25) & (y <= 50)
    right = (x >= 25) & (x <= 50) & (y >= -25) & (y <= 50)
    return bottom | left | right


SCENARIOS = {
    'center': (ROOT / 'muon_analysis' / 'muon_tracks.csv', u_target_mask),
    'offcenter': (ROOT / 'muon_analysis' / 'muon_tracks_offcenter.csv', shifted_u_mask),
    'small': (ROOT / 'muon_analysis' / 'muon_tracks_small.csv', small_u_mask),
    'medium_offset': (ROOT / 'muon_analysis' / 'muon_tracks_medium_offset.csv', medium_offset_mask),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS.keys(),
                        default=list(SCENARIOS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    xgrid, ygrid = np.meshgrid(cx, cy, indexing='ij')
    rows = []
    for scene in args.scenarios:
        csv_path, mask_fn = SCENARIOS[scene]
        if not csv_path.exists():
            raise FileNotFoundError(f'{scene}: {csv_path}; first run Geant4 for this scene')
        data = pd.read_csv(csv_path)
        spacing = same_side_layer_spacing_mm(data)
        truth = mask_fn(cx, cy)  # 真值仅在计算评价指标时使用。
        true_x, true_y = float(xgrid[truth].mean()), float(ygrid[truth].mean())
        rng = np.random.default_rng(RANDOM_SEED + 23)
        for count in EVENT_COUNTS:
            if count > len(data):
                continue
            for repeat in range(REPEATS):
                subset = data.iloc[rng.choice(len(data), size=count, replace=False)].copy()
                # 同一次重复的四个 sigma 使用相同标准正态噪声，减少随机比较误差。
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
                    auto = localize_multiscale_window(image, cx, cy)
                    rows.append({
                        'scenario': scene, 'event_count': count, 'repeat': repeat + 1,
                        'position_sigma_mm': sigma, 'true_x_mm': true_x,
                        'true_y_mm': true_y,
                        'global_error_mm': float(np.hypot(gx - true_x, gy - true_y)),
                        'fixed24_error_mm': float(np.hypot(fixed['x_mm'] - true_x,
                                                          fixed['y_mm'] - true_y)),
                        'multiscale_error_mm': float(np.hypot(auto['x_mm'] - true_x,
                                                             auto['y_mm'] - true_y)),
                        'selected_window_bins': auto['window_bins'],
                        'multiscale_x_mm': auto['x_mm'], 'multiscale_y_mm': auto['y_mm'],
                        'score': auto['score'],
                    })
        print(f'Finished {scene}: {sum(r["scenario"] == scene for r in rows)} images', flush=True)
    results = pd.DataFrame(rows)
    suffix = '_'.join(args.scenarios)
    results.to_csv(OUT_DIR / f'multiscale_by_repeat_{suffix}.csv', index=False)
    summary = results.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                              as_index=False).agg(
        repeats=('repeat', 'size'),
        global_error_mm_mean=('global_error_mm', 'mean'),
        global_error_mm_std=('global_error_mm', 'std'),
        fixed24_error_mm_mean=('fixed24_error_mm', 'mean'),
        fixed24_error_mm_std=('fixed24_error_mm', 'std'),
        multiscale_error_mm_mean=('multiscale_error_mm', 'mean'),
        multiscale_error_mm_std=('multiscale_error_mm', 'std'),
        selected_window_bins_mean=('selected_window_bins', 'mean'),
    )
    summary.to_csv(OUT_DIR / f'multiscale_summary_{suffix}.csv', index=False)
    (OUT_DIR / f'multiscale_config_{suffix}.json').write_text(json.dumps({
        'scenarios': args.scenarios, 'event_counts': EVENT_COUNTS,
        'sigmas_mm': POSITION_SIGMAS_MM, 'repeats': REPEATS,
        'seed': RANDOM_SEED + 23, 'refine_fraction': REFINE_FRACTION,
        'candidate_window_bins': MULTISCALE_WINDOW_BINS,
        'selection_score': '(smoothed window mean - smoothed global mean) * window bins',
        'truth_used_for_reconstruction': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(len(args.scenarios), 1, figsize=(8, 3 * len(args.scenarios)),
                             sharex=True, squeeze=False, constrained_layout=True)
    for ax, scene in zip(axes[:, 0], args.scenarios):
        part = summary[(summary.scenario == scene) & (summary.event_count == 5000)]
        for column, label, marker in (
            ('global_error_mm_mean', 'Global', 'o'),
            ('fixed24_error_mm_mean', 'Fixed 24', 's'),
            ('multiscale_error_mm_mean', 'Multiscale', '^'),
        ):
            ax.errorbar(part.position_sigma_mm, part[column],
                        yerr=part[column.replace('_mean', '_std')],
                        marker=marker, capsize=2, label=label)
        ax.set_title(scene)
        ax.set_ylabel('Localization error [mm]')
        ax.grid(alpha=.25)
        ax.legend()
    axes[-1, 0].set_xlabel('Hit-position sigma [mm]')
    fig.savefig(OUT_DIR / f'multiscale_comparison_{suffix}.png', dpi=170)
    plt.close(fig)
    print(summary[summary.event_count == 5000][[
        'scenario', 'position_sigma_mm', 'global_error_mm_mean',
        'fixed24_error_mm_mean', 'multiscale_error_mm_mean',
        'selected_window_bins_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
