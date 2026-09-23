"""不调参验证事件级误差方法在中心、偏心、小尺寸 U 目标上的泛化。

三个场景使用同一组事件数、探测器误差、重复次数和算法参数。
不同场景的目标真值掩膜仅用于最后计算 IoU/定位误差，重建函数无法访问它。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import (
    RANDOM_SEED, THETA_CAP_MRAD, THETA_MIN_MRAD, adaptive_volume,
    imaging_edges, prepare_events,
)
from run_baseline_experiment import evaluate_image
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import (
    MIN_PASS_FRACTION, POSTERIOR_DRAWS, REFINE_FRACTION,
    measured_hits, propagated_events,
)
from run_offcenter_generalization import evaluate_shifted
from run_small_target_generalization import evaluate_small
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'muon_analysis' / 'uncertainty_cross_scenario_results'
SCENARIOS = {
    'center': (ROOT / 'muon_analysis' / 'muon_tracks.csv', evaluate_image),
    'offcenter': (ROOT / 'muon_analysis' / 'muon_tracks_offcenter.csv', evaluate_shifted),
    'small': (ROOT / 'muon_analysis' / 'muon_tracks_small.csv', evaluate_small),
}
EVENT_COUNTS = (1000, 3000, 5000)
POSITION_SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)
REPEATS = 5
METHODS = ('observed', 'hard', 'propagated', 'majority')


def plot_5000(summary: pd.DataFrame) -> None:
    """每个场景一行，左 IoU、右定位误差；全部使用同一量程。"""
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 9.5), constrained_layout=True,
                             sharex=True, sharey='col')
    labels = {'observed': 'Observed', 'hard': 'Hard correction',
              'propagated': 'All candidates', 'majority': 'Majority gate'}
    colors = {'observed': '#7f7f7f', 'hard': '#1f77b4',
              'propagated': '#d62728', 'majority': '#2ca02c'}
    for row, scenario in enumerate(SCENARIOS):
        for method in METHODS:
            part = summary[(summary.scenario == scenario) &
                           (summary.event_count == 5000) & (summary.method == method)]
            axes[row, 0].errorbar(part.position_sigma_mm, part.iou_mean, yerr=part.iou_std,
                                  marker='o', capsize=2, color=colors[method], label=labels[method])
            axes[row, 1].errorbar(part.position_sigma_mm, part.localization_error_mm_mean,
                                  yerr=part.localization_error_mm_std, marker='o', capsize=2,
                                  color=colors[method], label=labels[method])
        axes[row, 0].set_ylabel(f'{scenario}\nIoU (higher better)')
        axes[row, 1].set_ylabel('Localization error [mm]\n(lower better)')
        for axis in axes[row]:
            axis.grid(alpha=.25)
            axis.legend(fontsize=7)
            axis.set_xlabel('Hit-position sigma [mm]')
    fig.suptitle('Detector uncertainty across target position and size: 5000 events')
    fig.savefig(OUT_DIR / 'uncertainty_cross_scenario_5000.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = imaging_edges(32)
    centers = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows = []
    for scenario, (csv_path, evaluate) in SCENARIOS.items():
        data = pd.read_csv(csv_path)
        spacing = same_side_layer_spacing_mm(data)
        # 每个场景从相同种子开始；不同事件集不能混用，但抽样过程保持一致。
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
                    all_candidates, majority, pass_rate, xy_spread, covered = propagated_events(
                        hits, sigma, posterior_rng
                    )
                    variants = {
                        'observed': prepare_events(observed),
                        'hard': corrected_events(observed, sigma, spacing),
                        'propagated': all_candidates,
                        'majority': majority,
                    }
                    for method, events in variants.items():
                        volume, _, refined, leaves = adaptive_volume(events, REFINE_FRACTION)
                        score = evaluate(volume.sum(axis=2), *centers)
                        rows.append({
                            'scenario': scenario, 'method': method, 'event_count': count,
                            'repeat': repeat + 1, 'position_sigma_mm': sigma,
                            'iou_equal_area': score[0], 'localization_error_mm': score[1],
                            'selected_candidate_rows': len(events),
                            'contributing_unique_events': (int(events.event_id.nunique())
                                if method in ('propagated', 'majority') and sigma > 0 else len(events)),
                            'event_pass_probability_mean': pass_rate if method in ('propagated', 'majority') else np.nan,
                            'poca_xy_spread_median_mm': xy_spread if method in ('propagated', 'majority') else np.nan,
                        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / 'uncertainty_cross_scenario_by_repeat.csv', index=False)
    summary = results.groupby(['scenario', 'method', 'event_count', 'position_sigma_mm'],
                              as_index=False).agg(
        repeats=('repeat', 'size'), iou_mean=('iou_equal_area', 'mean'),
        iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
        contributing_unique_events_mean=('contributing_unique_events', 'mean'),
    )
    summary.to_csv(OUT_DIR / 'uncertainty_cross_scenario_summary.csv', index=False)
    plot_5000(summary)
    config = {
        'scenarios': {key: str(path) for key, (path, _) in SCENARIOS.items()},
        'event_counts': EVENT_COUNTS, 'position_sigmas_mm': POSITION_SIGMAS_MM,
        'repeats': REPEATS, 'seed': RANDOM_SEED + 23,
        'posterior_draws': POSTERIOR_DRAWS, 'majority_min_pass_fraction': MIN_PASS_FRACTION,
        'theta_min_mrad': THETA_MIN_MRAD, 'theta_cap_mrad': THETA_CAP_MRAD,
        'refinement_fraction': REFINE_FRACTION,
        'note': 'All parameters fixed from center-target experiment; target masks used only for scoring.',
    }
    (OUT_DIR / 'uncertainty_cross_scenario_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(summary[(summary.event_count == 5000) &
                  summary.position_sigma_mm.isin([.05, .10, .30])]
          [['scenario', 'method', 'position_sigma_mm', 'iou_mean',
            'localization_error_mm_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
