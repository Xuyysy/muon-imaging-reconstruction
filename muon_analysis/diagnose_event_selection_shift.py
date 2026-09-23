"""配对追踪同一事件在理想击中和测量误差下的筛选状态与 PoCA 位移。

这是误差来源诊断，不是新筛选算法。理想击中和目标真值仅用于事后
给事件贴“原本通过/噪声后才通过”“目标内/外”的诊断标签。
后续真正的方法不能读取这些理想事件标签或目标掩膜。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_hotspot_failure import SCENE_NAMES
from run_adaptive_voxel_experiment import (
    RANDOM_SEED, THETA_MIN_MRAD, XY_LIMIT_MM, Z_LIMIT_MM,
    imaging_edges, prepare_events,
)
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_robust_localization_experiment import SCENARIOS
from run_uncertainty_aware_angle_experiment import (
    corrected_events, same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'muon_analysis' / 'event_selection_shift_results'
EVENT_COUNT = 5000
REPEATS = 5
SIGMAS_MM = (0.0, 0.05, 0.10, 0.30)


def inside_target(events: pd.DataFrame, mask: np.ndarray,
                  x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """按事件的 XY PoCA 落点判断是否在真值目标投影内，仅用于评价。"""
    ix = np.searchsorted(x_edges, events['poca_x_mm'].to_numpy(), side='right') - 1
    iy = np.searchsorted(y_edges, events['poca_y_mm'].to_numpy(), side='right') - 1
    ix = np.clip(ix, 0, mask.shape[0] - 1)
    iy = np.clip(iy, 0, mask.shape[1] - 1)
    return mask[ix, iy]


def analyze_pair(ideal: pd.DataFrame, selected: pd.DataFrame,
                 ideal_pass: np.ndarray, truth: np.ndarray,
                 x_edges: np.ndarray, y_edges: np.ndarray) -> dict:
    """输入行顺序必须对应同一批 5000 条缪子；不按空间位置匹配。"""
    if not ideal.index.equals(pd.RangeIndex(len(ideal))):
        raise ValueError('ideal events must have a stable 0..N-1 event index')
    ids = selected.index.to_numpy(dtype=int)
    if not np.all((ids >= 0) & (ids < len(ideal))):
        raise ValueError('selected event indices must refer to ideal-event rows')
    stable = ideal_pass[ids]
    new = ~stable
    weight = selected['capped_weight'].to_numpy()
    in_target = inside_target(selected, truth, x_edges, y_edges)
    outside = ~in_target
    total_weight = float(weight.sum())
    background_weight = float(weight[outside].sum())
    shift = np.hypot(
        selected['poca_x_mm'].to_numpy() - ideal.loc[ids, 'poca_x_mm'].to_numpy(),
        selected['poca_y_mm'].to_numpy() - ideal.loc[ids, 'poca_y_mm'].to_numpy(),
    )
    # noise-only 可能因理想 PoCA 在 ROI 外而不通过，位移诊断限定为
    # 理想 PoCA 原本就在 ROI 内的子集，避免拿极远的数值外推点解释图像。
    original_inside_roi = (
        ideal.loc[ids, 'poca_x_mm'].abs().to_numpy() <= XY_LIMIT_MM
    ) & (
        ideal.loc[ids, 'poca_y_mm'].abs().to_numpy() <= XY_LIMIT_MM
    ) & (
        ideal.loc[ids, 'poca_z_mm'].abs().to_numpy() <= Z_LIMIT_MM
    )
    new_roi = new & original_inside_roi
    ideal_theta = ideal.loc[ids, 'theta_mrad'].to_numpy()
    return {
        'ideal_selected_events': int(ideal_pass.sum()),
        'observed_selected_events': int(len(selected)),
        'stable_selected_events': int(stable.sum()),
        'newly_selected_events': int(new.sum()),
        'lost_ideal_events': int(ideal_pass.sum() - stable.sum()),
        'newly_selected_fraction': float(new.mean()) if len(selected) else np.nan,
        'newly_selected_weight_fraction': float(weight[new].sum() / total_weight)
        if total_weight else np.nan,
        'newly_selected_background_weight_fraction': float(
            weight[new & outside].sum() / background_weight
        ) if background_weight else np.nan,
        'stable_background_weight_fraction': float(
            weight[stable & outside].sum() / background_weight
        ) if background_weight else np.nan,
        'selected_target_weight_fraction': float(weight[in_target].sum() / total_weight)
        if total_weight else np.nan,
        'newly_selected_original_low_angle_fraction': float(
            (ideal_theta[new] < THETA_MIN_MRAD).mean()
        ) if new.any() else np.nan,
        'newly_selected_original_outside_roi_fraction': float(
            (~original_inside_roi[new]).mean()
        ) if new.any() else np.nan,
        'stable_poca_shift_median_mm': float(np.median(shift[stable]))
        if stable.any() else np.nan,
        'stable_poca_shift_p90_mm': float(np.quantile(shift[stable], .90))
        if stable.any() else np.nan,
        'newly_selected_in_roi_poca_shift_median_mm': float(np.median(shift[new_roi]))
        if new_roi.any() else np.nan,
        'newly_selected_in_roi_count': int(new_roi.sum()),
    }


def plot_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    metrics = [
        ('newly_selected_fraction_mean', 'Newly selected fraction'),
        ('newly_selected_background_weight_fraction_mean',
         'New-event share of background weight'),
        ('stable_poca_shift_median_mm_mean', 'Median PoCA shift of stable events [mm]'),
        ('selected_target_weight_fraction_mean', 'Selected-event weight inside target'),
    ]
    for ax, (name, ylabel) in zip(axes.ravel(), metrics):
        for scene, part in summary.groupby('scenario'):
            ax.errorbar(part.position_sigma_mm, part[name],
                        yerr=part[name.replace('_mean', '_std')].fillna(0),
                        marker='o', capsize=2, label=scene)
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Matched-event diagnosis: selection turnover and PoCA displacement')
    fig.savefig(OUT_DIR / 'event_selection_shift_diagnostics.png', dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = imaging_edges(32)
    x_edges, y_edges = edges[:2]
    cx = (x_edges[:-1] + x_edges[1:]) / 2
    cy = (y_edges[:-1] + y_edges[1:]) / 2
    rows = []
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
                    continue  # 复用前几轮完全相同的 5000 事件子样本。
                ideal = recompute_from_noisy_hits(subset, 0.0, np.random.default_rng(0))
                ideal_pass = np.zeros(count, dtype=bool)
                ideal_pass[prepare_events(ideal).index.to_numpy(dtype=int)] = True
                for sigma in SIGMAS_MM:
                    hits = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(hits, 0.0, np.random.default_rng(0))
                    selected = corrected_events(observed, sigma, spacing)
                    record = analyze_pair(ideal, selected, ideal_pass, truth,
                                          x_edges, y_edges)
                    record.update({'scenario': scene, 'repeat': repeat + 1,
                                   'position_sigma_mm': sigma,
                                   'theta_noise_floor_mrad': theta_noise_floor_mrad(sigma, spacing),
                                   'event_count': count})
                    rows.append(record)
        print(f'Finished {scene}', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / 'event_selection_shift_by_repeat.csv', index=False)
    numeric = [column for column in result.select_dtypes('number')
               if column not in ('repeat', 'event_count', 'position_sigma_mm')]
    summary = result.groupby(['scenario', 'position_sigma_mm'], as_index=False)[
        numeric].agg(['mean', 'std'])
    summary.columns = [first if not second else f'{first}_{second}'
                       for first, second in summary.columns.to_flat_index()]
    summary.to_csv(OUT_DIR / 'event_selection_shift_summary.csv', index=False)
    plot_summary(summary)
    (OUT_DIR / 'event_selection_shift_config.json').write_text(json.dumps({
        'scenes': SCENE_NAMES, 'event_count': EVENT_COUNT, 'repeats': REPEATS,
        'sigmas_mm': SIGMAS_MM, 'seed': RANDOM_SEED + 23,
        'comparison': 'event row index, paired ideal and measured detector hits',
        'selection': 'theta >= 2 mrad after hard noise-floor correction, PoCA in ROI',
        'truth_usage': 'retrospective diagnostics only; not used in selection',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[['scenario', 'position_sigma_mm',
                   'ideal_selected_events_mean', 'observed_selected_events_mean',
                   'newly_selected_fraction_mean',
                   'newly_selected_background_weight_fraction_mean',
                   'stable_poca_shift_median_mm_mean',
                   'selected_target_weight_fraction_mean']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
