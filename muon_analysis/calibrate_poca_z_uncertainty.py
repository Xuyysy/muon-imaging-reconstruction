"""检验逐事件 PoCA-Z 不确定度是否能解释测量误差下的深度漂移。

运行前方案见 POCA_Z_CALIBRATION_PLAN.md。仅使用 U/L/双铝块开发集；
不使用目标掩膜，也不修改现有重建规则。这里的“参考”是无注入击中误差
时的 PoCA 算法估计，不是真实散射位置。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_poca_z_stability import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM
from diagnose_selection_gate_transitions import gate_masks
from run_adaptive_voxel_experiment import RANDOM_SEED, Z_LIMIT_MM
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

BOOTSTRAP_DRAWS = 32
KEYS = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']


def estimate_z_samples(measured: pd.DataFrame, sigma_mm: float,
                       seed: int, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    """只从本次测得的四层击中出发，重复传播已知的位置测量误差。

    一行对应一条缪子，一列对应一次扰动。它是测量敏感性近似，
    不是已校准的真实散射点后验分布；绝不能在这里读取理想 PoCA。
    """
    if draws < 2:
        raise ValueError('At least two perturbation draws are required')
    if sigma_mm < 0:
        raise ValueError('Hit-position sigma must be non-negative')
    rng = np.random.default_rng(seed)
    return np.column_stack([
        recompute_from_noisy_hits(measured, sigma_mm, rng).poca_z_mm.to_numpy()
        for _ in range(draws)
    ])


def evaluate_calibration(reference_z: np.ndarray, observed_z: np.ndarray,
                         z_samples: np.ndarray) -> tuple[dict, list[dict]]:
    """检查参考 PoCA-Z 的区间覆盖率，以及不确定度和越界率的关联。

    三分位按本条件预测标准差排序，仅描述风险，不把目标真值或越界标签
    用于设定重建阈值。相关抽样不是独立 Geant4 重复，不做显著性检验。
    """
    reference_z = np.asarray(reference_z, dtype=float)
    observed_z = np.asarray(observed_z, dtype=float)
    if z_samples.ndim != 2 or z_samples.shape[0] != len(reference_z):
        raise ValueError('Unpaired Z perturbation samples')
    if len(observed_z) != len(reference_z) or not np.isfinite(z_samples).all():
        raise ValueError('Invalid or nonfinite paired PoCA-Z values')
    n = len(reference_z)
    if not n:
        raise ValueError('No reference-selected events')
    sd = z_samples.std(axis=1, ddof=1)
    q05, q16, q84, q95 = np.quantile(z_samples, [.05, .16, .84, .95], axis=1)
    coverage68 = (q16 <= reference_z) & (reference_z <= q84)
    coverage90 = (q05 <= reference_z) & (reference_z <= q95)
    displacement = np.abs(reference_z - observed_z)
    z_exit = np.abs(observed_z) > Z_LIMIT_MM
    nonzero = sd > 0
    ratios = displacement[nonzero] / sd[nonzero]
    summary = {
        'reference_selected': n,
        'z_exit': int(z_exit.sum()),
        'z_exit_fraction': float(z_exit.mean()),
        'predicted_sd_median_mm': float(np.median(sd)),
        'predicted_sd_p90_mm': float(np.quantile(sd, .9)),
        'abs_dz_median_mm': float(np.median(displacement)),
        'coverage68': float(coverage68.mean()),
        'coverage90': float(coverage90.mean()),
        'ratio_median': float(np.median(ratios)) if len(ratios) else np.nan,
        'ratio_p90': float(np.quantile(ratios, .9)) if len(ratios) else np.nan,
        'zero_sd_events': int((~nonzero).sum()),
    }
    # 稳定排序确保即使出现相同预测标准差，三组仍是互斥且覆盖全部事件。
    order = np.argsort(sd, kind='stable')
    tertile = np.empty(n, dtype=int)
    for group, indices in enumerate(np.array_split(order, 3)):
        tertile[indices] = group
    strata = []
    for group in range(3):
        mask = tertile == group
        strata.append({
            'uncertainty_tertile': group + 1,
            'events': int(mask.sum()),
            'predicted_sd_median_mm': float(np.median(sd[mask])),
            'abs_dz_median_mm': float(np.median(displacement[mask])),
            'z_exit_fraction': float(z_exit[mask].mean()),
            'coverage68': float(coverage68[mask].mean()),
            'coverage90': float(coverage90[mask].mean()),
        })
    return summary, strata


def run(output_dir: Path) -> None:
    """复用上一轮完全相同的事件抽样与测量噪声，再独立抽样误差传播。"""
    previous = pd.read_csv(ROOT / 'muon_analysis/poca_z_stability_results/by_repeat.csv')
    rows: list[dict] = []
    strata_rows: list[dict] = []
    for scene_index, (scene, (csv_name, _)) in enumerate(SCENES.items()):
        data = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                normal_draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                                for prefix in ('up1', 'up2', 'down1', 'down2')
                                for axis in ('x', 'y')}
                reference = recompute_from_noisy_hits(
                    measured_hits(subset, 0.0, normal_draws), 0.0,
                    np.random.default_rng(0))
                selected = gate_masks(reference, 0.0, spacing)['selected']
                reference_z = reference.poca_z_mm.to_numpy()[selected]
                for sigma in SIGMAS_MM:
                    hits = measured_hits(subset, sigma, normal_draws)
                    observed = recompute_from_noisy_hits(
                        hits, 0.0, np.random.default_rng(0))
                    observed_z = observed.poca_z_mm.to_numpy()[selected]
                    seed = RANDOM_SEED + 200000 * scene_index + 10000 * count \
                        + 100 * repeat + int(round(sigma * 1000))
                    samples = estimate_z_samples(hits.iloc[selected], sigma, seed)
                    metrics, strata = evaluate_calibration(reference_z, observed_z, samples)
                    key = {'scenario': scene, 'event_count': count, 'repeat': repeat,
                           'position_sigma_mm': sigma}
                    rows.append({**key, **metrics})
                    strata_rows.extend({**key, **item} for item in strata)
                print(f'Z uncertainty: {scene}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    strata_frame = pd.DataFrame(strata_rows)
    if frame.duplicated(KEYS).any() or len(strata_frame) != 3 * len(frame):
        raise AssertionError('Missing or duplicate paired conditions')
    matched = frame.merge(previous[KEYS + ['baseline_selected', 'z_exit']],
                          on=KEYS, validate='one_to_one', suffixes=('', '_previous'))
    if len(matched) != len(previous):
        raise AssertionError('Previous paired conditions do not match')
    if not np.array_equal(matched.reference_selected, matched.baseline_selected):
        raise AssertionError('Reference-selected event count changed')
    if not np.array_equal(matched.z_exit, matched.z_exit_previous):
        raise AssertionError('Z-exit count changed from previous diagnostic')
    control = frame[frame.position_sigma_mm == 0]
    if control.z_exit.any() or (control.abs_dz_median_mm != 0).any() \
            or (control.coverage68 != 1).any():
        raise AssertionError('Zero-noise control failed')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    strata_frame.to_csv(output_dir / 'by_tertile.csv', index=False)
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm'],
                            as_index=False).agg(
        repeats=('repeat', 'size'),
        reference_selected_mean=('reference_selected', 'mean'),
        z_exit_fraction_mean=('z_exit_fraction', 'mean'),
        predicted_sd_median_mm_mean=('predicted_sd_median_mm', 'mean'),
        abs_dz_median_mm_mean=('abs_dz_median_mm', 'mean'),
        coverage68_mean=('coverage68', 'mean'),
        coverage90_mean=('coverage90', 'mean'),
        ratio_median_mean=('ratio_median', 'mean'),
        ratio_p90_mean=('ratio_p90', 'mean'),
        zero_sd_events_mean=('zero_sd_events', 'mean'),
    )
    summary.to_csv(output_dir / 'summary.csv', index=False)
    tertile_summary = strata_frame.groupby(
        ['scenario', 'event_count', 'position_sigma_mm', 'uncertainty_tertile'],
        as_index=False).agg(
        events_mean=('events', 'mean'),
        predicted_sd_median_mm_mean=('predicted_sd_median_mm', 'mean'),
        abs_dz_median_mm_mean=('abs_dz_median_mm', 'mean'),
        z_exit_fraction_mean=('z_exit_fraction', 'mean'),
        coverage68_mean=('coverage68', 'mean'),
        coverage90_mean=('coverage90', 'mean'),
    )
    tertile_summary.to_csv(output_dir / 'tertile_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, sigma in zip(axes, (.05, .10)):
        for scene in SCENES:
            part = tertile_summary[(tertile_summary.scenario == scene)
                                   & (tertile_summary.event_count == 5000)
                                   & (tertile_summary.position_sigma_mm == sigma)]
            ax.plot(part.uncertainty_tertile, part.z_exit_fraction_mean,
                    marker='o', label=scene)
        ax.set(xlabel='Predicted PoCA-Z uncertainty tertile',
               title=f'Hit-position sigma = {sigma:g} mm',
               xticks=[1, 2, 3], ylim=(0, 1))
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Z-exit fraction of reference-selected events')
    axes[1].legend()
    fig.suptitle('Development scenes: event-level Z-instability risk ranking')
    fig.savefig(output_dir / 'z_exit_by_uncertainty_tertile.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'POCA_Z_CALIBRATION_PLAN.md',
        'role': 'measurement repeatability diagnostic, not imaging improvement',
        'scenes': list(SCENES), 'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'bootstrap_draws': BOOTSTRAP_DRAWS,
        'reference': 'sigma=0 algorithmic PoCA, not true scattering point',
        'calibration_estimator': 'hit perturbations around noisy measured hits',
        'target_truth_used': False, 't_scene_used': False,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary[(summary.event_count == 5000)
                  & (summary.position_sigma_mm == .05)].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/poca_z_calibration_results')
    run(parser.parse_args().output_dir)
