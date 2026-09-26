"""在已见 Geant4 数据上诊断软 Z 门收益的观测侧特征。

只做描述，不训练阈值/分类器。先读 OBSERVED_Z_GATE_RISK_DIAGNOSTIC_PLAN.md。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from run_adaptive_voxel_experiment import RANDOM_SEED, THETA_MIN_MRAD
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_soft_z_gate_experiment import (
    COUNTS, GRIDS, REPEATS, ROOT, SIGMAS_MM, candidate_events,
    z_acceptance_weights,
)
from run_uncertainty_aware_angle_experiment import (
    same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

SOURCES = (
    ('u_center', 'muon_tracks.csv', 'old', 0),
    ('l_shape', 'muon_tracks_l_shape.csv', 'old', 1),
    ('double_block', 'muon_tracks_double_block.csv', 'old', 2),
    ('frame_seed_a', 'muon_tracks_frame_seed_a.csv', 'frame', 0),
    ('frame_seed_b', 'muon_tracks_frame_seed_b.csv', 'frame', 1),
    ('iron_seed_a', 'muon_tracks_plus_iron_seed_a.csv', 'iron', 0),
    ('iron_seed_b', 'muon_tracks_plus_iron_seed_b.csv', 'iron', 1),
)
KEYS = ['event_count', 'repeat', 'position_sigma_mm', 'grid']


def experiment_parameters(family: str, index: int) -> tuple[int, int]:
    """严格复用原实验主 RNG 与独立传播 RNG 的场景偏移。"""
    if family == 'old':
        return 2107 + 100 * index, 300000 * index
    if family == 'frame':
        return 5107 + 1000 * index, 600000 + 300000 * index
    if family == 'iron':
        return 8107 + 1000 * index, 900000 + 300000 * index
    raise ValueError(family)


def source_scores(source: str, family: str,
                  old: pd.DataFrame, frame: pd.DataFrame,
                  iron: pd.DataFrame) -> pd.DataFrame:
    """读取现存的 H/P 同事件评分；分数只会出现在输出评价列。"""
    if family == 'old':
        scored, p_name = old[old.scenario == source], 'soft_z'
    elif family == 'frame':
        scored, p_name = frame[frame.stream == source.removeprefix('frame_')], 'propagated'
    else:
        scored, p_name = iron[iron.stream == source.removeprefix('iron_')], 'propagated'
    h = scored[scored.method == 'hard'].set_index(KEYS).sort_index()
    p = scored[scored.method == p_name].set_index(KEYS).sort_index()
    if len(h) != len(p) or not h.index.equals(p.index):
        raise AssertionError(f'{source}: H/P labels are not paired')
    return pd.DataFrame({'hard_iou': h.iou, 'propagated_iou': p.iou,
                         'saved_candidates': h['xy_angle_candidates' if family == 'old'
                                                else 'candidates'],
                         'saved_hard_z_count': h.effective_z_accepted_events
                         if family != 'iron' else np.nan,
                         'saved_soft_z_count': p.effective_z_accepted_events
                         if family != 'iron' else np.nan})


def observed_features(observed: pd.DataFrame, ids: np.ndarray,
                      theta_weight: np.ndarray, sampled_z: np.ndarray,
                      sigma: float, spacing: float,
                      input_count: int) -> dict[str, float]:
    """所有特征来自观测击中及误差传播，绝不读取目标掩膜/IoU。"""
    z = observed.poca_z_mm.to_numpy()[ids]
    h, p = z_acceptance_weights(z, sampled_z)
    floor = theta_noise_floor_mrad(sigma, spacing)
    theta_obs = observed.theta_mrad.to_numpy()[ids]
    theta = np.sqrt(np.maximum(theta_obs**2 - floor**2, 0.))
    if len(ids) == 0:
        raise ValueError('No candidate events in diagnostic condition')
    soft_mass = float(np.sum(theta_weight * p))
    hard_mass = float(np.sum(theta_weight * h))
    rescue = float(np.sum(theta_weight * (1 - h) * p))
    suppression = float(np.sum(theta_weight * h * (1 - p)))
    return {
        'noise_floor_ratio': float(floor / THETA_MIN_MRAD),
        'candidate_count': int(len(ids)),
        'candidate_fraction': float(len(ids) / input_count),
        'hard_z_count': float(h.sum()),
        'soft_z_count': float(p.sum()),
        'theta_corrected_median_mrad': float(np.median(theta)),
        'theta_corrected_p25_mrad': float(np.quantile(theta, .25)),
        'fractional_gate_fraction': float(np.mean((p > 0) & (p < 1))),
        'rescue_mass_fraction': rescue / soft_mass if soft_mass > 0 else 0.,
        'suppression_mass_fraction': suppression / hard_mass if hard_mass > 0 else 0.,
        'soft_to_hard_mass_ratio': soft_mass / hard_mass if hard_mass > 0 else np.nan,
    }


def run(output_dir: Path) -> None:
    old = pd.read_csv(ROOT / 'muon_analysis/soft_z_gate_results/by_repeat.csv')
    frame = pd.read_csv(ROOT / 'muon_analysis/frame_z_gate_results/by_repeat.csv')
    iron = pd.read_csv(ROOT / 'muon_analysis/plus_iron_fallback_results/by_repeat.csv')
    rows = []
    for source, filename, family, index in SOURCES:
        data = pd.read_csv(ROOT / 'muon_analysis' / filename)
        spacing = same_side_layer_spacing_mm(data)
        main_offset, propagation_offset = experiment_parameters(family, index)
        rng = np.random.default_rng(RANDOM_SEED + main_offset)
        scores = source_scores(source, family, old, frame, iron)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                normals = {f'{plane}_{axis}': rng.normal(0, 1, count)
                           for plane in ('up1', 'up2', 'down1', 'down2')
                           for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, normals)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    ids, weight = candidate_events(observed, sigma, spacing)
                    seed = (RANDOM_SEED + propagation_offset + 10000 * count
                            + 100 * repeat + int(round(sigma * 1000)))
                    z_samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                   seed, BOOTSTRAP_DRAWS)
                    feature = observed_features(observed, ids, weight,
                                                z_samples, sigma, spacing, count)
                    for grid in GRIDS:
                        key = count, repeat, sigma, grid
                        saved = scores.loc[key]
                        if int(saved.saved_candidates) != feature['candidate_count']:
                            raise AssertionError(f'{source} {key}: candidate mismatch')
                        if family != 'iron' and (not np.isclose(
                                saved.saved_hard_z_count, feature['hard_z_count'])
                                or not np.isclose(saved.saved_soft_z_count,
                                                  feature['soft_z_count'], atol=1e-10)):
                            raise AssertionError(f'{source} {key}: Z-gate mismatch')
                        rows.append({
                            'source': source, 'family': family,
                            'event_count': count, 'repeat': repeat,
                            'position_sigma_mm': sigma, 'grid': grid,
                            **feature,
                            'hard_iou': float(saved.hard_iou),
                            'propagated_iou': float(saved.propagated_iou),
                            'iou_gain': float(saved.propagated_iou - saved.hard_iou),
                        })
                print(f'Observed risk: {source}, N={count}, repeat={repeat}/{REPEATS}',
                      flush=True)
    result = pd.DataFrame(rows)
    expected = len(SOURCES) * len(COUNTS) * REPEATS * len(SIGMAS_MM) * len(GRIDS)
    if len(result) != expected or result.duplicated(['source'] + KEYS).any():
        raise AssertionError('Missing or duplicate paired conditions')
    feature_columns = [
        'noise_floor_ratio', 'candidate_count', 'candidate_fraction',
        'hard_z_count', 'soft_z_count', 'theta_corrected_median_mrad',
        'theta_corrected_p25_mrad', 'fractional_gate_fraction',
        'rescue_mass_fraction', 'suppression_mass_fraction',
        'soft_to_hard_mass_ratio',
    ]
    summary = result.groupby(['source', 'family', 'event_count',
                              'position_sigma_mm', 'grid'], as_index=False).agg(
        repeats=('repeat', 'size'),
        iou_gain_mean=('iou_gain', 'mean'),
        iou_gain_std=('iou_gain', 'std'),
        positive_repeats=('iou_gain', lambda x: int((x > 1e-12).sum())),
        negative_repeats=('iou_gain', lambda x: int((x < -1e-12).sum())),
        **{name: (name, 'mean') for name in feature_columns})
    if not np.allclose(result[result.position_sigma_mm == 0].iou_gain, 0):
        raise AssertionError('Zero-noise H/P label identity failed')
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    focused = summary[(summary.event_count == 5000) & (summary.grid == 32)
                      & (summary.position_sigma_mm == .30)].copy()
    focused.to_csv(output_dir / 'focused_high_noise.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    focus_grid = summary[(summary.event_count == 5000) & (summary.grid == 32)
                         & (summary.position_sigma_mm > 0)]
    colors = {'old': 'tab:blue', 'frame': 'tab:orange', 'iron': 'tab:green'}
    markers = {0.05: 'o', 0.10: 's', 0.30: '^'}
    for family, color in colors.items():
        for sigma, marker in markers.items():
            part = focus_grid[(focus_grid.family == family)
                              & (focus_grid.position_sigma_mm == sigma)]
            axes[0].scatter(part.candidate_fraction, part.iou_gain_mean,
                            color=color, marker=marker, alpha=.85)
            axes[1].scatter(part.rescue_mass_fraction, part.iou_gain_mean,
                            color=color, marker=marker, alpha=.85)
    for ax, xlabel in zip(axes, ('Angle/XY candidate fraction',
                                 'Rescued scattering-weight fraction')):
        ax.axhline(0, color='black', lw=1)
        ax.set(xlabel=xlabel, ylabel='Paired mean IoU gain (P − H)')
        ax.grid(alpha=.25)
    family_legend = axes[0].legend(
        handles=[Line2D([0], [0], marker='o', linestyle='None',
                        color=color, label=family)
                 for family, color in colors.items()], title='Target family',
        loc='upper left', fontsize=8)
    axes[0].add_artist(family_legend)
    axes[0].legend(handles=[Line2D([0], [0], marker=marker, linestyle='None',
                                   color='black', label=f'{sigma:.2f} mm')
                            for sigma, marker in markers.items()],
                   title='Hit error σ', loc='lower left', fontsize=8)
    fig.savefig(output_dir / 'observed_features_vs_gain_5000_32.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'OBSERVED_Z_GATE_RISK_DIAGNOSTIC_PLAN.md',
        'role': 'post-hoc development diagnosis; no new test',
        'sources': [row[0] for row in SOURCES],
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM,
        'repeats': REPEATS, 'grids': GRIDS,
        'truth_used_only_for_saved_iou_labels': True,
        'candidate_and_gate_counts_reproduced': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(focused[['source', 'iou_gain_mean', 'candidate_count',
                   'theta_corrected_median_mrad', 'fractional_gate_fraction',
                   'rescue_mass_fraction', 'suppression_mass_fraction']]
          .sort_values('iou_gain_mean').to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/observed_z_gate_risk_results')
