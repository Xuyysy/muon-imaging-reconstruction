"""同形状十字 G4_Al/G4_Fe 控制诊断；不把同种子误称同缪子。

先读 PLUS_MATERIAL_CONTROL_PLAN.md。H/P 原算法不变；目标真值仅用于评价。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from diagnose_observed_z_gate_risk import observed_features
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_plus_iron_fallback_validation import plus_mask, score_plus
from run_soft_z_gate_experiment import (
    COUNTS, GRIDS, REPEATS, ROOT, SIGMAS_MM, candidate_events, xy_image,
    z_acceptance_weights,
)
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

MATERIALS = ('al', 'iron')
STREAMS = ('a', 'b')
KEYS = ['material', 'stream', 'event_count', 'repeat', 'position_sigma_mm', 'grid']
FEATURES = (
    'candidate_count', 'candidate_fraction', 'hard_z_count', 'soft_z_count',
    'theta_corrected_median_mrad', 'theta_corrected_p25_mrad',
    'fractional_gate_fraction', 'rescue_mass_fraction',
    'suppression_mass_fraction', 'soft_to_hard_mass_ratio',
)


def csv_for(material: str, stream: str) -> Path:
    if material not in MATERIALS or stream not in STREAMS:
        raise ValueError('Unknown material or Geant4 stream')
    name = 'plus_al' if material == 'al' else 'plus_iron'
    return ROOT / 'muon_analysis' / f'muon_tracks_{name}_seed_{stream}.csv'


def upstream_exact_match_fraction(al: pd.DataFrame,
                                  iron: pd.DataFrame) -> tuple[int, float]:
    """事件编号相同不保证源/上游轨迹相同；只做限制核查。"""
    merged = al.merge(iron, on='event', suffixes=('_al', '_iron'),
                      validate='one_to_one')
    fields = ('up1_x_mm', 'up1_y_mm', 'up2_x_mm', 'up2_y_mm')
    same = np.column_stack([
        merged[f'{field}_al'].to_numpy() == merged[f'{field}_iron'].to_numpy()
        for field in fields
    ]).all(axis=1)
    return len(merged), float(same.mean())


def run(output_dir: Path) -> None:
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = plus_mask(cx, cy)
    data = {(material, stream): pd.read_csv(csv_for(material, stream))
            for material in MATERIALS for stream in STREAMS}
    pairing_rows = []
    for stream in STREAMS:
        n, fraction = upstream_exact_match_fraction(data[('al', stream)],
                                                    data[('iron', stream)])
        pairing_rows.append({'stream': stream, 'common_event_ids': n,
                             'exact_four_upstream_coordinates_fraction': fraction})
    old_iron = pd.read_csv(ROOT / 'muon_analysis/plus_iron_fallback_results/by_repeat.csv')
    rows = []
    for material in MATERIALS:
        for stream_index, stream in enumerate(STREAMS):
            tracks = data[(material, stream)]
            if len(tracks) < max(COUNTS):
                raise ValueError(f'{material} {stream}: too few complete tracks')
            spacing = same_side_layer_spacing_mm(tracks)
            # 同一分析 RNG 种子给两种材料提供同样的索引/噪声生成规则；
            # 因原始 Geant4 轨迹和记录条数已分叉，这仍不是逐事件物理配对。
            rng = np.random.default_rng(RANDOM_SEED + 8107 + 1000 * stream_index)
            for count in COUNTS:
                for repeat in range(1, REPEATS + 1):
                    subset = tracks.iloc[rng.choice(len(tracks), count,
                                                   replace=False)].copy()
                    normals = {f'{plane}_{axis}': rng.normal(0, 1, count)
                               for plane in ('up1', 'up2', 'down1', 'down2')
                               for axis in ('x', 'y')}
                    for sigma in SIGMAS_MM:
                        measured = measured_hits(subset, sigma, normals)
                        observed = recompute_from_noisy_hits(
                            measured, 0., np.random.default_rng(0))
                        ids, weight = candidate_events(observed, sigma, spacing)
                        z = observed.poca_z_mm.to_numpy()[ids]
                        # 与旧铁目标分析相同；铝目标用同一传播种子方案。
                        seed = (RANDOM_SEED + 900000 + 300000 * stream_index
                                + 10000 * count + 100 * repeat
                                + int(round(sigma * 1000)))
                        samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                     seed, BOOTSTRAP_DRAWS)
                        hard, propagated = z_acceptance_weights(z, samples)
                        if sigma == 0 and not np.array_equal(hard, propagated):
                            raise AssertionError('Zero-noise H/P differ')
                        feature = observed_features(observed, ids, weight,
                                                    samples, sigma, spacing, count)
                        xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                        for grid in GRIDS:
                            for method, gate in (('hard', hard),
                                                 ('propagated', propagated)):
                                image = xy_image(xy, weight * gate, grid)
                                rows.append({
                                    'material': material, 'stream': stream,
                                    'event_count': count, 'repeat': repeat,
                                    'position_sigma_mm': sigma, 'grid': grid,
                                    'method': method, **feature,
                                    **score_plus(image, truth, cx, cy),
                                })
                    print(f'Plus material: {material}, {stream}, N={count},'
                          f' repeat={repeat}/{REPEATS}', flush=True)
    by_repeat = pd.DataFrame(rows)
    expected = len(MATERIALS) * len(STREAMS) * len(COUNTS) * REPEATS \
        * len(SIGMAS_MM) * len(GRIDS) * 2
    if len(by_repeat) != expected or by_repeat.duplicated(KEYS + ['method']).any():
        raise AssertionError('Incomplete or duplicate conditions')
    h = by_repeat[by_repeat.method == 'hard'].set_index(KEYS).sort_index()
    p = by_repeat[by_repeat.method == 'propagated'].set_index(KEYS).sort_index()
    if not h.index.equals(p.index):
        raise AssertionError('Unpaired H/P methods')
    paired = h[list(FEATURES) + ['iou', 'corner_false_positive_fraction']].copy()
    paired = paired.rename(columns={'iou': 'hard_iou',
                                    'corner_false_positive_fraction': 'hard_corner_fp'})
    paired['propagated_iou'] = p.iou
    paired['propagated_corner_fp'] = p.corner_false_positive_fraction
    paired['iou_gain'] = p.iou - h.iou
    paired = paired.reset_index()
    if not np.allclose(paired[paired.position_sigma_mm == 0].iou_gain, 0):
        raise AssertionError('Zero-noise score identity failed')
    # 旧铁数据必须逐条件重现，才能把铝比较接到原研究链路中。
    for stream in STREAMS:
        old = old_iron[(old_iron.stream == f'seed_{stream}')
                       & old_iron.method.isin(['hard', 'propagated'])]
        new = by_repeat[(by_repeat.material == 'iron')
                        & (by_repeat.stream == stream)]
        key = ['event_count', 'repeat', 'position_sigma_mm', 'grid', 'method']
        old = old.set_index(key).sort_index()
        new = new.set_index(key).sort_index()
        if len(old) != len(new) or not old.index.equals(new.index) \
                or not np.allclose(old.iou, new.iou, rtol=0, atol=1e-12) \
                or not np.array_equal(old.candidates, new.candidate_count):
            raise AssertionError(f'Iron stream {stream} could not be reproduced')
    summary = paired.groupby(['material', 'stream', 'event_count',
                              'position_sigma_mm', 'grid'], as_index=False).agg(
        repeats=('repeat', 'size'),
        hard_iou_mean=('hard_iou', 'mean'),
        propagated_iou_mean=('propagated_iou', 'mean'),
        iou_gain_mean=('iou_gain', 'mean'),
        hard_corner_fp_mean=('hard_corner_fp', 'mean'),
        propagated_corner_fp_mean=('propagated_corner_fp', 'mean'),
        **{name: (name, 'mean') for name in FEATURES})
    # 与铁随机流同编号只作为成组描述，不视作同一入射粒子的配对估计。
    contrast_rows = []
    for stream in STREAMS:
        for count in COUNTS:
            for sigma in SIGMAS_MM:
                for grid in GRIDS:
                    key = ((summary.stream == stream)
                           & (summary.event_count == count)
                           & (summary.position_sigma_mm == sigma)
                           & (summary.grid == grid))
                    one = summary[key].set_index('material')
                    if set(one.index) != set(MATERIALS):
                        raise AssertionError('Missing material-control cell')
                    fields = ('candidate_fraction', 'candidate_count',
                              'theta_corrected_median_mrad',
                              'rescue_mass_fraction',
                              'suppression_mass_fraction', 'iou_gain_mean')
                    contrast_rows.append({
                        'stream': stream, 'event_count': count,
                        'position_sigma_mm': sigma, 'grid': grid,
                        **{f'{field}_al': float(one.loc['al', field])
                           for field in fields},
                        **{f'{field}_iron': float(one.loc['iron', field])
                           for field in fields},
                        **{f'{field}_iron_minus_al': float(one.loc['iron', field]
                                                         - one.loc['al', field])
                           for field in fields},
                    })
    contrast = pd.DataFrame(contrast_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_repeat.to_csv(output_dir / 'by_repeat.csv', index=False)
    paired.to_csv(output_dir / 'paired_by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    contrast.to_csv(output_dir / 'material_contrast.csv', index=False)
    pd.DataFrame(pairing_rows).to_csv(output_dir / 'upstream_pairing_check.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for material, color in (('al', 'tab:blue'), ('iron', 'tab:red')):
        for stream, marker in (('a', 'o'), ('b', 's')):
            part = summary[(summary.material == material)
                           & (summary.stream == stream)
                           & (summary.event_count == 5000)
                           & (summary.grid == 32)].sort_values('position_sigma_mm')
            label = f'{material}, seed {stream}'
            axes[0].plot(part.position_sigma_mm,
                         part.theta_corrected_median_mrad,
                         color=color, marker=marker, label=label)
            axes[1].plot(part.position_sigma_mm, part.iou_gain_mean,
                         color=color, marker=marker, label=label)
    axes[0].set(ylabel='Median corrected scattering angle [mrad]')
    axes[1].set(ylabel='Mean equal-area IoU gain (P − H)')
    axes[1].axhline(0, color='black', lw=1)
    for ax in axes:
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / 'plus_material_control_5000_32.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'PLUS_MATERIAL_CONTROL_PLAN.md',
        'role': 'same plus geometry material-control diagnosis, not rule validation',
        'materials': MATERIALS, 'streams': STREAMS,
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'grids': GRIDS,
        'repeats': REPEATS, 'draws': BOOTSTRAP_DRAWS,
        'source_seeds_reused': True,
        'event_level_material_pairing_assumed': False,
        'old_iron_iou_and_candidate_count_reproduced': True,
        'target_truth_only_for_iou_and_corner_fp': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary[(summary.event_count == 5000) & (summary.grid == 32)]
          [['material', 'stream', 'position_sigma_mm', 'candidate_count',
            'theta_corrected_median_mrad', 'rescue_mass_fraction',
            'iou_gain_mean']].to_string(index=False))
    print(pd.DataFrame(pairing_rows).to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/plus_material_control_results')
