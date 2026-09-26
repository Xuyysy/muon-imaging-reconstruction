"""冻结 Q 失败后的 N=1000/5000 局部支撑机制诊断；不改门控。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from local_support_soft_z_gate import SUPPORT_GRID, SUPPORT_SHRINK, local_support_gate
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_local_support_gate_validation import (
    COUNTS, GRID, KEY, MATERIALS, REPEATS, ROOT, SIGMAS_MM, STREAMS, source_csv,
)
from run_plus_iron_fallback_validation import plus_mask, score_plus
from run_soft_z_gate_experiment import candidate_events, xy_image, z_acceptance_weights
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm


def support_details(xy: np.ndarray, weight: np.ndarray,
                    hard: np.ndarray, propagated: np.ndarray,
                    q_gate: np.ndarray) -> dict[str, float]:
    """只用观测位置、原权重和门系数；目标真值不在函数输入中。"""
    xy = np.asarray(xy, float)
    w = np.asarray(weight, float)
    h = np.asarray(hard, float)
    p = np.asarray(propagated, float)
    q_gate = np.asarray(q_gate, float)
    if xy.shape != (len(w), 2) or not (len(w) == len(h) == len(p) == len(q_gate)):
        raise ValueError('Unaligned event arrays')
    if not (np.isfinite(xy).all() and np.isfinite(w).all()
            and np.isfinite(h).all() and np.isfinite(p).all()
            and np.isfinite(q_gate).all()):
        raise ValueError('Nonfinite event values')
    x_edge, y_edge = imaging_edges(SUPPORT_GRID)[:2]
    hard_weight = w * h
    hard_image, _, _ = np.histogram2d(xy[:, 0], xy[:, 1],
                                      bins=(x_edge, y_edge), weights=hard_weight)
    hard_square, _, _ = np.histogram2d(xy[:, 0], xy[:, 1],
                                       bins=(x_edge, y_edge), weights=hard_weight**2)
    neff = np.divide(hard_image**2, hard_square,
                     out=np.zeros_like(hard_image), where=hard_square > 0)
    q_map = neff / (neff + SUPPORT_SHRINK)
    ix = np.clip(np.searchsorted(x_edge, xy[:, 0], side='right') - 1,
                 0, SUPPORT_GRID - 1)
    iy = np.clip(np.searchsorted(y_edge, xy[:, 1], side='right') - 1,
                 0, SUPPORT_GRID - 1)
    local_q = q_map[ix, iy]
    expected = h * p + (1 - h) * p * local_q
    if not np.allclose(expected, q_gate, rtol=1e-12, atol=1e-12):
        raise AssertionError('Q implementation and diagnostic q-map disagree')
    rescue = w * (1 - h) * p
    mass = float(rescue.sum())
    kept = float(np.dot(rescue, local_q))
    return {
        'candidate_count': len(w),
        'hard_accepted_count': int(h.sum()),
        'hard_occupied_cells': int(np.sum(hard_image > 0)),
        'hard_neff_ge2_cells': int(np.sum(neff >= 2)),
        'rescue_mass': mass,
        'kept_rescue_mass': kept,
        'kept_rescue_fraction': kept / mass if mass > 0 else np.nan,
        'rescue_mass_in_empty_fraction':
            float(rescue[neff[ix, iy] == 0].sum() / mass) if mass > 0 else np.nan,
        'rescue_mass_in_neff_ge2_fraction':
            float(rescue[neff[ix, iy] >= 2].sum() / mass) if mass > 0 else np.nan,
        'rescue_weighted_neff':
            float(np.dot(rescue, neff[ix, iy]) / mass) if mass > 0 else np.nan,
    }


def run(output_dir: Path) -> None:
    previous = pd.read_csv(ROOT / 'muon_analysis/local_support_gate_validation_results/paired_by_repeat.csv')
    if previous.duplicated(KEY).any():
        raise AssertionError('Duplicate prior results')
    previous = previous.set_index(KEY).sort_index()
    edges = imaging_edges(GRID)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = plus_mask(cx, cy)  # 仅供最终复现旧 IoU，不进入 support_details。
    rows = []
    for material_index, material in enumerate(MATERIALS):
        for stream_index, stream in enumerate(STREAMS):
            tracks = pd.read_csv(source_csv(material, stream))
            spacing = same_side_layer_spacing_mm(tracks)
            rng = np.random.default_rng(RANDOM_SEED + 12013
                                        + 1000 * stream_index + 100 * material_index)
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
                        ids, w = candidate_events(observed, sigma, spacing)
                        xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                        z = observed.poca_z_mm.to_numpy()[ids]
                        seed = (RANDOM_SEED + 1200000 + 300000 * material_index
                                + 100000 * stream_index + 10000 * count
                                + 100 * repeat + int(round(sigma * 1000)))
                        samples = estimate_z_samples(measured.iloc[ids], sigma,
                                                     seed, BOOTSTRAP_DRAWS)
                        h, p = z_acceptance_weights(z, samples)
                        q = local_support_gate(xy, w, h, p)
                        features = support_details(xy, w, h, p, q)
                        key = material, stream, count, repeat, sigma
                        old = previous.loc[key]
                        if (int(old.candidate_count) != features['candidate_count']
                                or int(old.hard_z_count) != features['hard_accepted_count']):
                            raise AssertionError(f'Prior event counts mismatch: {key}')
                        # 对三幅图分别重新评分，保证读取的旧 IoU 标签逐次对应同一批事件。
                        scores = {name: score_plus(xy_image(xy, w * gate, GRID),
                                                   truth, cx, cy)['iou']
                                  for name, gate in (('h', h), ('p', p), ('q', q))}
                        if any(not np.isclose(scores[name], float(old[f'{name}_iou']),
                                               rtol=0, atol=1e-12)
                               for name in ('h', 'p', 'q')):
                            raise AssertionError(f'Prior H/P/Q IoU mismatch: {key}')
                        if sigma == 0 and (features['rescue_mass'] != 0
                                           or not np.isnan(features['kept_rescue_fraction'])):
                            raise AssertionError('Zero-noise rescue fraction should be undefined')
                        rows.append({
                            'material': material, 'stream': stream,
                            'event_count': count, 'repeat': repeat,
                            'position_sigma_mm': sigma, **features,
                            'q_minus_p_label': float(old.q_minus_p),
                            'q_minus_h_label': float(old.q_minus_h),
                            'p_minus_h_label': float(old.p_minus_h),
                        })
                    print(f'Count mechanism: {material}, {stream}, N={count}, '
                          f'repeat {repeat}/{REPEATS}', flush=True)
    frame = pd.DataFrame(rows)
    expected = len(MATERIALS) * len(STREAMS) * len(COUNTS) * REPEATS * len(SIGMAS_MM)
    if len(frame) != expected or frame.duplicated(KEY).any():
        raise AssertionError('Incomplete diagnostic matrix')
    summary = frame.groupby(['material', 'stream', 'event_count',
                             'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{col: (col, 'mean') for col in frame.columns if col not in KEY})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    focus = summary[summary.position_sigma_mm == .30]
    for material, color in (('al', 'tab:blue'), ('iron', 'tab:red')):
        for stream, marker in (('a', 'o'), ('b', 's')):
            part = focus[(focus.material == material) & (focus.stream == stream)]
            part = part.sort_values('event_count')
            label = f'{material} {stream}'
            axes[0].plot(part.event_count, part.kept_rescue_fraction,
                         color=color, marker=marker, label=label)
            axes[1].plot(part.event_count, part.q_minus_p_label,
                         color=color, marker=marker, label=label)
    axes[0].set(ylabel='Fraction of P rescue mass retained by Q')
    axes[1].set(ylabel='Mean equal-area IoU difference Q − P')
    axes[1].axhline(0, color='black', lw=1)
    for ax in axes:
        ax.set_xlabel('Input event count')
        ax.set_xticks(COUNTS)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Development-only 75 mm count mechanism, hit σ=0.30 mm')
    fig.savefig(output_dir / 'local_support_count_mechanism.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'LOCAL_SUPPORT_COUNT_MECHANISM_PLAN.md',
        'role': 'post-hoc diagnosis of frozen Q failure, not new validation',
        'materials': MATERIALS, 'streams': STREAMS, 'counts': COUNTS,
        'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS, 'draws': BOOTSTRAP_DRAWS,
        'old_h_p_q_results_reproduced': True,
        'truth_only_for_reproduction_checks': True,
        'no_new_gate_parameters_or_threshold_fit': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(focus[['material', 'stream', 'event_count', 'hard_occupied_cells',
                 'hard_neff_ge2_cells', 'rescue_mass', 'kept_rescue_fraction',
                 'rescue_mass_in_empty_fraction', 'q_minus_p_label',
                 'q_minus_h_label']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/local_support_count_mechanism_results')
