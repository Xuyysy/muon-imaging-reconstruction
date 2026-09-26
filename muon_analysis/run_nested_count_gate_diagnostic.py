"""同粒子嵌套 N 诊断：不改已冻结的 H/P/Q 门，也不作新独立验证。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from diagnose_local_support_count_mechanism import support_details
from local_support_soft_z_gate import local_support_gate
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_local_support_gate_validation import (
    COUNTS, GRID, MATERIALS, REPEATS, ROOT, SIGMAS_MM, STREAMS, source_csv,
)
from run_plus_iron_fallback_validation import plus_mask, score_plus
from run_soft_z_gate_experiment import candidate_events, xy_image, z_acceptance_weights
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm


def run(output_dir: Path) -> None:
    """先计算完整 5000 组，再让 1000 组只读取前缀事件及传播抽样。"""
    if tuple(COUNTS) != (1000, 5000):
        raise AssertionError('Nested diagnostic is designed for N=1000/5000 only')
    edges = imaging_edges(GRID)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = plus_mask(cx, cy)  # 只进入 score_plus，不参与门控。
    rows = []
    checks = []
    source_counts = []
    for mi, material in enumerate(MATERIALS):
        for si, stream in enumerate(STREAMS):
            tracks = pd.read_csv(source_csv(material, stream))
            if len(tracks) < 5000:
                raise ValueError(f'Insufficient complete tracks: {material} {stream}')
            source_counts.append({'material': material, 'stream': stream,
                                  'complete_tracks': len(tracks)})
            spacing = same_side_layer_spacing_mm(tracks)
            rng = np.random.default_rng(RANDOM_SEED + 24017 + 1000 * si + 100 * mi)
            for repeat in range(1, REPEATS + 1):
                # 无放回随机排列；N=1000 为这一次 N=5000 的严格前缀。
                subset = tracks.iloc[rng.choice(len(tracks), 5000, replace=False)].copy()
                normals = {f'{plane}_{axis}': rng.normal(size=5000)
                           for plane in ('up1', 'up2', 'down1', 'down2')
                           for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, normals)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    full_ids, full_weight = candidate_events(observed, sigma, spacing)
                    full_xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[full_ids]
                    full_z = observed.poca_z_mm.to_numpy()[full_ids]
                    seed = (RANDOM_SEED + 2400000 + 300000 * mi + 100000 * si
                            + 100 * repeat + int(round(sigma * 1000)))
                    # 按 5000 组候选顺序只生成一次；若分别按 N 调用，向量长度
                    # 会改变 RNG 消耗顺序，同一事件传播误差将不再一致。
                    full_samples = estimate_z_samples(measured.iloc[full_ids],
                                                       sigma, seed, BOOTSTRAP_DRAWS)
                    full_h, full_p = z_acceptance_weights(full_z, full_samples)
                    prefix_observed = observed.iloc[:1000]
                    prefix_ids, prefix_weight = candidate_events(
                        prefix_observed, sigma, spacing)
                    in_prefix = full_ids < 1000
                    if not (np.array_equal(prefix_ids, full_ids[in_prefix])
                            and np.array_equal(prefix_weight, full_weight[in_prefix])):
                        raise AssertionError('Candidate events differ for shared particles')
                    prefix_xy = full_xy[in_prefix]
                    prefix_z = full_z[in_prefix]
                    prefix_samples = full_samples[in_prefix]
                    prefix_h, prefix_p = z_acceptance_weights(prefix_z, prefix_samples)
                    if not (np.array_equal(prefix_h, full_h[in_prefix])
                            and np.array_equal(prefix_p, full_p[in_prefix])):
                        raise AssertionError('H/P differ for shared particles')
                    # 记录逐事件对齐的机器可审计检查；零差异来自数组恒等。
                    checks.append({'material': material, 'stream': stream,
                                   'repeat': repeat, 'position_sigma_mm': sigma,
                                   'shared_candidates': len(prefix_ids),
                                   'candidate_ids_equal': True,
                                   'weights_equal': True, 'z_samples_equal': True,
                                   'h_p_equal': True})
                    for count, xy, weight, h, p in (
                        (1000, prefix_xy, prefix_weight, prefix_h, prefix_p),
                        (5000, full_xy, full_weight, full_h, full_p),
                    ):
                        q = local_support_gate(xy, weight, h, p)
                        if sigma == 0 and not (np.array_equal(h, p)
                                               and np.array_equal(h, q)):
                            raise AssertionError('H/P/Q must agree at zero noise')
                        detail = support_details(xy, weight, h, p, q)
                        scores = {
                            name: score_plus(xy_image(xy, weight * gate, GRID),
                                             truth, cx, cy)['iou']
                            for name, gate in (('h', h), ('p', p), ('q', q))
                        }
                        rows.append({'material': material, 'stream': stream,
                                     'event_count': count, 'repeat': repeat,
                                     'position_sigma_mm': sigma, **detail,
                                     'h_iou': scores['h'], 'p_iou': scores['p'],
                                     'q_iou': scores['q'],
                                     'q_minus_h': scores['q'] - scores['h'],
                                     'q_minus_p': scores['q'] - scores['p']})
                print(f'Nested N: {material} {stream}, repeat {repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    key = ['material', 'stream', 'repeat', 'position_sigma_mm', 'event_count']
    if (len(frame) != 4 * REPEATS * len(SIGMAS_MM) * 2
            or frame.duplicated(key).any()):
        raise AssertionError('Incomplete or duplicated nested matrix')
    summary = frame.groupby(['material', 'stream', 'position_sigma_mm',
                             'event_count'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{col: (col, 'mean') for col in frame.columns if col not in key})
    wide = frame.pivot(index=['material', 'stream', 'repeat', 'position_sigma_mm'],
                       columns='event_count', values=['q_minus_p', 'q_minus_h',
                                                      'kept_rescue_fraction',
                                                      'hard_neff_ge2_cells'])
    delta = pd.DataFrame(index=wide.index)
    for metric in ('q_minus_p', 'q_minus_h', 'kept_rescue_fraction',
                   'hard_neff_ge2_cells'):
        delta[f'delta_{metric}_5000_minus_1000'] = wide[metric][5000] - wide[metric][1000]
    delta = delta.reset_index()
    if (frame.loc[frame.position_sigma_mm == 0, ['q_minus_p', 'q_minus_h']]
            .to_numpy() != 0).any():
        raise AssertionError('Zero-noise IoU differences must vanish')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    delta.to_csv(output_dir / 'paired_count_delta.csv', index=False)
    pd.DataFrame(checks).to_csv(output_dir / 'event_identity_checks.csv', index=False)
    pd.DataFrame(source_counts).to_csv(output_dir / 'source_counts.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'NESTED_COUNT_GATE_DIAGNOSTIC_PLAN.md',
        'role': 'post-hoc nested-count mechanism, not independent validation',
        'counts': COUNTS, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'grid': GRID, 'draws': BOOTSTRAP_DRAWS,
        'shared_event_measurement_and_propagation': True,
        'truth_only_for_final_scoring': True,
        'frozen_gate': 'Q = h*p + (1-h)*p*n_eff/(n_eff+2)',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    focus = summary[summary.position_sigma_mm == .30]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for material, color in (('al', 'tab:blue'), ('iron', 'tab:red')):
        for stream, marker in (('a', 'o'), ('b', 's')):
            part = focus[(focus.material == material) & (focus.stream == stream)]
            part = part.sort_values('event_count')
            axes[0].plot(part.event_count, part.kept_rescue_fraction,
                         color=color, marker=marker, label=f'{material} {stream}')
            axes[1].plot(part.event_count, part.q_minus_p,
                         color=color, marker=marker, label=f'{material} {stream}')
    axes[1].axhline(0, color='black', lw=1)
    for ax in axes:
        ax.set_xticks(COUNTS)
        ax.set_xlabel('Nested input event count')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('P rescue mass retained by Q')
    axes[1].set_ylabel('Mean equal-area IoU: Q − P')
    axes[0].legend(fontsize=8)
    fig.suptitle('Post-hoc nested-count diagnostic, hit σ=0.30 mm')
    fig.savefig(output_dir / 'nested_count_gate_diagnostic.png', dpi=180)
    plt.close(fig)
    print(focus[['material', 'stream', 'event_count', 'hard_neff_ge2_cells',
                 'kept_rescue_fraction', 'q_minus_p', 'q_minus_h']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/nested_count_gate_diagnostic_results')
