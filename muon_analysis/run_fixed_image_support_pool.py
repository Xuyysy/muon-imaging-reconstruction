"""固定 1000 条成像事件，只改变 Q 的硬门局部支撑估计池。"""
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
from run_adaptive_voxel_experiment import RANDOM_SEED, Z_LIMIT_MM, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_local_support_gate_validation import (
    GRID, MATERIALS, REPEATS, ROOT, SIGMAS_MM, STREAMS, source_csv,
)
from run_plus_iron_fallback_validation import plus_mask, score_plus
from run_soft_z_gate_experiment import candidate_events, xy_image, z_acceptance_weights
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

IMAGE_COUNT = 1000
SUPPORT_COUNTS = (1000, 5000)


def gate_from_support_pool(image_xy: np.ndarray, image_w: np.ndarray,
                           image_h: np.ndarray, image_p: np.ndarray,
                           pool_xy: np.ndarray, pool_w: np.ndarray,
                           pool_h: np.ndarray) -> tuple[np.ndarray, dict]:
    """只从支撑池建立硬门 n_eff；返回成像事件的冻结 Q 和诊断量。

    这里没有目标真值。pool 事件绝不加入成像图的权重；当 pool 恰为
    image 事件时，这个函数必须逐事件等于原 local_support_gate。
    """
    x_edges, y_edges = imaging_edges(SUPPORT_GRID)[:2]
    hard_w = pool_w * pool_h
    total, _, _ = np.histogram2d(pool_xy[:, 0], pool_xy[:, 1],
                                  bins=(x_edges, y_edges), weights=hard_w)
    squared, _, _ = np.histogram2d(pool_xy[:, 0], pool_xy[:, 1],
                                    bins=(x_edges, y_edges), weights=hard_w**2)
    neff = np.divide(total**2, squared, out=np.zeros_like(total), where=squared > 0)
    ix = np.clip(np.searchsorted(x_edges, image_xy[:, 0], side='right') - 1,
                 0, SUPPORT_GRID - 1)
    iy = np.clip(np.searchsorted(y_edges, image_xy[:, 1], side='right') - 1,
                 0, SUPPORT_GRID - 1)
    local_neff = neff[ix, iy]
    local_q = local_neff / (local_neff + SUPPORT_SHRINK)
    gate = image_h * image_p + (1 - image_h) * image_p * local_q
    rescue = image_w * (1 - image_h) * image_p
    rescue_mass = float(rescue.sum())
    if np.any(gate < -1e-12) or np.any(gate > image_p + 1e-12):
        raise AssertionError('Q must stay between 0 and P')
    return gate, {
        'support_hard_neff_ge2_cells': int((neff >= 2).sum()),
        'image_rescue_mass': rescue_mass,
        'image_rescue_kept_fraction': (
            float(np.dot(rescue, local_q) / rescue_mass)
            if rescue_mass > 0 else np.nan),
        'image_rescue_weighted_support_neff': (
            float(np.dot(rescue, local_neff) / rescue_mass)
            if rescue_mass > 0 else np.nan),
    }


def run(output_dir: Path) -> None:
    """重用嵌套实验的随机序列，核对自身的 S=1000 基线。"""
    previous_path = (ROOT / 'muon_analysis/nested_count_gate_diagnostic_results'
                     / 'by_repeat.csv')
    previous = pd.read_csv(previous_path)
    previous = previous[previous.event_count == IMAGE_COUNT]
    previous = previous.set_index(['material', 'stream', 'repeat',
                                   'position_sigma_mm']).sort_index()
    edges = imaging_edges(GRID)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    truth = plus_mask(cx, cy)  # 只用于图像评分。
    rows = []
    checks = []
    for mi, material in enumerate(MATERIALS):
        for si, stream in enumerate(STREAMS):
            tracks = pd.read_csv(source_csv(material, stream))
            spacing = same_side_layer_spacing_mm(tracks)
            rng = np.random.default_rng(RANDOM_SEED + 24017 + 1000 * si + 100 * mi)
            for repeat in range(1, REPEATS + 1):
                subset = tracks.iloc[rng.choice(len(tracks), 5000, replace=False)].copy()
                normals = {f'{plane}_{axis}': rng.normal(size=5000)
                           for plane in ('up1', 'up2', 'down1', 'down2')
                           for axis in ('x', 'y')}
                for sigma in SIGMAS_MM:
                    measured = measured_hits(subset, sigma, normals)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    ids, full_w = candidate_events(observed, sigma, spacing)
                    full_xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                    full_z = observed.poca_z_mm.to_numpy()[ids]
                    full_h = (np.abs(full_z) <= Z_LIMIT_MM).astype(float)
                    in_image = ids < IMAGE_COUNT
                    image_xy = full_xy[in_image]
                    image_w = full_w[in_image]
                    image_h = full_h[in_image]
                    seed = (RANDOM_SEED + 2400000 + 300000 * mi + 100000 * si
                            + 100 * repeat + int(round(sigma * 1000)))
                    # 上轮先为 5000 组全部候选生成传播抽样；这里保持同一
                    # RNG 消耗顺序，再仅使用前 1000 成像事件的样本行。
                    full_samples = estimate_z_samples(measured.iloc[ids],
                                                       sigma, seed, BOOTSTRAP_DRAWS)
                    h_check, full_p = z_acceptance_weights(full_z, full_samples)
                    if not np.array_equal(full_h, h_check):
                        raise AssertionError('Nominal H differs from z acceptance')
                    image_p = full_p[in_image]
                    p_image = xy_image(image_xy, image_w * image_p, GRID)
                    h_image = xy_image(image_xy, image_w * image_h, GRID)
                    p_iou = score_plus(p_image, truth, cx, cy)['iou']
                    h_iou = score_plus(h_image, truth, cx, cy)['iou']
                    old = previous.loc[(material, stream, repeat, sigma)]
                    if not (np.isclose(p_iou, old.p_iou, rtol=0, atol=1e-12)
                            and np.isclose(h_iou, old.h_iou, rtol=0, atol=1e-12)):
                        raise AssertionError('Imaging baseline differs from nested experiment')
                    per_support = {}
                    for support_count in SUPPORT_COUNTS:
                        in_pool = ids < support_count
                        q, detail = gate_from_support_pool(
                            image_xy, image_w, image_h, image_p,
                            full_xy[in_pool], full_w[in_pool], full_h[in_pool])
                        if support_count == IMAGE_COUNT:
                            frozen = local_support_gate(
                                image_xy, image_w, image_h, image_p)
                            if not np.array_equal(q, frozen):
                                raise AssertionError('Self-pool Q differs from frozen Q')
                        q_image = xy_image(image_xy, image_w * q, GRID)
                        q_iou = score_plus(q_image, truth, cx, cy)['iou']
                        if support_count == IMAGE_COUNT and not np.isclose(
                                q_iou, old.q_iou, rtol=0, atol=1e-12):
                            raise AssertionError('Self-pool Q IoU differs from archive')
                        if sigma == 0 and (not np.array_equal(q, image_h)
                                           or not np.array_equal(q_image, p_image)):
                            raise AssertionError('Zero-noise H/P/Q image mismatch')
                        per_support[support_count] = (q, q_image)
                        rows.append({
                            'material': material, 'stream': stream,
                            'repeat': repeat, 'position_sigma_mm': sigma,
                            'image_event_count': IMAGE_COUNT,
                            'support_event_count': support_count,
                            'image_candidate_count': len(image_w),
                            'support_candidate_count': int(in_pool.sum()),
                            **detail, 'h_iou': h_iou, 'p_iou': p_iou,
                            'q_iou': q_iou, 'q_minus_p': q_iou - p_iou,
                            'q_minus_h': q_iou - h_iou,
                        })
                    checks.append({
                        'material': material, 'stream': stream,
                        'repeat': repeat, 'position_sigma_mm': sigma,
                        'image_candidate_count': len(image_w),
                        'image_h_p_reproduced': True,
                        'self_pool_gate_equal_frozen': True,
                        'self_pool_q_iou_reproduced': True,
                        'h_p_images_fixed': True,
                        'q_image_max_abs_change': float(np.max(np.abs(
                            per_support[5000][1] - per_support[1000][1]))),
                    })
                print(f'Fixed-image support: {material} {stream}, '
                      f'repeat {repeat}/{REPEATS}', flush=True)
    frame = pd.DataFrame(rows)
    key = ['material', 'stream', 'repeat', 'position_sigma_mm',
           'image_event_count', 'support_event_count']
    if len(frame) != 4 * REPEATS * len(SIGMAS_MM) * 2 or frame.duplicated(key).any():
        raise AssertionError('Incomplete or duplicated fixed-image matrix')
    summary = frame.groupby(['material', 'stream', 'position_sigma_mm',
                             'image_event_count', 'support_event_count'],
                            as_index=False).agg(
        repeats=('repeat', 'size'),
        **{col: (col, 'mean') for col in frame.columns if col not in key})
    wide = frame.pivot(index=['material', 'stream', 'repeat', 'position_sigma_mm'],
                       columns='support_event_count',
                       values=['q_iou', 'q_minus_p', 'image_rescue_kept_fraction',
                               'support_hard_neff_ge2_cells'])
    delta = pd.DataFrame(index=wide.index)
    for metric in ('q_iou', 'q_minus_p', 'image_rescue_kept_fraction',
                   'support_hard_neff_ge2_cells'):
        delta[f'delta_{metric}_pool5000_minus_pool1000'] = (
            wide[metric][5000] - wide[metric][1000])
    delta = delta.reset_index()
    # 路径分解：先固定 1000 条图像扩充支撑池，再在同一个 5000 支撑池
    # 下扩充成像事件。加法严格成立，但顺序改变时分量也可能改变。
    final = pd.read_csv(ROOT / 'muon_analysis/nested_count_gate_diagnostic_results'
                        / 'by_repeat.csv')
    final = final[final.event_count == 5000].set_index(
        ['material', 'stream', 'repeat', 'position_sigma_mm']).sort_index()
    path = pd.DataFrame(index=wide.index)
    path['baseline_image1000_pool1000'] = wide['q_minus_p'][1000]
    path['intermediate_image1000_pool5000'] = wide['q_minus_p'][5000]
    if not path.index.equals(final.index):
        raise AssertionError('Full-image and support-only pairs differ')
    path['final_image5000_pool5000'] = final.q_minus_p
    path['support_only_delta'] = (path.intermediate_image1000_pool5000
                                  - path.baseline_image1000_pool1000)
    path['additional_image_delta_at_pool5000'] = (
        path.final_image5000_pool5000 - path.intermediate_image1000_pool5000)
    path['total_delta'] = (path.final_image5000_pool5000
                           - path.baseline_image1000_pool1000)
    if not np.allclose(path.support_only_delta
                       + path.additional_image_delta_at_pool5000,
                       path.total_delta, rtol=0, atol=1e-12):
        raise AssertionError('Path decomposition does not add up')
    path = path.reset_index()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    delta.to_csv(output_dir / 'paired_support_delta.csv', index=False)
    path.to_csv(output_dir / 'path_decomposition.csv', index=False)
    pd.DataFrame(checks).to_csv(output_dir / 'identity_checks.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'FIXED_IMAGE_SUPPORT_POOL_PLAN.md',
        'role': 'post-hoc support-only software counterfactual, not independent validation',
        'image_event_count': IMAGE_COUNT,
        'support_counts': SUPPORT_COUNTS,
        'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'grid': GRID, 'draws': BOOTSTRAP_DRAWS,
        'extra_events_do_not_enter_image': True,
        'path_order': 'support pool first at fixed image, then add imaging events',
        'truth_only_for_final_scoring': True,
        'formula': 'Q = h*p + (1-h)*p*n_eff/(n_eff+2)',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    focus = summary[summary.position_sigma_mm == .30]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for material, color in (('al', 'tab:blue'), ('iron', 'tab:red')):
        for stream, marker in (('a', 'o'), ('b', 's')):
            part = focus[(focus.material == material) & (focus.stream == stream)]
            part = part.sort_values('support_event_count')
            for ax, field in zip(axes, ('image_rescue_kept_fraction', 'q_minus_p')):
                ax.plot(part.support_event_count, part[field], color=color,
                        marker=marker, label=f'{material} {stream}')
    axes[1].axhline(0, color='black', lw=1)
    axes[0].set_ylabel('P rescue mass retained by Q')
    axes[1].set_ylabel('Fixed-image IoU: Q − P')
    for ax in axes:
        ax.set_xticks(SUPPORT_COUNTS)
        ax.set_xlabel('Support-pool events; image fixed at 1000')
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Support-only counterfactual, hit σ=0.30 mm')
    fig.savefig(output_dir / 'fixed_image_support_pool.png', dpi=180)
    plt.close(fig)
    print(focus[['material', 'stream', 'support_event_count',
                 'image_rescue_kept_fraction', 'q_minus_p',
                 'support_hard_neff_ge2_cells']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/fixed_image_support_pool_results')
