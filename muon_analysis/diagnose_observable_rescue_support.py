"""已见十字目标上的无真值回流空间支撑诊断；不拟合门控阈值。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_plus_thickness_control import (
    COUNT, KEY, MATERIALS, REPEATS, ROOT, SIGMAS_MM, STREAMS,
    THICKNESSES_MM, advance_to_count_5000, csv_for,
)
from run_soft_z_gate_experiment import candidate_events, z_acceptance_weights
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

SUPPORT_GRID = 16
MIN_EFFECTIVE_COUNT = 2.0


def spatial_support_features(xy: np.ndarray, hard_weight: np.ndarray,
                             rescue_weight: np.ndarray) -> dict[str, float]:
    """只由名义观测 PoCA-XY 及 H/R 权重组成，不读取目标掩膜。"""
    xy = np.asarray(xy, float)
    hard_weight = np.asarray(hard_weight, float)
    rescue_weight = np.asarray(rescue_weight, float)
    if xy.shape != (len(hard_weight), 2) or len(hard_weight) != len(rescue_weight):
        raise ValueError('Unaligned XY and weights')
    if (not np.isfinite(xy).all() or not np.isfinite(hard_weight).all()
            or not np.isfinite(rescue_weight).all()
            or (hard_weight < 0).any() or (rescue_weight < 0).any()):
        raise ValueError('Nonfinite or negative event weight')
    edges = imaging_edges(SUPPORT_GRID)[:2]

    def image(weight: np.ndarray) -> np.ndarray:
        return np.histogram2d(xy[:, 0], xy[:, 1], bins=edges,
                              weights=weight)[0]

    hard = image(hard_weight)
    hard_squared = image(hard_weight**2)
    rescue = image(rescue_weight)
    neff = np.divide(hard**2, hard_squared,
                     out=np.zeros_like(hard), where=hard_squared > 0)
    mass = float(rescue.sum())
    norm_product = float(np.linalg.norm(hard) * np.linalg.norm(rescue))
    return {
        'supported_rescue_fraction':
            float(rescue[neff >= MIN_EFFECTIVE_COUNT].sum() / mass)
            if mass > 0 else np.nan,
        'empty_rescue_fraction':
            float(rescue[hard == 0].sum() / mass) if mass > 0 else np.nan,
        'hard_rescue_cosine':
            float(np.sum(hard * rescue) / norm_product)
            if norm_product > 0 else np.nan,
        'supported_cells': int(np.sum(neff >= MIN_EFFECTIVE_COUNT)),
        'occupied_hard_cells': int(np.sum(hard > 0)),
        'rescue_mass_on_coarse_grid': mass,
    }


def run(output_dir: Path) -> None:
    # IoU 作为先前存档的评价标签读取，不进入任何空间特征计算。
    saved = pd.read_csv(ROOT / 'muon_analysis/plus_thickness_mechanism_results/by_repeat.csv')
    if saved.duplicated(KEY).any():
        raise AssertionError('Duplicate saved conditions')
    saved = saved.set_index(KEY).sort_index()
    rows = []
    for material in MATERIALS:
        for thickness in THICKNESSES_MM:
            for stream_index, stream in enumerate(STREAMS):
                data = pd.read_csv(csv_for(material, thickness, stream))
                spacing = same_side_layer_spacing_mm(data)
                rng = np.random.default_rng(RANDOM_SEED + 8107 + 1000 * stream_index)
                advance_to_count_5000(rng, len(data))
                for repeat in range(1, REPEATS + 1):
                    subset = data.iloc[rng.choice(len(data), COUNT, replace=False)].copy()
                    normals = {f'{plane}_{axis}': rng.normal(0, 1, COUNT)
                               for plane in ('up1', 'up2', 'down1', 'down2')
                               for axis in ('x', 'y')}
                    for sigma in SIGMAS_MM:
                        measured = measured_hits(subset, sigma, normals)
                        observed = recompute_from_noisy_hits(
                            measured, 0., np.random.default_rng(0))
                        ids, weight = candidate_events(observed, sigma, spacing)
                        seed = (RANDOM_SEED + 900000 + 300000 * stream_index
                                + 10000 * COUNT + 100 * repeat
                                + int(round(sigma * 1000)))
                        sampled = estimate_z_samples(measured.iloc[ids], sigma,
                                                     seed, BOOTSTRAP_DRAWS)
                        hard, soft = z_acceptance_weights(
                            observed.poca_z_mm.to_numpy()[ids], sampled)
                        xy = observed[['poca_x_mm', 'poca_y_mm']].to_numpy()[ids]
                        features = spatial_support_features(
                            xy, weight * hard, weight * (1 - hard) * soft)
                        key = material, thickness, stream, repeat, sigma
                        old = saved.loc[key]
                        if (int(old.candidate_count) != len(ids)
                                or not np.isclose(old.rescue_mass,
                                                  features['rescue_mass_on_coarse_grid'],
                                                  rtol=0, atol=1e-9)):
                            raise AssertionError(f'Prior candidate/rescue mass mismatch: {key}')
                        rows.append({
                            'material': material, 'thickness_mm': thickness,
                            'stream': stream, 'repeat': repeat,
                            'position_sigma_mm': sigma, **features,
                            'candidate_count': len(ids),
                            'iou_gain_label': float(old.total_iou_gain),
                            'rescue_step_iou_label': float(old.rescue_step_iou),
                            'truth_rescue_fraction_label':
                                float(old.rescue_mass_in_truth_fraction),
                        })
                    print(f'Observable rescue: {material}, {thickness} mm, '
                          f'{stream}, repeat {repeat}/{REPEATS}', flush=True)
    frame = pd.DataFrame(rows)
    expected = len(MATERIALS) * len(THICKNESSES_MM) * len(STREAMS) * REPEATS * len(SIGMAS_MM)
    if len(frame) != expected or frame.duplicated(KEY).any():
        raise AssertionError('Incomplete or duplicate condition matrix')
    summary = frame.groupby(['material', 'thickness_mm', 'stream',
                             'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{col: (col, 'mean') for col in frame.columns if col not in KEY})
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    focus = summary[summary.position_sigma_mm == .30]
    for row in focus.itertuples():
        color = 'tab:blue' if row.material == 'al' else 'tab:red'
        marker = 'o' if row.thickness_mm == 50 else 's'
        for ax, feature in zip(axes,
                               ('supported_rescue_fraction', 'hard_rescue_cosine')):
            ax.scatter(getattr(row, feature), row.iou_gain_label,
                       color=color, marker=marker, s=50)
            ax.annotate(row.stream, (getattr(row, feature), row.iou_gain_label),
                        xytext=(3, 3), textcoords='offset points', fontsize=7)
    for ax, feature in zip(axes,
                           ('supported_rescue_fraction', 'hard_rescue_cosine')):
        ax.set(xlabel=feature.replace('_', ' '), ylabel='P − H mean IoU')
        ax.axhline(0, color='black', lw=1)
        ax.grid(alpha=.25)
    fig.suptitle('Development-only observable support vs IoU, σ=0.30 mm')
    fig.savefig(output_dir / 'observable_rescue_support_5000_32.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'OBSERVABLE_RESCUE_SUPPORT_PLAN.md',
        'role': 'post-hoc development diagnostic, not classifier validation',
        'support_grid': SUPPORT_GRID, 'min_effective_count': MIN_EFFECTIVE_COUNT,
        'count': COUNT, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'source_hard_rescue_mass_reproduced': True,
        'truth_not_in_spatial_features': True,
        'labels_only_for_evaluation': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(focus[['material', 'thickness_mm', 'stream',
                 'supported_rescue_fraction', 'empty_rescue_fraction',
                 'hard_rescue_cosine', 'iou_gain_label',
                 'truth_rescue_fraction_label']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/observable_rescue_support_results')
