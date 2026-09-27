"""固定 neff8 的条件重采样稳定性；先不建立真假值驱动的回退规则。

复用已有分析模块，输出进入 muon_analysis 的独立结果目录。
"""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'muon_analysis'))
import numpy as np
import pandas as pd
from confidence_refinement import _child_weights
from diagnose_single_cell_refinement_signal import KEY, topk_iou
from diagnose_split_confidence import COUNTS, REPEATS, SCENES, SIGMAS_MM, topk_mask
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_fixed_budget_refinement import FOCUS_COUNTS, FOCUS_SIGMAS_MM, images_and_ranks, apply_selection
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

BOOTSTRAPS = 32
FIXED_K = (64, 128, 256)


def images(events, selection):
    """只由观测 POCA 和权重构图，无目标形状或真值面积参数。"""
    child = _child_weights(events[['poca_x_mm', 'poca_y_mm']].to_numpy(float),
                           events.capped_weight.to_numpy(float))
    mass = child.sum(axis=2)
    base = np.repeat(np.repeat(mass / 4, 2, axis=0), 2, axis=1)
    fine = child.reshape(16, 16, 2, 2).transpose(0, 2, 1, 3).reshape(32, 32)
    return base, apply_selection(base, fine, selection)


def observable_proxy(events, selection, seed):
    """固定名义选格的条件 bootstrap，不重算选格，不注入新击中误差。

    L1 比较各方案相对其自己的名义图的稳定性，不是二者图像质量。
    主/辅助代理均不读取真值；不把稳定性当成准确性。
    """
    if len(events) == 0:
        raise ValueError('No selected events')
    nominal = images(events, selection)
    nominal = [im / im.sum() for im in nominal]
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(BOOTSTRAPS):
        sampled = events.iloc[rng.integers(0, len(events), size=len(events))]
        pair = images(sampled, selection)
        normalized = [im / im.sum() for im in pair]
        distances = [float(np.abs(im - ref).sum())
                     for im, ref in zip(normalized, nominal)]
        row = {'bootstrap': draw, 'coarse_l1': distances[0], 'refined_l1': distances[1],
               'l1_instability_increment': distances[1] - distances[0]}
        for k in FIXED_K:
            changed = [float(np.logical_xor(topk_mask(im, k), topk_mask(ref, k)).mean())
                       for im, ref in zip(normalized, nominal)]
            row[f'mask{k}_instability_increment'] = changed[1] - changed[0]
        rows.append(row)
    draws = pd.DataFrame(rows)
    return {col: float(draws[col].mean()) for col in draws if col != 'bootstrap'}, draws


def negative_auc(score, gain):
    """描述性 AUC：负收益为正类；与非负类比较，并列计半分。"""
    a = np.asarray(score)[np.asarray(gain) < -1e-12]
    b = np.asarray(score)[np.asarray(gain) >= -1e-12]
    if not len(a) or not len(b):
        return np.nan
    return float(((a[:, None] > b).astype(float)
                  + .5 * (a[:, None] == b)).mean())


def run():
    archive = pd.read_csv(ROOT / 'muon_analysis/fixed_budget_refinement_results/by_repeat.csv')
    archive = archive[archive.method.isin(('coarse', 'neff8'))].set_index(KEY + ['method'])
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows, draws = [], []
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        tracks = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        spacing = same_side_layer_spacing_mm(tracks)
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                # 必须消耗 N=3000 的随机数，以复现已归档 N=5000 的抽样。
                subset = tracks.iloc[rng.choice(len(tracks), count, replace=False)].copy()
                noise = {f'{p}_{a}': rng.normal(size=count)
                         for p in ('up1', 'up2', 'down1', 'down2') for a in ('x', 'y')}
                if count not in FOCUS_COUNTS:
                    continue
                for sigma in SIGMAS_MM:
                    if sigma not in FOCUS_SIGMAS_MM:
                        continue
                    events = corrected_events(recompute_from_noisy_hits(
                        measured_hits(subset, sigma, noise), 0., np.random.default_rng(0)),
                        sigma, spacing)
                    base, fine, neff, _, occupied = images_and_ranks(events)
                    selection = sorted(occupied, key=lambda cell: (-neff[cell], *cell))[:8]
                    refined = apply_selection(base, fine, selection)
                    key = dict(zip(KEY, (scene, count, repeat, sigma)))
                    index = (scene, count, repeat, sigma)
                    # 以下真值只用于复现检查及终点标签，不进入 observable_proxy。
                    truth = mask_fn(cx, cy)
                    if not all(np.isclose(topk_iou(im, truth), archive.loc[(*index, method)].iou,
                                           atol=1e-12, rtol=0)
                               for im, method in ((base, 'coarse'), (refined, 'neff8'))):
                        raise AssertionError(f'Archived image mismatch: {key}')
                    seed = (RANDOM_SEED + 951001 + scene_index * 100000
                            + count * 10 + repeat * 1000 + int(round(sigma * 1000)))
                    proxy, trial = observable_proxy(events, selection, seed)
                    rows.append({**key, 'selected_events': len(events), **proxy,
                                 'iou_gain': float(archive.loc[(*index, 'neff8')].iou_gain)})
                    draws.append(trial.assign(**key))
                print(f'Combination stability: {scene} N={count} repeat={repeat}', flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 60 or frame.duplicated(KEY).any():
        raise AssertionError('Incomplete diagnostic matrix')
    correlations = []
    for name, part in [('all', frame)] + [
            (f'N={n},sigma={s}', group)
            for (n, s), group in frame.groupby(['event_count', 'position_sigma_mm'])]:
        for metric in ['l1_instability_increment'] + [f'mask{k}_instability_increment' for k in FIXED_K]:
            correlations.append({'group': name, 'metric': metric, 'conditions': len(part),
                                 'negative_conditions': int((part.iou_gain < -1e-12).sum()),
                                 # 平均秩的 Pearson 即 Spearman，避免增加 scipy 依赖。
                                 'spearman_with_gain': part[metric].rank().corr(part.iou_gain.rank()),
                                 'negative_gain_auc': negative_auc(part[metric], part.iou_gain)})
    output = ROOT / 'muon_analysis/combination_stability_results'
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'by_repeat.csv', index=False)
    pd.concat(draws, ignore_index=True).to_csv(output / 'bootstrap_draws.csv', index=False)
    pd.DataFrame(correlations).to_csv(output / 'associations.csv', index=False)
    frame.groupby(['scenario', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'), iou_gain_mean=('iou_gain', 'mean'),
        instability_mean=('l1_instability_increment', 'mean')).to_csv(output / 'summary.csv', index=False)
    (output / 'config.json').write_text(json.dumps({
        'plan': 'B1/COMBINATION_STABILITY_PLAN.md', 'bootstraps': BOOTSTRAPS,
        'fixed_k': FIXED_K, 'primary': 'l1_instability_increment',
        'selection': 'fixed nominal neff8, not reselected in bootstrap',
        'truth_usage': 'archived outcome check and endpoint label only',
        'role': 'development-only association, no tuned threshold or validation',
    }, ensure_ascii=False, indent=2) + '\n')
    print(pd.DataFrame(correlations).to_string(index=False))


if __name__ == '__main__':
    run()
