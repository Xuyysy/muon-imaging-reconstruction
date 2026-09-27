"""固定 neff8 的条件重采样稳定性；先不建立真假值驱动的回退规则。

复用已有分析模块，输出进入 muon_analysis 的独立结果目录。
"""
from pathlib import Path
import sys
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'muon_analysis'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
TIE_POLICIES = ('index_asc', 'index_desc', 'fixed_permutation', 'fractional')


def tie_membership(image, k, policy):
    """完全相同权重的次键不读取目标真值，不向图像注入随机噪声。

    fractional 在第 k 名并列块中平均分配剩余名额；用于消除坐标偏好，
    但成员图之间的 L1 不能与硬掩膜 XOR 当成同一个指标。
    """
    flat = np.asarray(image).ravel()
    if not 0 < k < flat.size or not np.isfinite(flat).all():
        raise ValueError('Invalid image or k')
    result = np.zeros(flat.size, dtype=float)
    if policy == 'fractional':
        boundary = np.partition(flat, -k)[-k]
        above = flat > boundary
        equal = flat == boundary
        result[above] = 1.
        result[equal] = (k - int(above.sum())) / int(equal.sum())
    else:
        priority = np.arange(flat.size)
        if policy == 'index_desc':
            priority = -priority
        elif policy == 'fixed_permutation':
            priority = np.random.default_rng(314159).permutation(flat.size)
        elif policy != 'index_asc':
            raise ValueError(policy)
        result[np.lexsort((priority, -flat))[:k]] = 1.
    if not np.isclose(result.sum(), k, atol=1e-12, rtol=0):
        raise AssertionError('Membership area changed')
    return result.reshape(image.shape)


def tie_boundary_info(image, k):
    """记录名义 Top-k 边界的并列和零权重，避免误把空像素当轮廓。"""
    flat = image.ravel()
    boundary = np.partition(flat, -k)[-k]
    equal = int((flat == boundary).sum())
    slots = k - int((flat > boundary).sum())
    return {'cross_boundary_tie': int(0 < slots < equal),
            'boundary_equal_pixels': equal, 'boundary_slots': slots,
            'zero_pixels_required': max(0, k - int((flat > 0).sum()))}


def test_tie_membership():
    """小型不变量检查，无额外依赖或真值标签。"""
    distinct = np.arange(16, dtype=float).reshape(4, 4)
    for policy in TIE_POLICIES:
        if not np.array_equal(tie_membership(distinct, 5, policy), topk_mask(distinct, 5)):
            raise AssertionError('Tie policy reordered unequal pixels')
    tied = np.array([3., 2., 2., 2., 0., 0.])
    perm = np.array([5, 3, 0, 4, 2, 1])
    fraction = tie_membership(tied, 2, 'fractional')
    if not np.allclose(fraction, [1., 1/3, 1/3, 1/3, 0., 0.]):
        raise AssertionError('Fractional boundary allocation incorrect')
    if not np.allclose(tie_membership(tied[perm], 2, 'fractional'), fraction[perm]):
        raise AssertionError('Fractional policy depends on index')


def images(events, selection):
    """只由观测 POCA 和权重构图，无目标形状或真值面积参数。"""
    child = _child_weights(events[['poca_x_mm', 'poca_y_mm']].to_numpy(float),
                           events.capped_weight.to_numpy(float))
    mass = child.sum(axis=2)
    base = np.repeat(np.repeat(mass / 4, 2, axis=0), 2, axis=1)
    fine = child.reshape(16, 16, 2, 2).transpose(0, 2, 1, 3).reshape(32, 32)
    return base, apply_selection(base, fine, selection)


def observable_proxy(events, selection, seed, tie_audit=False):
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
            if tie_audit:
                for policy in TIE_POLICIES:
                    # 硬策略的 L1 等于 XOR；分数成员图这里只量化 L1。
                    distance = [float(np.abs(tie_membership(im, k, policy)
                                             - tie_membership(ref, k, policy)).mean())
                                for im, ref in zip(normalized, nominal)]
                    row[f'mask{k}_{policy}_instability_increment'] = distance[1] - distance[0]
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


def run(tie_audit=False):
    if tie_audit:
        test_tie_membership()
    archive = pd.read_csv(ROOT / 'muon_analysis/fixed_budget_refinement_results/by_repeat.csv')
    archive = archive[archive.method.isin(('coarse', 'neff8'))].set_index(KEY + ['method'])
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows, draws, boundaries, endpoints = [], [], [], []
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
                    proxy, trial = observable_proxy(events, selection, seed, tie_audit)
                    rows.append({**key, 'selected_events': len(events), **proxy,
                                 'iou_gain': float(archive.loc[(*index, 'neff8')].iou_gain)})
                    draws.append(trial.assign(**key))
                    if tie_audit:
                        for image, method in ((base, 'coarse'), (refined, 'neff8')):
                            for k in FIXED_K:
                                boundaries.append({**key, 'method': method, 'k': k,
                                                   **tie_boundary_info(image, k)})
                        # 真值仅在终点评分。原标签不因本轮审计而覆盖或改写。
                        for policy in TIE_POLICIES[:-1]:
                            scores = []
                            for image in (base, refined):
                                mask = tie_membership(image, int(truth.sum()), policy).astype(bool)
                                scores.append(float((mask & truth).sum() / (mask | truth).sum()))
                            endpoints.append({**key, 'policy': policy,
                                              'coarse_iou': scores[0], 'refined_iou': scores[1],
                                              'iou_gain': scores[1] - scores[0],
                                              'original_gain': float(archive.loc[(*index, 'neff8')].iou_gain)})
                print(f'Combination stability: {scene} N={count} repeat={repeat}', flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 60 or frame.duplicated(KEY).any():
        raise AssertionError('Incomplete diagnostic matrix')
    correlations = []
    for name, part in [('all', frame)] + [
            (f'N={n},sigma={s}', group)
            for (n, s), group in frame.groupby(['event_count', 'position_sigma_mm'])]:
        for metric in [col for col in frame if col.endswith('instability_increment')]:
            correlations.append({'group': name, 'metric': metric, 'conditions': len(part),
                                 'negative_conditions': int((part.iou_gain < -1e-12).sum()),
                                 # 平均秩的 Pearson 即 Spearman，避免增加 scipy 依赖。
                                 'spearman_with_gain': part[metric].rank().corr(part.iou_gain.rank()),
                                 'negative_gain_auc': negative_auc(part[metric], part.iou_gain)})
    output = ROOT / 'muon_analysis' / ('topk_tie_audit_results' if tie_audit
                                     else 'combination_stability_results')
    if tie_audit:
        previous = pd.read_csv(ROOT / 'muon_analysis/combination_stability_results/by_repeat.csv')
        merged = frame.merge(previous, on=KEY, suffixes=('', '_old'), validate='one_to_one')
        for metric in ['l1_instability_increment'] + [f'mask{k}_instability_increment' for k in FIXED_K]:
            if not np.allclose(merged[metric], merged[f'{metric}_old'], atol=1e-12, rtol=0):
                raise AssertionError(f'Original proxy changed: {metric}')
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'by_repeat.csv', index=False)
    pd.concat(draws, ignore_index=True).to_csv(output / 'bootstrap_draws.csv', index=False)
    pd.DataFrame(correlations).to_csv(output / 'associations.csv', index=False)
    if tie_audit:
        pd.DataFrame(boundaries).to_csv(output / 'nominal_boundaries.csv', index=False)
        endpoint_frame = pd.DataFrame(endpoints)
        # 把严格小于容差的值视为零；不以四舍五入改变非零标签。
        signs = lambda x: np.where(x > 1e-12, 1, np.where(x < -1e-12, -1, 0))
        endpoint_frame['sign_changed'] = signs(endpoint_frame.iou_gain) != signs(endpoint_frame.original_gain)
        endpoint_frame.to_csv(output / 'endpoint_sensitivity.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        focus = pd.DataFrame(correlations)
        focus = focus[focus.group == 'N=1000,sigma=0.3'].set_index('metric')
        for policy in ('legacy',) + TIE_POLICIES:
            values = [focus.loc[(f'mask{k}_instability_increment' if policy == 'legacy'
                                else f'mask{k}_{policy}_instability_increment')].negative_gain_auc
                      for k in FIXED_K]
            axes[0].plot(FIXED_K, values, marker='o', label=policy)
        axes[0].axhline(.5, color='gray', ls='--')
        axes[0].set(xlabel='Fixed k (not truth area)', ylabel='Descriptive negative-gain AUC',
                    title='N=1000, hit sigma=0.30 mm', ylim=(0, 1))
        axes[0].legend(fontsize=7)
        change = endpoint_frame.groupby('policy').sign_changed.sum()
        axes[1].bar(change.index, change.values, color='tab:orange')
        axes[1].set(ylabel='Gain-sign changes vs legacy / 60',
                    title='Endpoint tie-policy sensitivity', ylim=(0, 60))
        axes[1].tick_params(axis='x', labelsize=8)
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
        fig.savefig(output / 'topk_tie_audit.png', dpi=180)
        plt.close(fig)
    frame.groupby(['scenario', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'size'), iou_gain_mean=('iou_gain', 'mean'),
        instability_mean=('l1_instability_increment', 'mean')).to_csv(output / 'summary.csv', index=False)
    (output / 'config.json').write_text(json.dumps({
        'plan': ('B1/TOPK_TIE_AUDIT_PLAN.md' if tie_audit else 'B1/COMBINATION_STABILITY_PLAN.md'),
        'bootstraps': BOOTSTRAPS, 'tie_policies': TIE_POLICIES if tie_audit else [],
        'tie_priority_seed': 314159 if tie_audit else None,
        'fixed_k': FIXED_K, 'primary': 'l1_instability_increment',
        'selection': 'fixed nominal neff8, not reselected in bootstrap',
        'truth_usage': 'archived outcome check and endpoint label only',
        'role': 'development-only association, no tuned threshold or validation',
    }, ensure_ascii=False, indent=2) + '\n')
    print(pd.DataFrame(correlations).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tie-audit', action='store_true', help='并列敏感性审计，写入独立结果目录')
    run(parser.parse_args().tie_audit)
