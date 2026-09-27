"""已见开发集上的固定预算 XY 细分排序诊断；真值贪心不可部署。"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_refinement import _child_weights, _effective_count
from diagnose_heldout_subcell_score import heldout_subcell_score
from diagnose_single_cell_refinement_signal import topk_iou
from diagnose_split_confidence import COUNTS, REPEATS, ROOT, SCENES, SIGMAS_MM, topk_mask
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import corrected_events, same_side_layer_spacing_mm

BUDGET = 8
RANDOM_DRAWS = 16
FOCUS_COUNTS = (1000, 5000)
FOCUS_SIGMAS_MM = (.05, .30)
KEY = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
METHODS = ('coarse', 'neff8', 'heldout_positive', 'random8_mean', 'truth_greedy8')
EPS = 1e-12
CONTINUOUS_COLUMNS = ('target_mass_fraction', 'target_mass_gain',
                      'footprint_proxy_distance', 'footprint_proxy_gain',
                      'concentration_penalty', 'target_mass_reward')


def continuous_image_metrics(image, base, truth):
    """不排序、不强制分割面积，但真值足迹仍只可用于事后评价。

    p 是归一化观测权重，q 是均匀二值足迹代理而非材料密度真值。
    集中度项衡量平方权重变化，不可冒称为探测器噪声的方差。
    """
    if (image.shape != base.shape or image.shape != truth.shape
            or not np.isfinite(image).all() or not np.isfinite(base).all()
            or (image < 0).any() or (base < 0).any()
            or image.sum() <= 0 or base.sum() <= 0):
        raise ValueError('Invalid continuous images')
    mask = np.asarray(truth, dtype=bool)
    k = int(mask.sum())
    if not 0 < k < image.size:
        raise ValueError('Degenerate target footprint')
    p, p0 = image / image.sum(), base / base.sum()
    q = mask.astype(float) / k
    m = image.size
    foreground, foreground0 = float(p[mask].sum()), float(p0[mask].sum())
    distance, distance0 = float(m * np.square(p-q).sum()), float(m * np.square(p0-q).sum())
    mass_gain = foreground - foreground0
    penalty = float(m * (np.square(p).sum() - np.square(p0).sum()))
    reward = 2*m/k * mass_gain
    proxy_gain = distance0 - distance
    if not np.isclose(proxy_gain, reward-penalty, atol=1e-10, rtol=1e-10):
        raise AssertionError('Continuous proxy decomposition did not close')
    return {'target_mass_fraction': foreground, 'target_mass_gain': mass_gain,
            'footprint_proxy_distance': distance, 'footprint_proxy_gain': proxy_gain,
            'concentration_penalty': penalty, 'target_mass_reward': reward}


def test_continuous_metrics():
    truth = np.array([[True, True], [False, False]])
    base = np.ones((2, 2))
    image = np.array([[2., 2.], [0., 0.]])
    metrics = continuous_image_metrics(image, base, truth)
    if not (np.isclose(metrics['target_mass_fraction'], 1.)
            and np.isclose(metrics['footprint_proxy_distance'], 0.)):
        raise AssertionError('Ideal footprint score incorrect')
    for candidate in (continuous_image_metrics(image*7, base*3, truth),
                      continuous_image_metrics(image[::-1], base[::-1], truth[::-1])):
        for name in CONTINUOUS_COLUMNS:
            if not np.isclose(candidate[name], metrics[name]):
                raise AssertionError('Continuous metric scale/permutation mismatch')
    identical = continuous_image_metrics(base, base.copy(), truth)
    for name in ('target_mass_gain', 'footprint_proxy_gain', 'concentration_penalty', 'target_mass_reward'):
        if identical[name] != 0:
            raise AssertionError('Identical image has nonzero continuous gain')


def fractional_iou_bounds(image: np.ndarray, truth: np.ndarray) -> dict:
    """无坐标偏好的分数边界评分及合法硬并列 IoU 范围。

    真值面积只用于终点评分，不参加构图或选格。分数 IoU 是软成员度
    的交并比，不是随机硬掩膜 IoU 的期望；上下界不是统计置信区间。
    """
    if image.shape != truth.shape or not np.isfinite(image).all():
        raise ValueError('Invalid image/truth alignment')
    values = image.ravel()
    target = np.asarray(truth, dtype=bool).ravel()
    k = int(target.sum())
    if not 0 < k < values.size:
        raise ValueError('Degenerate truth area')
    boundary = np.partition(values, -k)[-k]
    above, tied = values > boundary, values == boundary
    slots = k - int(above.sum())
    n_tied = int(tied.sum())
    target_tied = int((tied & target).sum())
    fixed_overlap = int((above & target).sum())
    fraction_overlap = fixed_overlap + slots * target_tied / n_tied
    low_overlap = fixed_overlap + max(0, slots - (n_tied - target_tied))
    high_overlap = fixed_overlap + min(slots, target_tied)
    score = lambda overlap: float(overlap / (2*k - overlap))
    result = {'fractional_iou': score(fraction_overlap),
              'hard_iou_min': score(low_overlap), 'hard_iou_max': score(high_overlap),
              'boundary_equal_pixels': n_tied, 'boundary_slots': slots,
              'zero_pixels_required': max(0, k - int((values > 0).sum()))}
    if not (result['hard_iou_min'] - EPS <= result['fractional_iou']
            <= result['hard_iou_max'] + EPS):
        raise AssertionError('Fractional IoU outside tie bounds')
    return result


def fractional_pair_metrics(image, base, truth):
    """固定图像的双报告；完全相同图的收益必须严格为零。"""
    refined, coarse = fractional_iou_bounds(image, truth), fractional_iou_bounds(base, truth)
    same = np.array_equal(image, base)
    metrics = {**refined, 'fractional_gain': refined['fractional_iou'] - coarse['fractional_iou'],
            'gain_bound_min': 0. if same else refined['hard_iou_min'] - coarse['hard_iou_max'],
            'gain_bound_max': 0. if same else refined['hard_iou_max'] - coarse['hard_iou_min']}
    if not (metrics['gain_bound_min'] - EPS <= metrics['fractional_gain']
            <= metrics['gain_bound_max'] + EPS):
        raise AssertionError('Fractional gain outside pair bounds')
    return metrics


def test_fractional_metrics():
    """指标定义的最小不变量测试，不生成或修改实验数据。"""
    from itertools import combinations
    image = np.array([4., 2., 2., 2., 0., 0.])
    truth = np.array([True, True, False, False, True, False])
    result = fractional_iou_bounds(image, truth)
    # 枚举边界的合法硬选择，核查解析上下界确实可达。
    scores = []
    for choice in combinations([1, 2, 3], 2):
        mask = np.zeros(6, dtype=bool)
        mask[[0, *choice]] = True
        scores.append(float((mask & truth).sum() / (mask | truth).sum()))
    if not (np.isclose(min(scores), result['hard_iou_min'])
            and np.isclose(max(scores), result['hard_iou_max'])):
        raise AssertionError('Analytic bounds mismatch enumeration')
    perm = np.array([5, 2, 1, 4, 0, 3])
    if not np.isclose(result['fractional_iou'],
                      fractional_iou_bounds(image[perm], truth[perm])['fractional_iou']):
        raise AssertionError('Fractional score depends on coordinates')
    distinct = np.arange(16.).reshape(4, 4)
    distinct_truth = np.eye(4, dtype=bool)
    if not np.isclose(fractional_iou_bounds(distinct, distinct_truth)['fractional_iou'],
                      topk_iou_small(distinct, distinct_truth)):
        raise AssertionError('Distinct-weight score differs from hard IoU')
    zero = fractional_iou_bounds(np.zeros(6), truth)
    # 六像素均分三名额，交集1.5、并集4.5，评分1/3。
    if not np.isclose(zero['fractional_iou'], 1/3):
        raise AssertionError('Zero-image fractional area incorrect')
    pair = fractional_pair_metrics(image, image.copy(), truth)
    if pair['fractional_gain'] != 0 or pair['gain_bound_min'] != 0 or pair['gain_bound_max'] != 0:
        raise AssertionError('Identical image pair has nonzero gain')


def topk_iou_small(image, truth):
    """仅供非32x32小测试；正式原评分仍调用已有 topk_iou。"""
    mask = topk_mask(image, int(truth.sum()))
    return float((mask & truth).sum() / (mask | truth).sum())


def images_and_ranks(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """从同一入选事件构造粗图、细子格和两个无真值排序特征。"""
    points = events[['poca_x_mm', 'poca_y_mm']].to_numpy(float)
    weight = events.capped_weight.to_numpy(float)
    child = _child_weights(points, weight)
    neff = _effective_count(child, _child_weights(points, weight**2))
    counts = _child_weights(points, np.ones(len(points))).sum(axis=2)
    mass = child.sum(axis=2)
    base = np.repeat(np.repeat(mass / 4., 2, axis=0), 2, axis=1)
    fine = child.reshape(16, 16, 2, 2).transpose(0, 2, 1, 3).reshape(32, 32)
    score, _ = heldout_subcell_score(points)
    cells = [tuple(map(int, pair)) for pair in np.argwhere(counts > 0)]
    if len(cells) < BUDGET or not np.isclose(base.sum(), fine.sum(), atol=1e-9):
        raise AssertionError('Insufficient occupied cells or nonconserved image')
    return base, fine, neff, score, cells


def apply_selection(base: np.ndarray, fine: np.ndarray,
                    selected: list[tuple[int, int]]) -> np.ndarray:
    """替换选中粗格的四子格；其他像素严格沿用粗图。"""
    result = base.copy()
    for ix, iy in selected:
        sl = np.s_[2*ix:2*ix+2, 2*iy:2*iy+2]
        result[sl] = fine[sl]
    if not np.isclose(result.sum(), base.sum(), rtol=1e-12, atol=1e-9):
        raise AssertionError('Selection changed image total weight')
    return result


def greedy_with_truth(base: np.ndarray, fine: np.ndarray, truth: np.ndarray,
                      cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """只作诊断参照：真值直接参与逐步最大边际 IoU 选择，无法部署。"""
    selected: list[tuple[int, int]] = []
    remaining = set(cells)
    current = base.copy()
    current_iou = topk_iou(current, truth)
    for _ in range(BUDGET):
        best, best_gain = None, EPS
        for cell in sorted(remaining):
            ix, iy = cell
            sl = np.s_[2*ix:2*ix+2, 2*iy:2*iy+2]
            candidate = current.copy()
            candidate[sl] = fine[sl]
            gain = topk_iou(candidate, truth) - current_iou
            if gain > best_gain:
                best, best_gain = cell, gain
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        ix, iy = best
        current[2*ix:2*ix+2, 2*iy:2*iy+2] = fine[2*ix:2*ix+2, 2*iy:2*iy+2]
        current_iou += best_gain
    return selected


def image_metrics(image: np.ndarray, base: np.ndarray,
                  truth: np.ndarray, selected_count: int) -> dict:
    """Top-k 看轮廓，L1 只看连续图像改变量而非成像质量。"""
    k = int(truth.sum())
    return {
        'selected_cells': selected_count,
        'iou': topk_iou(image, truth),
        'topk_changed_pixels': int(np.logical_xor(
            topk_mask(image, k), topk_mask(base, k)).sum()),
        'relative_image_l1': (float(np.abs(image - base).sum() / base.sum())
                              if base.sum() > 0 else 0.),
    }


def run(output_dir: Path, fractional_audit: bool = False, continuous_audit: bool = False) -> None:
    if continuous_audit:
        fractional_audit = True
        test_continuous_metrics()
    if fractional_audit:
        test_fractional_metrics()
        frozen = pd.read_csv(ROOT / 'muon_analysis/fixed_budget_refinement_results/selected_cells.csv')
        original = pd.read_csv(ROOT / 'muon_analysis/fixed_budget_refinement_results/by_repeat.csv')
    archive = pd.read_csv(ROOT / 'muon_analysis/single_cell_refinement_signal_results'
                          / 'by_repeat.csv').set_index(KEY).sort_index()
    edges = imaging_edges(32)
    cx, cy = [(axis[:-1] + axis[1:]) / 2 for axis in edges[:2]]
    rows, random_rows, selected_rows = [], [], []
    inputs = {}
    for scene_index, (scene, (csv_name, mask_fn)) in enumerate(SCENES.items()):
        tracks = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        spacing = same_side_layer_spacing_mm(tracks)
        truth = mask_fn(cx, cy)
        inputs[scene] = {'csv': csv_name, 'complete_tracks': len(tracks)}
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        for count in COUNTS:
            for repeat in range(1, REPEATS + 1):
                # 3000 组虽然不评分，仍必须抽样并消耗噪声 RNG，才能让
                # 后续 5000 组逐次复现上一轮已归档的抽样顺序。
                subset = tracks.iloc[rng.choice(len(tracks), count, replace=False)].copy()
                draws = {f'{plane}_{axis}': rng.normal(size=count)
                         for plane in ('up1', 'up2', 'down1', 'down2')
                         for axis in ('x', 'y')}
                if count not in FOCUS_COUNTS:
                    continue
                for sigma in SIGMAS_MM:
                    if sigma not in FOCUS_SIGMAS_MM:
                        continue
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(
                        measured, 0., np.random.default_rng(0))
                    events = corrected_events(observed, sigma, spacing)
                    key = {'scenario': scene, 'event_count': count,
                           'repeat': repeat, 'position_sigma_mm': sigma}
                    old = archive.loc[(scene, count, repeat, sigma)]
                    base, fine, neff, score, cells = images_and_ranks(events)
                    base_iou = topk_iou(base, truth)
                    if not (len(events) == int(old.selected_events)
                            and np.isclose(base_iou, old.coarse_iou,
                                           rtol=0, atol=1e-12)):
                        raise AssertionError(f'Prior event/image mismatch: {key}')
                    # 并列分数用 ix,iy 升序；正分缺失者不可混作零分。
                    neff_selection = sorted(cells,
                                            key=lambda cell: (-neff[cell], *cell))[:BUDGET]
                    scored = [cell for cell in cells
                              if np.isfinite(score[cell]) and score[cell] > 0]
                    heldout_selection = sorted(
                        scored, key=lambda cell: (-score[cell], *cell))[:BUDGET]
                    if fractional_audit:
                        part = frozen[(frozen.scenario == scene) & (frozen.event_count == count)
                                      & (frozen.repeat == repeat) & (frozen.position_sigma_mm == sigma)]
                        selections_archived = {
                            method: list(zip(group.ix.astype(int), group.iy.astype(int)))
                            for method, group in part.groupby('method', sort=False)}
                        for method, selection in (('neff8', neff_selection),
                                                  ('heldout_positive', heldout_selection)):
                            if selection != selections_archived.get(method, []):
                                raise AssertionError(f'Frozen selections changed: {key, method}')
                        # 原真值贪心的格保持不变，不用新指标重新优化。
                        oracle_selection = selections_archived.get('truth_greedy8', [])
                    else:
                        oracle_selection = greedy_with_truth(base, fine, truth, cells)
                    selections = {
                        'coarse': [], 'neff8': neff_selection,
                        'heldout_positive': heldout_selection,
                        'truth_greedy8': oracle_selection,
                    }
                    for method, selected in selections.items():
                        image = apply_selection(base, fine, selected)
                        metrics = image_metrics(image, base, truth, len(selected))
                        if fractional_audit:
                            metrics.update(fractional_pair_metrics(image, base, truth))
                        if continuous_audit:
                            metrics.update(continuous_image_metrics(image, base, truth))
                        rows.append({**key, 'method': method,
                                     'occupied_cells': len(cells),
                                     'scored_cells': int(np.isfinite(score).sum()),
                                     **metrics, 'iou_gain': metrics['iou'] - base_iou})
                        selected_rows.extend({**key, 'method': method,
                                              'ix': ix, 'iy': iy}
                                             for ix, iy in selected)
                    random_rng = np.random.default_rng(
                        RANDOM_SEED + 730001 + scene_index * 100000
                        + count * 10 + repeat * 1000 + int(round(sigma * 1000)))
                    trial_metrics = []
                    for draw_index in range(RANDOM_DRAWS):
                        choice = random_rng.choice(len(cells), BUDGET, replace=False)
                        selection = [cells[index] for index in choice]
                        image = apply_selection(base, fine, selection)
                        metrics = image_metrics(image, base, truth, BUDGET)
                        if fractional_audit:
                            metrics.update(fractional_pair_metrics(image, base, truth))
                        if continuous_audit:
                            metrics.update(continuous_image_metrics(image, base, truth))
                        random_rows.append({**key, 'draw': draw_index,
                                            **metrics, 'iou_gain': metrics['iou'] - base_iou})
                        trial_metrics.append(metrics)
                    rows.append({**key, 'method': 'random8_mean',
                                 'occupied_cells': len(cells),
                                 'scored_cells': int(np.isfinite(score).sum()),
                                 **{name: float(np.mean([m[name] for m in trial_metrics]))
                                    for name in ('selected_cells', 'iou',
                                                 'topk_changed_pixels', 'relative_image_l1')},
                                 'iou_gain': float(np.mean([m['iou'] - base_iou
                                                            for m in trial_metrics])),
                                 **({name: float(np.mean([m[name] for m in trial_metrics]))
                                     for name in ('fractional_iou', 'fractional_gain',
                                                  'hard_iou_min', 'hard_iou_max',
                                                  'gain_bound_min', 'gain_bound_max',
                                                  'boundary_equal_pixels', 'boundary_slots',
                                                  'zero_pixels_required')}
                                    if fractional_audit else {}),
                                 **({name: float(np.mean([m[name] for m in trial_metrics]))
                                     for name in CONTINUOUS_COLUMNS} if continuous_audit else {})})
                print(f'Fixed budget: {scene}, N={count}, repeat {repeat}/{REPEATS}',
                      flush=True)
    frame = pd.DataFrame(rows)
    if (len(frame) != len(SCENES) * len(FOCUS_COUNTS) * len(FOCUS_SIGMAS_MM)
            * REPEATS * len(METHODS) or frame.duplicated(KEY + ['method']).any()):
        raise AssertionError('Incomplete fixed-budget comparison')
    if fractional_audit:
        matched = frame.merge(original, on=KEY + ['method'], suffixes=('', '_old'),
                              validate='one_to_one')
        if len(matched) != 300:
            raise AssertionError('Original archive match incomplete')
        for name in ('iou', 'iou_gain', 'selected_cells', 'relative_image_l1'):
            if not np.allclose(matched[name], matched[name + '_old'], atol=EPS, rtol=0):
                raise AssertionError(f'Original metric changed: {name}')
        # 原硬评分也必须处于合法并列界中；随机方法的界逐次平均。
        for value, low, high in (('iou', 'hard_iou_min', 'hard_iou_max'),
                                 ('iou_gain', 'gain_bound_min', 'gain_bound_max')):
            if not ((frame[value] >= frame[low] - EPS).all()
                    and (frame[value] <= frame[high] + EPS).all()):
                raise AssertionError(f'Original {value} outside tie bounds')
        signs = lambda x: np.where(x > EPS, 1, np.where(x < -EPS, -1, 0))
        frame['sign_changed'] = signs(frame.fractional_gain) != signs(frame.iou_gain)
        frame['strict_positive_bound'] = frame.gain_bound_min > EPS
        frame['strict_negative_bound'] = frame.gain_bound_max < -EPS
    if continuous_audit:
        prior = pd.read_csv(ROOT / 'muon_analysis/fractional_iou_audit_results/by_repeat.csv')
        matched = frame.merge(prior, on=KEY + ['method'], suffixes=('', '_prior'),
                              validate='one_to_one')
        if len(matched) != 300:
            raise AssertionError('Fractional archive match incomplete')
        for name in ('fractional_iou', 'fractional_gain', 'gain_bound_min', 'gain_bound_max'):
            if not np.allclose(matched[name], matched[name+'_prior'], atol=EPS, rtol=0):
                raise AssertionError(f'Archived fractional metric changed: {name}')
        if (frame.concentration_penalty < -1e-10).any():
            raise AssertionError('Uniform coarse-cell replacement reduced square concentration')
    summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm',
                             'method'], as_index=False).agg(
        repeats=('repeat', 'size'),
        **{f'{name}_mean': (name, 'mean') for name in
           ('selected_cells', 'occupied_cells', 'scored_cells', 'iou',
            'iou_gain', 'topk_changed_pixels', 'relative_image_l1')},
        **{f'{name}_std': (name, 'std') for name in ('iou_gain', 'selected_cells')})
    if fractional_audit:
        audit_summary = frame.groupby(['scenario', 'event_count', 'position_sigma_mm', 'method'],
                                      as_index=False).agg(
            **{f'{name}_mean': (name, 'mean') for name in
               ('fractional_iou', 'fractional_gain', 'gain_bound_min', 'gain_bound_max')},
            sign_changes=('sign_changed', 'sum'),
            strict_positive_conditions=('strict_positive_bound', 'sum'),
            strict_negative_conditions=('strict_negative_bound', 'sum'))
        summary = summary.merge(audit_summary,
                                on=['scenario', 'event_count', 'position_sigma_mm', 'method'],
                                validate='one_to_one')
    if continuous_audit:
        group = ['scenario', 'event_count', 'position_sigma_mm', 'method']
        continuous_summary = frame.groupby(group, as_index=False).agg(
            **{f'{name}_mean': (name, 'mean') for name in CONTINUOUS_COLUMNS},
            mass_positive_conditions=('target_mass_gain', lambda x: int((x > EPS).sum())),
            proxy_positive_conditions=('footprint_proxy_gain', lambda x: int((x > EPS).sum())))
        summary = summary.merge(continuous_summary, on=group, validate='one_to_one')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    pd.DataFrame(random_rows).to_csv(output_dir / 'random_draws.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / 'selected_cells.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': ('B1/CONTINUOUS_IMAGE_AUDIT_PLAN.md' if continuous_audit else
                 'B1/FRACTIONAL_IOU_AUDIT_PLAN.md' if fractional_audit
                 else 'FIXED_BUDGET_REFINEMENT_PLAN.md'),
        'role': ('development-only frozen-image score audit' if fractional_audit
                 else 'development-only fixed-budget ranking diagnostic'),
        'fractional_score_is_expected_hard_iou': False,
        'gain_bounds_are_confidence_intervals': False,
        'continuous_proxy_is_material_density': False,
        'inputs': inputs, 'counts': FOCUS_COUNTS, 'sigmas_mm': FOCUS_SIGMAS_MM,
        'repeats': REPEATS, 'budget': BUDGET, 'random_draws': RANDOM_DRAWS,
        'truth_usage': 'IoU scoring and nondeployable truth_greedy8 only',
        'greedy_is_global_upper_bound': False,
        'image_type': '2.5D XY projection, not full 3D density',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    focus = summary[(summary.event_count == 1000)
                    & (summary.position_sigma_mm == .30)]
    labels = list(SCENES)
    positions = np.arange(len(labels))
    colors = {'neff8': 'tab:blue', 'heldout_positive': 'tab:orange',
              'random8_mean': 'tab:gray', 'truth_greedy8': 'tab:green'}
    for offset, method in zip((-.27, -.09, .09, .27), colors):
        part = focus[focus.method == method].set_index('scenario').loc[labels]
        ax.bar(positions + offset, part.iou_gain_mean, width=.17,
               color=colors[method], label=method)
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel('Mean equal-area IoU gain vs coarse')
    ax.set_title('Fixed budget ≤8 XY cells; N=1000, hit σ=0.30 mm')
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / 'fixed_budget_refinement.png', dpi=180)
    plt.close(fig)
    if fractional_audit:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, (count, sigma) in zip(axes, ((1000, .30), (5000, .05))):
            part = summary[(summary.method == 'neff8') & (summary.event_count == count)
                           & (summary.position_sigma_mm == sigma)].set_index('scenario').loc[labels]
            ax.bar(positions - .18, part.iou_gain_mean, width=.36, label='Legacy hard Top-k')
            ax.bar(positions + .18, part.fractional_gain_mean, width=.36, label='Fractional boundary IoU')
            ax.axhline(0, color='black', lw=1)
            ax.set_xticks(positions, labels)
            ax.set(title=f'Frozen neff8: N={count}, sigma={sigma:.2f} mm',
                   ylabel='Mean gain vs coarse')
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(fontsize=8)
        fig.savefig(output_dir / 'fractional_iou_comparison.png', dpi=180)
        plt.close(fig)
    if continuous_audit:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for column, (count, sigma) in enumerate(((1000, .30), (5000, .05))):
            part = summary[(summary.method == 'neff8') & (summary.event_count == count)
                           & (summary.position_sigma_mm == sigma)].set_index('scenario').loc[labels]
            axes[0, column].bar(positions, part.target_mass_gain_mean, color='tab:blue')
            axes[0, column].set(title=f'Frozen neff8: N={count}, sigma={sigma:.2f} mm',
                                ylabel='Target mass fraction gain')
            ax = axes[1, column]
            ax.bar(positions-.2, part.target_mass_reward_mean, width=.2, label='Target-mass reward')
            ax.bar(positions, part.concentration_penalty_mean, width=.2, label='Concentration cost')
            ax.bar(positions+.2, part.footprint_proxy_gain_mean, width=.2, label='Proxy distance gain')
            ax.set(ylabel='Scaled proxy-distance components')
        for ax in axes.ravel():
            ax.axhline(0, color='black', lw=1)
            ax.set_xticks(positions, labels)
            ax.grid(axis='y', alpha=.2)
        axes[1, 0].legend(fontsize=7)
        fig.savefig(output_dir / 'continuous_image_audit.png', dpi=180)
        plt.close(fig)
    print(focus[['scenario', 'method', 'selected_cells_mean', 'iou_gain_mean',
                 'topk_changed_pixels_mean', 'relative_image_l1_mean']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fractional-audit', action='store_true',
                        help='冻结图像分数边界评价，另存结果，不覆盖原比较')
    parser.add_argument('--continuous-audit', action='store_true',
                        help='增加连续图像配对评价（含分数IoU），输出独立目录')
    args = parser.parse_args()
    run(ROOT / 'muon_analysis' / ('continuous_image_audit_results' if args.continuous_audit
                                 else 'fractional_iou_audit_results' if args.fractional_audit
                                 else 'fixed_budget_refinement_results'),
        args.fractional_audit, args.continuous_audit)
