"""汇总已归档细化证据，不重跑仿真、不重新选格或调整指标。

本轮是回顾性核查，不是新算法实验。所有记录来自已见数据；
五次重复共享场景CSV，不能从行数推导独立物理样本数。
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEY = ['scenario', 'event_count', 'repeat', 'position_sigma_mm', 'method']
EPS = 1e-12


def sign_counts(values):
    """采用既有1e-12容差；缺失值不能偷偷计为零收益。"""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite archived gains')
    return {'positive': int((values > EPS).sum()),
            'negative': int((values < -EPS).sum()),
            'zero': int((np.abs(values) <= EPS).sum())}


def aligned_archives(continuous, fractional, original):
    """必须一对一匹配全部300条记录，再核对历史评分。

    不能按文件行号拼接：若某文件顺序变动，行号会配错场景/抽样。
    不同指标并存不意味着它们测量同一物理量。
    """
    frames = []
    for frame in (continuous, fractional, original):
        if len(frame) != 300 or frame.duplicated(KEY).any():
            raise ValueError('Expected 300 unique archived method records')
        frames.append(frame.set_index(KEY).sort_index())
    current, frac, old = frames
    if not current.index.equals(frac.index) or not current.index.equals(old.index):
        raise ValueError('Archive keys differ')
    for other, columns in ((frac, ['fractional_iou', 'fractional_gain',
                                    'gain_bound_min', 'gain_bound_max']),
                           (old, ['iou', 'iou_gain', 'selected_cells'])):
        for column in columns:
            if not np.allclose(current[column], other[column], rtol=1e-12, atol=EPS):
                raise ValueError(f'Historical metric changed: {column}')
    return current.reset_index()


def method_evidence(frame):
    """分开记录轮廓、权重集中和均匀足迹代理，不能择优报指标。"""
    result = {}
    for method, rows in frame.groupby('method'):
        result[method] = {
            'conditions': len(rows),
            'fractional_gain': sign_counts(rows.fractional_gain),
            'original_hard_gain': sign_counts(rows.iou_gain),
            'target_mass_gain': sign_counts(rows.target_mass_gain),
            'uniform_proxy_gain': sign_counts(rows.footprint_proxy_gain),
            'strict_negative_tie_bound': int((rows.gain_bound_max < -EPS).sum()),
            'strict_positive_tie_bound': int((rows.gain_bound_min > EPS).sum()),
            'joint_contour_mass_proxy_positive': int(((rows.fractional_gain > EPS)
                & (rows.target_mass_gain > EPS) & (rows.footprint_proxy_gain > EPS)).sum()),
            'contour_positive_proxy_negative': int(((rows.fractional_gain > EPS)
                & (rows.footprint_proxy_gain < -EPS)).sum()),
            'mean_fractional_gain': float(rows.fractional_gain.mean()),
            'mean_selected_cells': float(rows.selected_cells.mean()),
        }
    return result


def run():
    # 哈希用于复查归档内容；不是声称本轮方案在历史实验前已注册。
    inputs = {
        'continuous': 'continuous_image_audit_results/by_repeat.csv',
        'fractional': 'fractional_iou_audit_results/by_repeat.csv',
        'original': 'fixed_budget_refinement_results/by_repeat.csv',
        'support': 'split_confidence_diagnosis_results/by_repeat.csv',
        'single_cell': 'single_cell_refinement_signal_results/by_repeat.csv',
        'h_verdict': 'edge_h_heldout_results/verdict.json',
    }
    hashes = {name: {'path': path, 'sha256': hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
              for name, path in inputs.items()}
    frame = aligned_archives(*(pd.read_csv(ROOT/inputs[name])
                               for name in ('continuous', 'fractional', 'original')))
    methods = method_evidence(frame)
    support = pd.read_csv(ROOT/inputs['support'])
    cells = pd.read_csv(ROOT/inputs['single_cell'])
    condition_key = KEY[:-1]
    for table in (support, cells):
        if len(table) != 180 or table.duplicated(condition_key).any():
            raise ValueError('Expected 180 unique diagnosis conditions')
    support_low = support[(support.event_count == 1000) & (support.position_sigma_mm == .30)]
    cell_low = cells[(cells.event_count == 1000) & (cells.position_sigma_mm == .30)]
    if len(support_low) != 15 or len(cell_low) != 15:
        raise ValueError('Incomplete low-statistics focus')
    # 以下逐格收益仍来自原硬Top-k，不与分数审计混成一个“可达上界”。
    low = {scene: {'mean_positive_hard_gain_cells': float(rows.positive_gain_cells.mean()),
                   'mean_negative_hard_gain_cells': float(rows.negative_gain_cells.mean()),
                   'mean_scored_cells': float(rows.scored_cells.mean())}
           for scene, rows in cell_low.groupby('scenario')}
    report = {
        'scope': 'retrospective consolidation; no new reconstruction or independent validation',
        'input_hashes': hashes, 'archive_alignment_passed': True,
        'methods': methods,
        'support': {'conditions': len(support),
                    'conditions_with_hard_mask_change': int((support.topk_changed_pixels != 0).sum()),
                    'low_stats_supported_cells_sum': int(support_low.split_supported_xy_cells.sum())},
        'single_cell_low_stats_original_hard_only': low,
        'edge_h_frozen_verdict': json.loads((ROOT/inputs['h_verdict']).read_text()),
        'prohibitions': ['truth_greedy8 is not a deployable algorithm or global upper bound',
                         'uniform footprint proxy is not material density truth',
                         'S has not been paired against these exact archived selections',
                         'subsampling repetitions are not independent physics streams'],
    }
    output = ROOT/'refinement_checkpoint_results'
    output.mkdir(exist_ok=True)
    (output/'evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 真正粗细体素选择：证据收敛检查点', '',
             '本轮只核对已归档记录，不新增算法、仿真或参数搜索。三份历史评分归档的300条记录按键逐条一致。', '',
             '|方法|分数IoU正/负/零|均匀代理正/负/零|轮廓正而代理负|并列稳健正/负|',
             '|---|---|---|---:|---|']
    fmt = lambda counts: '/'.join(str(counts[k]) for k in ('positive','negative','zero'))
    for name, record in methods.items():
        lines.append(f"|{name}|{fmt(record['fractional_gain'])}|{fmt(record['uniform_proxy_gain'])}|"
                     f"{record['contour_positive_proxy_negative']}|"
                     f"{record['strict_positive_tie_bound']}/{record['strict_negative_tie_bound']}|")
    lines += ['', '## 已确认与未确认', '',
              '1. neff8未稳定获益，部分损失不依赖合法硬并列选择；不能继续把旧微小硬评分正差当成功。',
              '2. 稀疏条件的原硬评分有潜在单格正收益，但并列审计不能保证这些格在分数指标下仍有效；真值贪心不是全局上界。',
              '3. 分半规则在180条开发条件均未改变硬掩膜，低统计量高误差15条条件的分半支撑为零。简单放宽门槛和已有留出正分都已做过，不重复包装为新规则。',
              '4. 固定S在另一些实验中构成强基线，但没有与这里完全相同的选格/事件做配对，所以不能跨归档推断S胜过所有细化方法。',
              '5. E在新H几何的四组预设判据只通过三组；减少空域权重不等于稳定提高轮廓。', '',
              '## 下一轮开发方案（尚未运行，不是结果）', '',
              '将候选问题改为“是否有足够证据偏离粗格均匀分配”，而不是固定选八格或仅按有效数排序。先用已知四子格概率的合成统计实验比较方差扣除的连续收缩、neff收缩和零细节对照；再决定是否值得进入已见Geant4开发集。',
              '候选使用四子格逐事件权重和、平方权重和估计对比噪声。风险估计对加权比值只是近似，角度权重与落点相关、POCA偏差和测量误差都可能破坏解释，必须先测试这些限制；不能当成已证明的物理规则。',
              '若合成检查通过，再使用同一事件/扰动配对比较粗图、原细图、候选及固定S；同时报告分数IoU、空域泄漏、目标内权重和局限明确的连续代理。不以目标真值调整实际收缩系数，不以零动作称为改进。',
              '开发数据包括已经看过的H等所有配置；它们不能再用作新候选的独立验证。方法冻结后才另建未见配置与随机流。三维叶体素结构、z方向分辨率和统一成本尚需后续单独验证；二维子格试验不是三维创新完成。', '',
              '输入摘要、精确计数和方法限制见 evidence.json。']
    (output/'research_note.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'methods': methods, 'support': report['support']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
