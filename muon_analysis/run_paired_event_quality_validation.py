"""配对复核事件质量实验：报告跨场景增益及其代价，而不是挑最佳图片。

输入是上一轮 run_continuous_event_quality_experiment.py 保存的逐重复 CSV。
同一 (场景、事件数、重复编号、位置误差) 下四种方法共享原始事件，
因此这里先计算“相对 hard”的配对差值，再做分组汇总。

本脚本不会重跑 Geant4、不会修改原始 CSV，也不根据真值重新选择参数。
输出文件仅用于研究诊断，不代表独立物理实验的显著性检验。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'muon_analysis/continuous_event_quality_results/continuous_event_quality_by_repeat.csv'
OUTPUT = ROOT / 'muon_analysis/paired_event_quality_results'
KEY = ['scenario', 'event_count', 'repeat', 'position_sigma_mm']
METHODS = ['hard', 'majority', 'pass_prob', 'pass_spread']
METRICS = ['iou', 'global_error_mm', 'auto_error_mm']


def load_and_validate() -> pd.DataFrame:
    """确保每个配对单元恰好有四种方法，避免缺行扭曲平均值。"""
    data = pd.read_csv(INPUT)
    required = set(KEY + ['method'] + METRICS + ['contributing_events', 'weight_fraction_of_hard'])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'输入缺少字段: {sorted(missing)}')
    if data.duplicated(KEY + ['method']).any():
        raise ValueError('存在重复配对键；请先检查上一轮输出。')
    if set(data.method) != set(METHODS):
        raise ValueError(f'方法集合不符: {sorted(set(data.method))}')
    counts = data.groupby(KEY).method.nunique()
    if (counts != len(METHODS)).any():
        raise ValueError('有场景/重复组合缺少方法，不可直接做配对比较。')
    if data[METRICS].isna().any().any():
        raise ValueError('评价指标有空值；请先查明是空图还是数据损坏。')
    return data


def paired_rows(data: pd.DataFrame) -> pd.DataFrame:
    """统一符号：正增益意味着相对 hard 更好。"""
    baseline = data[data.method == 'hard'].set_index(KEY)
    rows = []
    for method in METHODS[1:]:
        current = data[data.method == method].set_index(KEY)
        block = current.copy()
        block['method'] = method
        block['iou_gain'] = current.iou - baseline.iou
        block['global_error_gain_mm'] = baseline.global_error_mm - current.global_error_mm
        block['auto_error_gain_mm'] = baseline.auto_error_mm - current.auto_error_mm
        block['contributing_event_delta'] = current.contributing_events - baseline.contributing_events
        rows.append(block.reset_index()[KEY + ['method', 'iou_gain',
            'global_error_gain_mm', 'auto_error_gain_mm', 'contributing_event_delta',
            'weight_fraction_of_hard']])
    return pd.concat(rows, ignore_index=True)


def aggregate(pairs: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """同时给出均值、重复波动和获胜率，不只报告单个最大值。"""
    return pairs.groupby(groups, as_index=False).agg(
        paired_n=('iou_gain', 'size'),
        iou_gain_mean=('iou_gain', 'mean'),
        iou_gain_std=('iou_gain', 'std'),
        iou_win_fraction=('iou_gain', lambda x: float((x > 0).mean())),
        global_error_gain_mm_mean=('global_error_gain_mm', 'mean'),
        auto_error_gain_mm_mean=('auto_error_gain_mm', 'mean'),
        contributing_event_delta_mean=('contributing_event_delta', 'mean'),
        weight_fraction_of_hard_mean=('weight_fraction_of_hard', 'mean'),
    )


def plot_tradeoff(conditions: pd.DataFrame) -> None:
    """把低/中/高位置误差下的两种主要指标放在同一张图中。"""
    subset = conditions[conditions.event_count == 5000]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    colors = {'majority': '#1f77b4', 'pass_prob': '#ff7f0e', 'pass_spread': '#2ca02c'}
    for method in METHODS[1:]:
        group = subset[subset.method == method].groupby('position_sigma_mm', as_index=False).agg(
            iou=('iou_gain_mean', 'mean'), auto=('auto_error_gain_mm_mean', 'mean'))
        axes[0].plot(group.position_sigma_mm, group.iou, 'o-', color=colors[method], label=method)
        axes[1].plot(group.position_sigma_mm, group.auto, 'o-', color=colors[method], label=method)
    axes[0].set_ylabel('Paired IoU gain vs hard (higher is better)')
    axes[1].set_ylabel('Paired auto-localization error reduction [mm]')
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.8, linestyle='--')
        ax.set_xlabel('Hit-position sigma [mm]')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Locked event-quality rules: five scenes, 5000 input events')
    fig.savefig(OUTPUT / 'paired_tradeoff_5000.png', dpi=180)
    plt.close(fig)


def main() -> None:
    data = load_and_validate()
    pairs = paired_rows(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT / 'paired_by_repeat.csv', index=False)
    conditions = aggregate(pairs, ['scenario', 'event_count', 'position_sigma_mm', 'method'])
    conditions.to_csv(OUTPUT / 'paired_by_condition.csv', index=False)
    # sigma=0 时四种方法理论上一致；主汇总只纳入非零位置误差。
    noisy = pairs[pairs.position_sigma_mm > 0]
    aggregate(noisy, ['method']).to_csv(OUTPUT / 'paired_overall_noisy.csv', index=False)
    aggregate(noisy, ['scenario', 'method']).to_csv(OUTPUT / 'paired_by_scenario_noisy.csv', index=False)
    aggregate(noisy, ['event_count', 'position_sigma_mm', 'method']).to_csv(
        OUTPUT / 'paired_by_count_sigma_noisy.csv', index=False)
    plot_tradeoff(conditions)
    (OUTPUT / 'paired_validation_config.json').write_text(json.dumps({
        'source': str(INPUT), 'pairing_key': KEY, 'reference_method': 'hard',
        'metric_sign': 'positive means improvement over hard',
        'main_summary_excludes_sigma_zero': True,
        'parameters_reselected_from_truth': False,
        'note': 'Descriptive paired simulation analysis; repeats are not independent physical runs.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = aggregate(noisy, ['method'])
    print(summary[['method', 'paired_n', 'iou_gain_mean', 'iou_win_fraction',
                   'global_error_gain_mm_mean', 'auto_error_gain_mm_mean']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
