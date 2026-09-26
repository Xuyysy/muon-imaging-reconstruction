"""在已见数据上诊断固定噪声底回退；不用于独立验证或阈值扫描。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import THETA_MIN_MRAD
from run_soft_z_gate_experiment import ROOT
from run_uncertainty_aware_angle_experiment import (
    same_side_layer_spacing_mm, theta_noise_floor_mrad,
)

RISK_THRESHOLD = 2.0  # 后验开发决定；见 NOISE_FLOOR_FALLBACK_PROTOCOL.md。
DEV_CSV = {
    'u_center': 'muon_tracks.csv',
    'l_shape': 'muon_tracks_l_shape.csv',
    'double_block': 'muon_tracks_double_block.csv',
    'frame_seed_a': 'muon_tracks_frame_seed_a.csv',
    'frame_seed_b': 'muon_tracks_frame_seed_b.csv',
}


def choose_propagated(position_sigma_mm: float, spacing_mm: float) -> bool:
    """只根据外部标定的 σ、几何层距和既有角度下限选择 P/H。"""
    if position_sigma_mm < 0 or spacing_mm <= 0:
        raise ValueError('Invalid detector calibration')
    ratio = theta_noise_floor_mrad(position_sigma_mm, spacing_mm) / THETA_MIN_MRAD
    return bool(ratio < RISK_THRESHOLD)


def paired_scores(frame: pd.DataFrame, source: str, p_name: str,
                  spacing_mm: float) -> pd.DataFrame:
    keys = ['event_count', 'repeat', 'position_sigma_mm', 'grid']
    hard = frame[frame.method == 'hard'].set_index(keys).sort_index()
    propagated = frame[frame.method == p_name].set_index(keys).sort_index()
    if not hard.index.equals(propagated.index):
        raise AssertionError(f'{source}: H/P events not paired')
    result = pd.DataFrame({
        'hard_iou': hard.iou, 'propagated_iou': propagated.iou,
    }).reset_index()
    result.insert(0, 'source', source)
    result['noise_floor_ratio'] = result.position_sigma_mm.map(
        lambda sigma: theta_noise_floor_mrad(sigma, spacing_mm) / THETA_MIN_MRAD)
    result['choose_propagated'] = result.position_sigma_mm.map(
        lambda sigma: choose_propagated(sigma, spacing_mm))
    result['fallback_iou'] = np.where(result.choose_propagated,
                                      result.propagated_iou, result.hard_iou)
    result['fallback_gain_vs_hard'] = result.fallback_iou - result.hard_iou
    result['propagated_gain_vs_hard'] = result.propagated_iou - result.hard_iou
    # 这个真值选择只能作为开发诊断的非部署上界，绝不能拿来选门。
    result['oracle_iou'] = np.maximum(result.hard_iou, result.propagated_iou)
    result['regret_to_oracle'] = result.oracle_iou - result.fallback_iou
    return result


def run(output_dir: Path) -> None:
    rows = []
    old = pd.read_csv(ROOT / 'muon_analysis/soft_z_gate_results/by_repeat.csv')
    new = pd.read_csv(ROOT / 'muon_analysis/frame_z_gate_results/by_repeat.csv')
    for source, csv_name in DEV_CSV.items():
        data = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        spacing = same_side_layer_spacing_mm(data)
        if source.startswith('frame_'):
            stream = source.removeprefix('frame_')
            frame = new[new.stream == stream]
            p_name = 'propagated'
        else:
            frame = old[old.scenario == source]
            p_name = 'soft_z'
        rows.append(paired_scores(frame, source, p_name, spacing))
    paired = pd.concat(rows, ignore_index=True)
    expected = len(DEV_CSV) * 3 * 5 * 4 * 2
    if len(paired) != expected:
        raise AssertionError('Missing development conditions')
    summary = paired.groupby(['source', 'event_count', 'position_sigma_mm', 'grid'],
                             as_index=False).agg(
        repeats=('repeat', 'size'),
        noise_floor_ratio=('noise_floor_ratio', 'first'),
        choose_propagated=('choose_propagated', 'first'),
        hard_iou_mean=('hard_iou', 'mean'),
        propagated_iou_mean=('propagated_iou', 'mean'),
        fallback_iou_mean=('fallback_iou', 'mean'),
        fallback_gain_mean=('fallback_gain_vs_hard', 'mean'),
        propagated_gain_mean=('propagated_gain_vs_hard', 'mean'),
        regret_to_oracle_mean=('regret_to_oracle', 'mean'))
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'protocol': 'NOISE_FLOOR_FALLBACK_PROTOCOL.md',
        'role': 'post-hoc development diagnosis only',
        'risk_ratio': 'sqrt(8)*sigma_mm/layer_spacing_mm*1000/theta_min_mrad',
        'risk_threshold': RISK_THRESHOLD,
        'fallback': 'hard if ratio>=2 else propagated',
        'oracle': 'truth-dependent upper bound; never deployed',
    }, indent=2) + '\n')
    print(summary[(summary.event_count == 5000) & (summary.grid == 32)
                  & (summary.position_sigma_mm > 0)][
        ['source', 'position_sigma_mm', 'noise_floor_ratio',
         'choose_propagated', 'propagated_gain_mean', 'fallback_gain_mean',
         'regret_to_oracle_mean']].to_string(index=False))


if __name__ == '__main__':
    run(ROOT / 'muon_analysis/noise_floor_fallback_results')
