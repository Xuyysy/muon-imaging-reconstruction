"""跨目标形状检查 PoCA-Z 误差传播估计的增量预测信息。

方案在 POCA_Z_INCREMENTAL_VALUE_PLAN.md。预测目标仅是两次 PoCA
算法估计的深度差，不是真实散射点误差或重建图像质量。U/L/双铝块都
是开发数据；留出场景检查不等于新的物理独立实验。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibrate_poca_z_uncertainty import BOOTSTRAP_DRAWS, estimate_z_samples
from diagnose_poca_z_stability import REPEATS, ROOT, SCENES
from diagnose_selection_gate_transitions import gate_masks
from run_adaptive_voxel_experiment import RANDOM_SEED, Z_LIMIT_MM
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm

COUNT = 5000
SIGMAS_MM = (.05, .10)
RIDGE_ALPHA = 1.0


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray,
                  test_x: np.ndarray) -> np.ndarray:
    """训练集标准化后做固定强度 ridge；截距不惩罚，测试集不参与拟合。"""
    train_x = np.asarray(train_x, float)
    test_x = np.asarray(test_x, float)
    train_y = np.asarray(train_y, float)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError('Train/test feature dimensions must match')
    if len(train_x) != len(train_y) or not (np.isfinite(train_x).all()
                                              and np.isfinite(test_x).all()
                                              and np.isfinite(train_y).all()):
        raise ValueError('Nonfinite or unpaired regression data')
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(train_x)), (train_x - center) / scale])
    test_design = np.column_stack([np.ones(len(test_x)), (test_x - center) / scale])
    penalty = np.diag([0.] + [RIDGE_ALPHA] * train_x.shape[1])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    if not np.isfinite(coefficients).all():
        raise ValueError('Nonfinite ridge coefficients')
    # 对小规模特征矩阵做逐行点积；同时显式检查极端 PoCA-Z 值导致的溢出。
    prediction = np.einsum('ij,j->i', test_design, coefficients)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite held-out prediction')
    return prediction


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """仅在 log1p(mm) 尺度比较，两模型共用完全相同的测试事件。"""
    residual = np.asarray(prediction) - np.asarray(truth)
    return {'mae': float(np.abs(residual).mean()),
            'rmse': float(np.sqrt(np.mean(residual ** 2)))}


def collect_events() -> pd.DataFrame:
    """复现上一轮配对测量，仅使用观测侧输入构造预测特征。"""
    records = []
    for scene_index, (scene, (csv_name, _)) in enumerate(SCENES.items()):
        data = pd.read_csv(ROOT / 'muon_analysis' / csv_name)
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED + 1523 + 100 * scene_index)
        # 上一轮在 N=1000/3000 后才抽 N=5000；跳过的抽样仍需消耗同样的
        # 随机数，才能逐事件核对前一轮的参考入选数和 Z 越界数。
        for count in (1000, 3000, COUNT):
            for repeat in range(1, REPEATS + 1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                noise_draws = {f'{prefix}_{axis}': rng.normal(0, 1, count)
                               for prefix in ('up1', 'up2', 'down1', 'down2')
                               for axis in ('x', 'y')}
                if count != COUNT:
                    continue
                reference = recompute_from_noisy_hits(
                    measured_hits(subset, 0., noise_draws), 0., np.random.default_rng(0))
                selected = gate_masks(reference, 0., spacing)['selected']
                reference_z = reference.poca_z_mm.to_numpy()[selected]
                input_rows = subset.index.to_numpy()[selected]
                for sigma in SIGMAS_MM:
                    hits = measured_hits(subset, sigma, noise_draws)
                    observed = recompute_from_noisy_hits(
                        hits, 0., np.random.default_rng(0))
                    selected_observed = observed.iloc[selected]
                    seed = RANDOM_SEED + 200000 * scene_index + 10000 * count \
                        + 100 * repeat + int(round(sigma * 1000))
                    z_samples = estimate_z_samples(hits.iloc[selected], sigma, seed,
                                                   BOOTSTRAP_DRAWS)
                    predicted_sd = z_samples.std(axis=1, ddof=1)
                    observed_z = selected_observed.poca_z_mm.to_numpy()
                    theta = selected_observed.theta_mrad.to_numpy()
                    if not (np.isfinite(predicted_sd).all() and np.isfinite(theta).all()
                            and np.isfinite(observed_z).all()):
                        raise ValueError('Nonfinite event-level input')
                    records.extend({
                        'scenario': scene, 'repeat': repeat,
                        'position_sigma_mm': sigma, 'input_row': int(input_rows[i]),
                        'observed_theta_mrad': float(theta[i]),
                        'observed_abs_z_over_limit': float(abs(observed_z[i]) / Z_LIMIT_MM),
                        'predicted_z_sd_mm': float(predicted_sd[i]),
                        'reference_z_mm': float(reference_z[i]),
                        'observed_z_mm': float(observed_z[i]),
                        'abs_delta_z_mm': float(abs(reference_z[i] - observed_z[i])),
                    } for i in range(len(reference_z)))
                print(f'Incremental-value events: {scene}, repeat={repeat}/{REPEATS}',
                      flush=True)
    return pd.DataFrame.from_records(records)


def feature_matrix(data: pd.DataFrame, add_uncertainty: bool) -> np.ndarray:
    """所有变换预先固定；不从参考 PoCA 或目标形状读取输入。"""
    columns = [np.log1p(data.observed_theta_mrad.to_numpy()),
               data.observed_abs_z_over_limit.to_numpy()]
    if add_uncertainty:
        columns.append(np.log1p(data.predicted_z_sd_mm.to_numpy()))
    return np.column_stack(columns)


def compare_models(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按场景留出；每个留出集上基础/增量模型使用同一批事件。"""
    rows = []
    for sigma in SIGMAS_MM:
        subset = events[events.position_sigma_mm == sigma]
        for heldout in SCENES:
            train = subset[subset.scenario != heldout]
            test = subset[subset.scenario == heldout]
            train_y = np.log1p(train.abs_delta_z_mm.to_numpy())
            test_y = np.log1p(test.abs_delta_z_mm.to_numpy())
            if train.empty or test.empty:
                raise ValueError('Missing train or held-out scene')
            for model, add_uncertainty in (('angle_z', False),
                                           ('angle_z_plus_uncertainty', True)):
                prediction = ridge_predict(feature_matrix(train, add_uncertainty),
                                           train_y, feature_matrix(test, add_uncertainty))
                for repeat in range(1, REPEATS + 1):
                    mask = test.repeat.to_numpy() == repeat
                    metrics = regression_metrics(test_y[mask], prediction[mask])
                    rows.append({'position_sigma_mm': sigma, 'heldout_scene': heldout,
                                 'repeat': repeat, 'model': model,
                                 'test_events': int(mask.sum()), **metrics})
    by_repeat = pd.DataFrame(rows)
    summary = by_repeat.groupby(['position_sigma_mm', 'heldout_scene', 'model'],
                                as_index=False).agg(
        repeats=('repeat', 'size'), test_events_mean=('test_events', 'mean'),
        mae_mean=('mae', 'mean'), rmse_mean=('rmse', 'mean'))
    return by_repeat, summary


def run(output_dir: Path) -> None:
    events = collect_events()
    # 严格核对上一轮配对数量及越界标签，防止重复抽样顺序漂移。
    previous = pd.read_csv(ROOT / 'muon_analysis/poca_z_calibration_results/by_repeat.csv')
    previous = previous[(previous.event_count == COUNT)
                        & previous.position_sigma_mm.isin(SIGMAS_MM)]
    check = events.groupby(['scenario', 'repeat', 'position_sigma_mm'], as_index=False).agg(
        reference_selected=('input_row', 'size'),
        z_exit=('observed_z_mm', lambda z: int((z.abs() > Z_LIMIT_MM).sum())))
    matched = check.merge(previous[['scenario', 'repeat', 'position_sigma_mm',
                                    'reference_selected', 'z_exit']],
                          on=['scenario', 'repeat', 'position_sigma_mm'],
                          validate='one_to_one', suffixes=('', '_previous'))
    if len(matched) != len(previous) or not np.array_equal(
            matched.reference_selected, matched.reference_selected_previous) or not \
            np.array_equal(matched.z_exit, matched.z_exit_previous):
        raise AssertionError('Cannot reproduce previous paired event set')
    by_repeat, summary = compare_models(events)
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / 'event_features.csv', index=False)
    by_repeat.to_csv(output_dir / 'by_repeat.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    paired = summary.pivot(index=['position_sigma_mm', 'heldout_scene'],
                           columns='model', values='mae_mean').reset_index()
    paired['mae_gain'] = paired.angle_z - paired.angle_z_plus_uncertainty
    paired.to_csv(output_dir / 'paired_mae.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    positions = np.arange(len(paired))
    ax.bar(positions, paired.mae_gain,
           color=np.where(paired.mae_gain >= 0, '#2a9d8f', '#d95f02'))
    ax.set_xticks(positions, [f'{r.heldout_scene}\n{r.position_sigma_mm:g} mm'
                              for r in paired.itertuples()])
    ax.axhline(0, color='black', lw=1)
    ax.set(ylabel='Held-out MAE gain in log1p(mm) (positive = added value)',
           title='Added PoCA-Z uncertainty vs observed angle and Z position')
    ax.grid(axis='y', alpha=.25)
    fig.savefig(output_dir / 'heldout_mae_gain.png', dpi=180)
    plt.close(fig)
    (output_dir / 'config.json').write_text(json.dumps({
        'plan': 'POCA_Z_INCREMENTAL_VALUE_PLAN.md',
        'role': 'development-only predictive-value diagnostic, no image change',
        'count': COUNT, 'sigmas_mm': SIGMAS_MM, 'repeats': REPEATS,
        'bootstrap_draws': BOOTSTRAP_DRAWS, 'ridge_alpha': RIDGE_ALPHA,
        'model_target': 'log1p(abs(noisy_poca_z - sigma0_poca_z))',
        'baseline_features': ['log1p(observed_theta_mrad)',
                              'abs(observed_poca_z_mm)/100'],
        'added_feature': 'log1p(predicted_z_sd_mm)',
        'holdout_unit': 'target shape, not independent Geant4 seed',
        'target_truth_used': False, 't_scene_used': False,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(paired.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'muon_analysis/poca_z_incremental_results')
    run(parser.parse_args().output_dir)
