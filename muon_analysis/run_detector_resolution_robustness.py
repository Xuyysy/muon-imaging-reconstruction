"""探测器空间分辨率误差下的 PoCA 重建鲁棒性验证。

不重新运行 Geant4，而是在四层理想击中坐标的 x、y 方向叠加独立高斯误差，
再从扰动后的击中重新拟合两条轨迹、计算散射角与 PoCA。这样可隔离研究
“探测器位置测量误差”对重建算法的影响。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import (
    RANDOM_SEED, adaptive_volume, fixed_fine_volume, imaging_edges, prepare_events,
)
from run_baseline_experiment import evaluate_image

ROOT = Path('/Users/xuyaoyang/geant4-test')
INPUT_CSV = ROOT / 'muon_analysis' / 'muon_tracks.csv'
OUT_DIR = ROOT / 'muon_analysis' / 'detector_resolution_results'
EVENT_COUNTS = [1000, 3000, 5000]
REPEATS = 5
# 每层 x/y 击中的独立高斯标准差；0 mm 为理想探测器对照。
# 两平面相距约 100 mm，1 mm 误差会带来约 10 mrad 方向不确定度，
# 显著大于当前散射角尺度。因此重点扫描更有物理意义的亚毫米范围，
# 并保留 1 mm 作为“明显失效”的压力测试点。
POSITION_SIGMAS_MM = [0.0, 0.05, 0.10, 0.30, 1.0]
# 只检验两个代表性方案，不为有噪声场景重新调参。
METHODS = {'fixed_32x32x32': None, 'adaptive_16_to_32_40pct': 0.40}


def recompute_from_noisy_hits(data: pd.DataFrame, sigma_mm: float, rng: np.random.Generator) -> pd.DataFrame:
    """对四层 x/y 击中添加误差，并向量化重建 theta 与 PoCA。"""
    n = len(data)
    hits = []
    for prefix in ['up1', 'up2', 'down1', 'down2']:
        x = data[f'{prefix}_x_mm'].to_numpy() + rng.normal(0.0, sigma_mm, n)
        y = data[f'{prefix}_y_mm'].to_numpy() + rng.normal(0.0, sigma_mm, n)
        z = data[f'{prefix}_z_mm'].to_numpy()
        hits.append(np.column_stack([x, y, z]))
    up1, up2, down1, down2 = hits
    incoming = up2 - up1
    outgoing = down2 - down1
    incoming /= np.linalg.norm(incoming, axis=1, keepdims=True)
    outgoing /= np.linalg.norm(outgoing, axis=1, keepdims=True)
    theta_x = np.arctan2(outgoing[:, 0], outgoing[:, 2]) - np.arctan2(incoming[:, 0], incoming[:, 2])
    theta_y = np.arctan2(outgoing[:, 1], outgoing[:, 2]) - np.arctan2(incoming[:, 1], incoming[:, 2])
    theta = np.hypot(theta_x, theta_y)
    w0 = up1 - down1
    a = np.einsum('ij,ij->i', incoming, incoming)
    b = np.einsum('ij,ij->i', incoming, outgoing)
    c = np.einsum('ij,ij->i', outgoing, outgoing)
    d = np.einsum('ij,ij->i', incoming, w0)
    e = np.einsum('ij,ij->i', outgoing, w0)
    denominator = a * c - b * b
    s = np.zeros(n)
    t = np.zeros(n)
    stable = np.abs(denominator) > 1.e-12
    s[stable] = (b[stable] * e[stable] - c[stable] * d[stable]) / denominator[stable]
    t[stable] = (a[stable] * e[stable] - b[stable] * d[stable]) / denominator[stable]
    poca = 0.5 * ((up1 + s[:, None] * incoming) + (down1 + t[:, None] * outgoing))
    return pd.DataFrame({'poca_x_mm': poca[:, 0], 'poca_y_mm': poca[:, 1], 'poca_z_mm': poca[:, 2],
                         'theta_mrad': theta * 1000.0})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    edges = imaging_edges(32)
    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for count in [n for n in EVENT_COUNTS if n <= len(data)]:
        for repeat in range(REPEATS):
            subset = data.iloc[rng.choice(len(data), size=count, replace=False)]
            for sigma in POSITION_SIGMAS_MM:
                noisy = recompute_from_noisy_hits(subset, sigma, rng)
                events = prepare_events(noisy)
                for method, fraction in METHODS.items():
                    if fraction is None:
                        volume, _ = fixed_fine_volume(events)
                    else:
                        volume, _, refined, leaves = adaptive_volume(events, fraction)
                    image = volume.sum(axis=2)
                    iou, error, contrast, truth_pixels = evaluate_image(image, x_centers, y_centers)
                    rows.append({'method': method, 'event_count': count, 'repeat': repeat + 1,
                                 'position_sigma_mm': sigma, 'selected_events': len(events),
                                 'iou_equal_area': iou, 'localization_error_mm': error,
                                 'target_background_contrast': contrast, 'truth_u_pixels': truth_pixels})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / 'detector_resolution_metrics_by_repeat.csv', index=False)
    summary = metrics.groupby(['method', 'event_count', 'position_sigma_mm'], as_index=False).agg(
        repeats=('repeat', 'count'), selected_events_mean=('selected_events', 'mean'),
        iou_mean=('iou_equal_area', 'mean'), iou_std=('iou_equal_area', 'std'),
        localization_error_mm_mean=('localization_error_mm', 'mean'),
        localization_error_mm_std=('localization_error_mm', 'std'),
    )
    summary.to_csv(OUT_DIR / 'detector_resolution_metrics_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method, group in summary.groupby('method'):
        for count, part in group.groupby('event_count'):
            label = f'{method}, {count} events'
            axes[0].errorbar(part['position_sigma_mm'], part['iou_mean'], yerr=part['iou_std'], marker='o', capsize=3, label=label)
            axes[1].errorbar(part['position_sigma_mm'], part['localization_error_mm_mean'], yerr=part['localization_error_mm_std'], marker='o', capsize=3, label=label)
    axes[0].set(xlabel='Hit position uncertainty sigma [mm]', ylabel='IoU (higher is better)')
    axes[1].set(xlabel='Hit position uncertainty sigma [mm]', ylabel='Localization error [mm] (lower is better)')
    for axis in axes: axis.grid(alpha=0.25)
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle('Robustness to detector hit-position uncertainty')
    fig.savefig(OUT_DIR / 'detector_resolution_metrics.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()
