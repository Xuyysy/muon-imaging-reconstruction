"""冻结的局部支撑软深度门 Q；只用观测硬门事件建立空间支撑。"""
from __future__ import annotations

import numpy as np

from run_adaptive_voxel_experiment import imaging_edges

SUPPORT_GRID = 16
SUPPORT_SHRINK = 2.0


def local_support_gate(xy: np.ndarray, angle_weight: np.ndarray,
                       hard: np.ndarray, propagated: np.ndarray) -> np.ndarray:
    """返回 Q 的**事件门系数**，调用者再乘原角权重。

    只用名义硬门图的 n_eff；格内 q=n_eff/(n_eff+2)，空格 q=0。
    Q=h*p+(1-h)*p*q。不能读取仿真目标掩膜或评价指标。
    """
    xy = np.asarray(xy, float)
    w = np.asarray(angle_weight, float)
    h = np.asarray(hard, float)
    p = np.asarray(propagated, float)
    if xy.shape != (len(w), 2) or not (len(w) == len(h) == len(p)):
        raise ValueError('Unaligned event arrays')
    if (not np.isfinite(xy).all() or not np.isfinite(w).all()
            or not np.isfinite(h).all() or not np.isfinite(p).all()
            or (w < 0).any() or not np.isin(h, [0., 1.]).all()
            or (p < 0).any() or (p > 1).any()):
        raise ValueError('Invalid event values')
    x_edge, y_edge = imaging_edges(SUPPORT_GRID)[:2]
    hard_w = w * h
    total, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edge, y_edge),
                                  weights=hard_w)
    squared, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edge, y_edge),
                                    weights=hard_w**2)
    neff = np.divide(total**2, squared, out=np.zeros_like(total), where=squared > 0)
    q = neff / (neff + SUPPORT_SHRINK)
    ix = np.clip(np.searchsorted(x_edge, xy[:, 0], side='right') - 1,
                 0, SUPPORT_GRID - 1)
    iy = np.clip(np.searchsorted(y_edge, xy[:, 1], side='right') - 1,
                 0, SUPPORT_GRID - 1)
    gate = h * p + (1 - h) * p * q[ix, iy]
    if np.any(gate < -1e-12) or np.any(gate > p + 1e-12):
        raise AssertionError('Q should be between 0 and P event-wise')
    return gate
