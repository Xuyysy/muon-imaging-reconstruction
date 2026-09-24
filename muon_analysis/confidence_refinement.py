"""Pre-registered split-half consistency refinement for a *future* scene.

Only accepted POCA coordinates, their capped angle weights, and the calibrated
hit-position sigma enter the reconstruction. No target mask, IoU, or held-out
scene data are read here. The consistency score is NOT a calibrated probability.

We first estimate a 2x2 XY child pattern from all z slices of each 16x16 XY
cell. Four fixed random split-halves ask whether that pattern repeats. A
noise-dependent shrinkage mixes the pooled pattern with the coarse uniform
image in every z slice, conserving each slice's scattering weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_adaptive_voxel_experiment import THETA_CAP_MRAD, imaging_edges
from run_uncertainty_aware_angle_experiment import theta_noise_floor_mrad

COARSE_BINS = 16
FINE_BINS = 32
SPLIT_REPEATS = 4
MIN_HALF_EFFECTIVE_EVENTS = 4.0  # Four XY children: at least one effective event per child.
SPLIT_SEED = 20261007


def _effective_count(weight: np.ndarray, weight_squared: np.ndarray) -> np.ndarray:
    """Effective count from *event-wise* sum(w) and sum(w²), not squared bins."""
    total = weight.sum(axis=-1)
    squared = weight_squared.sum(axis=-1)
    return np.divide(total * total, squared, out=np.zeros_like(total),
                     where=squared > 0)


def _child_weights(points_xy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return each coarse XY cell's four child weights, shape (16,16,4)."""
    fine_edges = imaging_edges(FINE_BINS)
    fine, _, _ = np.histogram2d(points_xy[:, 0], points_xy[:, 1],
                                bins=fine_edges[:2], weights=weights)
    return fine.reshape(COARSE_BINS, 2, COARSE_BINS, 2).transpose(0, 2, 1, 3).reshape(16, 16, 4)


def split_half_consistency(points_xy: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean positive cosine agreement of child *deviations* from uniform.

    The split seed and count are fixed ahead of validation. A split contributes
    only if both halves have >=4 effective events in that coarse XY cell.
    A positive score means the two halves favour similar sub-cell patterns;
    it is not a p-value, posterior probability, or guarantee of true detail.
    """
    n = len(weights)
    agreement = np.zeros((16, 16), dtype=float)
    supported = np.zeros((16, 16), dtype=int)
    rng = np.random.default_rng(SPLIT_SEED)
    for _ in range(SPLIT_REPEATS):
        order = rng.permutation(n)
        halves = (order[:n // 2], order[n // 2:])
        child_a = _child_weights(points_xy[halves[0]], weights[halves[0]])
        child_b = _child_weights(points_xy[halves[1]], weights[halves[1]])
        squared_a = _child_weights(points_xy[halves[0]], weights[halves[0]] ** 2)
        squared_b = _child_weights(points_xy[halves[1]], weights[halves[1]] ** 2)
        valid = (_effective_count(child_a, squared_a) >= MIN_HALF_EFFECTIVE_EVENTS) & (
            _effective_count(child_b, squared_b) >= MIN_HALF_EFFECTIVE_EVENTS)
        sum_a, sum_b = child_a.sum(axis=2), child_b.sum(axis=2)
        pa = np.divide(child_a, sum_a[:, :, None], out=np.zeros_like(child_a),
                       where=sum_a[:, :, None] > 0) - 0.25
        pb = np.divide(child_b, sum_b[:, :, None], out=np.zeros_like(child_b),
                       where=sum_b[:, :, None] > 0) - 0.25
        dot = np.sum(pa * pb, axis=2)
        norm = np.sqrt(np.sum(pa * pa, axis=2) * np.sum(pb * pb, axis=2))
        cosine = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
        agreement += np.where(valid, np.maximum(cosine, 0.0), 0.0)
        supported += valid
    # An unsupported split contributes zero, rather than being omitted from
    # the denominator and artificially inflating sparse cells' scores.
    return agreement / SPLIT_REPEATS, supported


def confidence_refined_volume(events: pd.DataFrame, sigma_mm: float,
                              spacing_mm: float) -> tuple[np.ndarray, dict]:
    """Return a weight-conserving 32x32x16 volume and diagnostics.

    q_xy = split_agreement * n_eff/(n_eff+4) /
           [1+(theta_noise/theta_cap)^2].
    The +4 term is the four XY children, not a parameter fitted to a target.
    The pooled 2x2 child distribution is projected back into each original
    coarse z slice; there is no movement of weight between z slices.
    """
    if sigma_mm < 0 or spacing_mm <= 0:
        raise ValueError('sigma must be nonnegative and spacing positive')
    points = events[['poca_x_mm', 'poca_y_mm', 'poca_z_mm']].to_numpy(dtype=float)
    weights = events['capped_weight'].to_numpy(dtype=float)
    if not np.isfinite(points).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Coordinates and weights must be finite; weights nonnegative')
    coarse, _ = np.histogramdd(points, bins=imaging_edges(COARSE_BINS), weights=weights)
    child = _child_weights(points[:, :2], weights)
    sum_xy = child.sum(axis=2)
    pattern = np.divide(child, sum_xy[:, :, None], out=np.zeros_like(child),
                        where=sum_xy[:, :, None] > 0)
    consistency, supported = split_half_consistency(points[:, :2], weights)
    n_eff = _effective_count(child, _child_weights(points[:, :2], weights ** 2))
    theta_noise = theta_noise_floor_mrad(sigma_mm, spacing_mm)
    noise_factor = 1.0 / (1.0 + (theta_noise / THETA_CAP_MRAD) ** 2)
    alpha = consistency * n_eff / (n_eff + 4.0) * noise_factor
    # All alpha terms lie in [0,1]; otherwise the convex combination below
    # would no longer guarantee nonnegative, conserved per-slice weights.
    if np.any((alpha < 0) | (alpha > 1)):
        raise AssertionError('Invalid shrinkage coefficient')
    base = coarse[:, :, :, None] / 4.0
    pooled_detail = coarse[:, :, :, None] * pattern[:, :, None, :]
    mixed = (1.0 - alpha[:, :, None, None]) * base + (
        alpha[:, :, None, None] * pooled_detail)
    output = mixed.reshape(16, 16, 16, 2, 2).transpose(0, 3, 1, 4, 2).reshape(32, 32, 16)
    if not np.allclose(output.sum(axis=(0, 1)), coarse.sum(axis=(0, 1)),
                       rtol=1e-10, atol=1e-8):
        raise AssertionError('Refinement changed per-z scattering weight')
    diagnostics = {
        'supported_xy_cells': int(np.count_nonzero(supported)),
        'mean_consistency_supported': float(consistency[supported > 0].mean())
            if np.any(supported) else 0.0,
        'mean_alpha_nonzero': float(alpha[alpha > 0].mean())
            if np.any(alpha > 0) else 0.0,
        'active_refined_xy_cells': int(np.count_nonzero(alpha > 0)),
        'theta_noise_mrad': float(theta_noise),
    }
    return output, diagnostics
