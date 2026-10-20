"""Minimal, interpretable two-child split/merge statistical utilities.

All optimizer parameters are dimensionless scales multiplying covariance
derivatives that already include the declared reference density. The module is
NumPy-only so the synthetic prototype does not add a SciPy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np


class SplitMergeError(ValueError):
    """Invalid split/merge model or synthetic likelihood input."""


def child_levels_from_parent_contrast(parent_mean: float, contrast: float) -> tuple[float, float]:
    """Map dimensionless parent mean and signed right-minus-left contrast.

    lambda_left/right = lambda_ref * (parent_mean ∓ contrast/2).
    The physical density unit is inherited from lambda_ref (mm^-1).
    """
    mean, difference = float(parent_mean), float(contrast)
    if not math.isfinite(mean) or not math.isfinite(difference):
        raise SplitMergeError("parent mean and contrast must be finite")
    left, right = mean - difference / 2.0, mean + difference / 2.0
    if left < 0.0 or right < 0.0:
        raise SplitMergeError("child scattering levels must remain nonnegative")
    return left, right


def expected_two_sided_power(effect: float, variance: float, *, alpha: float = 0.05) -> float:
    """Normal/Wald design power for a predeclared nonzero contrast effect."""
    effect, variance, alpha = float(effect), float(variance), float(alpha)
    if not math.isfinite(effect) or not math.isfinite(variance) or variance <= 0:
        raise SplitMergeError("effect must be finite and variance finite and positive")
    if not 0.0 < alpha < 1.0:
        raise SplitMergeError("alpha must be strictly between zero and one")
    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    noncentrality = abs(effect) / math.sqrt(variance)
    normal = NormalDist(noncentrality, 1.0)
    return float(1.0 - (normal.cdf(zcrit) - normal.cdf(-zcrit)))


def calibrate_lrt_threshold(null_lrt_values, *, alpha: float = 0.05) -> float:
    """Empirical upper-tail cutoff; the caller must supply development nulls only.

    The conservative `higher` order statistic avoids interpolation below an
    observed null score. Evaluation observations must never be passed here.
    """
    values = np.asarray(null_lrt_values, dtype=float)
    if values.ndim != 1 or len(values) < 20 or not np.all(np.isfinite(values)):
        raise SplitMergeError("at least 20 finite development-null LRT values are required")
    if not 0.0 < alpha < 1.0:
        raise SplitMergeError("alpha must be strictly between zero and one")
    if np.any(values < -1e-9):
        raise SplitMergeError("likelihood-ratio values cannot be negative")
    index = min(len(values) - 1, max(0, math.ceil((1.0 - alpha) * len(values)) - 1))
    return float(np.sort(np.maximum(values, 0.0))[index])


def decide_split(contrast_report: dict, lrt: float, threshold: float,
                 predicted_power: float, *, min_power: float = 0.8,
                 fit_converged: bool = True) -> dict:
    """Split only if contrast is estimable, powered, and exceeds null calibration."""
    lrt, threshold = float(lrt), float(threshold)
    power, min_power = float(predicted_power), float(min_power)
    if not all(math.isfinite(v) for v in (lrt, threshold, power, min_power)):
        raise SplitMergeError("decision statistics must be finite")
    if lrt < 0 or threshold < 0 or not 0 <= power <= 1 or not 0 <= min_power <= 1:
        raise SplitMergeError("invalid LRT, threshold, or power range")
    if not fit_converged:
        reason = "fit_failure"
    elif not bool(contrast_report.get("estimable", False)):
        reason = "contrast_not_estimable"
    elif power < min_power:
        reason = "insufficient_design_power"
    elif lrt <= threshold:
        reason = "evidence_below_calibrated_threshold"
    else:
        reason = "split"
    return {"split": reason == "split", "reason": reason}


def count_based_split(child_event_counts, *, minimum_per_child: int = 40) -> bool:
    """Simple event-count-only comparator; it does not inspect residual values."""
    counts = tuple(int(value) for value in child_event_counts)
    if len(counts) != 2 or any(value < 0 for value in counts):
        raise SplitMergeError("exactly two nonnegative child event counts are required")
    if isinstance(minimum_per_child, bool) or int(minimum_per_child) != minimum_per_child or minimum_per_child < 1:
        raise SplitMergeError("minimum_per_child must be a positive integer")
    return min(counts) >= int(minimum_per_child)


@dataclass(frozen=True)
class GaussianObservationGroup:
    """One path class with common covariance derivatives and observed residuals."""
    name: str
    residuals_mm: np.ndarray
    base_covariance_mm2: np.ndarray
    covariance_derivatives_mm2: np.ndarray  # (parameters, 4, 4)

    def __post_init__(self):
        y = np.asarray(self.residuals_mm, dtype=float)
        base = np.asarray(self.base_covariance_mm2, dtype=float)
        deriv = np.asarray(self.covariance_derivatives_mm2, dtype=float)
        if y.ndim != 2 or y.shape[1] != 4 or len(y) == 0:
            raise SplitMergeError("residual group must contain (n,4) values")
        if base.shape != (4, 4) or deriv.ndim != 3 or deriv.shape[1:] != (4, 4):
            raise SplitMergeError("base/derivative covariance shapes must be (4,4)/(p,4,4)")
        if not all(np.all(np.isfinite(a)) for a in (y, base, deriv)):
            raise SplitMergeError("observation group values must be finite")
        if not np.allclose(base, base.T, atol=1e-12) or not np.allclose(deriv, deriv.transpose(0, 2, 1), atol=1e-12):
            raise SplitMergeError("covariance matrices and derivatives must be symmetric")


def grouped_gaussian_nll_and_gradient(groups, parameters):
    """Zero-mean Gaussian NLL/gradient using exact group sufficient statistics."""
    theta = np.asarray(parameters, dtype=float)
    groups = tuple(groups)
    if theta.ndim != 1 or not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise SplitMergeError("parameters must be a finite nonnegative vector")
    if not groups:
        raise SplitMergeError("at least one observation group is required")
    nll, gradient = 0.0, np.zeros_like(theta)
    for group in groups:
        if group.covariance_derivatives_mm2.shape[0] != len(theta):
            raise SplitMergeError("parameter count does not match group derivatives")
        covariance = group.base_covariance_mm2 + np.einsum(
            "p,pab->ab", theta, group.covariance_derivatives_mm2
        )
        covariance = 0.5 * (covariance + covariance.T)
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise SplitMergeError("model covariance must be positive definite") from exc
        inv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(4)))
        scatter = np.asarray(group.residuals_mm).T @ np.asarray(group.residuals_mm)
        n = len(group.residuals_mm)
        nll += 0.5 * (n * 2.0 * np.log(np.diag(chol)).sum() + np.trace(inv @ scatter))
        for index, derivative in enumerate(group.covariance_derivatives_mm2):
            gradient[index] += 0.5 * (
                n * np.trace(inv @ derivative)
                - np.trace(inv @ derivative @ inv @ scatter)
            )
    return float(nll), gradient


def fit_covariance_model(groups, *, initial=None, max_iter: int = 500,
                         tolerance: float = 1e-6) -> tuple[np.ndarray, bool, int]:
    """Fit nonnegative covariance scales with projected gradient + Armijo search.

    `tolerance` is on the Fisher-diagonal-preconditioned projected gradient,
    not on the raw likelihood gradient. The default 1e-6 is above the
    roundoff-limited scaled gradient observed when NLL values are O(1e4),
    while remaining a local optimizer stopping tolerance rather than a
    scientific decision threshold.

    Returns parameters, convergence flag and iteration count. Failure to
    converge is explicit; callers must retain/report it rather than silently
    treating the final iterate as a successful fit.
    """
    groups = tuple(groups)
    if not groups:
        raise SplitMergeError("at least one observation group is required")
    n_parameters = groups[0].covariance_derivatives_mm2.shape[0]
    theta = np.ones(n_parameters) if initial is None else np.asarray(initial, dtype=float).copy()
    if theta.shape != (n_parameters,) or not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise SplitMergeError("initial values must be a finite nonnegative parameter vector")
    if max_iter < 1 or tolerance <= 0 or not math.isfinite(tolerance):
        raise SplitMergeError("max_iter and tolerance must be positive")
    value, gradient = grouped_gaussian_nll_and_gradient(groups, theta)
    step = 1.0
    for iteration in range(1, max_iter + 1):
        fisher_diagonal = np.zeros(n_parameters)
        for group in groups:
            covariance = group.base_covariance_mm2 + np.einsum(
                "p,pab->ab", theta, group.covariance_derivatives_mm2
            )
            inv = np.linalg.inv(covariance)
            for index, derivative in enumerate(group.covariance_derivatives_mm2):
                fisher_diagonal[index] += 0.5 * len(group.residuals_mm) * np.trace(
                    inv @ derivative @ inv @ derivative
                )
        fisher_scale = max(float(np.max(fisher_diagonal)), 1.0)
        preconditioner = np.maximum(fisher_diagonal, fisher_scale * 1e-12)
        scaled_gradient = gradient / preconditioner
        projected_gradient = theta - np.maximum(0.0, theta - scaled_gradient)
        if np.linalg.norm(projected_gradient, ord=np.inf) <= tolerance:
            return theta, True, iteration - 1
        trial_step = min(max(step, 1e-8), 10.0)
        accepted = False
        for _ in range(50):
            candidate = np.maximum(0.0, theta - trial_step * scaled_gradient)
            delta = candidate - theta
            if not np.any(delta):
                trial_step *= 0.5
                continue
            candidate_value, candidate_gradient = grouped_gaussian_nll_and_gradient(groups, candidate)
            armijo = value + 1e-4 * float(gradient @ delta)
            if candidate_value <= armijo:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            return theta, False, iteration
        theta, value, gradient = candidate, candidate_value, candidate_gradient
        step = min(trial_step * 1.5, 10.0)
        if np.linalg.norm(delta) <= tolerance * (1.0 + np.linalg.norm(theta)):
            return theta, True, iteration
    return theta, False, max_iter
