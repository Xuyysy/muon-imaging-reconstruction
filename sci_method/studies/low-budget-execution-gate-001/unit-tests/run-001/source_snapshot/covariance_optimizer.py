"""Small NumPy-only bound-constrained covariance likelihood solver.

Coordinates are dimensionless scattering scales; covariance units are mm^2.
No scientific split rule, prior, truth, or event-selection field is read here.
Convergence is fixed-scale KKT stationarity, never a small parameter update.
"""
from dataclasses import dataclass, field
import math
import numpy as np


def kkt_residual(parameters, gradient, coordinate_scale):
    """KKT score for theta>=0; a positive score is allowed only at exact zero."""
    residual = np.where(parameters > 0, gradient, np.minimum(gradient, 0.0))
    return residual, float(np.max(np.abs(coordinate_scale * residual)))


def _whiten(chol, matrix):
    left = np.linalg.solve(chol, matrix)
    return np.linalg.solve(chol, left.T).T


def _log1pmx(x):
    """log(1+x)-x without cancellation; eig x must be greater than -1."""
    x = np.asarray(x)
    result = np.empty_like(x)
    small = np.abs(x) < 0.01
    z = x[small]
    result[small] = sum((-1)**(k+1)*z**k/k for k in range(2, 13))
    result[~small] = np.log1p(x[~small])-x[~small]
    return result


class CovarianceObjective:
    """Cache observed sufficient statistics, retaining the original likelihood."""
    def __init__(self, groups):
        self.groups = tuple(groups)
        if not self.groups:
            raise ValueError("at least one observation group is required")
        self.p = self.groups[0].covariance_derivatives_mm2.shape[0]
        self.data = []
        for group in self.groups:
            if group.covariance_derivatives_mm2.shape[0] != self.p:
                raise ValueError("inconsistent parameter count")
            self.data.append((len(group.residuals_mm),
                              group.residuals_mm.T @ group.residuals_mm,
                              group.base_covariance_mm2,
                              group.covariance_derivatives_mm2))

    def evaluate(self, theta):
        """Return full unnormalised NLL, score, observed Hessian and data Fisher.

        Additive n*d/2*log(2*pi) is omitted consistently with the old API.
        Whitening evaluates the score as tr((nI-W)Di)/2, avoiding subtraction
        of two large independently computed inverse/trace expressions.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.p,) or np.any(theta < 0) or not np.isfinite(theta).all():
            raise ValueError("parameters must be finite and nonnegative")
        values, gradient = [], np.zeros(self.p)
        hessian, fisher = np.zeros((self.p, self.p)), np.zeros((self.p, self.p))
        for n, scatter, base, derivatives in self.data:
            covariance = base + np.einsum("i,iab->ab", theta, derivatives)
            chol = np.linalg.cholesky((covariance+covariance.T)/2)
            w = _whiten(chol, scatter)
            ds = np.array([_whiten(chol, d) for d in derivatives])
            score_matrix = n*np.eye(4)-w
            values.append(float(n*np.log(np.diag(chol)).sum()+np.trace(w)/2))
            gradient += np.einsum("ab,iba->i", score_matrix, ds)/2
            for i, di in enumerate(ds):
                for j, dj in enumerate(ds):
                    product = di @ dj
                    fisher[i, j] += n*np.trace(product)/2
                    hessian[i, j] += (
                        np.trace((product+dj@di)@w)-n*np.trace(product))/2
        return (math.fsum(values), gradient, (hessian+hessian.T)/2,
                (fisher+fisher.T)/2)

    def difference(self, theta, candidate):
        """Compute f(candidate)-f(theta) directly, not by subtracting full NLLs.

        With E=L^-1 DeltaC L^-T, W=L^-1 S L^-T:
        2 Deltaf = n tr(log(I+E)-E) + tr(E(nI-W))
                   + tr((I+E)^-1 E^2 W).
        The expansion is an exact rearrangement, not a quadratic approximation.
        """
        terms = []
        for n, scatter, base, derivatives in self.data:
            covariance = base+np.einsum("i,iab->ab", theta, derivatives)
            chol = np.linalg.cholesky((covariance+covariance.T)/2)
            w = _whiten(chol, scatter)
            e = _whiten(chol, np.einsum("i,iab->ab", candidate-theta, derivatives))
            e = (e+e.T)/2
            eigen = np.linalg.eigvalsh(e)
            if eigen.min() <= -1:
                return math.inf
            term = (n*float(_log1pmx(eigen).sum())
                    + np.trace(e@(n*np.eye(4)-w))
                    + np.trace(np.linalg.solve(np.eye(4)+e, e@e)@w))/2
            terms.append(float(term))
        return math.fsum(terms)


@dataclass
class CovarianceFitResult:
    parameters: np.ndarray
    status: str
    iterations: int
    nll: float
    gradient: np.ndarray
    kkt_inf: float
    reason: str
    history: list = field(default_factory=list)

    @property
    def converged(self):
        return self.status == "converged"


def fit_covariance_detailed(groups, *, initial=None, max_iter=500, tolerance=1e-6,
                            record_history=False):
    """Damped active-set Newton with stable Armijo differences.

    Observed-Hessian Newton is used only on positive-curvature free directions.
    Otherwise use Fisher diagonal descent. This changes numerical solution,
    not the NLL model. Unobserved coordinates are left unchanged, never certified
    identifiable. KKT proves local stationarity only, not global optimality.
    """
    objective = CovarianceObjective(groups)
    theta = np.ones(objective.p) if initial is None else np.array(initial, dtype=float)
    if theta.shape != (objective.p,) or np.any(theta < 0) or not np.isfinite(theta).all():
        raise ValueError("initial parameters must be finite and nonnegative")
    if max_iter < 1 or tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("max_iter and tolerance must be positive")
    scale = np.maximum(1.0, theta.copy())
    history = []
    for iteration in range(max_iter+1):
        try:
            value, gradient, hessian, fisher = objective.evaluate(theta)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return CovarianceFitResult(theta, "failed", iteration, math.nan,
                                       np.full(objective.p, math.nan), math.inf,
                                       f"invalid_covariance: {exc}", history)
        _, residual = kkt_residual(theta, gradient, scale)
        if not (math.isfinite(value) and np.isfinite(gradient).all()):
            return CovarianceFitResult(theta, "failed", iteration, value, gradient,
                                       residual, "nonfinite_objective_or_score", history)
        if residual <= tolerance:
            # First-order stationarity can also occur at a local maximum.
            # Reject resolved negative curvature on interior coordinates.
            # Flat directions are allowed but do not imply identifiability.
            interior = theta > 0
            if np.any(interior):
                curvature = np.linalg.eigvalsh(hessian[np.ix_(interior, interior)])
                curvature_floor = 64*np.finfo(float).eps*max(1.0, float(np.max(np.abs(curvature))))
                if curvature.min() < -curvature_floor:
                    return CovarianceFitResult(theta, "unproven", iteration, value,
                        gradient, residual, "stationary_negative_curvature", history)
            return CovarianceFitResult(theta, "converged", iteration, value, gradient,
                                       residual, "fixed_scale_kkt", history)
        if iteration == max_iter:
            return CovarianceFitResult(theta, "unproven", iteration, value, gradient,
                                       residual, "iteration_budget", history)
        free = (theta > 0) | (gradient < 0)
        hf = hessian[np.ix_(free, free)]
        eigen, vectors = np.linalg.eigh(hf)
        floor = 64*np.finfo(float).eps*max(1.0, float(np.max(np.abs(eigen))))
        positive = eigen > floor
        direction = np.zeros_like(theta)
        if eigen.min() >= -floor and np.any(positive):
            direction[free] = -vectors[:, positive] @ (
                (vectors[:, positive].T @ gradient[free])/eigen[positive])
            method = "observed_hessian"
        else:
            method = "fisher_diagonal"
        if method == "fisher_diagonal" or float(gradient @ direction) >= 0:
            diagonal = np.maximum(np.diag(fisher), 64*np.finfo(float).eps)
            direction[free] = -gradient[free]/diagonal[free]
            method = "fisher_diagonal"
        accepted = False
        all_changes_rounded_to_zero = True
        for halving in range(50):
            step = 0.5**halving
            candidate = np.maximum(0, theta+step*direction)
            delta = candidate-theta
            all_changes_rounded_to_zero &= not bool(np.any(delta))
            directional = float(gradient @ delta)
            try:
                difference = objective.difference(theta, candidate)
            except (ValueError, np.linalg.LinAlgError):
                difference = math.inf
            accepted = bool(np.any(delta) and directional < 0
                            and math.isfinite(difference)
                            and difference <= 1e-4*directional)
            if record_history:
                history.append(dict(iteration=iteration, parameters=theta.copy(),
                    gradient=gradient.copy(), kkt_inf=residual, method=method,
                    fisher_diagonal=np.diag(fisher).copy(), hessian_eigenvalues=eigen,
                    step=step, delta=delta.copy(), stable_nll_change=difference,
                    armijo_change=1e-4*directional, accepted=accepted))
            if accepted:
                theta = candidate
                break
        if not accepted:
            # Line-search exhaustion alone does not establish a roundoff cause.
            # Only label directly observed unrepresentable parameter changes;
            # other exhaustion remains unproven and requires diagnosis.
            status = "precision_limited" if all_changes_rounded_to_zero else "unproven"
            reason = "parameter_resolution_kkt_unmet" if all_changes_rounded_to_zero else "line_search_exhausted"
            return CovarianceFitResult(theta, status, iteration+1,
                value, gradient, residual, reason, history)
    raise AssertionError("unreachable")
