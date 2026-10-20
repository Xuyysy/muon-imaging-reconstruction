"""Local data-likelihood Fisher information utilities."""

from __future__ import annotations
import math
import numpy as np


def fisher_information(covariances, covariance_derivatives) -> np.ndarray:
    """Expected Gaussian covariance Fisher: F_ab=1/2 sum tr(C^-1 D_a C^-1 D_b).

    This is information in the observation likelihood only; no prior precision
    or regularization appears here.
    """
    c = np.asarray(covariances, dtype=float)
    d = np.asarray(covariance_derivatives, dtype=float)
    if c.ndim != 3 or c.shape[1:] != (4, 4) or d.ndim != 4 or d.shape[:3] != c.shape:
        raise ValueError("expected covariances (n,4,4) and derivatives (n,4,4,p)")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(d)):
        raise ValueError("Fisher inputs must be finite")
    p = d.shape[-1]
    f = np.zeros((p, p))
    for ci, di in zip(c, d):
        ci = 0.5*(ci+ci.T)
        np.linalg.cholesky(ci)  # Require an actual SPD likelihood covariance.
        inv = np.linalg.inv(ci)
        for a in range(p):
            for b in range(p):
                f[a,b] += 0.5*np.trace(inv@di[:,:,a]@inv@di[:,:,b])
    return 0.5*(f+f.T)


def contrast_information(fisher, contrast, *, rank_rtol: float = 1e-10) -> dict:
    """Return likelihood-only information for a linear parameter contrast.

    `contrast` is the coefficient vector c in c.T @ theta. It must be
    estimable in the full parameter model, including every nuisance parameter:
    c must be orthogonal to the numerical null space of the joint Fisher matrix.
    Only then is c.T F^+ c reported as its local Cramér–Rao variance. For a
    non-estimable contrast, variance is +inf and information is zero; a finite
    pseudoinverse quadratic form is deliberately not exposed as an estimate.

    Parameter units are inherited from `theta`; the returned variance is in
    squared contrast-parameter units. Priors/regularization must not be passed
    here as if they were data Fisher information.
    """
    f = np.asarray(fisher, dtype=float)
    c = np.asarray(contrast, dtype=float)
    rtol = float(rank_rtol)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or not np.all(np.isfinite(f)):
        raise ValueError("fisher must be a finite square matrix")
    if c.shape != (f.shape[0],) or not np.all(np.isfinite(c)) or not np.any(c):
        raise ValueError("contrast must be a finite, nonzero vector matching fisher")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rank_rtol must be finite and positive")
    scale = max(float(np.linalg.norm(f, ord=2)), 0.0)
    symmetry_tol = max(1e-12, scale * 1e-10)
    if not np.allclose(f, f.T, rtol=1e-10, atol=symmetry_tol):
        raise ValueError("fisher must be symmetric")
    f = 0.5 * (f + f.T)
    eigenvalues, eigenvectors = np.linalg.eigh(f)
    psd_tol = max(1e-12, scale * 1e-10)
    if eigenvalues[0] < -psd_tol:
        raise ValueError("fisher must be positive semidefinite")
    rank_tolerance = scale * rtol
    estimable_modes = eigenvalues > rank_tolerance
    null_modes = ~estimable_modes
    null_projection = eigenvectors[:, null_modes] @ (eigenvectors[:, null_modes].T @ c)
    null_projection_norm = float(np.linalg.norm(null_projection))
    projection_tolerance = max(1e-12, math.sqrt(rtol) * float(np.linalg.norm(c)))
    estimable = null_projection_norm <= projection_tolerance
    if estimable and np.any(estimable_modes):
        coefficients = eigenvectors[:, estimable_modes].T @ c
        variance = float(np.sum(coefficients**2 / eigenvalues[estimable_modes]))
        information = 1.0 / variance if variance > 0 else math.inf
    else:
        estimable = False
        variance, information = math.inf, 0.0
    return {
        "estimable": bool(estimable),
        "variance": variance,
        "information": information,
        "null_projection_norm": null_projection_norm,
        "projection_tolerance": projection_tolerance,
        "rank": int(np.count_nonzero(estimable_modes)),
        "rank_tolerance": rank_tolerance,
        "eigenvalues": np.maximum(eigenvalues, 0.0),
    }


def information_report(fisher, nuisance_indices=(), *, rank_rtol: float = 1e-10) -> dict:
    """Return data eigensystem and Schur-profiled information for non-nuisance.

    rank_rtol is a fixed numerical reporting tolerance, not an optimized
    scientific threshold. Nuisance profiling uses a pseudoinverse, so a
    singular nuisance block is handled conservatively.
    """
    f = np.asarray(fisher, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or not np.all(np.isfinite(f)):
        raise ValueError("fisher must be a finite square matrix")
    f = 0.5*(f+f.T)
    vals, vecs = np.linalg.eigh(f)
    scale = max(float(vals[-1]), 0.0)
    if vals[0] < -max(1e-12, scale*1e-10):
        raise ValueError("Fisher matrix is not positive semidefinite")
    tol = scale*rank_rtol
    rank = int(np.count_nonzero(vals > tol)) if scale else 0
    nuisance = tuple(sorted(set(int(i) for i in nuisance_indices)))
    if any(i < 0 or i >= len(f) for i in nuisance):
        raise ValueError("nuisance index out of range")
    keep = tuple(i for i in range(len(f)) if i not in nuisance)
    profiled = f[np.ix_(keep, keep)].copy()
    if nuisance and keep:
        fnn = f[np.ix_(nuisance, nuisance)]
        fkn = f[np.ix_(keep, nuisance)]
        profiled -= fkn@np.linalg.pinv(fnn, rcond=rank_rtol)@fkn.T
        profiled = 0.5*(profiled+profiled.T)
    return {
        "eigenvalues": np.maximum(vals, 0.0),
        "eigenvectors_columns": vecs,
        "rank": rank,
        "rank_tolerance": tol,
        "rank_rtol": rank_rtol,
        "null_directions": vecs[:, vals <= tol],
        "nuisance_indices": nuisance,
        "profiled_parameter_indices": keep,
        "profiled_information": profiled,
    }


def add_prior_precision(data_fisher, prior_precision) -> np.ndarray:
    """Form a separate prior-plus-data precision illustration, never mutate F."""
    f = np.asarray(data_fisher, dtype=float)
    prior = np.asarray(prior_precision, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or f.shape != prior.shape:
        raise ValueError("data Fisher and prior precision must be same-size square matrices")
    total = f+prior
    return 0.5*(total+total.T)
