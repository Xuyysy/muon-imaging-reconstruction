"""Fixed-momentum, small-angle Gaussian scattering forward model.

Coordinates/lengths are mm, angles are radians, momentum is GeV/c, and
lambda=1/X0 is mm^-1. The prototype assumes known 4 GeV/c momentum (beta=1),
known plane geometry, independent isotropic Gaussian hit x/y error, nearly
straight paths, Gaussian small-angle scattering, and no energy loss. It is not
an extrapolation to a natural cosmic-ray momentum spectrum.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


class ForwardModelError(ValueError):
    """Invalid or degenerate Gaussian forward-model input."""


@dataclass(frozen=True)
class BoxVoxel:
    """Axis-aligned voxel bounds, all in mm."""
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    z_min_mm: float
    z_max_mm: float

    def __post_init__(self):
        b = (self.x_min_mm, self.x_max_mm, self.y_min_mm,
             self.y_max_mm, self.z_min_mm, self.z_max_mm)
        if not all(math.isfinite(v) for v in b):
            raise ForwardModelError("voxel bounds must be finite")
        if not (self.x_min_mm < self.x_max_mm and self.y_min_mm < self.y_max_mm
                and self.z_min_mm < self.z_max_mm):
            raise ForwardModelError("voxel bounds must have positive extent")


def _plane_z(z_mm) -> np.ndarray:
    z = np.asarray(z_mm, dtype=float)
    if z.shape != (4,) or not np.all(np.isfinite(z)):
        raise ForwardModelError("z_mm must contain four finite plane positions")
    if np.any(np.diff(z) <= 0):
        raise ForwardModelError("plane order must be strictly upstream-to-downstream")
    return z


def residual_jacobian(z_mm) -> np.ndarray:
    """4x12 J maps plane-major [x,y,z] hits to [rx_down1,rx_down2,ry1,ry2].

    Plane z coordinates are known, not fitted. Each residual is a downstream
    coordinate minus extrapolation of the two upstream coordinates.
    """
    z = _plane_z(z_mm)
    j = np.zeros((4, 12), dtype=float)
    row = 0
    for axis in (0, 1):
        for down_index in (2, 3):
            t = (z[down_index] - z[0]) / (z[1] - z[0])
            j[row, axis] = -(1.0 - t)
            j[row, 3 + axis] = -t
            j[row, 3 * down_index + axis] = 1.0
            row += 1
    return j


def residuals_from_hits(hits_mm, z_mm) -> np.ndarray:
    """Convert (4,3) or (n,4,3) hits into residuals with shape (n,4)."""
    hits = np.asarray(hits_mm, dtype=float)
    if hits.shape == (4, 3):
        hits = hits[None]
    if hits.ndim != 3 or hits.shape[1:] != (4, 3) or not np.all(np.isfinite(hits)):
        raise ForwardModelError("hits_mm must be finite with shape (4,3) or (n,4,3)")
    flat = hits.reshape(len(hits), 12).copy()
    flat[:, 2::3] = 0.0  # z is fixed geometry; do not propagate hit z errors.
    return flat @ residual_jacobian(z_mm).T


def detector_residual_covariance(z_mm, sigma_hit_mm: float) -> np.ndarray:
    """J Sigma_hit J.T for independent x/y hit errors; output unit is mm^2.

    Shared upstream hits induce nonzero covariance between downstream residuals.
    Equal, independent x/y sigma is an explicit simplifying assumption.
    """
    sigma = float(sigma_hit_mm)
    if not math.isfinite(sigma) or sigma < 0:
        raise ForwardModelError("sigma_hit_mm must be finite and nonnegative")
    variances = np.zeros(12)
    variances[0::3] = sigma**2
    variances[1::3] = sigma**2
    j = residual_jacobian(z_mm)
    result = (j * variances[None, :]) @ j.T
    return 0.5 * (result + result.T)


def _clip_linear_interval(origin, slope, low, high, z_low, z_high):
    if abs(slope) < 1e-15:
        return (z_low, z_high) if low <= origin <= high else None
    a, b = (low-origin)/slope, (high-origin)/slope
    lo, hi = max(z_low, min(a, b)), min(z_high, max(a, b))
    return (lo, hi) if hi > lo else None


def scattering_covariance_per_density(path, voxel: BoxVoxel, z_mm,
                                      momentum_gev: float = 4.0) -> np.ndarray:
    """Fermi-Eyges derivative D=dC/dlambda for one voxel and one nominal path.

    `path=(x0_mm,y0_mm,dx/dz,dy/dz,z_start_mm,z_end_mm)` uses the line
    x(z)=x0+z*dx/dz, y(z)=y0+z*dy/dz. For downstream residual planes j,k,
    D_jk=(13.6 MeV/(p beta c))^2 integral_(path intersect voxel)
    (z_j-z)(z_k-z) ds. With p in GeV/c and beta=1, the coefficient is
    (0.0136/p)^2 rad^2; D has mm^3, so lambda [mm^-1]*D [mm^3] is mm^2.
    Each transverse projection has the same 2x2 covariance, and x-y blocks are
    zero under isotropic independent kicks. Cross-plane terms are retained.
    This straight-nominal-path approximation omits path bending, energy loss,
    momentum spread, large-angle tails, absorption and non-Gaussian scattering.
    """
    z = _plane_z(z_mm)
    if len(path) != 6:
        raise ForwardModelError("path must be (x0,y0,dx_dz,dy_dz,z_start,z_end)")
    x0, y0, sx, sy, z0, z1 = map(float, path)
    p = float(momentum_gev)
    if not all(math.isfinite(v) for v in (x0, y0, sx, sy, z0, z1, p)) or z1 <= z0 or p <= 0:
        raise ForwardModelError("path finite; z_end>z_start and momentum>0 are required")
    lo, hi = max(z0, voxel.z_min_mm), min(z1, voxel.z_max_mm)
    if hi <= lo:
        return np.zeros((4, 4))
    for origin, slope, low, high in ((x0, sx, voxel.x_min_mm, voxel.x_max_mm),
                                    (y0, sy, voxel.y_min_mm, voxel.y_max_mm)):
        interval = _clip_linear_interval(origin, slope, low, high, lo, hi)
        if interval is None:
            return np.zeros((4, 4))
        lo, hi = interval
    kappa_sq = (0.0136 / p) ** 2  # beta approximated as 1.
    ds_dz = math.sqrt(1 + sx*sx + sy*sy)
    projected = np.zeros((2, 2))
    for a, j in enumerate((2, 3)):
        for b, k in enumerate((2, 3)):
            def primitive(q):
                return z[j]*z[k]*q - 0.5*(z[j]+z[k])*q*q + q*q*q/3
            projected[a, b] = kappa_sq * ds_dz * (primitive(hi)-primitive(lo))
    projected = 0.5 * (projected + projected.T)
    result = np.zeros((4, 4))
    result[:2, :2] = projected
    result[2:, 2:] = projected
    return result


def gaussian_nll_and_gradient(residuals, detector_covariances, kernels,
                              density_mm_inv) -> tuple[float, np.ndarray]:
    """Gaussian NLL (constant omitted) and analytic gradient for lambda>=0.

    Shapes: y=(n,4), Sigma_det=(n,4,4), D=(n,4,4,p), lambda=(p,).
    Every event covariance must be SPD; no hidden ridge is added. For event i,
    NLL=1/2(log|C_i|+y_i.T C_i^-1 y_i) and gradient_v=1/2[tr(C^-1D_v)
    - y.T C^-1 D_v C^-1 y].
    """
    y = np.asarray(residuals, dtype=float)
    detector = np.asarray(detector_covariances, dtype=float)
    d = np.asarray(kernels, dtype=float)
    density = np.asarray(density_mm_inv, dtype=float)
    if y.ndim != 2 or y.shape[1] != 4:
        raise ForwardModelError("residuals must have shape (n,4)")
    n = len(y)
    if n == 0:
        raise ForwardModelError("at least one residual event is required")
    if detector.shape != (n, 4, 4) or d.ndim != 4 or d.shape[:3] != (n, 4, 4):
        raise ForwardModelError("covariance/kernel shapes do not match residuals")
    p = d.shape[-1]
    if density.shape != (p,) or np.any(density < 0):
        raise ForwardModelError("density must be nonnegative with one value per voxel")
    if not all(np.all(np.isfinite(a)) for a in (y, detector, d, density)):
        raise ForwardModelError("likelihood inputs must be finite")
    nll, gradient = 0.0, np.zeros(p)
    for i in range(n):
        c = detector[i] + np.einsum("abv,v->ab", d[i], density)
        c = 0.5*(c+c.T)
        try:
            chol = np.linalg.cholesky(c)
        except np.linalg.LinAlgError as exc:
            raise ForwardModelError("event covariance must be SPD; no ridge is added") from exc
        logdet = 2*np.log(np.diag(chol)).sum()
        cy = np.linalg.solve(chol.T, np.linalg.solve(chol, y[i]))
        inv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(4)))
        nll += 0.5*(logdet+y[i]@cy)
        for v in range(p):
            dv = 0.5*(d[i,:,:,v]+d[i,:,:,v].T)
            gradient[v] += 0.5*(np.trace(inv@dv)-cy@dv@cy)
    return float(nll), gradient
