"""固定物理窗的支撑与质量守恒局部有效尺度候选，不读目标真值。

这是二维原生网格上的观测权重再分配，不改变三维体素树。
所有常数运行前冻结；普通支撑调节/双边思想不自动构成论文创新。
"""
from __future__ import annotations
import numpy as np
from run_adaptive_voxel_experiment import imaging_edges, XY_LIMIT_MM
from run_frame_z_gate_validation import SMOOTH_SIGMA_MM

SUPPORT_HALF_WIDTH_MM = 16.875
SUPPORT_SCALE = 4.
LOG_CONTRAST_SCALE = float(np.log(2.))
GUIDE_FLOOR_RATIO = 1e-3


def physical_support(xy, weights, query):
    """直接查询事件的同物理方窗，不先按16/32格做直方图。

    相同query输入必得相同支撑。Σw²是事件权重平方之和，
    不能误用已经求和的像素质量平方。分块降低临时数组峰值。
    """
    xy, weights, query = np.asarray(xy, float), np.asarray(weights, float), np.asarray(query, float)
    if (xy.shape != (len(weights), 2) or query.ndim != 2 or query.shape[1] != 2
            or not np.isfinite(xy).all() or not np.isfinite(weights).all()
            or not np.isfinite(query).all() or (weights < 0).any()
            or (np.abs(xy) > XY_LIMIT_MM).any() or (np.abs(query) > XY_LIMIT_MM).any()):
        raise ValueError('Invalid physical support input')
    active = weights > 0
    xy, weights = xy[active], weights[active]
    mass, squared = np.zeros(len(query)), np.zeros(len(query))
    for start in range(0, len(query), 128):
        inside = (np.abs(query[start:start+128, None, :]-xy[None, :, :])
                  <= SUPPORT_HALF_WIDTH_MM).all(axis=2)
        mass[start:start+128] = np.sum(inside*weights[None, :], axis=1)
        squared[start:start+128] = np.sum(inside*np.square(weights)[None, :], axis=1)
    neff = np.divide(mass**2, squared, out=np.zeros_like(mass), where=squared > 0)
    # 域边缘方窗截断后，用实际窗口面积估计同单位观测引导密度。
    lower = np.maximum(query-SUPPORT_HALF_WIDTH_MM, -XY_LIMIT_MM)
    upper = np.minimum(query+SUPPORT_HALF_WIDTH_MM, XY_LIMIT_MM)
    area = np.prod(upper-lower, axis=1)
    guide = mass/area
    return neff, guide


def native_support(xy, weights, bins):
    if bins not in (16, 32):
        raise ValueError('Unsupported native grid')
    edges = imaging_edges(bins)[:2]
    centers = [(a[:-1]+a[1:])/2 for a in edges]
    x, y = np.meshgrid(*centers, indexing='ij')
    neff, guide = physical_support(xy, weights, np.column_stack((x.ravel(), y.ravel())))
    return neff.reshape(bins, bins), guide.reshape(bins, bins)


def local_scale_image(image, bins, neff, guide, support=True, edge=True):
    """原生源像素质量在有限域内重分配；两因子可取消作消融。

    alpha控制平滑与原位细节的混合，并不是宣称可靠的最佳尺度。
    边缘核按观测guide差异限制跨域分配，并非知道真实材料轮廓。
    对每源列归一化，因此局部强度变化不通过丢弃质量实现。
    """
    if (bins not in (16, 32) or image.shape != (32, 32)
            or neff.shape != (bins, bins) or guide.shape != neff.shape
            or not all(np.isfinite(a).all() for a in (image, neff, guide))
            or any((a < 0).any() for a in (image, neff, guide))):
        raise ValueError('Invalid local-scale image/support')
    native = image if bins == 32 else image.reshape(16, 2, 16, 2).sum(axis=(1, 3))
    alpha = SUPPORT_SCALE/(neff+SUPPORT_SCALE) if support else np.ones_like(neff)
    positive = guide[guide > 0]
    floor = float(positive.mean()*GUIDE_FLOOR_RATIO) if len(positive) else 1.
    log_guide = np.log(guide+floor)
    width = float(np.diff(imaging_edges(bins)[0])[0])
    sigma_pixel = SMOOTH_SIGMA_MM/width
    radius = int(np.ceil(3*sigma_pixel))
    # 索引方向：src区域向dst区域分配；切片不环绕。
    transfers, denominator = [], np.zeros_like(native)
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            sx = slice(max(0, -dx), min(bins, bins-dx))
            sy = slice(max(0, -dy), min(bins, bins-dy))
            tx = slice(max(0, dx), min(bins, bins+dx))
            ty = slice(max(0, dy), min(bins, bins+dy))
            spatial = float(np.exp(-.5*(dx*dx+dy*dy)/sigma_pixel**2))
            kernel = np.full(native[sx, sy].shape, spatial)
            if edge:
                contrast = (log_guide[tx, ty]-log_guide[sx, sy])/LOG_CONTRAST_SCALE
                kernel *= np.exp(-.5*contrast**2)
            denominator[sx, sy] += kernel
            transfers.append((sx, sy, tx, ty, kernel, (dx*dx+dy*dy)*width**2))
    output = native*(1-alpha)
    displacement_weight = 0.
    for sx, sy, tx, ty, kernel, squared_distance_mm2 in transfers:
        allocated = native[sx, sy]*alpha[sx, sy]*kernel/denominator[sx, sy]
        output[tx, ty] += allocated
        displacement_weight += float(allocated.sum()*squared_distance_mm2)
    if bins == 16:
        output = np.repeat(np.repeat(output/4, 2, axis=0), 2, axis=1)
    if (not np.isfinite(output).all() or (output < 0).any()
            or not np.isclose(output.sum(), image.sum(), rtol=1e-12, atol=1e-9)):
        raise AssertionError('Local-scale weight conservation/positivity failure')
    return output, {'alpha_mass_mean': float((alpha*native).sum()/native.sum()) if native.sum() else 0.,
                    'physical_neff_mass_mean': float((neff*native).sum()/native.sum()) if native.sum() else 0.,
                    'expected_squared_displacement_mm2': displacement_weight/float(native.sum()) if native.sum() else 0.}


def matched_isotropic_image(base, fixed, target_moment_mm2, fixed_moment_mm2):
    """H/S凸组合匹配源权重加权的一个二阶位移矩，不做真值优化。

    超出可行范围明确拒绝，不能静默夹断后假称严格匹配。输入矩
    属于核分配，不是输出图像的空间方差。极小超限只作浮点处理。
    """
    if (base.shape != fixed.shape or not np.isfinite(base).all() or not np.isfinite(fixed).all()
            or (base < 0).any() or (fixed < 0).any()
            or not np.isfinite(target_moment_mm2) or not np.isfinite(fixed_moment_mm2)
            or target_moment_mm2 < 0 or fixed_moment_mm2 < 0
            or not np.isclose(base.sum(), fixed.sum(), rtol=1e-12, atol=1e-9)):
        raise ValueError('Invalid matched-diffusion images/moments')
    if fixed_moment_mm2 == 0:
        if target_moment_mm2 != 0:
            raise ValueError('Positive displacement cannot match zero fixed displacement')
        beta = 0.
    else:
        beta = target_moment_mm2/fixed_moment_mm2
    if beta > 1+1e-12:
        raise ValueError('Target displacement outside H/S convex-control range')
    beta = min(beta, 1.)
    image = (1-beta)*base+beta*fixed
    matched_moment = beta*fixed_moment_mm2
    if not np.isclose(matched_moment, target_moment_mm2, atol=1e-9, rtol=1e-12):
        raise AssertionError('Diffusion-moment match failed')
    return image, {'alpha_mass_mean': beta, 'physical_neff_mass_mean': np.nan,
                   'expected_squared_displacement_mm2': matched_moment,
                   'match_beta': beta, 'moment_match_error_mm2': matched_moment-target_moment_mm2}
