"""四子格风险收缩的最小统计检查：不读Geant4数据或目标掩膜。

重建只接收四子格sum(w)、sum(w²)，真值仅在独立评分函数中使用。
待检验公式按上一轮方案原样实现；本轮不根据得分修补公式。
"""
from __future__ import annotations
import numpy as np

METHODS = ('coarse', 'raw', 'neff', 'candidate')


def reconstruct(child_weight, child_squared_weight):
    """返回每粗格四子格概率与诊断，最后一维固定为4。

    variance_proxy = Σ_i w_i² ||e_i-p_hat||² / W²。
    按子格合并后，残差为1-2*p_j+Σ_k p_k²，仍是逐事件平方
    权重，不是把子格sum(w)直接平方。这个代入估计并不无偏。
    """
    weight = np.asarray(child_weight, dtype=float)
    squared = np.asarray(child_squared_weight, dtype=float)
    if (weight.shape != squared.shape or weight.ndim < 1 or weight.shape[-1] != 4
            or not np.isfinite(weight).all() or not np.isfinite(squared).all()
            or (weight < 0).any() or (squared < 0).any()
            or ((weight == 0) != (squared == 0)).any()):
        raise ValueError('Invalid four-child event-weight moments')
    # 每行共同缩放，不改变概率/有效数/风险代理，避免大权重的平方溢出。
    scale = weight.max(axis=-1, keepdims=True)
    scaled = np.divide(weight, scale, out=np.zeros_like(weight), where=scale > 0)
    scaled_squared = np.divide(squared, scale, out=np.zeros_like(squared), where=scale > 0)
    scaled_squared = np.divide(scaled_squared, scale, out=np.zeros_like(squared), where=scale > 0)
    total = scaled.sum(axis=-1)
    square_total = scaled_squared.sum(axis=-1)
    uniform = np.full_like(weight, .25)
    probability = np.divide(scaled, total[..., None], out=uniform.copy(), where=total[..., None] > 0)
    effective = np.divide(total**2, square_total, out=np.zeros_like(total), where=square_total > 0)
    contrast = np.square(probability-uniform).sum(axis=-1)
    residual_squared = np.maximum(0., 1.-2.*probability+np.square(probability).sum(axis=-1, keepdims=True))
    variance = np.divide((scaled_squared*residual_squared).sum(axis=-1), total**2,
                         out=np.zeros_like(total), where=total > 0)
    ratio = np.divide(variance, contrast, out=np.ones_like(contrast), where=contrast > 0)
    alpha = np.where(total > 0, np.clip(1.-ratio, 0., 1.), 0.)
    neff_alpha = effective/(effective+4.)
    images = {'coarse':uniform, 'raw':probability,
              'neff':uniform+neff_alpha[..., None]*(probability-uniform),
              'candidate':uniform+alpha[..., None]*(probability-uniform)}
    for image in images.values():
        if not np.isfinite(image).all() or (image < -1e-14).any() or not np.allclose(image.sum(axis=-1), 1., atol=1e-14):
            raise AssertionError('Probability reconstruction not normalized/nonnegative')
    return images, {'candidate_alpha':alpha, 'neff_alpha':neff_alpha,
                    'effective_count':effective, 'contrast':contrast, 'variance_proxy':variance}


def weighted_population_target(event_probability, conditional_mean_weight):
    """预测的是加权质量比例q，不是未加权事件概率p。

    q_j=p_j E[w|j]/Σ_k p_k E[w|k]。此函数仅给合成实验评分真值；
    它不会传入reconstruct，因此不能帮助候选选alpha。
    """
    p = np.asarray(event_probability, dtype=float)
    mean = np.asarray(conditional_mean_weight, dtype=float)
    if (p.shape != (4,) or mean.shape != (4,) or not np.isfinite(p).all()
            or not np.isfinite(mean).all() or (p < 0).any() or (mean <= 0).any()
            or not np.isclose(p.sum(), 1.)):
        raise ValueError('Invalid population specification')
    weighted = p*mean
    return weighted/weighted.sum()


def squared_risk(image, truth):
    """每次独立模拟的四维平方误差，不是图像SSIM或材料密度误差。"""
    return np.square(np.asarray(image)-np.asarray(truth)).sum(axis=-1)
