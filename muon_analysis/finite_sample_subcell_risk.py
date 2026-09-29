"""原插件方差的条件小样本修正与先验上限，保持旧核心不变。

无偏性仅限于给定权重后，子格类别独立且概率相同的模型。
权重--类别相关压力条件不享有该保证；收缩风险也不因此无偏。
"""
import numpy as np
from synthetic_subcell_risk import reconstruct


def repaired_reconstruct(child_weight, child_squared_weight, child_cubed_weight):
    """输入均为逐事件权重矩在四子格的合计，不读取真值。

    事件级三阶矩不可用(sum w)^3替代。校正用s2/(s2-2s3+s2²)，
    不用n_eff/(n_eff-1)冒充任意变权下的精确因子。
    """
    old, diagnostic = reconstruct(child_weight, child_squared_weight)
    w=np.asarray(child_weight,dtype=float)
    w2=np.asarray(child_squared_weight,dtype=float)
    w3=np.asarray(child_cubed_weight,dtype=float)
    if (w3.shape != w.shape or not np.isfinite(w3).all() or (w3 < 0).any()
            or ((w3 == 0) != (w == 0)).any()):
        raise ValueError('Invalid event-wise third weight moment')
    scale=w.max(axis=-1,keepdims=True)
    normalized=[]
    for moment,power in ((w,1),(w2,2),(w3,3)):
        value=moment.copy()
        for _ in range(power):
            value=np.divide(value,scale,out=np.zeros_like(value),where=scale>0)
        normalized.append(value)
    total=normalized[0].sum(axis=-1)
    s2=np.divide(normalized[1].sum(axis=-1),total**2,out=np.zeros_like(total),where=total>0)
    s3=np.divide(normalized[2].sum(axis=-1),total**3,out=np.zeros_like(total),where=total>0)
    # 先除s2再算相对分母，避免极小s2乘积造成多余下溢。
    relative=1.-2.*np.divide(s3,s2,out=np.zeros_like(s2),where=s2>0)+s2
    valid=(total>0)&(s2<1.-1e-12)&(relative>1e-12)
    factor=np.divide(1.,relative,out=np.zeros_like(relative),where=valid)
    variance=diagnostic['variance_proxy']*factor
    contrast=diagnostic['contrast']
    ratio=np.divide(variance,contrast,out=np.ones_like(contrast),where=contrast>0)
    corrected_alpha=np.where(valid,np.clip(1.-ratio,0.,1.),diagnostic['neff_alpha'])
    guarded_alpha=np.minimum(corrected_alpha,diagnostic['neff_alpha'])
    uniform=old['coarse']
    images={**old,'corrected':uniform+corrected_alpha[...,None]*(old['raw']-uniform),
            'guarded':uniform+guarded_alpha[...,None]*(old['raw']-uniform)}
    for image in images.values():
        if not np.isfinite(image).all() or (image < -1e-14).any() or not np.allclose(image.sum(axis=-1),1.,atol=1e-14):
            raise AssertionError('Repaired probability invalid')
    return images,{**diagnostic,'variance_corrected':variance,'correction_factor':factor,
                   'correction_valid':valid,'corrected_alpha':corrected_alpha,
                   'guarded_alpha':guarded_alpha,'s2':s2,'s3':s3}
