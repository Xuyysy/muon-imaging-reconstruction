"""固定总预算的两粗格四子格借用；只使用观测计数。"""
import numpy as np

PRIOR=4.  # 四子格均匀先验，沿用旧温和收缩，不从真值调参。


def _uncertainty(counts, total):
    """经验比例的总方差；n=1用均匀先验0.75，非数据推得。"""
    frequency=counts/total
    if total==1:
        return np.full(len(counts),.75)
    return (1.-np.square(frequency).sum(axis=1))/(total-1.)


def reconstruct_pair(count_a, count_b):
    """两个形状相同的批量计数数组(batch,4)，返回各方法的A/B概率。

    每格物理总权重独立保持n_A或n_B；借用只作用于比例估计，
    不将邻居事件复制成新的物理散射权重。全共享估计本身
    可能跨边界混合，故输出必须分A/B评价。
    """
    a,b=np.asarray(count_a,dtype=float),np.asarray(count_b,dtype=float)
    if (a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=4
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or (a<0).any() or (b<0).any()):
        raise ValueError('Invalid paired four-child counts')
    total_a=a.sum(axis=1)
    total_b=b.sum(axis=1)
    if ((total_a<=0).any() or (total_b<=0).any()
            or not np.allclose(total_a,total_a[0]) or not np.allclose(total_b,total_b[0])):
        raise ValueError('Each batch must have fixed positive counts')
    na,nb=float(total_a[0]),float(total_b[0])
    u=np.full_like(a,.25)
    independent_a=(PRIOR*u+a)/(PRIOR+na)
    independent_b=(PRIOR*u+b)/(PRIOR+nb)
    pooled=(PRIOR*u+a+b)/(PRIOR+na+nb)
    # 邻域伪计数最多4，方向取经验概率；不改变可用物理事件预算。
    borrowed_a=min(nb,PRIOR)
    borrowed_b=min(na,PRIOR)
    frequency_a=a/na
    frequency_b=b/nb
    discrepancy=np.square(frequency_a-frequency_b).sum(axis=1)
    noise=_uncertainty(a,na)+_uncertainty(b,nb)
    gate=np.divide(noise,discrepancy,out=np.ones_like(noise),where=discrepancy>0)
    gate=np.clip(gate,0.,1.)

    def blend(own,own_n,other_frequency,borrowed):
        borrowed=np.asarray(borrowed)
        return (PRIOR*u+own+borrowed[:,None]*other_frequency)/(PRIOR+own_n+borrowed[:,None])

    limited_a=blend(a,na,frequency_b,np.full(len(a),borrowed_a))
    limited_b=blend(b,nb,frequency_a,np.full(len(a),borrowed_b))
    gated_a=blend(a,na,frequency_b,borrowed_a*gate)
    gated_b=blend(b,nb,frequency_a,borrowed_b*gate)
    outputs={'independent':(independent_a,independent_b),
             'pooled':(pooled,pooled.copy()),'limited':(limited_a,limited_b),
             'gated':(gated_a,gated_b)}
    for pair in outputs.values():
        for value in pair:
            if not np.isfinite(value).all() or (value<0).any() or not np.allclose(value.sum(axis=1),1.):
                raise AssertionError('Non-conserved/negative four-child probability')
    return outputs,{'gate':gate,'borrowed_a':borrowed_a*gate,
                    'borrowed_b':borrowed_b*gate,'discrepancy':discrepancy,'noise':noise,
                    'counts_a':na,'counts_b':nb}
