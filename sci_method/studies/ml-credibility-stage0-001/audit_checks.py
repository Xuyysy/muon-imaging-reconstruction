"""只读核验旧FE模块的确定性代数反例；不是新重建算法或训练入口。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sci_method/src'))
from sci_method.forward_model import (BoxVoxel, detector_residual_covariance,
    scattering_covariance_per_density, gaussian_nll_and_gradient)
from sci_method.identifiability import fisher_information, contrast_information

Z = np.array([-400., -300., 300., 400.])
LAYERS = [(-100,-50), (-50,0), (0,50), (50,100)]
V = np.array([-1.,3.,-3.,1.])
LAMBDA_REF = 1e-4


def exact_moments():
    """单位依次mm、mm²、mm³；Fraction避免原函数积分公式的浮点依赖。"""
    return [[Q(b**(r+1)-a**(r+1), r+1) for a,b in LAYERS] for r in range(3)]


def exact_kernels():
    """按独立矩展开计算D[mm³]；输出float仅用于与既有实现比较。"""
    m = exact_moments()
    factor = (Q(136,10000)/4)**2
    out = np.zeros((4,4,4))
    for l in range(4):
        for j,zj in enumerate((300,400)):
            for k,zk in enumerate((300,400)):
                integral = zj*zk*m[0][l]-(zj+zk)*m[1][l]+m[2][l]
                out[l,j,k] = out[l,j+2,k+2] = float(factor*integral)
    return out


def bound(matrix):
    return 256*np.finfo(float).eps*max(1., float(np.linalg.norm(matrix, np.inf)))


def path_kernels(slope=(0.,0.), half_xy=100.):
    path = (0.,0.,*slope,-100.,100.)
    return np.array([scattering_covariance_per_density(
        path, BoxVoxel(-half_xy,half_xy,-half_xy,half_xy,a,b), Z)
        for a,b in LAYERS])


def path_report(slope, half_xy):
    kernels = path_kernels(slope,half_xy)
    derivatives = LAMBDA_REF*kernels
    det = detector_residual_covariance(Z,.3)
    u0,u1 = np.ones(4),np.ones(4)+.1*V
    material0 = np.einsum('i,iab->ab',u0,derivatives)
    c0 = det+material0
    c1 = det+np.einsum('i,iab->ab',u1,derivatives)
    # 4个明确指定的数值向量只检查NLL恒等式，不从任何分布抽样。
    y = np.array([[0.,0.,0.,0.],[1.,0.,0.,1.],[2.,-1.,1.,-2.],[-.5,.2,1.,-3.]])
    d = np.repeat(kernels.transpose(1,2,0)[None],len(y),axis=0)
    detector = np.repeat(det[None],len(y),axis=0)
    # 保持旧API单位契约：kernel mm³乘λ[mm^-1]，而非把u冒充λ。
    n0,_ = gaussian_nll_and_gradient(y,detector,d,LAMBDA_REF*u0)
    n1,_ = gaussian_nll_and_gradient(y,detector,d,LAMBDA_REF*u1)
    f = fisher_information(c0[None],derivatives.transpose(1,2,0)[None])
    expected = exact_kernels()*math.sqrt(1+slope[0]**2+slope[1]**2)
    spectral = np.linalg.eigvalsh(f)
    return {
        'kernels_per_density_mm3': kernels.tolist(),
        'covariance_u0_mm2': c0.tolist(), 'covariance_u1_mm2': c1.tolist(),
        'kernel_matches_exact': bool(np.max(np.abs(kernels-expected)) <= bound(expected.reshape(4,-1))),
        'covariance_equal': bool(np.max(np.abs(c1-c0)) <= bound(c0)),
        'covariance_abs_difference_mm2': float(np.max(np.abs(c1-c0))),
        'covariance_rounding_budget_mm2': bound(c0),
        'relative_covariance_change': float(np.linalg.norm(c1-c0,np.inf)/np.linalg.norm(c0,np.inf)),
        'initial_clipping_threshold': .001,
        'initial_clipping_threshold_met': bool(np.linalg.norm(c1-c0,np.inf)/np.linalg.norm(c0,np.inf)>.001),
        'relative_material_covariance_change': float(np.linalg.norm(c1-c0,np.inf)/np.linalg.norm(material0,np.inf)),
        'nll_u0':n0,'nll_u1':n1,'nll_abs_difference':abs(n1-n0),
        'covariance_eigenvalues_mm2':np.linalg.eigvalsh(c0).tolist(),
        'fisher':f.tolist(),'fisher_eigenvalues':spectral.tolist(),
        'fisher_rank':int(np.count_nonzero(spectral>max(spectral)*1e-10)),
        'fisher_null_relative_residual':float(np.linalg.norm(f@V)/max(np.linalg.norm(f)*np.linalg.norm(V),1e-300)),
        'depth_contrast_estimable':contrast_information(f,V/20)['estimable'],
        'total_density_estimable':contrast_information(f,np.ones(4))['estimable'],
    }


def zero_failure_upper(n, confidence=.95):
    """0/n independent Bernoulli failures的单侧Clopper–Pearson上界。"""
    if n < 1 or not 0 < confidence < 1:
        raise ValueError('invalid Bernoulli bound inputs')
    return 1-(1-confidence)**(1/n)


def binomial_mass(n,k,p):
    return math.comb(n,k)*p**k*(1-p)**(n-k)


def paired_power(n, discordance, difference, alpha=.05):
    """有限和精确McNemar规划功效，不生成数据；差为P(ML-only)-P(MAP-only)。"""
    q,delta = discordance,difference
    if n<1 or not 0<q<=1 or abs(delta)>q or not 0<alpha<1:
        raise ValueError('invalid paired Bernoulli probabilities')
    positive = (q+delta)/(2*q)
    power = 0.
    for k in range(1,n+1):
        outer = binomial_mass(n,k,q)
        if outer == 0:
            continue
        half_cdf = np.cumsum([math.comb(k,x)/2**k for x in range(k+1)])
        reject = [x for x in range(k+1) if min(1.,2*half_cdf[min(x,k-x)])<=alpha]
        power += outer*math.fsum(binomial_mass(k,x,positive) for x in reject)
    return power


def analyze():
    vertical = path_report((0.,0.),100.)
    d = LAMBDA_REF*path_kernels()
    ds = np.concatenate([d,d.sum(axis=0,keepdims=True)],axis=0)
    det = detector_residual_covariance(Z,.3)
    obj_c = det+1.5*d.sum(axis=0)
    joint = fisher_information(obj_c[None],ds.transpose(1,2,0)[None])
    # 独立空场测同一背景b；4个目标参数对空场导数=0。
    cal_d = np.zeros_like(ds); cal_d[4] = d.sum(axis=0)
    cal_c = det+.5*d.sum(axis=0)
    cal_f = fisher_information(cal_c[None],cal_d.transpose(1,2,0)[None])
    total = np.r_[np.ones(4),0.]; depth = np.r_[V/20,0.]
    f = np.array(vertical['fisher'])
    return {
        'u0':np.ones(4).tolist(),'u1':(np.ones(4)+.1*V).tolist(),'null_direction':V.tolist(),
        'exact_moments':[[str(x) for x in row] for row in exact_moments()],
        'depth_contrast_u0':float((V/20)@np.ones(4)),
        'depth_contrast_u1':float((V/20)@(np.ones(4)+.1*V)),
        'detector_covariance_mm2':det.tolist(),'vertical':vertical,
        'tilted_full':path_report((.1,.05),100.),
        'tilted_clipped':path_report((.1,.05),5.),
        'nuisance':{
            'object_joint_fisher':joint.tolist(),'calibration_fisher':cal_f.tolist(),
            'unknown_common_total_estimable':contrast_information(joint,total)['estimable'],
            'calibrated_common_total_estimable':contrast_information(joint+cal_f,total)['estimable'],
            'calibrated_depth_contrast_estimable':contrast_information(joint+cal_f,depth)['estimable'],
            'unknown_common_total_estimable_after_duplicating_object':contrast_information(2*joint,total)['estimable']},
        'prior':{'data_curvature_null_direction':float(V@f@V),
                 'added_prior_curvature':float(V@np.eye(4)@V)},
        'planning':{
            'zero_failures_upper95_n64':zero_failure_upper(64),
            'zero_failures_upper95_n128':zero_failure_upper(128),
            'probability_zero_at_risk05_n64':.95**64,
            'conformal_n64_order':math.ceil(65*.95),
            'conformal_continuous_marginal_miscoverage':3/65,
            'unadjusted_min_of_20_uniform_pvalues_fwer':1-.95**20,
            'paired_power_n128_difference05':{str(q):paired_power(128,q,.05) for q in (.05,.1,.25,.5)}}}


def run(output):
    """新目录独占；先保存来源/状态；异常也留下证据，不运行任何研究入口。"""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    def dump(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    files = [Path(__file__),Path(__file__).with_name('test_audit_checks.py'),
             Path(__file__).with_name('CHECK_SPEC.md'),
             ROOT/'sci_method/src/sci_method/forward_model.py',
             ROOT/'sci_method/src/sci_method/identifiability.py']
    dump('run_state.json',{'status':'running','scope':'deterministic stage0; no observations sampled'})
    try:
        dump('config.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
             'specification_date':'2026-10-25','z_mm':Z.tolist(),'layers_mm':LAYERS,
             'lambda_ref_mm_inv':LAMBDA_REF,'sigma_hit_mm':.3,'p_gev_c':4.,
             'random_seed':None,'randomness':'none; exact algebra and fixed matrices only',
             'rounding_budget':'256*eps*max(1,norm_inf(reference))',
             'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'git_status_before_snapshot':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
             'command':sys.argv,'python':sys.version,'numpy':np.__version__,'platform':platform.platform()})
        input_paths = [ROOT/'sci_method/studies/ml-direction-feasibility-001'/name
                       for name in ('README.md','LITERATURE_REVIEW.md','RESEARCH_DECISION.md',
                                    'MINIMAL_POC_PLAN.md','ASSETS_AND_DATA.md')]
        dump('input_sources.json',{'role':'prior review documents, read-only; no sampled data',
             'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in input_paths}})
        snapshot = output/'source_snapshot'; snapshot.mkdir()
        for p in files:
            shutil.copy2(p,snapshot/p.name)
        report = analyze()
        dump('checks.json',report)
        # 仅认证下列显式恒等式；不把两个等式误当全部独立核检查通过。
        clipped = report['tilted_clipped']
        checks = {'exact_moment_null':all(sum(x*v for x,v in zip(row,(-1,3,-3,1)))==0
                                         for row in exact_moments()),
                  **{f'{path}_{key}':report[path][key]
                     for path in ('vertical','tilted_full')
                     for key in ('kernel_matches_exact','covariance_equal')},
                  'clipping_status_consistent':clipped['initial_clipping_threshold_met']==(
                      clipped['relative_covariance_change']>clipped['initial_clipping_threshold'])}
        dump('certification_checks.json',checks)
        if not all(checks.values()):
            raise AssertionError('declared mathematical checks failed: '+str(
                [name for name,passed in checks.items() if not passed]))
        failed = ([] if clipped['initial_clipping_threshold_met'] else [
            f"clipped total covariance change > {clipped['initial_clipping_threshold']}: NOT MET"])
        dump('run_state.json',{'status':'completed_with_failed_effect_expectation' if failed else 'completed',
             'numerical_identities':'passed',
             'certified_checks':list(checks),
             'failed_diagnostics':failed,
             'scope':'deterministic numerical checks, not performance validation'})
    except BaseException as exc:
        dump('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        dump('run_state.json',{'status':'failed'})
        raise
    finally:
        hashes = {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(output.rglob('*')) if p.is_file() and p.name!='evidence.json'}
        dump('evidence.json',{'scope':'deterministic audit only','sha256':hashes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
