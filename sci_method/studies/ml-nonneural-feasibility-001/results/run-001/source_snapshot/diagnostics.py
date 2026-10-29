"""Bounded study-local Gaussian diagnostics. Units: residual mm, density mm^-1.

Not a production reconstructor. Truth is accepted only by evaluation/oracle
helpers. Shared research modules are imported read-only; no neural training.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
from numpy.polynomial import Polynomial as Poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'sci_method/src'))
from sci_method.forward_model import (BoxVoxel, detector_residual_covariance,
                                      scattering_covariance_per_density)
from sci_method.covariance_optimizer import (fit_covariance_detailed,
                                             CovarianceObjective)
from sci_method.split_merge import (GaussianObservationGroup,
                                    grouped_gaussian_nll_and_gradient)

COUNTS = np.array([4,8,12,16,20,24,32,40])
SCENES = [
    ('empty', [0]*8, .3),
    ('uniform_boundary', [1]*8, .3),
    ('two_enhancements', [0,0,0,3,0,0,0,4], .3),
    ('weak_enhancements', [1,1,1.4,1,1,1.4,1,1], .3),
    ('coverage_challenge', [4,0,0,0,1.5,0,0,0], .3),
    ('high_hit_error', [0,0,0,3,0,0,0,4], 1.0),
]


def chi2_sf_even(q, df):
    """Exact integer incomplete-gamma finite sum, evaluated stably in float64."""
    if int(df) != df or df < 2 or df % 2:
        raise ValueError('positive even degrees of freedom required')
    if not math.isfinite(q):
        raise ValueError('finite statistic required')
    if q <= 0: return 1.0
    x=q/2
    terms=np.arange(df//2)*math.log(x)-np.array([math.lgamma(k+1) for k in range(df//2)])
    m=float(terms.max())
    return min(1.0, math.exp(-x+m)*float(np.exp(terms-m).sum()))


@lru_cache(None)
def chi2_isf_even(alpha,df):
    if not 0 < alpha < 1: raise ValueError('alpha in (0,1) required')
    lo,hi=0.,float(df+20*math.sqrt(2*df)+100)
    while chi2_sf_even(hi,df)>alpha: hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if chi2_sf_even(mid,df)>alpha: lo=mid
        else: hi=mid
    return (lo+hi)/2


@lru_cache(None)
def beta_quadrature(n,nodes):
    x,w=np.polynomial.legendre.leggauss(nodes)
    x=(x+1)/2; w=w/2
    lw=(n-1)*(np.log(x)+np.log1p(-x))
    weights=w*np.exp(lw-float(lw.max()))
    return x,weights/weights.sum()


def pair_tail(q,n,weights,nodes=128):
    """P(b0*chi²(2n)+b1*chi²(2n)>q), using Beta/Gamma independence.

    Gauss-Legendre integration is numerical, not a new exact-statistics claim.
    Equal weights and zero weights are evaluated analytically.
    """
    b=np.asarray(weights,float)
    if b.shape!=(2,) or np.any(b<0) or not np.isfinite(b).all() or n<1:
        raise ValueError('two nonnegative weights and positive n required')
    if q<0: return 1.
    if not np.any(b): return 0.
    if b[0]==b[1]: return chi2_sf_even(q/b[0],4*n)
    if np.any(b==0): return chi2_sf_even(q/float(b.max()),2*n)
    x,w=beta_quadrature(n,nodes)
    scales=b[0]*x+b[1]*(1-x)
    # Incomplete gamma, shape 2n, finite positive sum; no cancellation.
    z=q/(2*scales)
    if q==0: return 1.
    logs=np.arange(2*n)[:,None]*np.log(z)-np.array(
        [math.lgamma(k+1) for k in range(2*n)])[:,None]
    m=logs.max(axis=0)
    sf=np.exp(-z+m)*np.exp(logs-m).sum(axis=0)
    return float(w@sf)


def checked_tail(q,n,weights):
    p=pair_tail(q,n,weights,128); p2=pair_tail(q,n,weights,256)
    if abs(p-p2)>1e-8:
        raise ArithmeticError(f'quadrature tolerance failed: {p} vs {p2}')
    return p2


@lru_cache(None)
def weighted_isf(alpha,n,b0,b1):
    lo,hi=0.,chi2_isf_even(alpha,4*n)*max(b0,b1)
    for _ in range(70):
        mid=(lo+hi)/2
        if pair_tail(mid,n,[b0,b1])>alpha: lo=mid
        else: hi=mid
    result=(lo+hi)/2
    if abs(checked_tail(result,n,[b0,b1])-alpha)>1e-8:
        raise ArithmeticError('weighted reference quantile certification failed')
    return result


def physical_modes(sigma):
    """Same known FE response in each disjoint lateral ROI; no outside response.

    lambda_ref=.01 mm^-1, u dimensionless. Positive sigma required for C0 SPD.
    Eigen a are dimensionless derivatives after C0 whitening; transform maps
    physical residual mm to whitened eigenmodes, each repeated in x/y.
    """
    if sigma<=0: raise ValueError('positive known hit sigma required')
    z=(-400,-300,300,400)
    c0=detector_residual_covariance(z,sigma)
    d=.01*scattering_covariance_per_density((0,0,0,0,-100,100),
            BoxVoxel(-1,1,-1,1,-100,100),z,4.)
    L=np.linalg.cholesky(c0[:2,:2])
    whiten=np.linalg.solve(L,np.eye(2))
    h=whiten@d[:2,:2]@whiten.T
    a,V=np.linalg.eigh((h+h.T)/2)
    transform=V.T@whiten
    return c0,d,a,transform


def Q_pvalue(energy,n,a):
    # Energy is the sum of squares of x/y modes: one value per depth mode.
    q=float(np.sum(np.asarray(energy)/(1+np.asarray(a))))
    return chi2_sf_even(q,4*int(n))


def lr_weights(a):
    # Fixed simple alternative ua=3, null boundary u*=1.
    a=np.asarray(a)
    return 1-(1+a)/(1+3*a)


def map_mode(energy,n,a):
    """Global one-dimensional ridge-MAP via all polynomial stationary roots.

    f=n sum log(1+u*a)+.5 sum T/(1+u*a)+.25*u²/2.
    This diagnostic is separable ROI MAP; it is not a full-grid TV-MAP API.
    """
    a=np.asarray(a,float); t=np.asarray(energy,float)
    if a.shape!=(2,) or t.shape!=(2,) or np.any(a<0) or np.any(t<0):
        raise ValueError('nonnegative two-mode inputs required')
    p=[Poly([1,float(x)]) for x in a]
    numerator=.25*Poly([0,1])*p[0]**2*p[1]**2
    for m in range(2):
        numerator+=(n*a[m]*p[m]-.5*t[m]*a[m])*p[1-m]**2
    candidates=[0.]
    for root in numerator.roots():
        if abs(root.imag)<=1e-7*max(1,abs(root.real)) and root.real>0:
            candidates.append(float(root.real))
    def f(u): return float(np.sum(n*np.log1p(u*a)+.5*t/(1+u*a))+.125*u*u)
    u=min(candidates,key=f)
    g=float(np.sum(n*a/(1+u*a)-.5*t*a/(1+u*a)**2)+.25*u)
    if abs(g if u>0 else min(g,0))>1e-6:
        raise ArithmeticError('MAP stationary root fails fixed KKT precision')
    return u,g


def top2(scores):
    return np.argsort(-np.asarray(scores),kind='stable')[:2].tolist()


def propose(energy,counts,a):
    """Observables only. No truth/mask/target identity can be provided."""
    n=np.asarray(counts); a=np.asarray(a)
    maps=[map_mode(t,int(k),a)[0] for t,k in zip(energy,n)]
    r=a/(1+a)
    info=n*np.sum(r*r)
    score=.5*np.sum((energy/(1+a)-2*n[:,None])*r,axis=1)/np.sqrt(info)
    return dict(fixed=[0,1],map=top2(maps),score=top2(score),fisher=top2(info))


def oracle_indices(truth,powers):
    """Evaluation-only expected-discovery upper bound. No realized audit data."""
    return top2(np.where(np.asarray(truth)>1,powers,0))


def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]


def candidate_study(out):
    energies=np.empty((6,128,2,8,2))
    seed_keys=[]; records=[]; summaries={}; matrices={}; proposal_seconds=0.
    for si,(name,truth,sigma) in enumerate(SCENES):
        truth=np.array(truth,float); true=truth>1
        c0,d,a,transform=physical_modes(sigma)
        matrices[name]=dict(base_mm2=c0.tolist(),derivative_mm2=d.tolist(),
                            mode_derivative=a.tolist(),transform=transform.tolist())
        n=COUNTS//2
        split_power=np.array([checked_tail(chi2_isf_even(.025,4*int(k)),int(k),
                           (1+u*a)/(1+a)) for k,u in zip(n,truth)])
        full_power=np.array([checked_tail(chi2_isf_even(.00625,4*int(k)),int(k),
                           (1+u*a)/(1+a)) for k,u in zip(COUNTS,truth)])
        oracle=oracle_indices(truth,split_power)
        full_oracle=oracle_indices(truth,full_power)
        for rep in range(128):
            for part in range(2):
                seed=[20261029,si,rep,part]; seed_keys.append(seed)
                rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
                energies[si,rep,part]=rng.chisquare(2*n[:,None],size=(8,2))*(1+truth[:,None]*a)
            ep,ea=energies[si,rep]
            start=time.process_time()
            choices=propose(ep,n,a)
            proposal_seconds+=time.process_time()-start
            choices['oracle']=oracle
            ps=np.array([Q_pvalue(t,k,a) for t,k in zip(ea,n)])
            pf=np.array([Q_pvalue(t,k,a) for t,k in zip(ep+ea,COUNTS)])
            for method,selected in choices.items():
                supported=[j for j in selected if ps[j]<=.025]
                records.append(dict(scene=name,replicate=rep,method='split_'+method,
                    selected=selected,supported=supported,
                    true_support=int(sum(true[j] for j in supported)),
                    false_support=int(sum(not true[j] for j in supported)),
                    fwer=bool(any(not true[j] for j in supported)),
                    missed=int(true.sum()-sum(true[j] for j in supported)),
                    conditional_expected=float(sum(split_power[j] for j in selected if true[j])),
                    oracle_gap=float(sum(split_power[j] for j in oracle if true[j])-
                                     sum(split_power[j] for j in selected if true[j]))))
            # Same-data selection is protected against the entire fixed family.
            full_selected=propose(ep+ea,COUNTS,a)['score']
            for method,selected in [('full_score_top2',full_selected),
                                    ('full_oracle',full_oracle),('full_all8',list(range(8)))]:
                supported=[j for j in selected if pf[j]<=.00625]
                records.append(dict(scene=name,replicate=rep,method=method,
                    selected=selected,supported=supported,
                    true_support=int(sum(true[j] for j in supported)),
                    false_support=int(sum(not true[j] for j in supported)),
                    fwer=bool(any(not true[j] for j in supported)),
                    missed=int(true.sum()-sum(true[j] for j in supported)),
                    conditional_expected=None,oracle_gap=None))
        rows=[r for r in records if r['scene']==name]
        sums={}
        for method in sorted({r['method'] for r in rows}):
            subset=[r for r in rows if r['method']==method]
            k=sum(r['fwer'] for r in subset)
            sums[method]=dict(true_support_mean=float(np.mean([r['true_support'] for r in subset])),
                fwer_count=k,fwer_rate=k/128,fwer_wilson95=wilson(k,128),
                missed_mean=float(np.mean([r['missed'] for r in subset])),
                conditional_gap_mean=(float(np.mean([r['oracle_gap'] for r in subset]))
                                      if subset[0]['oracle_gap'] is not None else None))
        score=np.array([r['true_support'] for r in rows if r['method']=='split_score'])
        oracle_obs=np.array([r['true_support'] for r in rows if r['method']=='split_oracle'])
        diff=oracle_obs-score
        summaries[name]=dict(truth=truth.tolist(),sigma_hit_mm=sigma,total_events=int(COUNTS.sum()),
            methods=sums,split_power=split_power.tolist(),full_family_power=full_power.tolist(),
            expected_split_oracle=float(sum(split_power[j] for j in oracle if true[j])),
            expected_full_oracle=float(sum(full_power[j] for j in full_oracle if true[j])),
            paired_oracle_minus_score_mean=float(diff.mean()),
            paired_difference_mcse=float(diff.std(ddof=1)/math.sqrt(128)))
    np.savez_compressed(out/'mode_energies.npz',energy=energies,counts_total=COUNTS)
    dump(out/'seed_keys.json',dict(engine='numpy.PCG64',seedsequence_keys=seed_keys))
    dump(out/'physical_inputs.json',matrices)
    dump(out/'trial_records.json',records)
    dump(out/'candidate_summary.json',summaries)
    with (out/'candidate_comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['scene','method','true_support_mean','fwer_count',
                            'fwer_rate','missed_mean','conditional_gap_mean'])
        writer.writeheader()
        for name,summary in summaries.items():
            for method,row in summary['methods'].items():
                writer.writerow(dict(scene=name,method=method,**{k:row[k] for k in writer.fieldnames[2:]}))
    return dict(total_replicates=768,methods_per_replicate=8,
                proposal_all_methods_cpu_seconds=proposal_seconds)


def analytical_checks():
    checks={}
    for sigma in (.3,1.):
        c0,d,a,transform=physical_modes(sigma)
        r=a/(1+a); efficiency=float(r.sum()**2/(2*(r*r).sum()))
        w=lr_weights(a)
        cases=[]
        for n in (2,20):
            qcrit=chi2_isf_even(.025,4*n)
            lrcrit=weighted_isf(.025,n,*w)
            for u in (0.,1.,1.4,3.):
                b=(1+u*a)/(1+a)
                cases.append(dict(n=n,u=u,Q_power=checked_tail(qcrit,n,b),
                    LR_simple_alt3_power=checked_tail(lrcrit,n,w*b)))
        checks[str(sigma)]=dict(a=a.tolist(),Q_local_efficiency_vs_score=efficiency,
                              tail_comparison=cases)
    z=(-400,-300,300,400); v=np.array([-1,3,-3,1])
    kernels=np.array([scattering_covariance_per_density((0,0,0,0,-100,100),
        BoxVoxel(-1,1,-1,1,lo,lo+50),z,4.) for lo in (-100,-50,0,50)])
    null=np.einsum('i,ijk->jk',v,kernels)
    budget=256*np.finfo(float).eps*max(1,np.linalg.norm(kernels.reshape(4,-1),ord=np.inf))
    if np.max(abs(null))>budget: raise ArithmeticError('depth negative-control mismatch')
    checks['negative_controls']=dict(depth_null_max_abs=float(np.max(abs(null))),
        rounding_budget=budget,background_target_identical_derivatives='algebraic non-identifiability; no p-values assigned',
        unobserved_kernel='D=0 gives zero Fisher but Q noise test still has nominal size')
    crit=chi2_isf_even(.025,160)
    checks['exploratory_misspecification']=dict(N=40,relative_covariance_excess=.05,
           nominal_alpha=.025,true_null_rejection=chi2_sf_even(crit/1.05,160),
           limit_fixed_positive_excess_large_N=1.)
    return checks


def solver_cost():
    c0,d,_,_=physical_modes(.3)
    truth=np.array([.2,.5,1.,3.,.3,1.4,.7,4.])
    start=time.process_time(); groups=[]
    for j,n in enumerate(COUNTS):
        chol=np.linalg.cholesky(c0+truth[j]*d)
        y=np.tile(2*chol.T,(int(n)//4,1))
        ds=np.zeros((8,4,4)); ds[j]=d
        groups.append(GaussianObservationGroup(str(j),y,c0,ds))
    preparation=time.process_time()-start
    coarse=[GaussianObservationGroup(g.name,g.residuals_mm,c0,d[None]) for g in groups]
    records=[]
    for repeat in range(5):
        for name in ('zero','uniform','coarse_warm'):
            start=time.process_time(); coarse_result=None
            if name=='coarse_warm':
                coarse_result=fit_covariance_detailed(coarse,initial=[1.])
                if not coarse_result.converged: raise ArithmeticError('coarse warm-start failed')
                initial=np.full(8,coarse_result.parameters[0])
            else: initial=np.full(8,0. if name=='zero' else 1.)
            fit=fit_covariance_detailed(groups,initial=initial)
            fit_cpu=time.process_time()-start
            start=time.process_time()
            f,g=grouped_gaussian_nll_and_gradient(groups,fit.parameters)
            independent_kkt=float(np.max(abs(np.maximum(1,initial)*np.where(
                fit.parameters>0,g,np.minimum(g,0)))))
            certification=time.process_time()-start
            if not fit.converged or independent_kkt>1e-6 or abs(f-fit.nll)>1e-7:
                raise ArithmeticError('solver cost comparison did not meet fixed certification')
            records.append(dict(repeat=repeat,initialization=name,initial=initial.tolist(),
                parameters=fit.parameters.tolist(),nll=f,kkt=independent_kkt,
                iterations=fit.iterations,status=fit.status,fit_cpu_seconds=fit_cpu,
                certification_cpu_seconds=certification,
                coarse_fit_iterations=coarse_result.iterations if coarse_result else None))
    if np.ptp([r['nll'] for r in records])>1e-7:
        raise ArithmeticError('initializations differ beyond NLL budget')
    start=time.process_time(); objective=CovarianceObjective(groups)
    for _ in range(20): objective.evaluate(np.ones(8))
    objective_time=(time.process_time()-start)/20
    _,_,a,transform=physical_modes(.3)
    start=time.process_time()
    for _ in range(20):
        for g,n in zip(groups,COUNTS):
            x=g.residuals_mm[:,:2]@transform.T
            y=g.residuals_mm[:,2:]@transform.T
            Q_pvalue(np.sum(x*x+y*y,axis=0),n,a)
    audit_time=(time.process_time()-start)/20
    return dict(task='deterministic 8-dimensional covariance MLE; not a generic MAP benchmark',
        observations='S_j=n_j*C_j exactly; not random scans',preparation_cpu_seconds=preparation,
        objective_score_hessian_fisher_seconds=objective_time,
        all8_Qaudit_seconds=audit_time,records=records)


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not args.run_id.startswith('run-') or Path(args.run_id).name!=args.run_id:
        raise ValueError('simple run ID required')
    out=HERE/'results'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    start=time.process_time(); status='started'
    sources=[HERE/'DIAGNOSTIC_PLAN.md',HERE/'diagnostics.py',HERE/'test_diagnostics.py',HERE/'budget_run.py']
    sources += [ROOT/'sci_method/src/sci_method'/f for f in
                 ('forward_model.py','split_merge.py','covariance_optimizer.py')]
    inputs=[ROOT/'sci_method/studies/ml-credibility-stage0-001'/f for f in
             ('MATHEMATICAL_REVIEW.md','METHODOLOGY_REVIEW.md','CHECK_SPEC.md','NEXT_STAGE_PLAN.md','VERIFICATION.md')]
    inputs += [ROOT/'sci_method'/f for f in ('FORWARD_MODEL.md','OPTIMIZER_STABILITY_REVIEW.md','RESEARCH_EVIDENCE_INDEX.md')]
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    for path in sources: (snapshot/path.name).write_bytes(path.read_bytes())
    config=dict(classification='predeclared diagnostic; analytical LR comparison exploratory',
        baseline='b57ba0e8bbe2c930a7b473a34a0d90901708bda1',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        command=sys.argv,python=sys.version,executable=sys.executable,numpy=np.__version__,
        platform=platform.platform(),p_gev_c=4.,lambda_ref_mm_inv=.01,u_star=1.,
        alpha=.05,K=2,counts=COUNTS.tolist(),replicates=128,scenes=SCENES,
        rng='PCG64; SeedSequence([20261029,scene,replicate,part])',
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    dump(out/'config.json',config)
    try:
        dump(out/'analytical_checks.json',analytical_checks())
        dump(out/'candidate_execution.json',candidate_study(out))
        dump(out/'solver_cost.json',solver_cost())
        status='completed'
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc()); status='failed'
        raise
    finally:
        dump(out/'state.json',dict(status=status,cpu_seconds=time.process_time()-start,
            rusage_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            rusage_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime))
        dump(out/'evidence.json',dict(code_commit=config['code_commit'],
            sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(status=status,result=str(out),cpu_seconds=time.process_time()-start)))


if __name__=='__main__': main()
