"""按生成前方案配对复现旧合成抽样，再检查两个开发性修正。"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from synthetic_subcell_risk import squared_risk, weighted_population_target
from finite_sample_subcell_risk import repaired_reconstruct
from run_synthetic_subcell_risk import COUNTS, TRIALS, SEED, PROBABILITIES, REGIMES, MEAN_WEIGHTS, paired_difference

ROOT=Path(__file__).resolve().parent


def run():
    archive_path=ROOT/'synthetic_subcell_risk_results/results.json'
    archive=json.loads(archive_path.read_text())
    for name,digest in archive['code_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise AssertionError('Old synthetic core changed; do not replace archived hash')
    keys=lambda r:(r['probability_case'],r['weight_regime'],r['event_count'])
    old={keys(r):r for r in archive['results']}
    results=[]
    for pi,(name,probability) in enumerate(PROBABILITIES.items()):
        for ri,regime in enumerate(REGIMES):
            truth=weighted_population_target(probability,MEAN_WEIGHTS[regime])
            for ni,count in enumerate(COUNTS):
                # 与旧入口一致，方法间共享事件和权重，不根据结果另抽样。
                rng=np.random.default_rng(SEED+1000*pi+100*ri+ni)
                children=rng.choice(4,size=(TRIALS,count),p=probability)
                if regime=='independent_lognormal':
                    weights=rng.lognormal(mean=-.5,sigma=1.,size=children.shape)
                elif regime=='child_correlated':
                    weights=np.asarray(MEAN_WEIGHTS[regime])[children]
                else:
                    weights=np.ones_like(children,dtype=float)
                moments=[]
                for power in (1,2,3):
                    moments.append(np.stack([(weights**power*(children==child)).sum(axis=1)
                                             for child in range(4)],axis=1))
                images,d=repaired_reconstruct(*moments)
                risks={method:squared_risk(image,truth) for method,image in images.items()}
                metrics={method:{'mean_squared_risk':float(risk.mean()),
                                 'mc_standard_error':float(risk.std(ddof=1)/np.sqrt(TRIALS))}
                         for method,risk in risks.items()}
                previous=old[(name,regime,count)]
                for method in ('coarse','raw','neff','candidate'):
                    for metric,value in metrics[method].items():
                        if not np.isclose(value,previous['methods'][method][metric],rtol=1e-12,atol=1e-12):
                            raise AssertionError(f'Archive mismatch: {name}/{regime}/{count}/{method}')
                # 仅独立类别模型有条件方差的解析参照，相关配置不冒称无偏。
                variance_check=None
                if count>1 and regime!='child_correlated':
                    expected=d['s2']*(1.-np.square(probability).sum())
                    variance_check=paired_difference(d['variance_corrected'],expected)
                results.append({'probability_case':name,'weight_regime':regime,'event_count':count,
                    'trials':TRIALS,'weighted_target':truth.tolist(),'methods':metrics,
                    'paired_minus_neff':{m:paired_difference(risks[m],risks['neff']) for m in ('corrected','guarded')},
                    'guarded_minus_corrected':paired_difference(risks['guarded'],risks['corrected']),
                    'mean_alpha':{m:float(d[m+'_alpha'].mean()) for m in ('corrected','guarded')},
                    'fallback_fraction':float((~d['correction_valid']).mean()),
                    'mean_correction_factor_when_valid':float(d['correction_factor'][d['correction_valid']].mean())
                        if d['correction_valid'].any() else None,
                    'conditional_variance_bias_check':variance_check})
    if len(results)!=63 or len({keys(r) for r in results})!=63:
        raise AssertionError('Incomplete paired design')
    single=next(r for r in results if keys(r)==('uniform','equal',1))
    focus=[r for r in results if r['probability_case']!='uniform' and r['weight_regime']=='equal'
           and r['event_count'] in (2,4,8)]
    decisions=[{'probability_case':r['probability_case'],'event_count':r['event_count'],
                **r['paired_minus_neff']['guarded'],
                'passes':r['paired_minus_neff']['guarded']['mean']<=1e-12} for r in focus]
    sanity=single['paired_minus_neff']['guarded']['mean']<=1e-12
    screening=len(decisions)==6 and all(r['passes'] for r in decisions)
    verdict={'single_event_sanity_passed':sanity,'nonuniform_focus_groups':len(decisions),
             'focus_groups_passing':sum(r['passes'] for r in decisions),
             'development_screening_passed':sanity and screening,
             'next_action':'paired Geant4 development only' if sanity and screening
                           else 'remain synthetic; inspect shrinkage bias before Geant4',
             'scope':'development screening, not unseen physics validation or imaging innovation'}
    # 同一局部观测能来自两种不同总体。真值只在这里作解释性评分。
    example_images,example_d=repaired_reconstruct([1,1,0,0],[1,1,0,0],[1,1,0,0])
    ambiguity={'observed_counts':[1,1,0,0],
               'alpha_guarded':float(example_d['guarded_alpha']),
               'alpha_neff':float(example_d['neff_alpha']),
               'risk_given_truth':{
                   label:{m:float(squared_risk(example_images[m],truth)) for m in ('neff','guarded')}
                   for label,truth in (('uniform',np.full(4,.25)),('two_child_boundary',np.array([.5,.5,0,0])))}}
    report={'config':archive['config'],'original_archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'old_method_alignment_records':63*4,
            'code_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                           for n in ('finite_sample_subcell_risk.py','run_finite_sample_subcell_risk.py')},
            'results':results,'focus_decisions':decisions,'verdict':verdict,
            'same_observation_ambiguity':ambiguity}
    out=ROOT/'finite_sample_subcell_risk_results'
    out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
    for pi,name in enumerate(PROBABILITIES):
        for ri,regime in enumerate(REGIMES):
            part=[r for r in results if r['probability_case']==name and r['weight_regime']==regime]
            ax=axes[pi,ri]
            for method in ('corrected','guarded'):
                ax.errorbar(COUNTS,[r['paired_minus_neff'][method]['mean'] for r in part],
                    yerr=[r['paired_minus_neff'][method]['mc_standard_error'] for r in part],
                    fmt='o-',ms=3,capsize=2,label=f'{method} - neff')
            ax.axhline(0,color='black',lw=.7)
            ax.set_xscale('symlog',linthresh=1)
            ax.set_xlim(-.1,160)  # errorbar在变换后自动边距过大；无负事件数。
            ax.set(title=f'{name}\n{regime}',xlabel='Events per coarse cell',ylabel='Paired squared-risk difference')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'finite_sample_risk.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'verdict':verdict,'focus_decisions':decisions},indent=2))


if __name__=='__main__':
    run()
